import os
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
import torchmetrics
import torch
from model import RegionAwareModel, VulDetectionModel
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from pytorch_lightning.utilities.rank_zero import rank_zero_info

class HierContrastiveLoss(nn.Module):
    def __init__(self, model_args, temp=0.07, weights=[1.0, 0.5, 0.3, 0.1, 0.1]):
        super().__init__()
        self.temp = temp
        self.register_buffer('weights', torch.tensor(weights))
        self.modal_weights = nn.Parameter(torch.ones(5))

    def forward(self, all_outputs, batch_data):
        device = next(self.parameters()).device
        total_loss = 0
        loss_terms = {}

        # Compare sample pair types
        pair_types = ['intra_pos', 'intra_neg', 'cross_pos', 'cross_neg']

        # Dynamically traverse the existing pair types
        for ptype in pair_types:
            if ptype not in all_outputs:
                continue

            anchor_out, paired_out = all_outputs[ptype]
            if not (anchor_out and paired_out):
                continue

            # Calculate this type of loss
            if 'pos' in ptype:
                loss = self._pair_contrast(anchor_out, paired_out, is_positive=True, device=device)
                weight = self.weights[0] if 'intra' in ptype else self.weights[2]
            else:
                loss = self._pair_contrast(anchor_out, paired_out, is_positive=False, device=device)
                weight = self.weights[1] if 'intra' in ptype else self.weights[3]

            total_loss += weight.to(device) * loss.to(device)
            loss_terms[f"{ptype}_loss"] = loss.detach()

        # Regression loss calculation
        reg_loss = self._calc_reg_loss(all_outputs, batch_data, device)
        total_loss += self.weights[4].to(device)  * reg_loss.to(device)
        loss_terms["reg_loss"] = reg_loss.detach()

        return total_loss

    def _pair_contrast(self, anchor_out, paired_out, is_positive, device):
        fused_loss = 0
        a = anchor_out['contrast_emb']
        p = paired_out['contrast_emb']
        a_feat = F.normalize(a, dim=1)
        p_feat = F.normalize(p, dim=1)
        # Pairwise similarity calculation
        sim_pairs = torch.sum(a_feat * p_feat, dim=1)  # (B,)

        temp = self.temp
        sim_pairs = sim_pairs / temp

        # Loss calculation
        if is_positive:
            # Positive sample pair: maximize similarity
            fused_loss = torch.mean(torch.exp(-sim_pairs))
        else:
            # Negative sample pairs: Minimize similarity
            fused_loss = torch.mean(torch.relu(sim_pairs + 0.3))

        # Multimodal hierarchical comparison
        modalities = ['global_graph', 'local_graph',
                      'global_semantic', 'local_semantic']

        modal_losses = []
        for i, mod in enumerate(modalities):
            a = anchor_out['modality_embs'][mod]
            p = paired_out['modality_embs'][mod]
            a_feat = F.normalize(a, dim=1)
            p_feat = F.normalize(p, dim=1)

            # Pairwise similarity calculation
            sim_pairs = torch.sum(a_feat * p_feat, dim=1)  # (B,)
            sim_pairs = sim_pairs / temp

            # Modality specific losses
            if is_positive:
                loss = torch.mean(torch.exp(-sim_pairs))
            else:
                loss = torch.mean(torch.relu(sim_pairs + 0.3))

            modal_losses.append(loss.to(device)  * torch.sigmoid(self.modal_weights[i]))

        modal_losses.append(fused_loss.to(device)  * torch.sigmoid(self.modal_weights[-1]))
        final_loss = sum(modal_losses) / (len(modalities) + 1)
        return final_loss

    def _calc_reg_loss(self, all_outputs, batch_data, device):
        """Regression loss calculation"""
        reg_loss = torch.tensor(0.0, device=device)
        count = 0

        for ptype in all_outputs.keys():
            anchor_out, paired_out = all_outputs[ptype]
            if not (anchor_out and paired_out):
                continue

            score_key = f'{ptype}_pairs'
            if score_key not in batch_data or len(batch_data[score_key]) < 1:
                continue

            try:
                # Anchor point regression calculation
                anchor_pred = anchor_out['pred_scores'].to(device)
                target_anchor = batch_data[score_key][0][0]['region_score'].clone().detach().to(device).float()
                reg_loss += F.mse_loss(anchor_pred, target_anchor)
                count += 1

                # Paired regression calculation
                if len(batch_data[score_key][0]) > 1:
                    paired_pred = paired_out['pred_scores'].to(device)
                    target_paired = batch_data[score_key][0][1]['region_score'].clone().detach().to(device).float()
                    reg_loss += F.mse_loss(paired_pred, target_paired)
                    count += 1

            except KeyError as e:
                print(f"Skip {ptype}: {str(e)}")

        return reg_loss / count if count > 0 else torch.tensor(0.0, device=device)

class ContrastiveLearner(pl.LightningModule):
    """Contrastive Learning Pre-training Module"""
    def __init__(self, model_args, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model_args = model_args
        # Model initialization
        self.encoder = RegionAwareModel(model_args)
        self.criterion = HierContrastiveLoss(model_args)

    def forward(self, sample_data, graph_data):
        return self.encoder(sample_data, graph_data)

    def training_step(self, batch, batch_idx):

        graph_data = batch['graph_data']
        intra_pos_anchor_data, intra_pos_paired_data = batch['intra_pos_pairs'][0] if batch['intra_pos_pairs'] else (None, None)
        intra_neg_anchor_data, intra_neg_paired_data = batch['intra_neg_pairs'][0] if batch['intra_neg_pairs'] else (None, None)
        cross_pos_anchor_data, cross_pos_paired_data = batch['cross_pos_pairs'][0] if batch['cross_pos_pairs'] else (None, None)
        cross_neg_anchor_data, cross_neg_paired_data = batch['cross_neg_pairs'][0] if batch['cross_neg_pairs'] else (None, None)

        def process_pairs(anchor_data, paired_data):
            anchor_out = self.forward(anchor_data, graph_data)
            paired_out = self.forward(paired_data, graph_data)
            return anchor_out, paired_out

        def safe_process(anchor_data, paired_data):
            if anchor_data and paired_data and len(anchor_data) and len(paired_data) > 0:
                return process_pairs(anchor_data, paired_data)
            return None, None

        intra_pos_anchor_out, intra_pos_paired_out = safe_process(intra_pos_anchor_data, intra_pos_paired_data)
        intra_neg_anchor_out, intra_neg_paired_out = safe_process(intra_neg_anchor_data, intra_neg_paired_data)
        cross_pos_anchor_out, cross_pos_paired_out = safe_process(cross_pos_anchor_data, cross_pos_paired_data)
        cross_neg_anchor_out, cross_neg_paired_out = safe_process(cross_neg_anchor_data, cross_neg_paired_data)

        # Forward propagation of each sample pair
        outputs = {
            'intra_pos': (intra_pos_anchor_out, intra_pos_paired_out),
            'intra_neg': (intra_neg_anchor_out, intra_neg_paired_out),
            'cross_pos': (cross_pos_anchor_out, cross_pos_paired_out),
            'cross_neg': (cross_neg_anchor_out, cross_neg_paired_out)
        }

        outputs = {
            k: v for k, v in outputs.items()
            if v[0] is not None and v[1] is not None
        }

        if not outputs:
            # Maintaining the integrity of the computation graph
            dummy_loss = torch.tensor(0.0, requires_grad=True)
            dummy_loss = dummy_loss * sum(p.sum() for p in self.parameters())
            self.log('train_skip_batch', 1.0, sync_dist=True, batch_size=graph_data.y.size(0))
            return dummy_loss

        # Calculate total loss
        loss = self.criterion(outputs, batch)

        # Monitoring Metrics
        self.log('train_loss', loss, prog_bar=True, batch_size=graph_data.y.size(0), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        graph_data = batch['graph_data']
        intra_pos_anchor_data, intra_pos_paired_data = batch['intra_pos_pairs'][0] if batch['intra_pos_pairs'] else (None, None)
        intra_neg_anchor_data, intra_neg_paired_data = batch['intra_neg_pairs'][0] if batch['intra_neg_pairs'] else (None, None)
        cross_pos_anchor_data, cross_pos_paired_data = batch['cross_pos_pairs'][0] if batch['cross_pos_pairs'] else (None, None)
        cross_neg_anchor_data, cross_neg_paired_data = batch['cross_neg_pairs'][0] if batch['cross_neg_pairs'] else (None, None)

        def process_pairs(anchor_data, paired_data):
            anchor_out = self.forward(anchor_data, graph_data)
            paired_out = self.forward(paired_data, graph_data)
            return anchor_out, paired_out

        def safe_process(anchor_data, paired_data):
            if anchor_data and paired_data and len(anchor_data) and len(paired_data) > 0:
                return process_pairs(anchor_data, paired_data)
            return None, None

        intra_pos_anchor_out, intra_pos_paired_out = safe_process(intra_pos_anchor_data, intra_pos_paired_data)
        intra_neg_anchor_out, intra_neg_paired_out = safe_process(intra_neg_anchor_data, intra_neg_paired_data)
        cross_pos_anchor_out, cross_pos_paired_out = safe_process(cross_pos_anchor_data, cross_pos_paired_data)
        cross_neg_anchor_out, cross_neg_paired_out = safe_process(cross_neg_anchor_data, cross_neg_paired_data)

        # Forward propagation of each sample pair
        outputs = {
            'intra_pos': (intra_pos_anchor_out, intra_pos_paired_out),
            'intra_neg': (intra_neg_anchor_out, intra_neg_paired_out),
            'cross_pos': (cross_pos_anchor_out, cross_pos_paired_out),
            'cross_neg': (cross_neg_anchor_out, cross_neg_paired_out)
        }

        outputs = {
            k: v for k, v in outputs.items()
            if v[0] is not None and v[1] is not None
        }

        if not outputs:
            dummy_loss = torch.tensor(0.0, requires_grad=True)
            dummy_loss = dummy_loss * sum(p.sum() for p in self.parameters())  # Create fake correlations with model parameters
            self.log('val_skip_batch', 1.0, sync_dist=True, batch_size=graph_data.y.size(0))
            return dummy_loss

        loss = self.criterion(outputs, batch)

        self.log('val_loss', loss, prog_bar=True, batch_size=graph_data.y.size(0), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('threshold', torch.tensor(threshold))

    def forward(self, inputs, targets):
        p = inputs
        pt = torch.where(targets == 1, p, 1 - p)
        BCE_loss = F.binary_cross_entropy(p, targets, reduction='none')
        F_loss = self.alpha * (1-pt).pow(self.gamma) * BCE_loss
        return F_loss.mean()

class VulDetectionSystem(pl.LightningModule):
    def __init__(self, pretrained_path, model_args, lr):
        super().__init__()
        self.save_hyperparameters()
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint {pretrained_path} not found")

        self.model = VulDetectionModel(pretrained_path, model_args)
        self.lr = lr

        # Defining the loss function
        self.criterion = FocalLoss(
            alpha=0.25,
            gamma=2,
            threshold=0.5
        )

        self.train_metrics = MetricCollection({
            "acc": Accuracy(task="binary"),
            "prec": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1": F1Score(task="binary")
        })
        self.val_metrics = MetricCollection({
            "acc": Accuracy(task="binary"),
            "prec": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1": F1Score(task="binary")
        })
        self.test_metrics = MetricCollection({
            "acc": Accuracy(task="binary"),
            "prec": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1": F1Score(task="binary")
        })

    def forward(self, graph_data):
        return self.model(graph_data)

    def _log_epoch_metrics(self, metrics, prefix):
        """Dedicated to recording epoch-level indicators"""
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value,
                    prog_bar=True,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True)

    def training_step(self, batch, batch_idx):
        labels = batch.y.float()
        outputs = self(batch)

        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, batch_size=labels.size(0), sync_dist=True)

        train_preds = (outputs > self.criterion.threshold).float()
        # Update training metrics
        self.train_metrics.update(train_preds, labels)

        return loss

    def on_train_epoch_end(self):
        # Calculate and record training metrics
        metrics = self.train_metrics.compute()
        self._log_epoch_metrics(metrics, "train")
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        labels = batch.y.float()
        outputs = self(batch)

        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True,
                 batch_size=labels.size(0), sync_dist=True)

        val_preds = (outputs > self.criterion.threshold).float()
        # Update val metrics
        self.val_metrics.update(val_preds, labels)

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        metrics = self.val_metrics.compute()
        self._log_epoch_metrics(metrics, "val")
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        labels = batch.y.float()
        outputs = self(batch)

        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, batch_size=labels.size(0), sync_dist=True)

        test_preds = (outputs > self.criterion.threshold).float()

        self.test_metrics.update(test_preds, labels)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self._log_epoch_metrics(metrics, "test")
        self.test_metrics.reset()

    def configure_optimizers(self):
        # Hierarchical configuration
        optimizer = torch.optim.AdamW([
            {"params": self.model.pretrained_model.parameters(), "lr": self.lr * 0.1},
            {"params": self.model.classifier.parameters(), "lr": self.lr}
        ], weight_decay=1e-4)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.lr * 0.01,
                max_lr=self.lr,
                step_size_up=500,
                cycle_momentum=False
            ),
            "interval": "step"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}