import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import List

import pytorch_lightning as pl
import torch
from data_loading import GlobalDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from training import ContrastiveLearner, VulDetectionSystem

torch.set_float32_matmul_precision('medium')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', help='List of GPU indices to use')
    parser.add_argument('--data_path', type=str, default="FFMPeg+Qemu.pkl")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--clip_model_path', type=str, default=None, help='Path to cli pretrained checkpoint')
    parser.add_argument('--pre_clip_model_path', type=str, default=None,
                        help="If you want to continue training from a certain interruption point, specify the .ckpt file path")
    parser.add_argument('--cls_model_path', type=str, default=None,
                        help='(Optional) Path to pretrained classification checkpoint for direct testing')
    parser.add_argument('--cwe_mode', action='store_true',
                        help='If set, treat data_path as a directory of CWE .pkl files and merge them')
    parser.add_argument('--sim_threshold', type=int, default=0.7)
    parser.add_argument('--delt1', type=int, default=0.2)
    parser.add_argument('--delt2', type=int, default=0.3)

    args = parser.parse_args()
    rank_zero_info(vars(args))

    # ================= Data loading =================
    data_module = GlobalDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_regions=5,
        emb_dim=128,
        seed=42,
        sim_threshold=args.sim_threshold,
        delt1=args.delt1,
        delt2=args.delt2,
        cwe_mode=args.cwe_mode
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # ================= Model configuration =================
    model_args = ModelArgs(
        hidden_dim=128,
        max_epochs=args.max_epochs,
        gpus=args.gpus
    )
    lr = 1e-4

    # If there is no pre-trained contrastive model, retrain one, otherwise just load the clip model directly.
    if not args.clip_model_path:
        # ================= Stage 1: Contrastive Learning Pre-training =================
        if args.pre_clip_model_path:  # If a model that was previously interrupted during training is specified, continue training for the remaining epochs.
            contrast_model = ContrastiveLearner.load_from_checkpoint(
                args.pre_clip_model_path,
                model_args=model_args,
                lr=lr
            )
        else:
            # Start training from scratch as usual.
            contrast_model = ContrastiveLearner(model_args, lr)

        # Configuration comparison and learning of the best callback
        contrast_best_checkpoint = ModelCheckpoint(
            filename="best-contrast-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        # Configure and compare each callback
        contrast_all_checkpoint = ModelCheckpoint(
            filename="{epoch:03d}",
            every_n_epochs=1,
            save_top_k=-1
        )

        contrast_trainer = pl.Trainer(
            accelerator='gpu' if args.gpus else 'cpu',
            devices=args.gpus if args.gpus else "auto",
            # strategy='ddp_find_unused_parameters_true' if args.gpus else 'auto',
            max_epochs=args.max_epochs,
            callbacks=[contrast_best_checkpoint, contrast_all_checkpoint],
            log_every_n_steps=10,
            deterministic=True,
            logger=pl.loggers.CSVLogger(save_dir=f"./cl_logs"),
            enable_progress_bar=True
        )

        # Performing contrastive learning training
        contrast_trainer.fit(
            contrast_model,
            train_dataloaders=data_module.contrast_train_dataloader(),
            val_dataloaders=data_module.contrast_val_dataloader()
        )
        # Load the best comparison model
        last_contrast_path = contrast_best_checkpoint.best_model_path
    else:
        last_contrast_path = args.clip_model_path

    # ================= Stage 2: Classification task training =================
    if args.cls_model_path:  # If a trained classification model is provided, skip training and proceed directly to testing.
        cls_model = VulDetectionSystem(last_contrast_path, model_args, lr)
        rank_zero_info(
            f"\n🔍 cls_model_path provided, skipping training. Testing with checkpoint: {args.cls_model_path}")
        tester = pl.Trainer(
            accelerator='gpu' if args.gpus else 'cpu',
            devices=args.gpus if args.gpus else "auto",
        )
        tester.test(
            cls_model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=args.cls_model_path
        )
        return

    rank_zero_info(f"\n🔍 Loading contrast model from: {last_contrast_path}")
    # Initializing the classification model
    cls_model = VulDetectionSystem(last_contrast_path, model_args, lr)

    # Configure classification task callback
    acc_checkpoint = ModelCheckpoint(
        monitor='val_acc',
        filename='best-cls-{epoch}-{val_acc:.2f}',
        mode='max',
        save_top_k=1
    )

    rec_checkpoint = ModelCheckpoint(
        monitor='val_rec',
        filename='best-cls-{epoch}-{val_rec:.2f}',
        mode='max',
        save_top_k=1
    )

    f1_checkpoint = ModelCheckpoint(
        monitor='val_f1',
        filename='best-cls-{epoch}-{val_f1:.2f}',
        mode='max',
        save_top_k=1
    )

    last_checkpoint = ModelCheckpoint(
        filename='last-{epoch}',
        save_last=True,
        save_on_train_epoch_end=True
    )

    cls_trainer = pl.Trainer(
        accelerator='gpu' if args.gpus else 'cpu',
        devices=args.gpus if args.gpus else "auto",
        strategy='ddp_find_unused_parameters_true' if args.gpus else 'auto',
        max_epochs=args.max_epochs,
        callbacks=[acc_checkpoint, rec_checkpoint, f1_checkpoint, last_checkpoint],
        logger=pl.loggers.CSVLogger(save_dir="./cls_logs"),
        enable_progress_bar=True
    )

    # Perform classification training
    cls_trainer.fit(
        cls_model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )

    # ================= Final Test =================
    rank_zero_info("\n🔥 Running Final Test")
    # Classification Test
    rank_zero_info("\n=== Starting classification testing ===")
    ckpt_paths = {
        "Best Acc": acc_checkpoint.best_model_path,
        "Best Rec": rec_checkpoint.best_model_path,
        "Best F1": f1_checkpoint.best_model_path,
        "Last Model": last_checkpoint.last_model_path
    }

    for model_name, path in ckpt_paths.items():
        if not path:
            rank_zero_info(f"✅ Checkpoint for {model_name} not found, skipping test")
            continue

        rank_zero_info(f"\n✅ Testing {model_name} model")
        cls_trainer.test(
            dataloaders=data_module.test_dataloader(),
            ckpt_path=path
        )


@dataclass
class ModelArgs:
    input_dim: int = 128  # Must strictly match AST node feature dimension
    hidden_dim: int = 128
    contrast_dim: int = 64
    cls_hidden: int = 64
    num_layers: int = 10
    max_epochs: int = 50
    gpus: List[int] = field(default_factory=lambda: [0])
    device: torch.device = field(init=False)

    def __post_init__(self):
        if self.gpus and torch.cuda.is_available():
            primary_gpu = self.gpus[0]
            self.device = torch.device(f'cuda:{primary_gpu}')
        else:
            self.device = torch.device('cpu')


if __name__ == "__main__":
    main()