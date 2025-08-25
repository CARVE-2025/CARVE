import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from pytorch_lightning.utilities.rank_zero import rank_zero_info

class Global_Graph_Encoder(nn.Module):
    """Shared global graph encoder"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn_layer = GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        x = data.ast_x
        rest_ast_edge_index = data.rest_ast_edge_index
        all_edges_to_remove_tensors = [e for e in data.all_edges_to_remove_tensors]

        # ===== Processing AST graph =====
        x = self.process_graph(x, all_edges_to_remove_tensors, self.gcn_layer)
        x = self.gcn_layer(x, rest_ast_edge_index)
        x = F.relu(x)

        return x

    def process_graph(self, x, all_edges_to_remove_tensors, gcn_layer):
        initial_x = x
        for i in range(len(all_edges_to_remove_tensors)):
            residual = x
            edges = all_edges_to_remove_tensors[i]

            x = gcn_layer[i](x, edges)
            x = F.layer_norm(x, x.size()[1:])
            x = F.gelu(x)

            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
            else:
                x = x + nn.Linear(residual.shape[-1], x.shape[-1]).to(x.device)(residual)

        x = x + initial_x
        return x

class Local_Graph_Encoder(nn.Module):
    """Shared local graph encoder"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Conv1 = GCNConv(input_dim, hidden_dim)
        self.Conv2 = GCNConv(input_dim, hidden_dim)
        self.Conv3 = GCNConv(input_dim, hidden_dim)

    def forward(self, region_features, region_edges, region_features_batch):
        region_edges = region_edges.to(torch.int64)
        region_x = self.Conv1(region_features, region_edges)
        region_x = F.gelu(region_x)

        region_x = self.Conv2(region_x, region_edges)
        region_x = F.gelu(region_x)

        region_x = self.Conv3(region_x, region_edges)
        region_x = F.gelu(region_x)

        region_x = global_mean_pool(region_x, region_features_batch)

        return region_x

class AutoMaskGenerator(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, embeddings):
        """
        Input: embeddings [B, L, D]
        Output: mask [B, L] (True indicates the filled position)
        """
        # Calculate the L2 norm at each position
        norms = torch.norm(embeddings, p=2, dim=-1)  # [B, L]

        # Automatically detect fill positions (norm close to 0)
        mask = norms < self.epsilon  # [B, L]

        # Make sure there is at least one valid position
        all_padded = mask.all(dim=1)
        mask[all_padded, 0] = False

        return mask

class IntraMHAFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads: int = 4):
        super().__init__()
        # Multi‐Head Attention fusion over the two modalities
        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        # Learnable weights for modalities
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5]))  # shape: [2]

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor):
        """
        global_feat: (B, R, D)
        local_feat:  (B, R, D)
        returns fused: (B, R, D)
        Pre-training stage B is 1.
        """
        B, R, D = global_feat.size()

        # Stack modalities: shape (B, 2, R, D)
        stacked = torch.stack([global_feat, local_feat], dim=1)

        # Softmax weights over modality dimension, then reshape to broadcast
        # alpha: [2] -> [1, 2, 1, 1]
        weights = self.alpha.softmax(dim=0).view(1, 2, 1, 1)

        # Apply weights: (B,2,R,D) * (1,2,1,1) -> (B,2,R,D)
        weighted = stacked * weights

        # Prepare for nn.MultiheadAttention which expects (L (=2), N (=B*R), E (=D))
        # First reshape to (2, B*R, D)
        attn_input = weighted.permute(1, 0, 2, 3).reshape(2, B * R, D)

        # Self‐attention
        attn_output, _ = self.fusion_attn(
            query=attn_input,
            key=attn_input,
            value=attn_input
        )  # shape: (2, B*R, D)

        # Bring back to (B, R, D): average the two modality outputs
        attn_output = attn_output.view(2, B, R, D).mean(dim=0)  # (B, R, D)

        return attn_output

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_q = nn.Linear(dim, dim)
        self.W_kv = nn.Linear(dim, dim * 2)  # Merge K, V to generate

    def forward(self, global_feat, local_feat):
        # Projection
        Q = self.W_q(global_feat)  # [B, R, D]
        K, V = self.W_kv(local_feat).chunk(2, dim=-1)  # Split

        # Attention calculation
        attn = torch.matmul(Q, K.transpose(1, 2)) / (self.dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Residual connection
        return global_feat + torch.matmul(attn, V)  # [B, R, D]

class RegionAwareModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Global Graph Structure Encoder
        self.Global_Graph_Encoder = Global_Graph_Encoder(args.input_dim, args.hidden_dim)

        # Local Graph Structure Encoder
        self.Local_Graph_Encoder = Local_Graph_Encoder(args.input_dim, args.hidden_dim)

        # Automatic Mask Generator
        self.mask_gen = AutoMaskGenerator()

        # Global semantic encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=4,
            dim_feedforward=4 * args.hidden_dim,
            dropout=0.2,
            activation= 'gelu',
            layer_norm_eps= 1e-5,
            batch_first=True,
        )
        self.global_semantic_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # Local semantic encoder
        self.local_semantic_encoder = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=2,
            dim_feedforward=2 * args.hidden_dim,
            dropout=0.2,
            activation='gelu',
            layer_norm_eps=1e-5,
            batch_first=True
        )

        # Fusion Layer
        self.Intrafusion = IntraMHAFusion(args.hidden_dim)
        self.Crossfusion = CrossAttentionFusion(args.hidden_dim)

        # Comparison of learning projection heads
        self.projection = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, 128)
        )

        # Regression prediction head
        self.regressor = nn.Linear(args.hidden_dim, 1)

    def forward(self, sample_data, graph_data):
        # =================Global graph structure learning=================
        ast_x = self.Global_Graph_Encoder(graph_data)
        region_node_indices = sample_data['node_indices']

        global_region_features = self.region_features_extract(ast_x, region_node_indices)  # [R, D]
        # print(f"global_region_features:{global_region_features.shape}")

        # =================Local graph structure learning=================
        region_features = sample_data['features']
        region_features_batch = sample_data['features_batch']
        region_edges = sample_data['edges']

        local_region_features = self.Local_Graph_Encoder(region_features, region_edges, region_features_batch)  # [R, D]
        # print(f"local_region_features:{local_region_features.shape}")

        # =================Global semantic learning=================
        global_code_emb = graph_data.global_code_embeddings    # [B, L, D]
        line_indices = sample_data['line_indices']  # [R, L_num]
        batch_indices = sample_data['batch_indices']

        # Automatically generate src_key_padding_mask encoding, True means that the position is invalid, which is the opposite of our global_mask
        safa_global_mask = self.mask_gen(global_code_emb)  # [B, L]
        global_semantic = self.global_semantic_encoder(global_code_emb, src_key_padding_mask=safa_global_mask)  # [B, L, D]

        global_semantic_features = self.extract_global_semantic_features(global_semantic, line_indices, batch_indices)  # [R, D]
        # print(f"global_semantic_features: {global_semantic_features.shape}")

        # =================Local Semantic Learning=================
        local_code_emb = sample_data['code_emb']    # [R, L, D]
        local_emb_mask = sample_data['emb_mask']    # [R, L]

        safa_local_mask = self.mask_gen(local_code_emb)  # [R, L]
        local_semantic = self.local_semantic_encoder(local_code_emb, src_key_padding_mask=safa_local_mask) # [R, L, D]

        local_semantic_features = self.extract_local_semantic_features(local_semantic, local_emb_mask)  # [R, D]
        # print(f"local_semantic_features: {local_semantic_features.shape}")

        structure_fused = self.Intrafusion(global_region_features.unsqueeze(0), local_region_features.unsqueeze(0)).squeeze(0)   # [R, D]
        semantic_fused = self.Intrafusion(global_semantic_features.unsqueeze(0), local_semantic_features.unsqueeze(0)).squeeze(0)    # [R, D]
        fused_embs = self.Crossfusion(structure_fused.unsqueeze(0), semantic_fused.unsqueeze(0)).squeeze(0)  # [R, D]

        # Contrastive Learning Projection
        contrast_emb = self.projection(fused_embs)

        # Regression prediction
        pred_scores = torch.sigmoid(self.regressor(fused_embs).squeeze(-1))

        return {
            'contrast_emb': contrast_emb,  # For contrastive learning
            'pred_scores': pred_scores,  # For regression supervision
            'modality_embs': {  # For layered comparison
                'global_graph': global_region_features,
                'local_graph': local_region_features,
                'global_semantic': global_semantic_features,
                'local_semantic': local_semantic_features
            }
        }

    def region_features_extract(self, x, region_indices):
        # Create valid node mask [sample_num, max_node_size]
        mask = region_indices != -1

        # Replace -1 with 0 to avoid index out of bounds (actually it will be filtered by mask)
        indices = region_indices.clone()
        indices[region_indices == -1] = 0

        # Extract node features [sample_num, max_node_size, d]
        region_nodes = x[indices]

        # Apply a mask to filter invalid nodes
        region_nodes = region_nodes * mask.unsqueeze(-1).float()  # [sample_num, max_node_size, d]

        # Calculate the number of valid nodes [sample_num,1]
        valid_counts = torch.sum(mask.float(), dim=1, keepdim=True)
        valid_counts = valid_counts.clamp(min=1e-6)  # 防止除零

        # Weighted average calculation of regional characteristics
        sum_features = torch.sum(region_nodes, dim=1)  # [sample_num, d]
        region_features = sum_features / valid_counts  # [sample_num, d]

        return region_features

    def extract_global_semantic_features(self, encoded, line_indices, batch_indices):
        """
        New feature extraction function
        Parameters:
        encoded: [B, L, D] Encoded global semantic features
        line_indices: [R, L_num] The row index matrix corresponding to each region
        batch_indices: [R] The original batch to which each region belongs
        region_indices: [R] The index of each region within the original batch
        """
        # Get the maximum valid index
        max_valid_idx = encoded.shape[1] - 1
        # Create a mask to mark invalid indices (> max_valid_idx)
        invalid_mask = (line_indices > max_valid_idx)
        # Replace illegal indices with -1 (or fill value)
        line_indices = torch.where(invalid_mask, torch.full_like(line_indices, -1), line_indices)
        # Safely handling padding values
        adj_line_indices = line_indices.clone()
        adj_line_indices[line_indices == -1] = 0  # Replace fill value with valid index

        # Get the global row features for each region [R, L_num, D]
        region_features = encoded[batch_indices[:, None], adj_line_indices, :]

        # Create a valid feature mask [R, L_num]
        valid_mask = line_indices != -1

        # Weighted Average Pooling
        summed_features = region_features * valid_mask.unsqueeze(-1).float()  # [sample_num, max_node_size, d]
        valid_counts = torch.sum(valid_mask.float(), dim=1, keepdim=True)
        valid_counts = valid_counts.clamp(min=1e-6)  # Preventing division by zero
        # Weighted average calculation of regional characteristics
        sum_features = torch.sum(summed_features, dim=1)  # [sample_num, d]
        region_features = sum_features / valid_counts  # [sample_num, d]

        return region_features

    def extract_local_semantic_features(self, local_semantic, local_emb_mask):
        """
        New version of feature extraction function
        Parameters:
        local_semantic: [R, L, D] Encoded local semantic features
        local_emb_mask: [R, L] Valid index matrix corresponding to each region
        """
        # Apply mask to filter invalid rows
        region_lines = local_semantic * local_emb_mask.unsqueeze(-1).float()  # [R, L, D]

        # Calculate the number of valid nodes [R,1]
        valid_counts = torch.sum(local_emb_mask.float(), dim=1, keepdim=True)
        valid_counts = valid_counts.clamp(min=1e-6)  # Preventing division by zero

        # Weighted average calculation of regional characteristics
        sum_features = torch.sum(region_lines, dim=1)  # [R, D]
        region_features = sum_features / valid_counts  # [R, D]

        return region_features

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_mu = nn.Parameter(torch.randn(out_dim, in_dim))
        self.w_rho = nn.Parameter(torch.randn(out_dim, in_dim))
        self.b_mu = nn.Parameter(torch.randn(out_dim))
        self.b_rho = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        # Reparameterization Techniques
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)

        b_sigma = torch.log1p(torch.exp(self.b_rho))
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)

        return F.linear(x, w, b)

class VulDetectionModel(nn.Module):
    def __init__(self, pretrained_path, args, freeze_backbone=True):
        super().__init__()
        try:
            from training import ContrastiveLearner
            # Loading the full model via ContrastiveLearner
            cl_model = ContrastiveLearner.load_from_checkpoint(
                pretrained_path,
                args=args,
                map_location='cpu',
                strict=False
            )
            target_device = f"cuda:{args.gpus[0]}" if args.gpus else "cpu"
            cl_model = cl_model.to(target_device)
            self.pretrained_model = cl_model.encoder
            rank_zero_info(f"\n✅ Successfully loaded cl pretrained model from: {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {str(e)}") from e

        # Fusion Layer
        self.Intrafusion = IntraMHAFusion(args.hidden_dim)
        self.Crossfusion = CrossAttentionFusion(args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            BayesianLinear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_data):
        # print(f"graph_data information:{graph_data}")
        # =================Global graph structural features=================
        ast_x = self.pretrained_model.Global_Graph_Encoder(graph_data)  # [N, D]

        region_node_indices = graph_data.node_index_lists  # [B, R, max_nodes]

        global_graph_feats = self._batch_region_features(ast_x, region_node_indices)  # [B, R, D]
        # print(f"global_graph_feats:{global_graph_feats.shape}")

        # =================Local graph structural features=================
        region_features = graph_data.region_features  # [B, R, M, D]
        region_edges = graph_data.region_edges  # [2, E]
        B, R, M, D = region_features.shape

        # Flatten all node features
        flat_features = region_features.view(-1, D)  # [B*R*M, D]

        # Generate region batch identification
        region_batch = torch.arange(B * R, device=region_features.device)
        region_batch = region_batch.repeat_interleave(M).view(B, R, M)
        region_batch = region_batch.reshape(-1)  # Final shape (B*R*M,)

        # Call Graph Encoder
        local_graph = self.pretrained_model.Local_Graph_Encoder(flat_features, region_edges, region_batch)  # [B*R, D]
        local_graph_feats = local_graph.view(B, R, -1)  # [B, R, D]
        # print(f"local_graph_feats:{local_graph_feats.shape}")

        # =================Global semantic features=================
        global_code_emb = graph_data.global_code_embeddings    # [B, L, D]
        # Generate security mask (adapt pre-trained model interface)
        safa_global_mask = self.pretrained_model.mask_gen(global_code_emb)  # [B, L]
        line_indices = graph_data.region_line_numbers_lists # [B, R, l]

        global_semantic = self.pretrained_model.global_semantic_encoder(global_code_emb, src_key_padding_mask=safa_global_mask)  # [B, L, D]
        global_semantic_feats = self._batch_extract_global_semantic(global_semantic, line_indices) # [B, R, D]
        # print(f"global_semantic_feats:{global_semantic_feats.shape}")

        # =================Local semantic features=================
        local_code_emb = graph_data.region_code_embeddings  # [B, R, l, D]
        local_emb_mask = graph_data.region_emb_mask  # [B, R, l]

        # Merge batch and region dimensions
        B, R, l, D = local_code_emb.shape
        merged_emb = local_code_emb.view(B * R, l, D)  # [B*R, l, D]
        # Generate security mask (adapt pre-trained model interface)
        safa_local_mask = self.pretrained_model.mask_gen(merged_emb)  # [B*R, L]

        local_semantic = self.pretrained_model.local_semantic_encoder(merged_emb, src_key_padding_mask=safa_local_mask)  # [B*R, l, D]
        local_semantic = local_semantic.view(B, R, -1, D)   # [B, R, l, D]
        local_semantic_feats = self._batch_extract_local_semantic(local_semantic, local_emb_mask)   # [B, R, D]
        # print(f"local_semantic_feats:{local_semantic_feats.shape}")

        # ================= Feature Fusion =================
        structure_fused = self.Intrafusion(global_graph_feats, local_graph_feats)
        semantic_fused = self.Intrafusion(global_semantic_feats, local_semantic_feats)
        fused_embs = self.Crossfusion(structure_fused, semantic_fused)

        # ================= Regional Weighted Fusion =================
        region_scores = torch.sigmoid(self.pretrained_model.regressor(fused_embs))  # [B, R, 1]
        weighted_emb = torch.sum(fused_embs * region_scores, dim=1)  # [B, D]

        # ================= Classification prediction =================
        outputs = self.classifier(weighted_emb).squeeze(-1)
        # print(f"outputs:{outputs.shape}")
        return outputs

    def _batch_region_features(self, ast_x, region_indices):
        """Batch structural feature extraction"""
        B, R, M = region_indices.shape
        d = ast_x.size(-1)

        # Create a mask to identify valid nodes (excluding -1)
        mask = (region_indices != -1)  # (B, R, M)

        # Replace invalid indexes with 0 to avoid index out of bounds
        safe_indices = region_indices.clone().masked_fill(~mask, 0)  # Replace -1 with 0

        # Collect node features (B, R, M, D)
        flat_indices = safe_indices.view(-1)  # Flatten to 1D index

        features = ast_x[flat_indices].view(B, R, M, d)

        # Apply mask to zero out padding positions
        masked_features = features * mask.unsqueeze(-1).to(features.dtype)  # (B, R, M, D)

        # Calculate the number of valid nodes and avoid division by zero
        valid_counts = mask.sum(dim=2, keepdim=True)  # (B, R, 1)
        valid_counts = valid_counts.clamp(min=1)  # Handle the situation where all the items are filled

        # Calculate the average features (B, R, D)
        region_embeddings = masked_features.sum(dim=2) / valid_counts

        return region_embeddings

    def _batch_extract_global_semantic(self, encoded, line_indices):
        """Batch semantic feature extraction"""
        B, L, D = encoded.shape
        B_r, R, l = line_indices.shape

        # Get the maximum valid index
        max_valid_idx = encoded.shape[1] - 1
        # Create a mask to mark illegal indices (>max_valid_idx)
        invalid_mask = (line_indices > max_valid_idx)
        # Replace illegal indices with -1 (or fill value)
        line_indices = torch.where(invalid_mask, torch.full_like(line_indices, -1), line_indices)

        # Generate valid row mask (B, R, l)
        valid_mask = (line_indices != -1)

        # Safe index processing (replace -1 with 0)
        safe_indices = line_indices.masked_fill(~valid_mask, 0)  # (B, R, l)

        # Generate batch index
        batch_indices = torch.arange(B, device=encoded.device)[:, None, None]  # (B, 1, 1)

        batch_indices = batch_indices.expand(-1, R, l)  # (B, R, l)

        # 3D index collection features (B, R, l, D)
        region_semantic = encoded[batch_indices, safe_indices]

        # Apply mask and calculate running mean
        masked_features = region_semantic * valid_mask[..., None].to(encoded.dtype)  # (B,R,l,D)

        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B,R,1)

        region_embeddings = masked_features.sum(dim=2) / valid_counts  # (B,R,D)

        return region_embeddings

    def _batch_extract_local_semantic(self, local_semantic, local_emb_mask):
        # Expand mask dimensions to match features
        valid_mask = local_emb_mask.unsqueeze(-1)  # [B, R, l, 1]
        mask_tensor = valid_mask.to(local_semantic.dtype)  # [B, R, l, 1]

        # Apply mask and sum
        masked_features = local_semantic * mask_tensor  # [B, R, l, D]
        sum_features = masked_features.sum(dim=2)  # [B, R, D]

        # Calculate the number of valid elements
        valid_counts = valid_mask.sum(dim=2).float()  # [B, R, 1]
        valid_counts = valid_counts.clamp(min=1)

        # Perform safety averaging
        local_semantic_feats = sum_features / valid_counts  # [B, R, D]

        return local_semantic_feats