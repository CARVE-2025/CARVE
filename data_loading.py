import os
import torch
import random
import pickle
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch_geometric.data import Dataset, Batch
from typing import Optional, List, Dict
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import hashlib
import psutil

class GlobalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 32,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            max_regions: int = 5,
            emb_dim: int = 128,
            seed: int = 42,
            sim_threshold: int = 0.7,
            delt1: int = 0.2,
            delt2: int = 0.3,
            cwe_mode: bool = False
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_regions = max_regions
        self.emb_dim = emb_dim
        self.seed = seed

        self.global_train = None
        self.global_val = None
        self.global_test = None
        self.contrast_train = None
        self.contrast_val = None

        self.sim_threshold = sim_threshold
        self.delt1 = delt1
        self.delt2 = delt2
        self.cwe_mode = cwe_mode

    def prepare_data(self):
        if self.cwe_mode:
            if not os.path.isdir(self.data_path):
                raise FileNotFoundError(f"In CWE mode, please specify a directory containing multiple .pkl files, rather than a single file.：{self.data_path}")
        else:
            if not os.path.isfile(self.data_path):
                raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    def setup(self, stage: str = None):
        if self.cwe_mode:    # If it is CWE mode: divide each .pkl under data_path separately, then concatenate them.
            train_list, val_list, test_list = [], [], []
            g = torch.Generator().manual_seed(self.seed)

            for fname in sorted(os.listdir(self.data_path)):
                rank_zero_info(f"Loading {fname}")
                if not fname.endswith(".pkl"):
                    continue
                full_path = os.path.join(self.data_path, fname)
                with open(full_path, "rb") as f:
                    data_i = pickle.load(f)

                n = len(data_i)
                n_train = int(self.train_ratio * n)
                n_val   = int(self.val_ratio   * n)
                n_test  = n - n_train - n_val

                # Randomly divide each CWE subset.
                subt = random_split(data_i, [n_train, n_val, n_test], generator=g)
                train_list.extend([subt[0].dataset[i] for i in subt[0].indices])
                val_list.extend([subt[1].dataset[i] for i in subt[1].indices])
                test_list.extend([subt[2].dataset[i] for i in subt[2].indices])

            # Finally, concatenate all subsets of CWE.
            self.global_train = train_list
            self.global_val = val_list
            self.global_test = test_list

        else:    # Otherwise: According to the original logic, divide 8:1:1 from a single .pkl file.
            with open(self.data_path, "rb") as f:
                full_dataset = pickle.load(f)
            self._safe_data_split(full_dataset)

        # If it is not CWE mode, perform contrastive learning segmentation again; in CWE mode, also perform 8:2 from global_train.
        if self.cwe_mode:
            # contrast split on the concatenated global_train
            n_ct = int(0.8 * len(self.global_train))
            n_cv = len(self.global_train) - n_ct
            g = torch.Generator().manual_seed(self.seed)
            self.contrast_train, self.contrast_val = random_split(
                self.global_train, [n_ct, n_cv], generator=g
            )
        else:
            # Original contrast learning classification
            pass

        # In CWE mode, skip global overlap checks; otherwise, keep the original settings.
        if not self.cwe_mode:
            self._verify_data_isolation()

    def get_split_fingerprint(self):
        return {
            "global_train": tuple(sorted(self.global_train.indices)),
            "global_val": tuple(sorted(self.global_val.indices)),
            "global_test": tuple(sorted(self.global_test.indices)),
            "contrast_train": tuple(sorted(self.contrast_train.indices)),
            "contrast_val": tuple(sorted(self.contrast_val.indices))
        }

    def save_split_fingerprint(self, path):
        """Save complete partition information"""
        torch.save(self.get_split_fingerprint(), path)

    def _safe_data_split(self, full_dataset):
        """Perform hierarchical data partitioning"""
        # Global partition (8:1:1)
        total = len(full_dataset)
        train_size = int(self.train_ratio * total)
        val_size = int(self.val_ratio * total)
        test_size = total - train_size - val_size

        # Fixed random seed to ensure reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        self.global_train, self.global_val, self.global_test = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        # Contrastive learning split (8:2 inside global_train)
        contrast_train_size = int(0.8 * len(self.global_train))
        contrast_val_size = len(self.global_train) - contrast_train_size
        self.contrast_train, self.contrast_val = random_split(
            self.global_train,
            [contrast_train_size, contrast_val_size],
            generator=generator
        )

    def _verify_data_isolation(self):
        """验证数据隔离性"""
        # Get the original dataset index
        train_indices = set(self.global_train.indices)
        val_indices = set(self.global_val.indices)
        test_indices = set(self.global_test.indices)

        # Verify that the global partitions have no overlap
        assert train_indices.isdisjoint(val_indices), "There is overlap between the training set and the validation set"
        assert train_indices.isdisjoint(test_indices), "There is overlap between the training set and the test set"
        assert val_indices.isdisjoint(test_indices), "There is overlap between the validation set and the test set"

        # Get the true index of the comparison learning set
        contrast_train_real_indices = {
            self.global_train.indices[i]
            for i in self.contrast_train.indices
        }
        contrast_val_real_indices = {
            self.global_train.indices[i]
            for i in self.contrast_val.indices
        }

        # Verify that contrastive learning is split within the training set
        assert contrast_train_real_indices.issubset(train_indices), "Comparison of training set data leakage"
        assert contrast_val_real_indices.issubset(train_indices), "Comparison of validation set data leakage"

    # Weight Sampling
    def _create_weighted_sampler(self, subset):
        labels = [1 if p.y == 1 else 0 for p in subset]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = [class_weights[label] for label in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    # Contrastive Learning Data Loader ----------------------------------------------
    def contrast_train_dataloader(self):
        return DataLoader(
            self.contrast_train,
            batch_size=self.batch_size,
            sampler=self._create_weighted_sampler(self.contrast_train),
            shuffle=False,
            collate_fn=self._contrastive_collate,
            num_workers=8
        )

    def contrast_val_dataloader(self):
        return DataLoader(
            self.contrast_val,
            batch_size=self.batch_size,
            collate_fn=self._contrastive_collate,
            num_workers=8
        )

    # Classification Task Data Loader ----------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.global_train,
            batch_size=self.batch_size,
            sampler=self._create_weighted_sampler(self.global_train),
            shuffle=False,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.global_val,
            batch_size=self.batch_size,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.global_test,
            batch_size=self.batch_size,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def _classification_collate(self, batch):
        try:
            mem = psutil.virtual_memory()
            # print(f"Memory usage: {mem.percent}% | Available: {mem.available / 1024 ** 3:.1f}GB")
            return self._build_contrastive_batch(batch)
        except Exception as e:
            print(f"Collate error: {str(e)}")
            torch.save(batch, "error_batch.pt")
            raise

    def _contrastive_collate(self, batch):
        # Original graph structure batch construction
        batch_data = self._build_contrastive_batch(batch)

        # Get key feature tensors
        B = batch_data.num_graphs

        ########################################
        # Phase 1: Build region index map
        ########################################

        # Create a region-to-graph mapping
        region_to_graph = []
        region_indices = []
        for b in range(B):
            valid_regions = torch.where(batch_data.region_semantic_emb_mask[b])[0]
            region_indices.extend([(b, r) for r in valid_regions.tolist()])
            region_to_graph.extend([b] * len(valid_regions))
        total_regions = len(region_indices)
        region_to_graph = torch.tensor(region_to_graph)

        ########################################
        # Phase 2: Same-file sample pair generation
        ########################################
        # Generates a mask of regions in the same file
        intra_mask = torch.zeros(total_regions, total_regions, dtype=torch.bool)
        for i in range(total_regions):
            for j in range(i + 1, total_regions):
                if region_to_graph[i] == region_to_graph[j]:
                    intra_mask[i, j] = True

        # Calculate the similarity matrix
        batch_indices = [b for (b, r) in region_indices]
        region_indices = [r for (b, r) in region_indices]

        batch_indices = torch.LongTensor(batch_indices)
        region_indices = torch.LongTensor(region_indices)

        # Extracting effective area features
        sem_emb = batch_data.region_semantic_emb[batch_indices, region_indices]  # (valid_region_num, d)
        stru_emb = batch_data.region_structural_emb[batch_indices, region_indices]  # (valid_region_num, d)
        combined_emb = torch.cat([sem_emb, stru_emb], dim=-1)
        sim_matrix = F.cosine_similarity(combined_emb.unsqueeze(1), combined_emb.unsqueeze(0), dim=-1)
        # Score Difference Matrix
        scores = batch_data.region_score[batch_indices, region_indices]
        score_diff = torch.abs(scores.unsqueeze(1) - scores.unsqueeze(0))
        # Generate positive and negative sample indexes
        pos_pairs = torch.where(intra_mask & (score_diff < self.delt1) & (sim_matrix > self.sim_threshold))
        neg_pairs = torch.where(intra_mask & (score_diff > self.delt2) & (sim_matrix > self.sim_threshold))

        ########################################
        # Stage 3: Generation of cross-file sample pairs
        ########################################
        # Generate cross-file masks
        cross_mask = (region_to_graph.unsqueeze(1) != region_to_graph.unsqueeze(0))
        # Cross-file positive and negative samples
        cross_pos = torch.where(cross_mask & (score_diff < self.delt1) & (sim_matrix > self.sim_threshold))
        cross_neg = torch.where(cross_mask & (score_diff > self.delt2) & (sim_matrix > self.sim_threshold))

        ########################################
        # Phase 4: Building regional data packages
        ########################################
        def build_region_package(indices):
            """Build a sample package with complete zone information"""
            packages = []
            edges_offset = 0
            for i, idx in enumerate(indices):
                b = batch_indices[idx]
                r = region_indices[idx]
                pkg = {
                    'code_emb': batch_data.region_code_embeddings[b, r],
                    'emb_mask': batch_data.region_emb_mask[b, r],
                    'features': batch[b]['region_features'][r],
                    'edges': batch[b]['region_edges'][r] + edges_offset,
                    'region_score': batch_data.region_score[b, r],
                    'node_indices': batch_data.node_index_lists[b, r],
                    'line_indices': batch_data.region_line_numbers_lists[b, r],
                    'batch_idx': b.clone().detach().long().requires_grad_(False),
                    'region_idx': r.clone().detach().long().requires_grad_(False)
                }

                edges_offset += batch[b]['region_features'][r].size(0)
                packages.append(pkg)

            code_emb_list = [p['code_emb'] for p in packages]
            emb_mask_list = [p['emb_mask'] for p in packages]
            features_list = [p['features'] for p in packages]
            edges_list = [p['edges'] for p in packages]
            region_score_list = [p['region_score'] for p in packages]
            node_indices_list = [p['node_indices'] for p in packages]
            line_indices_list = [p['line_indices'] for p in packages]

            code_emb = torch.stack(code_emb_list)
            emb_mask = torch.stack(emb_mask_list)
            features = torch.cat(features_list, dim=0)
            features_batch = torch.cat([torch.full((x.size(0),), i, dtype=torch.long) for i, x in enumerate(features_list)], dim=0)
            edges = torch.cat(edges_list, dim=1)
            region_score = torch.stack(region_score_list)
            node_indices = torch.stack(node_indices_list)
            line_indices = torch.stack(line_indices_list)

            # Concatenate the tensors of all regions
            merged = {
                'code_emb': code_emb,
                'emb_mask': emb_mask,
                'features': features,
                'features_batch': features_batch,
                'edges': edges,
                'region_score': region_score,
                'node_indices': node_indices,
                'line_indices': line_indices,
                'batch_indices': torch.stack([p['batch_idx'] for p in packages]),
                'region_indices': torch.stack([p['region_idx'] for p in packages])
            }
            return merged

        def safe_build(pair_list):
            # Convert to a valid index list
            indices = pair_list.tolist()

            # Secondary validity check
            if len(indices) == 0:
                return None

            return build_region_package(indices)

        # Construct the final sample pair
        contrastive_pairs = {
            'graph_data': batch_data,
            'intra_pos_pairs': [
                (safe_build(pos_pairs[0]), safe_build(pos_pairs[1]))
            ] if pos_pairs[0].size(0) > 0 and pos_pairs[1].size(0) > 0 else [],

            'intra_neg_pairs': [
                (safe_build(neg_pairs[0]), safe_build(neg_pairs[1]))
            ] if neg_pairs[0].size(0) > 0 and neg_pairs[1].size(0) > 0 else [],

            'cross_pos_pairs': [
                (safe_build(cross_pos[0]), safe_build(cross_pos[1]))
            ] if cross_pos[0].size(0) > 0 and cross_pos[1].size(0) > 0 else [],

            'cross_neg_pairs': [
                (safe_build(cross_neg[0]), safe_build(cross_neg[1]))
            ] if cross_neg[0].size(0) > 0 and cross_neg[1].size(0) > 0 else []
        }

        return contrastive_pairs

    def _build_contrastive_batch(self, samples):
        max_regions = self.max_regions
        d = self.emb_dim
        all_region_edges = []
        all_semantic_embs = []
        all_structural_embs = []

        """Constructing comparative batch data"""
        ast_x_list, edge_index_list = [], []
        region_data = defaultdict(list)
        node_offset = 0

        # First pass: Calculate the maximum dimension
        max_region_nodes = max([r.size(0) for data in samples for r in data.node_index_lists], default=0)
        max_region_lines = max([r.size(0) for data in samples for r in data.region_line_numbers_lists], default=0)
        max_k = max(len(data.all_edges_to_remove_list) for data in samples)

        rest_ast_edge_index_list = []
        all_file_name = []
        # Second pass: filling data
        for data in samples:
            all_file_name.append(data.file_name)
            all_region_edges.append(data.region_edges)
            all_semantic_embs.append(data.region_semantic_emb)
            all_structural_embs.append(data.region_structural_emb)
            # Processing regional data
            region_features = self._process_regions(
                data.node_index_lists,
                data.region_line_numbers_lists,
                data.region_score,
                data.region_features,
                node_offset,
                max_regions,
                max_region_nodes,
                max_region_lines,
                d
            )
            for k, v in region_features.items():
                region_data[k].append(v)

            # Processing graph-structured data
            ast_x_list.append(data.ast_x)
            edge_index_list.append(data.ast_edge_index + node_offset)
            rest_ast_edge_index_list.append(data.rest_ast_edge_index + node_offset)
            all_edges_to_remove_list = self.all_edges_to_remove(data.all_edges_to_remove_list, node_offset, max_k)
            node_offset += data.ast_x.size(0)

        all_edges_to_remove_tensors = self.all_edges_to_remove_tensor(all_edges_to_remove_list)

        # Embedded alignment processing
        region_emb_data = self._process_region_embeddings(
            [data.region_code_embeddings_lists for data in samples]
        )
        global_emb_data = self._process_global_embeddings(
            [data.global_code_embedding for data in samples]
        )
        ast_x = torch.cat(ast_x_list, dim=0)

        region_features = torch.stack(region_data['features'])   # (batch, max_regions, max_nodes, d)

        region_edges = self._process_region_edges(all_region_edges, max_regions, max_region_nodes)
        region_semantic_emb = self._process_region_embs(all_semantic_embs, max_regions, d)
        region_structural_emb = self._process_region_embs(all_structural_embs, max_regions, d)

        # Create a Batch object
        batch_data = Batch(
            ast_x=ast_x,
            num_nodes=ast_x.size(0),
            ast_edge_index=torch.cat(edge_index_list, dim=1),
            rest_ast_edge_index=torch.cat(rest_ast_edge_index_list, dim=1),
            all_edges_to_remove_tensors=all_edges_to_remove_tensors,

            node_index_lists=torch.stack(region_data['nodes']),
            region_line_numbers_lists=torch.stack(region_data['lines']),
            region_score=torch.stack(region_data['scores']),
            ast_batch=torch.cat([torch.full((x.size(0),), i) for i, x in enumerate(ast_x_list)]),
            y=torch.cat([data.y for data in samples]),

            region_code_embeddings=region_emb_data['region_embeddings'],  # (batch, max_regions, max_nodes, d)
            global_code_embeddings=global_emb_data['global_embeddings'],  # (batch, max_lines, d)
            region_emb_mask=region_emb_data['region_emb_mask'],  # (batch, max_regions, max_nodes)
            global_emb_mask=global_emb_data['global_emb_mask'],  # (batch, max_lines)

            region_features=region_features,    # (total_regions, max_nodes, d)
            region_edges=region_edges,
            region_semantic_emb=region_semantic_emb['embeddings'],
            region_semantic_emb_mask=region_semantic_emb['mask'],
            region_structural_emb=region_structural_emb['embeddings'],
            region_structural_emb_mask=region_structural_emb['mask'],
            file_name = all_file_name,
        )
        batch_data.num_graphs = batch_data.y.size(0)

        # print(f"batch_data information:{batch_data}")

        return batch_data

    def _process_region_embeddings(self, all_region_embeddings):
        """Region embedding alignment"""
        # Calculate the maximum dimension
        max_regions = self.max_regions
        max_nodes = max(
            emb.size(0)
            for regions in all_region_embeddings
            for emb in regions
        )
        emb_dim = self.emb_dim

        padded_embeddings = []
        masks = []

        for batch_idx, regions in enumerate(all_region_embeddings):
            region_embs = []
            region_mask = []

            # Step 1: Filling the node dimensions
            for region_idx, emb in enumerate(regions):
                node_pad = max_nodes - emb.size(0)
                padded_emb = F.pad(emb, (0, 0, 0, node_pad))  # (max_nodes, emb_dim)
                region_embs.append(padded_emb)

                # Generate mask
                mask = torch.cat([
                    torch.ones(emb.size(0), dtype=torch.bool),
                    torch.zeros(node_pad, dtype=torch.bool)
                ])
                region_mask.append(mask)

            # Step 2: Fill in the area dimension
            region_pad = max_regions - len(regions)
            if region_pad > 0:
                dummy_emb = torch.zeros((max_nodes, emb_dim))
                dummy_mask = torch.zeros((max_nodes,), dtype=torch.bool)
                # Adding multiple virtual zones
                for _ in range(region_pad):
                    region_embs.append(dummy_emb)
                    region_mask.append(dummy_mask)

            # Step 3: Dimension Verification
            try:
                stacked_emb = torch.stack(region_embs)  # (max_regions, max_nodes, emb_dim)
                stacked_mask = torch.stack(region_mask)
            except RuntimeError as e:
                print(f"Stacking failed! Error message: {str(e)}")
                raise

            padded_embeddings.append(stacked_emb)
            masks.append(stacked_mask)

        return {
            'region_embeddings': torch.stack(padded_embeddings),  # (batch, max_regions, max_nodes, emb_dim)
            'region_emb_mask': torch.stack(masks)
        }

    def _process_global_embeddings(self, all_global_embeddings):
        """Handling global embedding alignment"""
        max_lines = max(emb.size(0) for emb in all_global_embeddings)

        padded_embs = []
        masks = []

        for emb in all_global_embeddings:
            # Row number padding
            line_pad = max_lines - emb.size(0)
            padded_emb = F.pad(emb, (0, 0, 0, line_pad), value=0)  # (max_lines, d)
            padded_embs.append(padded_emb)

            # Generate row-level mask
            mask = torch.cat([
                torch.ones(emb.size(0), dtype=torch.bool),
                torch.zeros(line_pad, dtype=torch.bool)
            ])
            masks.append(mask)

        return {
            'global_embeddings': torch.stack(padded_embs),  # (batch, max_lines, d)
            'global_emb_mask': torch.stack(masks)  # (batch, max_lines)
        }

    def all_edges_to_remove(self, all_edges_to_remove, node_offset, max_k):
        all_edges_to_remove_list = [[] for _ in range(max_k)]  # Initialize K lists, each list is used to store all edges of a certain layer
        for k in range(max_k):  # Handle all_edges_to_remove
            if k < len(all_edges_to_remove):
                edge_to_remove = all_edges_to_remove[k] + node_offset
                all_edges_to_remove_list[k].append(edge_to_remove)
            else:
                all_edges_to_remove_list[k].append(torch.empty((2, 0), dtype=torch.long)) # If the number of layers of the current data is less than max_k, fill in the empty edge index

        return all_edges_to_remove_list

    def all_edges_to_remove_tensor(self, all_edges_to_remove_list):
        all_edges_to_remove_tensors = []
        for k, edges_in_k in enumerate(all_edges_to_remove_list): # Merge all_edges_to_remove_list
            if edges_in_k:
                concatenated_edges = torch.cat(edges_in_k, dim=1)
            else:
                concatenated_edges = torch.empty((2, 0), dtype=torch.long)
            all_edges_to_remove_tensors.append(concatenated_edges)

        return all_edges_to_remove_tensors

    def _process_regions(self, nodes, lines, scores, region_features, offset, max_regions, max_nodes, max_lines, d):
        """Processing area data filling"""
        padded_nodes = []
        padded_lines = []
        padded_scores = []
        padded_features = []
        for i in range(max_regions):
            if i < len(nodes):
                node = nodes[i] + offset
                line = lines[i]
                score = scores[i] if i < len(scores) else 0.0
            else:
                node = torch.full((max_nodes,), -1, dtype=torch.long)
                line = torch.full((max_lines,), -1, dtype=torch.long)
                score = torch.tensor(0.0)
            # Node filling
            if node.size(0) < max_nodes:
                node = F.pad(node, (0, max_nodes - node.size(0)), value=-1)
            padded_nodes.append(node)

            if i < len(region_features):
                # Node dimension filling (n,d) -> (max_nodes,d)
                feat = F.pad(region_features[i], (0, 0, 0, max_nodes - region_features[i].size(0)))
            else:
                feat = torch.zeros((max_nodes, d))
            padded_features.append(feat)
            # Row number padding
            if line.size(0) < max_nodes:
                line = F.pad(line, (0, max_lines - line.size(0)), value=-1)
            padded_lines.append(line)
            padded_scores.append(score)

        processed_nodes = torch.stack(padded_nodes)
        processed_lines = torch.stack(padded_lines)
        processed_score = torch.stack(padded_scores)
        processed_features = torch.stack(padded_features)

        return {
            'nodes': processed_nodes,   # (max_regions, max_nodes)
            'lines': processed_lines,   # (max_regions, max_line_num)
            'scores': processed_score,  # (max_regions)
            'features': processed_features,  # (max_regions, max_nodes, d)
        }

    def _process_region_edges(self, all_region_edges, max_regions, max_region_nodes):
        """Handling offset and padding of region edge indices"""
        batch_edges = []
        for batch_idx, regions_edges in enumerate(all_region_edges):
            region_edges_list = []
            for region_idx, edges in enumerate(regions_edges):
                # Calculate the offset
                offset = (max_regions * batch_idx + region_idx) * max_region_nodes
                shifted_edges = edges + offset
                region_edges_list.append(shifted_edges)

            batch_edges.extend(region_edges_list)

        return torch.cat(batch_edges, dim=1) if batch_edges else torch.empty((2, 0), dtype=torch.long)


    def _process_region_embs(self, all_embs, max_regions, d):
        """Processing region-level embedding alignment"""
        padded_embs = []
        masks = []

        for embs in all_embs:
            # Area dimension padding
            emb_pad = max_regions - len(embs)
            padded = F.pad(embs, (0, 0, 0, emb_pad))  # (max_regions, d)
            mask = torch.cat([torch.ones(len(embs)), torch.zeros(emb_pad)]).bool()

            padded_embs.append(padded)
            masks.append(mask)

        return {
            'embeddings': torch.stack(padded_embs),  # (batch, max_regions, d)
            'mask': torch.stack(masks)  # (batch, max_regions)
        }