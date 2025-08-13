import os
import gensim
from nltk.tokenize import RegexpTokenizer
import logging
import torch
import json
from torch_geometric.data import Data
import traceback
import pickle
import re
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from collections import Counter
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
from ordered_set import OrderedSet
import math
import argparse
from collections import defaultdict
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self,
                 c_dir: str,
                 js_dir: str,
                 vul_lines_path: str,
                 w2v_path: str,
                 cwe_map_path: str,
                 dataset_name: str,
                 seed: int = 42):
        c_regexp = (
            r'\w+|->|\+\+|--|<=|>=|==|!=|'
            r'<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|'
            r'&=|<<=|>>=|^=|\|=|::|'
            r'[!@#$%^&*()_+\-=\[\]{};\':"\|,.<>/?]'
        )
        self.tokenizer = RegexpTokenizer(c_regexp)

        self.original_c = c_dir
        self.original_js = js_dir
        self.vul_lines = vul_lines_path
        self.w2v_path = w2v_path
        self.cwe_map_path = cwe_map_path
        self.dataset_name = dataset_name

        if not os.path.exists(self.w2v_path):
            raise FileNotFoundError(f"Word2Vec file not found: {self.w2v_path}")
        self.word_vectors   = gensim.models.KeyedVectors.load(self.w2v_path, mmap='r')
        self.embedding_size = self.word_vectors.vector_size

        self.rng = random.Random(seed)

    def run(self):
        print(f"Preprocessing dataset at:\n  C dir:   {self.original_c}\n  JSON dir:{self.original_js}\n  Vul lines:{self.vul_lines}\n  W2V:      {self.w2v_path}\n  W2V:      {self.cwe_map_path}")
        # Path Configuration
        path_config = {
            "original_c": self.original_c,
            "original_js": self.original_js,
            "vul_lines": self.vul_lines,
            "cwe_map": self.cwe_map_path
        }
        # Reading the CWE Map
        with open(path_config["cwe_map"], 'r', encoding='utf-8') as f:
            cwe_map = json.load(f)

        # Read vulnerability row data
        with open(path_config["vul_lines"], 'r', encoding='utf-8') as f:
            vul_lines_data = json.load(f)

        out_dir = f"./CWE/{self.dataset_name}/"
        os.makedirs(out_dir, exist_ok=True)

        # Construct and save a dataset for each CWE separately
        for cwe in tqdm(sorted(cwe_map.keys()), desc=f"Building for Each CWE"):
            stems = set(cwe_map[cwe])
            if not stems:
                continue  # Skip if there is no corresponding file

            dataset = self.build_classify_dataset(
                stems,
                vul_lines_data,
                path_config["original_c"],
                path_config["original_js"]
            )

            out_file = os.path.join(out_dir, f"{cwe}.pkl")
            self.save_processed_data(dataset, out_file)
            print(f"✔ Saved {out_file}")

    def build_classify_dataset(self,
                               stems: set,
                               vul_lines_data: dict,
                               c_dir: str,
                               js_dir: str):
        """
        Constructs a classification dataset for the specified collection of files.
        :param stems: a set of file stem names (without suffix)
        :param vul_lines_data: a mapping of C file names -> vulnerability line numbers
        :param c_dir: C source file directory
        :param js_dir: Joern exported JSON directory
        :return: list of processed samples
        """
        dataset = []
        error_log = []
        pbar = tqdm(sorted(stems), desc="  Processing stems", leave=False)

        for stem in pbar:
            c_file = f"{stem}.c"
            js_file = f"{stem}.json"
            c_path = os.path.join(c_dir, c_file)
            js_path = os.path.join(js_dir, js_file)

            # Check if the file exists
            if not os.path.exists(c_path):
                error_log.append(f"Missing C file: {c_path}")
                continue
            if not os.path.exists(js_path):
                error_log.append(f"Missing JSON file: {js_path}")
                continue

            # Get the corresponding vulnerability line list
            vul_lines = vul_lines_data.get(c_file, [])

            try:
                data = self.prepare_torch_graph(
                    js_path=js_path,
                    c_path=c_path,
                    vul_lines=vul_lines
                )
                dataset.append(data)

            except Exception as e:
                error_msg = f"[{c_file} joern extraction error: {e}, please check source code integrity.]"
                error_log.append(error_msg)
                continue

        self._save_error_log(error_log)
        return dataset

    def _save_error_log(self, error_log: List[str]):
        if error_log:
            with open("data_preprocess_cwe_errors.log", "w") as f:
                f.write("\n".join(error_log))
            logger.warning(f"Encountered {len(error_log)} errors, see log file for details")

    def prepare_torch_graph(self, js_path: str, c_path: str, vul_lines: List[int]) -> Data:
        with open(js_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        file_name = os.path.basename(js_path)

        ast_nodes = json_data['ast_nodes']
        ast_edges = json_data['ast_edges']
        ast_nodes, ast_edges = self.filter_nodes(ast_nodes, ast_edges)
        ast_nodes_dict = {node['id']: node for node in ast_nodes}
        # Build adjacency list and node dictionary
        adjacency_map = defaultdict(list)
        for edge in ast_edges:
            parent, child = edge[0], edge[1]
            adjacency_map[parent].append(child)

        # Region Partition based on AST syntax structure
        regions = self.process_ast(ast_nodes, adjacency_map)

        max_depth = 2
        prev_length = -1
        while True:
            # Perform the finest-grained partitioning
            refined = self.refine_regions(regions, max_depth=max_depth)
            current_length = len(refined)
            if current_length == prev_length:
                break
            max_depth += 1
            prev_length = current_length

        node_index_lists = list(refined.values())
        region_count = len(node_index_lists)

        sorted_node_index_lists = [sorted(sublist) for sublist in node_index_lists]
        # The Final Partition
        if len(sorted_node_index_lists) > 1:
            last_list = sorted_node_index_lists[-1]
            sublists = self.split_continuous_sublists(last_list)
            sorted_node_index_lists = sorted_node_index_lists[:-1] + sublists

        # Filter out sublists of length 1
        filtered_lists = [sublist for sublist in sorted_node_index_lists if len(sublist) > 1]
        filtered_lists = self.deduplicate_nested_list(filtered_lists)
        region_line_numbers = self.line_number_extract(ast_nodes_dict, filtered_lists)  # 提取各区域的行号

        # Performing a Merge
        merged_regions = self.merge_regions(filtered_lists, ast_nodes_dict, adjacency_map, region_line_numbers)

        # Get the set of vulnerable nodes
        vul_line_node_ids = self.get_vul_line_nodes(ast_nodes_dict, vul_lines)
        risk_node_ids = set(vul_line_node_ids)

        def compute_N_avg(regions):
            """
            Calculate the average number of nodes N_avg in all regions.
            If regions is empty, return 1.0 to avoid subsequent division by zero errors.

            regions: List[Dict], each dict contains at least 'nodes' key
            Return: float, average number of nodes or default value 1.0
            """
            if not regions:
                return 1.0
            total_nodes = sum(len(r['nodes']) for r in regions)
            return total_nodes / len(regions)
        N_avg = compute_N_avg(merged_regions)
        # Calculate risk score for each region
        for region in merged_regions:
            region_lines = self.line_number_extract(ast_nodes_dict, [region['nodes']])[0]
            region['lines'] = region_lines
            region['high_risk_count'] = sum(1 for nid in region['nodes']
                                            if nid in risk_node_ids)
            region['score'] = self.region_quality_evaluation(region, vul_lines, N_avg)
        # Select the final area
        sorted_regions = sorted(merged_regions, key=lambda x: x['score'], reverse=True)
        final_regions = self.select_final_regions(sorted_regions)
        final_regions_count = len(final_regions)
        if final_regions_count == 0 or region_count == 0:
            raise JoernError
        final_nodes_index_lists = [region['nodes'] for region in final_regions]
        normalized_node_index_lists = self.normalize_lists(ast_nodes, final_nodes_index_lists)
        normalized_node_index_lists_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in
                                               normalized_node_index_lists]

        final_score_lists = [region['score'] for region in final_regions]
        final_score_lists_tensor = torch.tensor(final_score_lists, dtype=torch.float32)
        final_region_line_numbers = self.line_number_extract(ast_nodes_dict, final_nodes_index_lists)
        region_line_numbers_lists_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in
                                             final_region_line_numbers]

        region_code_snippets = self.extract_code_snippets(c_path, final_region_line_numbers)
        region_code_embeddings_lists = self.code_snippets_embedding(region_code_snippets)

        global_code_embeddings = self.global_code_embedding(c_path)  # (n_lines, d)

        # AST edge index list for HierGCN
        rest_ast_edges, all_edges_to_remove = self.simplify_ast_iteratively(ast_nodes, ast_edges)
        normalized_ast_edges = self.normalize_graph(ast_nodes, ast_edges)
        normalized_rest_ast_edges = self.normalize_graph(ast_nodes, rest_ast_edges)
        normalized_all_edges_to_remove = self.normalized_edges(ast_nodes, all_edges_to_remove)

        ast_x = self.generate_node_embeddings(ast_nodes, "AST")

        ast_edge_index = torch.tensor(normalized_ast_edges, dtype=torch.int64).t().contiguous()

        subgraphs = self.extract_region_feature(normalized_node_index_lists_tensors, ast_x, ast_edge_index)

        all_edges_to_remove_list = [torch.tensor(edges, dtype=torch.int64).t().contiguous() for edges in
                                    normalized_all_edges_to_remove]
        rest_ast_edge_index = torch.tensor(normalized_rest_ast_edges, dtype=torch.int64).t().contiguous()

        region_features_list = []
        region_nodes_list = []
        region_edges_list = []
        for region in subgraphs:
            region_features_list.append(region['features'])
            region_nodes_list.append(region['nodes'])
            region_edges_list.append(region['edges'])

        data = Data(
            ast_x=ast_x,
            num_nodes=ast_x.size(0),
            region_code_embeddings_lists=region_code_embeddings_lists,  # Region-level code embedding list
            global_code_embedding=global_code_embeddings,  # Global code embedding

            region_features=region_features_list,  # Regional node feature matrix list
            region_nodes=region_nodes_list,  # Regional node index list
            region_edges=region_edges_list,  # Region edge index matrix list

            node_index_lists=normalized_node_index_lists_tensors,  # Regional node index list
            region_line_numbers_lists=region_line_numbers_lists_tensors,  # Area row number index list

            ast_edge_index=ast_edge_index,  # AST edge index matrix
            all_edges_to_remove_list=all_edges_to_remove_list,  # AST hierarchical edge index matrix
            rest_ast_edge_index=rest_ast_edge_index,  # AST hierarchical final edge index matrix

            y=torch.tensor([self.get_target(file_name)], dtype=torch.long),  # Graph level vulnerability label 0/1
            region_score=final_score_lists_tensor,  # Area level score label 0~1

            file_name=file_name[:-5]  # File name prefix
        )
        data.region_semantic_emb = torch.stack(self._get_overall_emb(data.region_code_embeddings_lists))  # Region original semantic embedding
        data.region_structural_emb = torch.stack(self._get_overall_emb(data.region_features))  # Region original structure embedding

        return data

    def _get_overall_emb(self, region_features_list):
        # Pooling of AST node features for each region
        return [torch.mean(feats, dim=0) for feats in region_features_list]

    def extract_region_feature(self, region_nodes_list, node_features, edge_index):
        subgraphs = []

        for region_nodes in region_nodes_list:
            region_set = set(region_nodes.tolist())
            sub_node_features = node_features[region_nodes]
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(region_nodes.tolist())}
            src, dst = edge_index
            mask = torch.tensor([(s.item() in region_set and d.item() in region_set)
                                 for s, d in zip(src, dst)], dtype=torch.bool)

            sub_edge_index = edge_index[:, mask]

            sub_edge_index = torch.stack([
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[0]]),
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[1]])
            ])

            subgraphs.append({
                'nodes': region_nodes,
                'features': sub_node_features,
                'edges': sub_edge_index
            })

        return subgraphs

    def global_code_embedding(self, c_path: str) -> torch.Tensor:
        """Generates a line-by-line embedding of the entire file,
        with empty lines filled with zero vectors"""
        with open(c_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = [line.rstrip('\n') for line in f]

        line_embeddings = []
        for line in all_lines:
            try:
                cleaned_line = line.strip()
                if not cleaned_line:
                    emb = np.zeros(self.word_vectors.vector_size)
                else:
                    cleaned_line = line.replace('\t', ' ').replace('\n', ' ')
                    tokens = self.tokenizer.tokenize(cleaned_line)
                    vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
                    emb = np.mean(vectors, axis=0) if vectors else np.zeros(self.word_vectors.vector_size)

                if emb.shape != (self.word_vectors.vector_size,):
                    emb = np.zeros(self.word_vectors.vector_size)
            except Exception as e:
                print(f"Error in line {len(line_embeddings) + 1}: {str(e)}")
                emb = np.zeros(self.word_vectors.vector_size)

            line_embeddings.append(emb)

        embeddings_array = np.array(line_embeddings, dtype=np.float32)
        embeddings_tensor = torch.tensor(embeddings_array)  # (n_lines, d)
        assert embeddings_tensor.ndim == 2, f"dim error: (n, d) but {embeddings_tensor.shape}"
        return embeddings_tensor

    def code_snippets_embedding(self, region_code_snippets):
        """Generates row-by-row embeddings of region codes,
        returning a list of tensors of shape (n_region, seq_len, d)"""
        region_embeddings = []

        for region in region_code_snippets:
            line_embeddings = []
            for line in region:
                try:
                    cleaned_line = line.strip().replace('\t', ' ').replace('\n', ' ')
                    if not cleaned_line:
                        continue

                    tokens = self.tokenizer.tokenize(cleaned_line)
                    if not tokens:
                        continue

                    vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
                    if vectors:
                        line_emb = np.mean(vectors, axis=0)
                    else:
                        line_emb = np.zeros(self.word_vectors.vector_size)

                    line_embeddings.append(line_emb)

                except Exception as e:
                    print(f"Error processing line: {line} - {str(e)}")
                    continue

            if line_embeddings:
                line_embeddings = np.array(line_embeddings, dtype=np.float32)
                region_tensor = torch.tensor(line_embeddings, dtype=torch.float32)
            else:
                region_tensor = torch.zeros(1, self.word_vectors.vector_size)

            region_embeddings.append(region_tensor)

        return region_embeddings  # List[Tensor(n_lines, d)]

    def extract_code_snippets(self, c_path: str, line_numbers_list: List[List[int]]) -> List[str]:
        """Enhanced code snippet extraction to handle blank lines"""
        with open(c_path, 'r', encoding='utf-8', errors='replace') as f:
            code_lines = [line.rstrip('\n') for line in f]

        code_snippets_lists = []
        for region_lines in line_numbers_list:
            valid_entries = []
            for ln in set(region_lines):
                if 0 < ln <= len(code_lines):
                    valid_entries.append({
                        "index": ln - 1,
                        "content": code_lines[ln - 1].strip(),
                        "is_empty": (code_lines[ln - 1].strip() == "")
                    })

            merged_code = []
            last_non_empty = -2
            for entry in sorted(valid_entries, key=lambda x: x["index"]):
                if entry["is_empty"]:
                    if merged_code and not merged_code[-1].endswith('\n'):
                        merged_code.append('\n')
                    continue

                if entry["index"] == last_non_empty + 1:
                    merged_code[-1] += " " + entry["content"]
                else:
                    merged_code.append(entry["content"])

                last_non_empty = entry["index"]

            code_snippets_lists.append(merged_code)

        return code_snippets_lists

    def select_final_regions(self, sorted_regions, min_regions=5, max_regions=5):
        """
        Improved region selection logic to avoid repeated replenishment of the same region
        :param sorted_regions: list of regions sorted by score (in descending order)
        :return: optimized list of regions
        """
        scores = [r['score'] for r in sorted_regions]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        quality_threshold = max(mean_score + 0.5 * std_score, 0.3)

        qualified = [r for r in sorted_regions if r['score'] >= quality_threshold]

        final_regions = qualified.copy()

        need_supplement = max(min_regions - len(qualified), 0)

        if need_supplement > 0:
            selected_identifiers = {tuple(r['nodes']) for r in qualified}

            candidate_pool = [
                r for r in sorted_regions
                if tuple(r['nodes']) not in selected_identifiers
            ]

            supplement = candidate_pool[:min(need_supplement, len(candidate_pool))]

            final_regions += supplement
        final_regions = final_regions[:max_regions]

        return final_regions

    def region_quality_evaluation(self, region, vul_lines, N_avg):
        """Final region score evaluation function"""
        # Dimension 1: Vulnerability Line Coverage
        line_coverage = len(set(region['lines']) & set(vul_lines)) / (len(vul_lines) + 1e-8)

        # Dimension 2: Structural complexity
        size_score = np.log1p(len(region['nodes'])) / np.log1p(N_avg)

        # Dimension 3: High-risk pattern density
        risk_density = region['high_risk_count'] / (len(region['nodes']) + 1e-8)

        score = 0.6 * line_coverage + 0.3 * risk_density + 0.1 * size_score

        return score

    def get_vul_line_nodes(self, ast_nodes_dict, vul_lines):
        vul_lines = set(vul_lines)
        return [
            nid for nid, node in ast_nodes_dict.items()
            if node.get('lineNumber', -1) != -1
               and int(node['lineNumber']) in vul_lines
        ]

    def calculate_region_score(self, node_id_list, nodes_dict, adjacency_map):
        node_scores = [self.calculate_risk_score(nodes_dict[nid], adjacency_map, nodes_dict)
                       for nid in node_id_list]

        cross_file_risk = any(
            any(kw in nodes_dict[nid]['code'].lower()
                for kw in ['extern', 'import', 'dll'])
            for nid in node_id_list
        )

        base_score = max(node_scores)
        weight = 1.0 + 0.15 * len([s for s in node_scores if s > 0.7])
        if cross_file_risk:
            weight *= 1.3

        raw_score = base_score * weight

        return raw_score

    def merge_regions(self, region_list, nodes_dict, adjacency_map, region_line_numbers, overlap_threshold=0.6,
                      score_diff=0.3):
        regions = [{
            'nodes': region,
            'score': self.calculate_region_score(region, nodes_dict, adjacency_map),    # A preliminary fuzzy score
            'lines': region_line_numbers[i]
        } for i, region in enumerate(region_list)]

        merged = []
        while regions:
            current = regions.pop(0)
            candidates = []

            for target in regions:
                overlap = len(set(current['lines']) & set(target['lines'])) / \
                          len(set(current['lines']) | set(target['lines']))
                score_gap = abs(current['score'] - target['score'])

                if overlap >= overlap_threshold and score_gap <= score_diff:
                    candidates.append(target)

            if candidates:
                merged_region = {
                    'nodes': current['nodes'] + [n for c in candidates for n in c['nodes']],
                    'score': max([current['score']] + [c['score'] for c in candidates]),
                    'lines': sorted(set(current['lines'] + [l for c in candidates for l in c['lines']]))
                }
                regions = [r for r in regions if r not in candidates]
                regions.insert(0, merged_region)
            else:
                merged.append(current)

        return merged

    def deduplicate_nested_list(self, nested_list):
        seen = {}
        for sublist in nested_list:
            key = tuple(sublist)
            if key not in seen:
                seen[key] = sublist
        return list(seen.values())

    def split_continuous_sublists(self, lst):
        if not lst:
            return []
        sublists = []
        if len(lst) > 2:
            start_id = 1
        else:
            start_id = 0
        current = [lst[start_id]]
        for num in lst[(start_id + 1):]:
            if num == current[-1] + 1:
                current.append(num)
            else:
                sublists.append(current)
                current = [num]
        sublists.append(current)
        return sublists

    def process_ast(self, ast_nodes, adjacency_map):
        nodes = {node['id']: node for node in ast_nodes}
        # Find the root node (label is BLOCK, and the parent node is of type METHOD and name is <global>)
        root_node = None
        root_node = min(ast_nodes, key=lambda node: node['id'])
        if not root_node:
            raise ValueError("Root BLOCK node not found")

        regions = defaultdict(dict)  # Region level dictionary

        def collect_subtree(node_id):
            """Collect the IDs of a node and all its children"""
            subtree = []
            stack = [node_id]
            while stack:
                current = stack.pop()
                subtree.append(current)
                for child in adjacency_map.get(current, []):
                    stack.append(child)
            return subtree

        def traverse(current_id, current_level):
            """Recursively traverse the AST and divide it into regions"""
            if current_id == root_node['id']:
                for child in adjacency_map.get(current_id, []):
                    traverse(child, current_level)
                return
            current_node = nodes.get(current_id)
            if not current_node:
                return
            # Check if it is a selected node
            if current_node.get('_label') in ['CONTROL_STRUCTURE', 'BLOCK']:
                new_level = current_level + 1
                subtree = collect_subtree(current_id)
                regions[new_level][current_id] = subtree
                for child in adjacency_map.get(current_id, []):
                    traverse(child, new_level)
            else:
                for child in adjacency_map.get(current_id, []):
                    traverse(child, current_level)

        # Start traversing from the child nodes of the root node, the initial level is 0
        for child_id in adjacency_map.get(root_node['id'], []):
            traverse(child_id, 0)

        return {level: dict(nodes) for level, nodes in regions.items()}

    def detect_high_risk_nodes(self, ast_nodes):
        HIGH_RISK_TYPES = {'CALL', 'CONTROL_STRUCTURE', 'JUMP_TARGET'}
        VUL_PATTERNS = {
            # Level Ⅰ: High-risk memory operations
            'memcpy', 'strcpy', 'memmove', 'sprintf', 'strcat', 'strncpy',
            'vsprintf', 'gets', 'scanf', 'strncat',

            # Level II: Resource management risk
            'malloc', 'free', 'realloc', 'calloc', 'strdup', 'fopen',
            'fclose', 'fdopen', 'popen',

            # Level III: Potential vulnerability patterns
            'strncmp', 'strlen', 'atoi', 'atol', 'atof', 'getenv',
            'system', 'sscanf', 'access',

            # Level IV: Context-sensitive structures
            'if', 'switch', 'for', 'while', 'goto', 'sizeof'
        }
        high_risk_nodes = []
        for node in ast_nodes:
            if node.get('_label') in HIGH_RISK_TYPES:
                code = node.get('code', '')
                matched_patterns = [pattern for pattern in VUL_PATTERNS if pattern in code]
                if matched_patterns:
                    high_risk_nodes.append(node['id'])
        return high_risk_nodes

    def calculate_risk_score(self, node, adj_map, nodes):
        static_weights = {
            # Level Ⅰ：High-risk memory operations（0.8-1.0）
            'memcpy': 1.0, 'strcpy': 1.0, 'memmove': 0.9, 'sprintf': 0.9, 'strcat': 0.95,
            'strncpy': 0.85, 'vsprintf': 0.9, 'gets': 1.0, 'scanf': 0.8, 'strncat': 0.8,

            # Level Ⅱ：Resource management risk（0.6-0.8）
            'malloc': 0.8, 'free': 0.8, 'realloc': 0.7, 'calloc': 0.75,
            'strdup': 0.7, 'fopen': 0.6, 'fclose': 0.6, 'fdopen': 0.6, 'popen': 0.65,

            # Level Ⅲ：Potential vulnerability patterns（0.4-0.6）
            'strncmp': 0.6, 'strlen': 0.6, 'atoi': 0.5, 'atol': 0.5, 'atof': 0.5,
            'getenv': 0.4, 'system': 0.8, 'sscanf': 0.5, 'access': 0.4,

            # Level Ⅳ：Context-sensitive structures（0.2-0.4）
            'if': 0.4, 'switch': 0.4, 'for': 0.4, 'while': 0.4, 'goto': 0.4, 'sizeof': 0.4
        }
        VUL_PATTERNS = {
            # Level Ⅰ：High-risk memory operations
            'memcpy', 'strcpy', 'memmove', 'sprintf', 'strcat', 'strncpy',
            'vsprintf', 'gets', 'scanf', 'strncat',

            # Level Ⅱ：Resource management risk
            'malloc', 'free', 'realloc', 'calloc', 'strdup', 'fopen',
            'fclose', 'fdopen', 'popen',

            # Level Ⅲ：Potential vulnerability patterns
            'strncmp', 'strlen', 'atoi', 'atol', 'atof', 'getenv',
            'system', 'sscanf', 'access',

            # Level Ⅳ：Context-sensitive structures
            'if', 'switch', 'for', 'while', 'goto', 'sizeof'
        }
        code = node.get('code', '')
        matched_patterns = [pattern for pattern in VUL_PATTERNS if pattern in code]
        if matched_patterns:
            for pattern in matched_patterns:
                base_score = static_weights.get(pattern, 0.4)
        else:
            base_score = 0.4

        # Context-based correction
        if len(adj_map.get(node['id'], [])) > 3:
            base_score *= 1.3

        if any('input' in nodes[c_id]['code']
               for c_id in adj_map.get(node['id'], [])):
            base_score *= 1.3

        return min(base_score, 1.0)

    def normalized_edges(self, nodes, edges):
        normalized_edges = []
        for edge in edges:
            normalized_edge = self.normalize_graph(nodes, edge)
            normalized_edges.append(normalized_edge)

        return normalized_edges

    def line_number_extract(self, ast_nodes_dict, node_index_lists):
        region_line_numbers = []

        for region in node_index_lists:
            line_set = set()
            for node_id in region:
                node = ast_nodes_dict.get(node_id)
                if node:
                    line_number = node.get('lineNumber', 0)
                    line_set.update([line_number])
            sorted_lines = sorted(line_set)
            region_line_numbers.append(sorted_lines)
        return region_line_numbers

    def filter_nodes(self, ast_nodes, ast_edges):
        """Filter the starting root node and its first two children in the AST
            and remove related edges."""
        # 1. Collect a list of all node IDs
        node_ids = {node['id'] for node in ast_nodes}

        # 2. Find the root node
        root_node_id = min(node_ids)

        # 3. Initialize an empty set to save the node IDs to be filtered
        nodes_to_remove = set()
        nodes_to_remove.add(root_node_id)

        # 4. Find the first and second child nodes of the root node
        first_child_id = None
        second_child_id = None

        for edge in ast_edges:
            if edge[0] == root_node_id:
                if first_child_id is None:
                    first_child_id = edge[1]
                    nodes_to_remove.add(first_child_id)
                else:
                    second_child_id = edge[1]
                    nodes_to_remove.add(second_child_id)
                    break

        # 5. Filter nodes: keep nodes not in the filtered set
        filtered_nodes = [node for node in ast_nodes if node['id'] not in nodes_to_remove]

        # 6. Filter edges: remove edges associated with the deleted node
        filtered_edges = []
        for edge in ast_edges:
            if edge[0] not in nodes_to_remove and edge[1] not in nodes_to_remove:
                filtered_edges.append(edge)

        return filtered_nodes, filtered_edges

    def normalize_lists(self, nodes, node_index_lists):
        # Normalize Node index
        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}
        normalized_lists = [[old_to_new_id_dict[node_id] for node_id in sublist] for sublist in node_index_lists]

        return normalized_lists

    def refine_regions(self, original_regions, max_depth):
        """
        Optimized region partitioning function
        :param original_regions: original region level dictionary {level: {node_id: [subtree_ids]}}
        :param max_depth: partition depth (K value)
        :return: region dictionary after partitioning at the specified depth
        """
        # Establish parent->child area mapping relationship
        parent_child_map = defaultdict(list)
        for level in sorted(original_regions.keys()):
            if level + 1 not in original_regions: continue
            for parent_id in original_regions[level]:
                parent_nodes = set(original_regions[level][parent_id])
                for child_id in original_regions[level + 1]:
                    if child_id in parent_nodes:
                        parent_child_map[(level, parent_id)].append(child_id)

        # Recursive partitioning function
        def split_region(current_level, current_id, current_nodes, processed_depth):
            if processed_depth >= max_depth:
                return {current_id: current_nodes}

            children = parent_child_map.get((current_level, current_id), [])
            if not children:
                return {current_id: current_nodes}

            split_result = {}
            remaining = set(current_nodes)

            for child_id in children:
                child_nodes = set(original_regions[current_level + 1][child_id])
                split_result[child_id] = list(child_nodes)
                remaining -= child_nodes

            if remaining:
                split_result[current_id] = list(remaining)

            final_result = {}
            for sub_id in list(split_result.keys()):
                if sub_id == current_id:
                    final_result.update({sub_id: split_result[sub_id]})
                else:
                    final_result.update(
                        split_region(
                            current_level + 1,
                            sub_id,
                            split_result[sub_id],
                            processed_depth + 1
                        )
                    )
            return final_result

        # Start processing from the first layer
        final_regions = {}
        for base_level in sorted(original_regions.keys()):
            if base_level > 1: break

            for region_id, region_nodes in original_regions[base_level].items():
                final_regions.update(
                    split_region(
                        current_level=base_level,
                        current_id=region_id,
                        current_nodes=region_nodes,
                        processed_depth=1
                    )
                )

        return final_regions

    def generate_node_embeddings(self, nodes, graph_type):
        node_embedding_dict = {}
        for n in nodes:
            if 'code' in n:
                n_code = n['code']
            else:
                n_code = ""
            try:
                n_code = n_code.replace('\\t', ' ')
                if not n_code:
                    code_embedding = np.zeros(self.word_vectors.vector_size)
                else:
                    tokens = self.tokenizer.tokenize(n_code)
                    vecs = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
                    if vecs:
                        # compute mean of all found token vectors
                        code_embedding = np.mean(vecs, axis=0)
                    else:
                        # no tokens → return zero‐vector
                        code_embedding = np.zeros(self.embedding_size, dtype=np.float32)
            except KeyError:
                raise Exception

            node_embedding_dict[n['id']] = code_embedding

        node_embeddings = np.array([node_embedding_dict[node['id']] for node in nodes], dtype=np.float32)
        return torch.tensor(node_embeddings, dtype=torch.float)

    # General normalization function
    def normalize_graph(self, nodes, edges):
        """
        Normalize nodes and edges, assign new IDs starting from 0 to nodes, and update edges according to the new IDs.

        :param nodes: list of nodes
        :param edges: list of edges
        :return: normalized nodes and edges, mapping dictionary from old to new IDs
        """
        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}
        normalized_edges = [[old_to_new_id_dict[edge[0]], old_to_new_id_dict[edge[1]]] for edge in edges]

        return normalized_edges

    # HierGCN node layering
    def simplify_ast_iteratively(self, ast_nodes, ast_edges):
        node_id_to_label = {node['id']: node.get('_label', '') for node in ast_nodes}

        stop_node_types = {"METHOD", "BLOCK", "CONTROL_STRUCTURE"}

        all_parents = {edge[0] for edge in ast_edges}

        all_edges_to_remove = []
        all_nodes_to_remove = set()

        while True:
            current_leaf_nodes = [node_id for node_id in node_id_to_label if node_id not in all_parents]

            nodes_to_remove = set()
            edges_to_remove = set()

            # Start from each leaf node and prune upwards
            for leaf in current_leaf_nodes:
                current_node = leaf
                parent_edges = [edge for edge in ast_edges if edge[1] == current_node]

                if not parent_edges:
                    continue

                parent_node = parent_edges[0][0]

                # If the current node type belongs to stop_node_types, the node will not be deleted
                if node_id_to_label.get(current_node) in stop_node_types:
                    continue

                # Otherwise, mark the current node for deletion
                nodes_to_remove.add(current_node)
                # Record the currently deleted edges
                edges_to_remove.add(tuple(parent_edges[0]))

            # If there is no node to be deleted, it means there is no node to be deleted in this step, and the iteration stops
            if not nodes_to_remove:
                break

            # Add the edges deleted in the current step to the all_edges_to_remove list
            all_edges_to_remove.append(edges_to_remove)

            # Delete the nodes and edges marked in the current iteration
            ast_edges = [edge for edge in ast_edges if tuple(edge) not in edges_to_remove]
            ast_nodes = [node for node in ast_nodes if node['id'] not in nodes_to_remove]

            # Update all_parents and all_nodes_to_remove
            all_parents = {edge[0] for edge in ast_edges}
            all_nodes_to_remove.update(nodes_to_remove)

        rest_ast_edges = ast_edges
        # Return the simplified graph, all deleted nodes and edges deleted in each round
        return rest_ast_edges, all_edges_to_remove

    def get_target(self, file_name: str) -> int:
        """Get the target given a file name."""
        target_str = file_name.split("_")[1]
        non_num = re.compile(r'[^\d]')
        target = int(non_num.sub('', target_str))
        return target

    def save_processed_data(self, data, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_processed_data(self, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

class JoernError(Exception):
    pass

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom.*")

if __name__ == "__main__":
    # Configuration corresponding to the two data sets
    DATASET_CONFIGS = {
        "BigVul": {
            "c_dir":        "./dataset/BigVul/c/",
            "js_dir":       "./dataset/BigVul/js/",
            "vul_lines":    "./dataset/BigVul/vulnerable_lines.json",
            "w2v_path":     "./dataset/BigVul/W2V/Fan-128-20.wordvectors",
            "cwe_map":      "./dataset/BigVul/cwe_mapping.json"
        },
        "DiverseVul": {
            "c_dir":        "./dataset/DiverseVul/c/",
            "js_dir":       "./dataset/DiverseVul/js/",
            "vul_lines":    "./dataset/DiverseVul/vulnerable_lines.json",
            "w2v_path":     "./dataset/DiverseVul/W2V/DiverseVul-128-20.wordvectors",
            "cwe_map":      "./dataset/DiverseVul/cwe_mapping.json"
        }
    }

    parser = argparse.ArgumentParser(description="Preprocess one of the two datasets")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Choose which dataset to preprocess"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    builder = DatasetBuilder(
        c_dir=cfg["c_dir"],
        js_dir=cfg["js_dir"],
        vul_lines_path=cfg["vul_lines"],
        w2v_path=cfg["w2v_path"],
        cwe_map_path=cfg["cwe_map"],
        seed=args.seed
    )

    builder.run()