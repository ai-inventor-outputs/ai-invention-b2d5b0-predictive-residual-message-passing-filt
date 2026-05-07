#!/usr/bin/env python3
"""Embedding-Space Cross-Table Predictability on rel-stack:
Third Dataset for Revised Regime Theory.

Uses PURE PyTorch (no NeighborLoader/pyg-lib/torch-sparse needed).
Loads rel-stack graph via relbench, trains Standard HeteroSAGE and PRMP HeteroSAGE
using full-graph forward passes, measuring embedding-space R² trajectories
across all FK links at training checkpoints.

Compares embedding-space predictability with Amazon (PRMP helps) and F1 (PRMP mixed)
to test whether low embedding-space R² explains PRMP's lack of benefit on rel-stack.
"""

import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
import gc
import json
import math
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ── Logging ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware Detection ────────────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(0.88, 0.88))
else:
    VRAM_GB = 0

RAM_BUDGET = int(min(38, TOTAL_RAM_GB * 0.65) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU}")
if HAS_GPU:
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM={VRAM_GB:.1f}GB")

# ── sklearn imports ───────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
CHANNELS = 48
NUM_LAYERS = 2
NUM_EPOCHS = 30
CHECKPOINT_EVERY = 15
LR = 0.003
WEIGHT_DECAY = 1e-4
R2_SUBSAMPLE = 2000
PATIENCE = 15

DATA_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus/full_data_out.json"
)

# ═══════════════════════════════════════════════════════════════════════════
# Scatter operations (pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-aggregate src by index. src:[E,D], index:[E] -> [dim_size,D]."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx_exp = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_exp, src)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1),
                       torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype))
    count = count.clamp(min=1)
    return out / count


# ═══════════════════════════════════════════════════════════════════════════
# Model components
# ═══════════════════════════════════════════════════════════════════════════
class ManualSAGEConv(nn.Module):
    """GraphSAGE-style convolution using pure PyTorch scatter ops."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        messages = x_src[src_idx]
        agg = scatter_mean(messages, dst_idx, x_dst.size(0))
        return self.lin_neigh(agg) + self.lin_self(x_dst)


class PRMPConv(nn.Module):
    """Predictive Residual Message Passing convolution.
    Predict child from parent, aggregate residuals."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.ReLU(),
            nn.Linear(in_channels, in_channels),
        )
        # Zero-init for smooth start
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)
        self.norm = nn.LayerNorm(in_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.last_r2 = None

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        x_dst_i = x_dst[dst_idx]
        pred = self.pred_mlp(x_dst_i.detach())
        residual = x_j - pred
        with torch.no_grad():
            ss_res = (residual ** 2).sum()
            ss_tot = ((x_j - x_j.mean(0)) ** 2).sum()
            self.last_r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
        normed = self.norm(residual)
        messages = self.lin(normed)
        return scatter_mean(messages, dst_idx, x_dst.size(0))


class HeteroConvLayer(nn.Module):
    """Manual heterogeneous convolution layer."""
    def __init__(self, convs_dict: dict):
        super().__init__()
        self.convs = nn.ModuleDict()
        self._keys = []
        for key, conv in convs_dict.items():
            str_key = "__".join(key)
            self.convs[str_key] = conv
            self._keys.append(key)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        updates: dict[str, list] = defaultdict(list)
        for key in self._keys:
            src_type, _rel, dst_type = key
            str_key = "__".join(key)
            conv = self.convs[str_key]
            if src_type in x_dict and dst_type in x_dict and key in edge_index_dict:
                out = conv(x_dict[src_type], x_dict[dst_type], edge_index_dict[key])
                updates[dst_type].append(out)
        result = {}
        for nt in x_dict:
            if updates[nt]:
                result[nt] = sum(updates[nt])
            else:
                result[nt] = x_dict[nt]
        return result

    def collect_prmp_r2(self) -> dict:
        r2s = {}
        for key in self._keys:
            str_key = "__".join(key)
            conv = self.convs[str_key]
            if isinstance(conv, PRMPConv) and conv.last_r2 is not None:
                r2s[str_key] = round(conv.last_r2, 6)
        return r2s


class HeteroGNN(nn.Module):
    """Heterogeneous GNN for rel-stack with embedding caching."""
    def __init__(self, node_feat_dims: dict, edge_types: list,
                 hidden: int = 128, use_prmp: bool = False):
        super().__init__()
        self.hidden = hidden
        self.use_prmp = use_prmp
        self.node_types = list(node_feat_dims.keys())

        # Per-node-type projection to hidden dim
        self.projections = nn.ModuleDict()
        for nt, dim in node_feat_dims.items():
            self.projections[nt] = nn.Linear(dim, hidden)

        # Two conv layers
        ChildConv = PRMPConv if use_prmp else ManualSAGEConv
        self.conv1 = HeteroConvLayer({
            et: ChildConv(hidden, hidden) for et in edge_types
        })
        self.conv2 = HeteroConvLayer({
            et: ChildConv(hidden, hidden) for et in edge_types
        })

        # Task-specific prediction head (set before training)
        self.pred_head = None
        self.cached_embeddings: dict = {}

    def set_pred_head(self, out_dim: int = 1):
        """Create a linear prediction head."""
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, out_dim),
        )
        return self

    def predict(self, h_dict: dict, entity_table: str, indices: torch.Tensor) -> torch.Tensor:
        """Get scalar predictions for given node indices using the pred head."""
        emb = h_dict[entity_table][indices]
        if self.pred_head is not None:
            return self.pred_head(emb).squeeze(-1)
        return emb.mean(dim=-1)

    def forward(self, x_dict: dict, edge_index_dict: dict,
                cache: bool = False) -> dict:
        """Full-graph forward. Returns x_dict after 2 layers."""
        # Project to hidden dim
        h = {}
        for nt in self.node_types:
            if nt in x_dict:
                h[nt] = F.relu(self.projections[nt](x_dict[nt]))
                h[nt] = torch.nan_to_num(h[nt], nan=0.0)
        if cache:
            self.cached_embeddings["layer_0"] = {
                k: v.detach().cpu() for k, v in h.items()
            }
        # Layer 1
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        if cache:
            self.cached_embeddings["layer_1"] = {
                k: v.detach().cpu() for k, v in h.items()
            }
        # Layer 2
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        if cache:
            self.cached_embeddings["layer_2"] = {
                k: v.detach().cpu() for k, v in h.items()
            }
        return h

    def collect_prmp_r2(self) -> dict:
        r2s = {}
        r2s.update(self.conv1.collect_prmp_r2())
        r2s.update({f"L2_{k}": v for k, v in self.conv2.collect_prmp_r2().items()})
        return r2s


# ═══════════════════════════════════════════════════════════════════════════
# R² computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_embedding_predictability(
    parent_emb: np.ndarray, child_emb: np.ndarray,
    n_subsample: int = 2000, num_cpus: int = 1,
    fast: bool = False,
) -> dict:
    """Ridge R², RF R², MI between aligned parent/child embeddings.
    fast=True skips RF and MI for speed during training checkpoints."""
    if len(parent_emb) < 10:
        return {"ridge_r2": 0.0, "rf_r2": 0.0, "mi": 0.0}
    if len(parent_emb) > n_subsample:
        idx = np.random.choice(len(parent_emb), n_subsample, replace=False)
        parent_emb, child_emb = parent_emb[idx], child_emb[idx]

    parent_emb = np.nan_to_num(parent_emb, nan=0.0, posinf=0.0, neginf=0.0)
    child_emb = np.nan_to_num(child_emb, nan=0.0, posinf=0.0, neginf=0.0)
    if parent_emb.std() < 1e-10 or child_emb.std() < 1e-10:
        return {"ridge_r2": 0.0, "rf_r2": 0.0, "mi": 0.0}

    n_comp = min(10, child_emb.shape[1])
    if child_emb.shape[1] > 10:
        try:
            child_reduced = PCA(n_components=n_comp).fit_transform(child_emb)
        except Exception:
            child_reduced = child_emb[:, :10]
    else:
        child_reduced = child_emb

    ridge_r2 = 0.0
    try:
        ridge = MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0]))
        scores = cross_val_score(ridge, parent_emb, child_reduced,
                                 cv=3, scoring="r2")
        ridge_r2 = float(np.mean(scores))
    except Exception as e:
        logger.debug(f"Ridge R² failed: {e}")

    rf_r2 = 0.0
    mi = 0.0
    if not fast:
        try:
            rf = RandomForestRegressor(n_estimators=30, max_depth=6,
                                       n_jobs=min(num_cpus, 4), random_state=42)
            scores = cross_val_score(MultiOutputRegressor(rf), parent_emb, child_reduced,
                                     cv=3, scoring="r2")
            rf_r2 = float(np.mean(scores))
        except Exception as e:
            logger.debug(f"RF R² failed: {e}")

        try:
            mi_vals = []
            for d in range(min(child_reduced.shape[1], 3)):
                mi_vals.append(float(np.mean(
                    mutual_info_regression(parent_emb, child_reduced[:, d], n_neighbors=5)
                )))
            mi = float(np.mean(mi_vals))
        except Exception as e:
            logger.debug(f"MI failed: {e}")

    return {"ridge_r2": round(ridge_r2, 6), "rf_r2": round(rf_r2, 6), "mi": round(mi, 6)}


def extract_r2_from_cached(cached_embeddings, edge_index_dict, fk_edge_map,
                           n_subsample=R2_SUBSAMPLE, num_cpus=NUM_CPUS,
                           fast=False, layers=None):
    """Compute R² for each FK link from cached embeddings.
    fast=True: only Ridge R² (skip RF/MI) for speed.
    layers: list of layer names to compute (default: all)."""
    results = {}
    target_layers = layers or sorted(cached_embeddings.keys())
    for fk_name, (child_type, parent_type, edge_key) in fk_edge_map.items():
        # Skip reverse edges (only measure original FK direction)
        if 'rev_' in str(edge_key[1]):
            continue
        if edge_key not in edge_index_dict:
            continue
        ei = edge_index_dict[edge_key]
        src_idx = ei[0].cpu().numpy()
        dst_idx = ei[1].cpu().numpy()
        if len(src_idx) > n_subsample:
            perm = np.random.choice(len(src_idx), n_subsample, replace=False)
            src_idx, dst_idx = src_idx[perm], dst_idx[perm]

        results[fk_name] = {}
        for layer_name in target_layers:
            if layer_name not in cached_embeddings:
                continue
            embs = cached_embeddings[layer_name]
            if child_type not in embs or parent_type not in embs:
                continue
            child_emb = embs[child_type].numpy()
            parent_emb = embs[parent_type].numpy()
            valid = (src_idx < child_emb.shape[0]) & (dst_idx < parent_emb.shape[0])
            if valid.sum() < 10:
                continue
            aligned_child = child_emb[src_idx[valid]]
            aligned_parent = parent_emb[dst_idx[valid]]
            metrics = compute_embedding_predictability(
                aligned_parent, aligned_child, n_subsample, num_cpus, fast=fast)
            results[fk_name][layer_name] = metrics
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Data loading: build heterogeneous graph from relbench
# ═══════════════════════════════════════════════════════════════════════════
def load_relstack_graph():
    """Load rel-stack via relbench, build full-graph data structures.
    Returns node features, edge indices, task labels, and FK edge map."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame import stype as tf_stype

    logger.info("Loading rel-stack dataset via relbench...")
    t0 = time.time()
    dataset = get_dataset("rel-stack", download=True)
    db = dataset.get_db()
    logger.info(f"  Tables: {list(db.table_dict.keys())}")

    col_to_stype = get_stype_proposal(db)
    for tbl in list(col_to_stype.keys()):
        text_cols = [c for c, s in col_to_stype[tbl].items()
                     if s in (tf_stype.text_embedded, tf_stype.text_tokenized)]
        for c in text_cols:
            del col_to_stype[tbl][c]

    cache_dir = str(WORKSPACE / "cache" / "rel-stack")
    logger.info(f"  Building heterogeneous graph...")
    data, col_stats = make_pkey_fkey_graph(
        db, col_to_stype_dict=col_to_stype,
        text_embedder_cfg=None, cache_dir=cache_dir,
    )
    logger.info(f"  Loaded in {time.time() - t0:.1f}s")
    logger.info(f"  Node types: {list(data.node_types)}")
    logger.info(f"  Edge types: {len(data.edge_types)}")

    # Extract raw features per node type using the encoder approach:
    # Each node type has data[nt].tf (TensorFrame). We need raw numeric features.
    node_feat_dict = {}
    for nt in data.node_types:
        if hasattr(data[nt], 'tf') and data[nt].tf is not None:
            tf = data[nt].tf
            # Get numeric features from TensorFrame
            if hasattr(tf, 'feat_dict'):
                feats = []
                for stype_key, feat_tensor in tf.feat_dict.items():
                    if feat_tensor is not None and feat_tensor.dim() >= 2:
                        # Flatten any extra dims
                        flat = feat_tensor.view(feat_tensor.size(0), -1).float()
                        flat = torch.nan_to_num(flat, nan=0.0)
                        feats.append(flat)
                if feats:
                    node_feat_dict[nt] = torch.cat(feats, dim=1)
                    logger.info(f"  {nt}: {node_feat_dict[nt].shape}")
                else:
                    # No features - use ones
                    n_nodes = data[nt].num_nodes
                    node_feat_dict[nt] = torch.ones(n_nodes, 1)
                    logger.info(f"  {nt}: fallback ones ({n_nodes}, 1)")
            else:
                n_nodes = data[nt].num_nodes
                node_feat_dict[nt] = torch.ones(n_nodes, 1)
                logger.info(f"  {nt}: fallback ones ({n_nodes}, 1)")
        else:
            n_nodes = data[nt].num_nodes if hasattr(data[nt], 'num_nodes') else 1
            node_feat_dict[nt] = torch.ones(n_nodes, 1)
            logger.info(f"  {nt}: no tf, ones ({n_nodes}, 1)")

    # Edge indices - only forward FK edges (skip reverse to save memory)
    edge_index_dict = {}
    for et in data.edge_types:
        if hasattr(data[et], 'edge_index') and data[et].edge_index is not None:
            edge_index_dict[et] = data[et].edge_index
            logger.info(f"  Edge {et}: {data[et].edge_index.size(1)} edges")
    logger.info(f"  Total edge types (incl. reverse): {len(edge_index_dict)}")

    # FK edge map for R² measurement
    fk_edge_map = {}
    for et in data.edge_types:
        src_type, rel, dst_type = et
        fk_name = f"{dst_type}_to_{src_type}"
        fk_edge_map[fk_name] = (src_type, dst_type, et)

    # Load task labels
    task_data = {}
    for task_cfg in [
        {"name": "post-votes", "type": "regression", "metric": "mae", "higher_better": False},
        {"name": "user-engagement", "type": "classification", "metric": "auroc", "higher_better": True},
    ]:
        try:
            task = get_task("rel-stack", task_cfg["name"], download=True)
            entity_table = task.entity_table
            entity_col = task.entity_col
            target_col = task.target_col
            pkey_col = db.table_dict[entity_table].pkey_col
            table_df = db.table_dict[entity_table].df
            pkey_to_idx = {v: i for i, v in enumerate(table_df[pkey_col].values)}

            splits = {}
            for split_name in ["train", "val", "test"]:
                try:
                    table = task.get_table(split_name)
                    df = table.df
                    if target_col not in df.columns:
                        logger.warning(f"  Task {task_cfg['name']}/{split_name}: "
                                       f"target col '{target_col}' not in df, skipping split")
                        continue
                    entities = df[entity_col].values
                    targets = df[target_col].values.astype(float)
                    node_indices = []
                    valid_targets = []
                    for ent, tgt in zip(entities, targets):
                        if ent in pkey_to_idx and np.isfinite(tgt):
                            node_indices.append(pkey_to_idx[ent])
                            valid_targets.append(tgt)
                    if node_indices:
                        splits[split_name] = {
                            "indices": torch.tensor(node_indices, dtype=torch.long),
                            "targets": torch.tensor(valid_targets, dtype=torch.float32),
                        }
                        logger.info(f"  Task {task_cfg['name']}/{split_name}: "
                                    f"{len(node_indices)} labeled nodes")
                    else:
                        logger.warning(f"  Task {task_cfg['name']}/{split_name}: "
                                       f"no valid nodes")
                except Exception as e:
                    logger.warning(f"  Task {task_cfg['name']}/{split_name} failed: {e}")

            if "train" in splits and "val" in splits:
                task_data[task_cfg["name"]] = {
                    "entity_table": entity_table,
                    "splits": splits,
                    "config": task_cfg,
                }
                logger.info(f"  Task {task_cfg['name']}: loaded {len(splits)} splits")
            else:
                logger.error(f"  Task {task_cfg['name']}: missing train/val, skipping")
        except Exception as e:
            logger.exception(f"  Failed to load task {task_cfg['name']}: {e}")

    return {
        "node_feat_dict": node_feat_dict,
        "edge_index_dict": edge_index_dict,
        "fk_edge_map": fk_edge_map,
        "task_data": task_data,
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Raw-Feature R² Baseline
# ═══════════════════════════════════════════════════════════════════════════
def compute_raw_feature_baseline(data_path: Path) -> dict:
    """Compute raw-feature R² for each FK link in the dependency data."""
    logger.info("Computing raw-feature R² baseline from dependency data...")
    raw = json.loads(data_path.read_text())
    baseline = {}
    for ds in raw["datasets"]:
        ds_name = ds["dataset"]
        examples = ds["examples"]
        if len(examples) < 20:
            continue
        try:
            parent_feats = [json.loads(ex["input"]) for ex in examples[:2000]]
            child_feats = [json.loads(ex["output"]) for ex in examples[:2000]]
            X = np.array([[float(v) for v in pf.values()] for pf in parent_feats])
            Y = np.array([[float(v) for v in cf.values()] for cf in child_feats])
        except Exception:
            continue
        ex0 = examples[0]
        metrics = compute_embedding_predictability(X, Y, n_subsample=2000, num_cpus=NUM_CPUS)
        baseline[ds_name] = {
            **metrics,
            "parent_dim": len(ex0.get("metadata_parent_feature_names", [])),
            "child_dim": len(ex0.get("metadata_child_feature_names", [])),
            "parent_table": ex0.get("metadata_parent_table", ""),
            "child_table": ex0.get("metadata_child_table", ""),
            "cardinality_mean": round(ex0.get("metadata_cardinality_mean", 0), 4),
        }
        logger.info(f"  {ds_name}: ridge_r2={metrics['ridge_r2']:.4f}")
    return baseline


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════
def _zero_nan_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            bad = ~torch.isfinite(p.grad)
            if bad.any():
                p.grad[bad] = 0.0


def _compute_metric(preds, labels, task_config):
    if task_config['metric'] == 'auroc':
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(labels, preds)
        except Exception:
            return 0.5
    elif task_config['metric'] == 'mae':
        return float(np.mean(np.abs(labels - preds)))
    return float('nan')


def train_full_graph(
    model: HeteroGNN,
    x_dict_dev: dict,
    edge_dict_dev: dict,
    task_info: dict,
    fk_edge_map: dict,
    edge_index_dict_cpu: dict,
    model_name: str,
    start_time: float,
    time_budget: float = 4200,
) -> dict:
    """Full-graph training with R² checkpoints."""
    config = task_info["config"]
    entity_table = task_info["entity_table"]
    splits = task_info["splits"]

    train_idx = splits["train"]["indices"].to(DEVICE)
    train_y = splits["train"]["targets"].to(DEVICE)
    val_idx = splits["val"]["indices"].to(DEVICE)
    val_y = splits["val"]["targets"].to(DEVICE)
    has_test = "test" in splits
    test_idx = splits["test"]["indices"].to(DEVICE) if has_test else None
    test_y = splits["test"]["targets"].to(DEVICE) if has_test else None
    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, "
                f"Test: {len(test_idx) if has_test else 'N/A'}")

    if config["type"] == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    epoch_records = []
    r2_timeseries = {fk: {f"layer_{l}": {"epochs": [], "ridge_r2": [], "rf_r2": [], "mi": []}
                          for l in range(3)} for fk in fk_edge_map}
    best_val = None
    best_state = None
    patience_ctr = 0

    # Subsample training/val for memory and speed
    max_train = min(len(train_idx), 30000)
    perm = torch.randperm(len(train_idx))[:max_train]
    train_idx_sub = train_idx[perm]
    train_y_sub = train_y[perm]

    max_val = min(len(val_idx), 10000)
    perm = torch.randperm(len(val_idx))[:max_val]
    val_idx_sub = val_idx[perm]
    val_y_sub = val_y[perm]

    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            logger.warning(f"Time budget ({elapsed:.0f}s), stopping ep {epoch}")
            break

        t0 = time.time()

        # === TRAIN ===
        model.train()
        optimizer.zero_grad()
        h_dict = model(x_dict_dev, edge_dict_dev, cache=False)

        if entity_table not in h_dict:
            logger.error(f"Entity table {entity_table} not in model output")
            break

        pred_train = model.predict(h_dict, entity_table, train_idx_sub)
        loss = loss_fn(pred_train, train_y_sub)

        if not torch.isfinite(loss):
            logger.warning(f"NaN loss at epoch {epoch}")
            continue

        loss.backward()
        _zero_nan_grads(model)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()

        # === VALIDATE ===
        model.eval()
        with torch.no_grad():
            h_dict = model(x_dict_dev, edge_dict_dev, cache=False)
            pred_val = model.predict(h_dict, entity_table, val_idx_sub)
            if config["type"] == "classification":
                val_preds_np = torch.sigmoid(pred_val).cpu().numpy()
            else:
                val_preds_np = pred_val.cpu().numpy()
            val_labels_np = val_y_sub.cpu().numpy()
            val_preds_np = np.nan_to_num(val_preds_np, nan=0.5 if config["type"] == "classification" else 0.0)
            val_metric = _compute_metric(val_preds_np, val_labels_np, config)

        # === PRMP R² ===
        prmp_r2 = model.collect_prmp_r2() if model.use_prmp else {}

        # === R² checkpoint ===
        r2_snapshot = {}
        is_ckpt = (epoch % CHECKPOINT_EVERY == 0) or (epoch == NUM_EPOCHS - 1)
        if is_ckpt:
            try:
                with torch.no_grad():
                    model.eval()
                    # Cache embeddings on current device then extract to CPU
                    _ = model(x_dict_dev, edge_dict_dev, cache=True)
                    # Use fast=True (Ridge only), only layer_2 for speed
                    is_final = (epoch >= NUM_EPOCHS - 2) or (patience_ctr >= PATIENCE - 1)
                    r2_snapshot = extract_r2_from_cached(
                        model.cached_embeddings, edge_index_dict_cpu,
                        fk_edge_map, R2_SUBSAMPLE, NUM_CPUS,
                        fast=not is_final,
                        layers=["layer_2"] if not is_final else None)
                    # Store in timeseries
                    for fk_name, layers in r2_snapshot.items():
                        for layer_name, metrics in layers.items():
                            if fk_name in r2_timeseries and layer_name in r2_timeseries[fk_name]:
                                ts = r2_timeseries[fk_name][layer_name]
                                ts["epochs"].append(epoch)
                                ts["ridge_r2"].append(metrics["ridge_r2"])
                                ts["rf_r2"].append(metrics["rf_r2"])
                                ts["mi"].append(metrics["mi"])
                    model.cached_embeddings = {}
                    gc.collect()
            except Exception as e:
                logger.warning(f"R² checkpoint failed ep {epoch}: {e}")
                model.cached_embeddings = {}
                gc.collect()
                if HAS_GPU:
                    torch.cuda.empty_cache()

        epoch_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_metric": round(float(val_metric), 6),
            "prmp_r2": prmp_r2,
            "r2_snapshot": r2_snapshot,
            "time_s": round(time.time() - t0, 1),
        })

        if epoch % 10 == 0 or is_ckpt:
            logger.info(f"  [{model_name}] ep{epoch:2d} loss={train_loss:.4f} "
                        f"val={val_metric:.4f} {time.time()-t0:.1f}s")

        # Early stopping
        improved = False
        if best_val is None:
            improved = True
        elif config["higher_better"] and val_metric > best_val:
            improved = True
        elif not config["higher_better"] and val_metric < best_val:
            improved = True
        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val": best_val,
        "epoch_records": epoch_records,
        "r2_timeseries": r2_timeseries,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Embedding-Space Cross-Table Predictability on rel-stack")
    logger.info("=" * 60)

    # Step 1: Raw-Feature R² Baseline
    raw_baseline = compute_raw_feature_baseline(DATA_PATH)

    # Step 2: Load rel-stack graph
    graph_data = load_relstack_graph()
    node_feat_dict = graph_data["node_feat_dict"]
    edge_index_dict = graph_data["edge_index_dict"]
    fk_edge_map = graph_data["fk_edge_map"]
    task_data = graph_data["task_data"]

    # Build node_feat_dims
    node_feat_dims = {nt: feat.size(1) for nt, feat in node_feat_dict.items()}
    # Use only forward FK edges for model (skip reverse to save memory)
    edge_types = [et for et in edge_index_dict.keys() if 'rev_' not in et[1]]
    logger.info(f"Node feat dims: {node_feat_dims}")
    logger.info(f"Model edge types (forward-only): {len(edge_types)}")
    logger.info(f"Total edge types (incl. reverse): {len(edge_index_dict)}")

    # Move to device
    x_dict_dev = {k: v.to(DEVICE) for k, v in node_feat_dict.items()}
    edge_dict_dev = {k: v.to(DEVICE) for k, v in edge_index_dict.items()}
    edge_index_dict_cpu = {k: v.cpu() for k, v in edge_index_dict.items()}

    # Step 3: Train models
    all_examples = []
    all_run_summaries = []
    all_r2_timeseries = {}

    task_configs = [
        {"name": "post-votes", "type": "regression", "metric": "mae", "higher_better": False},
        {"name": "user-engagement", "type": "classification", "metric": "auroc", "higher_better": True},
    ]

    for task_cfg in task_configs:
        task_name = task_cfg["name"]
        if task_name not in task_data:
            logger.warning(f"Task {task_name} not loaded, skipping")
            continue

        elapsed = time.time() - start_time
        if elapsed > 4200:
            logger.warning("Time budget exceeded")
            break

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Task: {task_name} ({task_cfg['type']})")

        for use_prmp in [False, True]:
            variant = "prmp" if use_prmp else "standard"
            elapsed = time.time() - start_time
            if elapsed > 4200:
                break

            logger.info(f"\n--- {task_name}/{variant} ---")
            t_run = time.time()

            try:
                torch.manual_seed(42)
                np.random.seed(42)
                if HAS_GPU:
                    torch.cuda.manual_seed(42)

                model = HeteroGNN(
                    node_feat_dims=node_feat_dims,
                    edge_types=edge_types,
                    hidden=CHANNELS,
                    use_prmp=use_prmp,
                )
                model.set_pred_head(out_dim=1)
                model = model.to(DEVICE)

                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"  Params: {param_count:,}")

                result = train_full_graph(
                    model, x_dict_dev, edge_dict_dev,
                    task_data[task_name], fk_edge_map,
                    edge_index_dict_cpu, f"{task_name}/{variant}",
                    start_time=start_time, time_budget=5400,
                )

                run_time = time.time() - t_run
                best_val = result["best_val"]
                logger.info(f"  {variant}: best_val={best_val} in {run_time:.1f}s")

                ts_key = f"{task_name}/{variant}"
                all_r2_timeseries[ts_key] = result["r2_timeseries"]

                # Build examples
                for rec in result["epoch_records"]:
                    example = {
                        "input": json.dumps({
                            "dataset": "rel-stack", "task": task_name,
                            "variant": variant, "task_type": task_cfg["type"],
                            "metric": task_cfg["metric"], "epoch": rec["epoch"],
                        }),
                        "output": json.dumps({
                            "train_loss": rec["train_loss"],
                            "val_metric": rec["val_metric"],
                            "prmp_r2": rec["prmp_r2"],
                            "r2_snapshot": rec["r2_snapshot"],
                        }),
                        "metadata_dataset": "rel-stack",
                        "metadata_task": task_name,
                        "metadata_variant": variant,
                        "metadata_epoch": rec["epoch"],
                        "predict_baseline": str(rec["val_metric"]) if variant == "standard" else "",
                        "predict_our_method": str(rec["val_metric"]) if variant == "prmp" else "",
                    }
                    all_examples.append(example)

                # Run summary
                all_examples.append({
                    "input": json.dumps({
                        "type": "run_summary", "dataset": "rel-stack",
                        "task": task_name, "variant": variant,
                    }),
                    "output": json.dumps({
                        "best_val": round(float(best_val), 6) if best_val is not None and np.isfinite(float(best_val)) else None,
                        "num_epochs": len(result["epoch_records"]),
                        "training_time_s": round(run_time, 1),
                    }),
                    "metadata_dataset": "rel-stack",
                    "metadata_task": task_name,
                    "metadata_variant": variant,
                    "metadata_epoch": -1,
                    "predict_baseline": str(round(float(best_val), 6)) if variant == "standard" and best_val is not None else "",
                    "predict_our_method": str(round(float(best_val), 6)) if variant == "prmp" and best_val is not None else "",
                })

                all_run_summaries.append({
                    "task": task_name, "variant": variant,
                    "task_type": task_cfg["type"],
                    "metric": task_cfg["metric"],
                    "higher_better": task_cfg["higher_better"],
                    "best_val": float(best_val) if best_val is not None else None,
                })

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  OOM for {variant}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                logger.exception(f"  Failed: {task_name}/{variant}")

            try:
                del model
            except Exception:
                pass
            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

    # Step 4: Analysis
    logger.info(f"\n{'=' * 60}")
    logger.info("Analysis...")

    # PRMP vs Standard
    per_task = {}
    for s in all_run_summaries:
        key = s["task"]
        if key not in per_task:
            per_task[key] = {"standard": None, "prmp": None,
                             "type": s["task_type"], "metric": s["metric"],
                             "higher_better": s["higher_better"]}
        per_task[key][s["variant"]] = s["best_val"]

    per_task_analysis = {}
    for tk, info in per_task.items():
        sv, pv = info["standard"], info["prmp"]
        if sv is not None and pv is not None:
            delta = (pv - sv) if info["higher_better"] else (sv - pv)
            per_task_analysis[tk] = {
                "standard": round(sv, 6), "prmp": round(pv, 6),
                "delta": round(delta, 6), "prmp_better": delta > 0,
            }
            logger.info(f"  {tk}: std={sv:.4f} prmp={pv:.4f} delta={delta:.4f}")

    # Embedding R² summary
    final_emb_r2 = []
    for ts_key, ts_data in all_r2_timeseries.items():
        for fk_name, layers in ts_data.items():
            if "layer_2" in layers and layers["layer_2"]["ridge_r2"]:
                final_emb_r2.append(layers["layer_2"]["ridge_r2"][-1])

    raw_r2_values = [info["ridge_r2"] for info in raw_baseline.values()]

    confound_analysis = {
        "raw_parent_dims": {k: v["parent_dim"] for k, v in raw_baseline.items()},
        "embedding_dims": CHANNELS,
        "raw_r2_mean": round(float(np.mean(raw_r2_values)), 6) if raw_r2_values else 0.0,
        "embedding_r2_mean": round(float(np.mean(final_emb_r2)), 6) if final_emb_r2 else 0.0,
        "confound_resolved": bool(final_emb_r2 and np.mean(final_emb_r2) > np.mean(raw_r2_values) + 0.05) if raw_r2_values and final_emb_r2 else False,
    }

    key_findings = {
        "prmp_helps_post_votes": per_task_analysis.get("post-votes", {}).get("prmp_better", False),
        "prmp_helps_user_engagement": per_task_analysis.get("user-engagement", {}).get("prmp_better", False),
        "embedding_r2_mean": confound_analysis["embedding_r2_mean"],
        "raw_r2_mean": confound_analysis["raw_r2_mean"],
        "parent_dim1_resolved": confound_analysis["confound_resolved"],
    }

    comparison = {
        "amazon": {"embedding_r2_range": "0.2-0.6", "prmp_helps": True},
        "rel_stack": {
            "embedding_r2_mean": key_findings["embedding_r2_mean"],
            "prmp_helps_regression": key_findings["prmp_helps_post_votes"],
            "prmp_helps_classification": key_findings["prmp_helps_user_engagement"],
        },
    }

    # Analysis example
    all_examples.append({
        "input": json.dumps({"type": "analysis_summary"}),
        "output": json.dumps({
            "per_task_analysis": per_task_analysis,
            "confound_analysis": confound_analysis,
            "key_findings": key_findings,
            "comparison_with_amazon": comparison,
        }),
        "metadata_dataset": "rel-stack",
        "metadata_task": "analysis",
        "metadata_variant": "comparison",
        "metadata_epoch": -1,
        "predict_baseline": json.dumps(key_findings),
        "predict_our_method": json.dumps(key_findings),
    })

    # Step 5: Save Output
    output = {
        "metadata": {
            "method_name": "Embedding_Space_Cross_Table_Predictability_RelStack",
            "description": (
                "Embedding-space cross-table predictability on rel-stack. "
                "Trains Standard and PRMP HeteroGNN, measures R² trajectories. "
                "Tests revised regime theory: low embedding R² -> PRMP doesn't help."
            ),
            "dataset": "rel-stack",
            "model_config": {
                "channels": CHANNELS, "num_layers": NUM_LAYERS,
                "num_epochs": NUM_EPOCHS, "lr": LR,
                "checkpoint_every": CHECKPOINT_EVERY,
            },
            "gpu_used": HAS_GPU,
            "device": str(DEVICE),
            "raw_feature_baseline": raw_baseline,
            "embedding_predictability": all_r2_timeseries,
            "task_performance": per_task_analysis,
            "comparison_with_amazon": comparison,
            "parent_dim_confound_analysis": confound_analysis,
            "key_findings": key_findings,
            "runtime_seconds": round(time.time() - start_time, 1),
        },
        "datasets": [
            {"dataset": "rel_stack_embedding_r2", "examples": all_examples}
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nSaved {len(all_examples)} examples to {out_path}")
    logger.info(f"Total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
