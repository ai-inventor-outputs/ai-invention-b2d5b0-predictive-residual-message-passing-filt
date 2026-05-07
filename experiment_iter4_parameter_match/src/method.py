#!/usr/bin/env python3
"""Parameter-Matched Control Experiments: Isolating PRMP's Predict-Subtract Mechanism.

Runs 5 model variants (Standard SAGEConv, PRMP, Wide SAGEConv, Auxiliary MLP,
Skip-Residual) on Amazon Video Games and F1 datasets with matched parameter
counts to determine if PRMP's improvement comes from its predict-subtract
mechanism or merely from having more parameters.

128-dim hidden, 2 GNN layers, 3 seeds, reports RMSE/MAE +/- std and Cohen's d.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
WS = Path(__file__).resolve().parent
LOG_DIR = WS / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware Detection (cgroup-aware) ────────────────────────────────────────

def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
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
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")

if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(0.90, 0.90 * _total / _total))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f} GB")
else:
    VRAM_GB = 0

# RAM budget: 20GB is plenty for these small graphs
RAM_BUDGET = int(min(20, TOTAL_RAM_GB * 0.35) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, device={DEVICE}")
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")

# ── Configuration ────────────────────────────────────────────────────────────
HIDDEN_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 15
SEEDS = [42, 123, 456]
GRAD_CLIP = 5.0

# Feature dimensions from Amazon data
REVIEW_FEAT_DIM = 21
PARENT_FEAT_DIM = 5

# Paths
AMAZON_DATA_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
    "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus/full_data_out.json"
)
F1_DATA_DEP_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
    "/3_invention_loop/iter_2/gen_art/data_id5_it2__opus"
)
F1_ZIP_PATH = F1_DATA_DEP_DIR / "temp" / "datasets" / "f1db_csv.zip"
F1_DEP_DATA_PATH = F1_DATA_DEP_DIR / "full_data_out.json"
OUTPUT_DIR = WS


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER OPERATIONS (manual, no PyG dependency)
# ══════════════════════════════════════════════════════════════════════════════

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean aggregation: group src rows by index, compute mean per group."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    ones = torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(-1), ones)
    count = count.clamp(min=1)
    return out / count


# ══════════════════════════════════════════════════════════════════════════════
# CORE CONVOLUTION MODULES (5 variants)
# ══════════════════════════════════════════════════════════════════════════════

class StandardConv(nn.Module):
    """(A) Standard SAGEConv-like: mean-aggregate neighbor features, concat with self, linear."""

    def __init__(self, in_src: int, in_dst: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_src + in_dst, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        agg = scatter_mean(x_j, dst_idx, dim_size=x_dst.size(0))
        return F.relu(self.lin(torch.cat([x_dst, agg], dim=-1)))


class PRMPConv(nn.Module):
    """(B) Predictive Residual Message Passing convolution.

    Predict child from parent, subtract, normalize residual, aggregate.
    """

    def __init__(self, in_src: int, in_dst: int, out_channels: int):
        super().__init__()
        hidden = min(in_src, in_dst)
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_dst, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_src)
        )
        # CRITICAL: Zero-init final layer so residuals ~= raw features at start
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)
        self.norm = nn.LayerNorm(in_src)
        self.lin = nn.Linear(in_src + in_dst, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]       # source (child) features per edge
        x_i = x_dst[dst_idx]       # destination (parent) features per edge
        # Predict source from destination — DETACH parent to prevent gradient competition
        predicted = self.pred_mlp(x_i.detach())
        residual = x_j - predicted
        normalized = self.norm(residual)
        agg = scatter_mean(normalized, dst_idx, dim_size=x_dst.size(0))
        return F.relu(self.lin(torch.cat([x_dst, agg], dim=-1)))


class WideStandardConv(nn.Module):
    """(C) Same as StandardConv, but with wider hidden dim to match PRMP's param count."""

    def __init__(self, in_src: int, in_dst: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_src + in_dst, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        agg = scatter_mean(x_j, dst_idx, dim_size=x_dst.size(0))
        return F.relu(self.lin(torch.cat([x_dst, agg], dim=-1)))


class AuxMLPConv(nn.Module):
    """(D) Standard aggregation + extra MLP that transforms child features BEFORE aggregation.

    Same extra params as PRMP's pred_mlp, but NO predict-subtract mechanism.
    """

    def __init__(self, in_src: int, in_dst: int, out_channels: int):
        super().__init__()
        hidden = min(in_src, in_dst)
        self.child_mlp = nn.Sequential(
            nn.Linear(in_src, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_src)
        )
        self.norm = nn.LayerNorm(in_src)  # same as PRMP norm for fair comparison
        self.lin = nn.Linear(in_src + in_dst, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        transformed = self.norm(self.child_mlp(x_j))  # transform child, NO subtraction
        agg = scatter_mean(transformed, dst_idx, dim_size=x_dst.size(0))
        return F.relu(self.lin(torch.cat([x_dst, agg], dim=-1)))


class SkipResidualConv(nn.Module):
    """(E) Standard aggregation + skip connection: concat raw parent features to agg.

    Tests if benefit comes from parent->child information pathway, not predict-subtract.
    """

    def __init__(self, in_src: int, in_dst: int, out_channels: int):
        super().__init__()
        # Extra params: wider final linear to handle concat: agg + x_dst + x_dst_skip
        self.lin = nn.Linear(in_src + in_dst + in_dst, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        agg = scatter_mean(x_j, dst_idx, dim_size=x_dst.size(0))
        # Skip: concat x_dst twice (once as neighbor context, once as skip)
        return F.relu(self.lin(torch.cat([x_dst, agg, x_dst], dim=-1)))


# ══════════════════════════════════════════════════════════════════════════════
# HETEROGENEOUS CONVOLUTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class HeteroConvLayer(nn.Module):
    """Per-edge-type convolutions, sum outputs per destination node type."""

    def __init__(self, conv_dict: dict):
        super().__init__()
        self.convs = nn.ModuleDict()
        self._edge_types = list(conv_dict.keys())
        for etype, conv in conv_dict.items():
            key = '__'.join(etype)
            self.convs[key] = conv

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        out_dict: dict[str, list[torch.Tensor]] = defaultdict(list)
        for etype in self._edge_types:
            key = '__'.join(etype)
            src_type, _, dst_type = etype
            if etype not in edge_index_dict:
                continue
            edge_index = edge_index_dict[etype]
            conv = self.convs[key]
            out = conv(x_dict[src_type], x_dict[dst_type], edge_index)
            out_dict[dst_type].append(out)
        # Sum across edge types targeting the same node type
        result = {}
        for ntype, outs in out_dict.items():
            result[ntype] = torch.stack(outs).sum(dim=0)
        # Keep node types that didn't receive messages
        for ntype in x_dict:
            if ntype not in result:
                result[ntype] = x_dict[ntype]
        return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL HETEROGENEOUS GNN
# ══════════════════════════════════════════════════════════════════════════════

class HeteroGNN(nn.Module):
    """Heterogeneous GNN with per-node-type projections + prediction head."""

    def __init__(self, node_types_with_dims: dict[str, int], edge_types: list,
                 hidden_dim: int, num_layers: int, conv_cls,
                 target_node_type: str):
        super().__init__()
        self.target_node_type = target_node_type

        # Input projections (different feature dims per node type)
        self.proj = nn.ModuleDict()
        for ntype, feat_dim in node_types_with_dims.items():
            self.proj[ntype] = nn.Linear(feat_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for etype in edge_types:
                conv_dict[etype] = conv_cls(hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(HeteroConvLayer(conv_dict))

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h_dict = {}
        for ntype, x in x_dict.items():
            h_dict[ntype] = F.relu(self.proj[ntype](x))
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
        return self.head(h_dict[self.target_node_type]).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER COUNTING & WIDE DIM COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_wide_hidden_dim(node_dims: dict[str, int], edge_types: list,
                         target_prmp_params: int, target_node_type: str) -> int:
    """Binary search for wide_hidden_dim so WideStandardConv matches PRMP param count."""
    lo, hi = HIDDEN_DIM + 1, 400
    best_dim = HIDDEN_DIM + 10
    best_diff = float('inf')

    for _ in range(50):
        mid = (lo + hi) // 2
        if mid == lo and lo == hi:
            break
        try:
            model = HeteroGNN(node_dims, edge_types, mid, NUM_LAYERS,
                              WideStandardConv, target_node_type).to('cpu')
            p = count_params(model)
            diff = abs(p - target_prmp_params)
            if diff < best_diff:
                best_diff = diff
                best_dim = mid
            if p < target_prmp_params:
                lo = mid + 1
            elif p > target_prmp_params:
                hi = mid - 1
            else:
                return mid
            del model
        except Exception:
            hi = mid - 1

    # Accept if within 5% of target
    pct_diff = best_diff / max(target_prmp_params, 1) * 100
    logger.info(f"  Wide hidden dim: {best_dim}, param diff: {best_diff} ({pct_diff:.1f}%)")
    return best_dim


# ══════════════════════════════════════════════════════════════════════════════
# AMAZON DATA LOADING & GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def load_amazon_data(data_path: Path) -> tuple:
    """Load and parse Amazon dataset from JSON."""
    logger.info(f"Loading Amazon data from {data_path}")
    data = json.loads(data_path.read_text())
    examples = data['datasets'][0]['examples']
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def parse_amazon_examples(examples: list) -> tuple:
    """Parse example inputs/outputs into numpy arrays."""
    features_list = []
    targets_list = []
    product_ids = []
    customer_ids = []
    folds = []

    for ex in examples:
        feat_dict = json.loads(ex['input'])
        feat_vals = [float(v) for v in feat_dict.values()]
        features_list.append(feat_vals)
        targets_list.append(float(ex['output']))
        product_ids.append(ex['metadata_product_id'])
        customer_ids.append(ex['metadata_customer_id'])
        folds.append(ex['metadata_fold'])

    features = np.array(features_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    return features, targets, product_ids, customer_ids, folds


def build_amazon_graph(features: np.ndarray, targets: np.ndarray,
                       product_ids: list, customer_ids: list,
                       folds: list, device: torch.device) -> tuple:
    """Build heterogeneous graph tensors for Amazon dataset."""
    num_reviews = len(features)
    unique_products = sorted(set(product_ids))
    unique_customers = sorted(set(customer_ids))
    prod_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    cust_to_idx = {cid: i for i, cid in enumerate(unique_customers)}
    num_products = len(unique_products)
    num_customers = len(unique_customers)
    logger.info(f"Amazon graph: {num_reviews} reviews, {num_products} products, "
                f"{num_customers} customers")

    review_x = torch.tensor(features, dtype=torch.float32)

    review_to_prod_src, review_to_prod_dst = [], []
    review_to_cust_src, review_to_cust_dst = [], []
    prod_reviews: dict[int, list[int]] = defaultdict(list)
    cust_reviews: dict[int, list[int]] = defaultdict(list)

    for i in range(num_reviews):
        pid = prod_to_idx[product_ids[i]]
        cid = cust_to_idx[customer_ids[i]]
        review_to_prod_src.append(i)
        review_to_prod_dst.append(pid)
        review_to_cust_src.append(i)
        review_to_cust_dst.append(cid)
        prod_reviews[pid].append(i)
        cust_reviews[cid].append(i)

    # Build product features by aggregating reviews (5 dims)
    prod_features = np.zeros((num_products, PARENT_FEAT_DIM), dtype=np.float32)
    for pid, rev_idxs in prod_reviews.items():
        revs = features[rev_idxs]
        rats = targets[rev_idxs]
        prod_features[pid, 0] = np.mean(rats)
        prod_features[pid, 1] = np.std(rats) if len(rats) > 1 else 0.0
        prod_features[pid, 2] = float(len(rats))
        prod_features[pid, 3] = np.mean(revs[:, 19]) if revs.shape[1] > 19 else 0.0
        prod_features[pid, 4] = np.mean(revs[:, 20]) if revs.shape[1] > 20 else 0.0

    # Build customer features by aggregating reviews (5 dims)
    cust_features = np.zeros((num_customers, PARENT_FEAT_DIM), dtype=np.float32)
    for cid, rev_idxs in cust_reviews.items():
        revs = features[rev_idxs]
        rats = targets[rev_idxs]
        cust_features[cid, 0] = np.mean(rats)
        cust_features[cid, 1] = np.std(rats) if len(rats) > 1 else 0.0
        cust_features[cid, 2] = float(len(rats))
        cust_features[cid, 3] = np.mean(revs[:, 19]) if revs.shape[1] > 19 else 0.0
        cust_features[cid, 4] = np.mean(revs[:, 20]) if revs.shape[1] > 20 else 0.0

    product_x = torch.tensor(prod_features, dtype=torch.float32)
    customer_x = torch.tensor(cust_features, dtype=torch.float32)

    edge_index_dict = {
        ('review', 'belongs_to', 'product'): torch.tensor(
            [review_to_prod_src, review_to_prod_dst], dtype=torch.long),
        ('review', 'written_by', 'customer'): torch.tensor(
            [review_to_cust_src, review_to_cust_dst], dtype=torch.long),
        ('product', 'rev_belongs_to', 'review'): torch.tensor(
            [review_to_prod_dst, review_to_prod_src], dtype=torch.long),
        ('customer', 'rev_written_by', 'review'): torch.tensor(
            [review_to_cust_dst, review_to_cust_src], dtype=torch.long),
    }

    x_dict = {
        'review': review_x.to(device),
        'product': product_x.to(device),
        'customer': customer_x.to(device),
    }
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    folds_arr = np.array(folds)
    train_mask = torch.tensor(np.isin(folds_arr, [0, 1, 2]), dtype=torch.bool).to(device)
    val_mask = torch.tensor(folds_arr == 3, dtype=torch.bool).to(device)
    test_mask = torch.tensor(folds_arr == 4, dtype=torch.bool).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).to(device)

    node_dims = {'review': REVIEW_FEAT_DIM, 'product': PARENT_FEAT_DIM,
                 'customer': PARENT_FEAT_DIM}

    logger.info(f"  Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, "
                f"Test: {test_mask.sum().item()}")
    return (x_dict, edge_index_dict, targets_tensor, train_mask, val_mask,
            test_mask, node_dims, list(edge_index_dict.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# F1 DATA LOADING & GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def load_f1_tables() -> dict[str, 'pd.DataFrame']:
    """Load Ergast F1 CSV tables from zip."""
    import pandas as pd

    csv_dir = WS / "ergast_csv"
    if not csv_dir.exists() or not list(csv_dir.glob("*.csv")):
        if F1_ZIP_PATH.exists():
            csv_dir.mkdir(exist_ok=True)
            logger.info(f"Extracting {F1_ZIP_PATH} to {csv_dir}")
            with zipfile.ZipFile(F1_ZIP_PATH, 'r') as zf:
                zf.extractall(csv_dir)
        else:
            logger.error(f"F1 CSV zip not found at {F1_ZIP_PATH}")
            return {}

    tables = {}
    for f in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False, na_values=["\\N", "NULL", ""])
            tables[f.stem] = df
            logger.debug(f"  Loaded {f.stem}: {len(df)} rows x {len(df.columns)} cols")
        except Exception:
            logger.exception(f"  Failed to load {f.name}")
    logger.info(f"Loaded {len(tables)} F1 tables")
    return tables


def build_f1_graph(tables: dict, dep_data: dict, device: torch.device) -> tuple:
    """Build heterogeneous graph from Ergast tables following FK links."""
    import pandas as pd

    node_dims = {}
    x_dict_raw = {}
    node_counts = {}

    # Build node features per table
    for tbl_meta in dep_data["metadata"]["tables"]:
        name = tbl_meta["name"]
        if name not in tables:
            continue
        df = tables[name]

        # Use numeric columns (excluding IDs) as features
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        id_cols = [c["name"] for c in tbl_meta.get("columns", []) if c.get("is_id")]
        feat_cols = [c for c in num_cols if c not in id_cols]

        if not feat_cols:
            x = np.ones((len(df), 1), dtype=np.float32)
            node_dims[name] = 1
        else:
            vals = df[feat_cols].values.astype(np.float32)
            vals = np.nan_to_num(vals, 0.0)
            std = vals.std(axis=0) + 1e-8
            vals = (vals - vals.mean(axis=0)) / std
            x = vals
            node_dims[name] = x.shape[1]

        x_dict_raw[name] = torch.from_numpy(x).to(device)
        node_counts[name] = len(df)

    # Build edges from FK links
    edge_index_dict = {}
    for fk in dep_data["metadata"]["fk_links"]:
        child_tbl = fk["child_table"]
        parent_tbl = fk["parent_table"]
        fk_col = fk["child_fk_col"]
        pk_col = fk.get("parent_pk_col", fk_col)

        if child_tbl not in tables or parent_tbl not in tables:
            continue
        child_df = tables[child_tbl]
        parent_df = tables[parent_tbl]
        if fk_col not in child_df.columns or pk_col not in parent_df.columns:
            continue

        # Map parent PK values to indices
        parent_ids = parent_df[pk_col].values
        pk_to_idx = {}
        for idx_val, pk in enumerate(parent_ids):
            try:
                pk_to_idx[int(pk)] = idx_val
            except (ValueError, TypeError):
                pk_to_idx[str(pk)] = idx_val

        src_indices, dst_indices = [], []
        for child_idx, fk_val in enumerate(child_df[fk_col].values):
            try:
                key = int(fk_val)
            except (ValueError, TypeError):
                key = str(fk_val)
            if key in pk_to_idx:
                src_indices.append(child_idx)
                dst_indices.append(pk_to_idx[key])

        if not src_indices:
            continue

        rel_name = f"fk_{fk_col}"
        etype_fwd = (child_tbl, rel_name, parent_tbl)
        etype_rev = (parent_tbl, f"rev_fk_{fk_col}", child_tbl)

        edge_index_dict[etype_fwd] = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long).to(device)
        edge_index_dict[etype_rev] = torch.tensor(
            [dst_indices, src_indices], dtype=torch.long).to(device)

    # Create task labels: driver-position (mean finishing position)
    target_node_type = "drivers"
    results_df = tables.get("results", pd.DataFrame())
    drivers_df = tables.get("drivers", pd.DataFrame())
    n_drivers = len(drivers_df)

    labels = torch.full((n_drivers,), 15.0, dtype=torch.float32)
    if "positionOrder" in results_df.columns and "driverId" in results_df.columns:
        driver_pos = results_df.groupby("driverId")["positionOrder"].mean()
        driver_ids = drivers_df["driverId"].values
        for i, did in enumerate(driver_ids):
            if did in driver_pos.index:
                labels[i] = float(driver_pos.loc[did])
    labels = labels.to(device)

    # Split: 70/15/15 random with fixed permutation
    torch.manual_seed(42)
    perm = torch.randperm(n_drivers)
    n_train = int(0.7 * n_drivers)
    n_val = int(0.15 * n_drivers)
    train_mask = torch.zeros(n_drivers, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_drivers, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_drivers, dtype=torch.bool, device=device)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    edge_types = list(edge_index_dict.keys())
    logger.info(f"F1 graph: {len(x_dict_raw)} node types, {len(edge_types)} edge types, "
                f"{n_drivers} drivers")
    logger.info(f"  Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, "
                f"Test: {test_mask.sum().item()}")

    return (x_dict_raw, edge_index_dict, labels, train_mask, val_mask,
            test_mask, node_dims, edge_types, target_node_type)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_variant(
    variant_name: str, conv_cls, hidden_dim: int,
    node_dims: dict[str, int], edge_types: list, target_node_type: str,
    x_dict: dict, edge_index_dict: dict, targets: torch.Tensor,
    train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
    loss_fn, primary_metric_name: str, device: torch.device,
) -> dict:
    """Train and evaluate a single variant across all seeds. Returns per-seed results."""
    logger.info(f"\n--- {variant_name} (hidden={hidden_dim}) ---")
    results = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if HAS_GPU:
            torch.cuda.manual_seed(seed)

        model = HeteroGNN(node_dims, edge_types, hidden_dim, NUM_LAYERS,
                          conv_cls, target_node_type).to(device)
        param_count = count_params(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        best_epoch = 0

        t0 = time.time()
        for epoch in range(MAX_EPOCHS):
            # Train step
            model.train()
            optimizer.zero_grad()
            pred = model(x_dict, edge_index_dict)
            loss = loss_fn(pred[train_mask], targets[train_mask])

            if torch.isnan(loss):
                logger.warning(f"  {variant_name} seed={seed}: NaN loss at epoch {epoch}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            # Val step
            model.eval()
            with torch.no_grad():
                val_pred = model(x_dict, edge_index_dict)
                val_loss = loss_fn(val_pred[val_mask], targets[val_mask])

            scheduler.step(val_loss)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        train_time = time.time() - t0
        epochs_trained = epoch + 1

        # Test evaluation
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        model.eval()
        with torch.no_grad():
            test_pred = model(x_dict, edge_index_dict)

        pred_np = test_pred[test_mask].cpu().numpy()
        true_np = targets[test_mask].cpu().numpy()

        rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))
        mae = float(np.mean(np.abs(pred_np - true_np)))
        ss_res = float(np.sum((pred_np - true_np) ** 2))
        ss_tot = float(np.sum((true_np - np.mean(true_np)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

        results[seed] = {
            'rmse': round(rmse, 6),
            'mae': round(mae, 6),
            'r2': round(r2, 6),
            'param_count': param_count,
            'best_epoch': best_epoch,
            'epochs_trained': epochs_trained,
            'train_time_s': round(train_time, 1),
        }
        logger.info(f"  seed={seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, "
                     f"epochs={epochs_trained}, best_ep={best_epoch}, "
                     f"params={param_count}, time={train_time:.1f}s")

        del model, optimizer, scheduler, best_state
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cohens_d(group1_values: list[float], group2_values: list[float]) -> float:
    """Cohen's d effect size: (mean1 - mean2) / pooled_std."""
    n1, n2 = len(group1_values), len(group2_values)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1_values), np.mean(group2_values)
    s1, s2 = np.std(group1_values, ddof=1), np.std(group2_values, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return float((m1 - m2) / max(pooled_std, 1e-8))


def analyze_results(all_results: dict, primary_metric: str, dataset_name: str) -> dict:
    """Compute Cohen's d and summary stats comparing PRMP vs all controls."""
    analysis = {}
    prmp_results = all_results.get("B_prmp", {})
    if not prmp_results:
        return analysis

    prmp_values = [prmp_results[s][primary_metric] for s in SEEDS if s in prmp_results]

    for variant_name in ["A_standard_sage", "C_wide_sage", "D_aux_mlp", "E_skip_residual"]:
        ctrl = all_results.get(variant_name, {})
        if not ctrl:
            continue
        ctrl_values = [ctrl[s][primary_metric] for s in SEEDS if s in ctrl]
        if not ctrl_values or not prmp_values:
            continue

        d = compute_cohens_d(prmp_values, ctrl_values)
        prmp_mean = float(np.mean(prmp_values))
        ctrl_mean = float(np.mean(ctrl_values))
        improvement_pct = (ctrl_mean - prmp_mean) / max(abs(ctrl_mean), 1e-8) * 100

        analysis[f"prmp_vs_{variant_name}"] = {
            "cohens_d": round(d, 4),
            "prmp_mean": round(prmp_mean, 6),
            "prmp_std": round(float(np.std(prmp_values, ddof=1)) if len(prmp_values) > 1 else 0.0, 6),
            "control_mean": round(ctrl_mean, 6),
            "control_std": round(float(np.std(ctrl_values, ddof=1)) if len(ctrl_values) > 1 else 0.0, 6),
            "improvement_pct": round(improvement_pct, 4),
            "prmp_values": [round(v, 6) for v in prmp_values],
            "control_values": [round(v, 6) for v in ctrl_values],
            "prmp_better": prmp_mean < ctrl_mean,  # Lower is better for RMSE/MAE
        }

    # Interpretation
    prmp_vs_wide = analysis.get("prmp_vs_C_wide_sage", {})
    prmp_vs_aux = analysis.get("prmp_vs_D_aux_mlp", {})
    prmp_vs_std = analysis.get("prmp_vs_A_standard_sage", {})

    if prmp_vs_wide.get("prmp_better") and prmp_vs_aux.get("prmp_better"):
        interpretation = (
            f"PRMP's predict-subtract mechanism provides genuine benefit on {dataset_name}. "
            f"PRMP outperforms both parameter-matched controls (Wide SAGEConv and AuxMLP), "
            f"indicating improvement is from the mechanism, not extra parameters."
        )
    elif prmp_vs_std.get("prmp_better"):
        interpretation = (
            f"PRMP improves over Standard on {dataset_name} but the improvement "
            f"may be partially attributable to extra parameters since one or more "
            f"parameter-matched controls also improve over Standard."
        )
    else:
        interpretation = (
            f"PRMP does not clearly outperform Standard on {dataset_name}. "
            f"The predict-subtract mechanism may not benefit this dataset's structure."
        )
    analysis["interpretation"] = interpretation
    return analysis


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_output(amazon_results: dict, f1_results: dict,
                 amazon_analysis: dict, f1_analysis: dict,
                 param_counts: dict, wide_hidden_dim_amazon: int,
                 wide_hidden_dim_f1: int, total_runtime: float) -> dict:
    """Build method_out.json in exp_gen_sol_out schema format."""

    def make_variant_summary(results: dict, metric: str) -> dict:
        """Aggregate per-seed results for a variant."""
        vals = [results[s][metric] for s in SEEDS if s in results]
        return {
            "mean": round(float(np.mean(vals)), 6) if vals else None,
            "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0,
            "values": [round(v, 6) for v in vals],
        }

    # Build per-variant summaries for both datasets
    amazon_summaries = {}
    for vname, vresults in amazon_results.items():
        amazon_summaries[vname] = {
            "rmse": make_variant_summary(vresults, "rmse"),
            "mae": make_variant_summary(vresults, "mae"),
            "r2": make_variant_summary(vresults, "r2"),
            "param_count": vresults[SEEDS[0]]["param_count"] if SEEDS[0] in vresults else 0,
        }

    f1_summaries = {}
    for vname, vresults in f1_results.items():
        f1_summaries[vname] = {
            "rmse": make_variant_summary(vresults, "rmse"),
            "mae": make_variant_summary(vresults, "mae"),
            "r2": make_variant_summary(vresults, "r2"),
            "param_count": vresults[SEEDS[0]]["param_count"] if SEEDS[0] in vresults else 0,
        }

    # Build examples for Amazon dataset
    amazon_examples = []
    variant_names = list(amazon_results.keys())
    for vname in variant_names:
        for seed in SEEDS:
            if seed not in amazon_results[vname]:
                continue
            metrics = amazon_results[vname][seed]
            inp = json.dumps({
                "dataset": "amazon_video_games",
                "variant": vname,
                "seed": seed,
                "hidden_dim": HIDDEN_DIM if "wide" not in vname else wide_hidden_dim_amazon,
            })
            out = json.dumps({
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "best_epoch": metrics["best_epoch"],
                "param_count": metrics["param_count"],
            })

            # Build predict_ fields for each variant at this seed
            ex = {"input": inp, "output": out}
            for other_vname in variant_names:
                if seed in amazon_results.get(other_vname, {}):
                    other_m = amazon_results[other_vname][seed]
                    ex[f"predict_{other_vname}"] = json.dumps({
                        "rmse": other_m["rmse"],
                        "mae": other_m["mae"],
                        "r2": other_m["r2"],
                    })
                else:
                    ex[f"predict_{other_vname}"] = json.dumps({})

            ex["metadata_dataset"] = "amazon_video_games"
            ex["metadata_variant"] = vname
            ex["metadata_seed"] = seed
            ex["metadata_task_type"] = "parameter_control"
            amazon_examples.append(ex)

    # Amazon summary example
    amazon_summary_inp = json.dumps({
        "dataset": "amazon_video_games",
        "analysis_type": "parameter_matched_control_summary",
    })
    amazon_summary_out = json.dumps({
        "summaries": amazon_summaries,
        "cohens_d": {k: v for k, v in amazon_analysis.items() if k != "interpretation"},
        "interpretation": amazon_analysis.get("interpretation", ""),
    })
    amazon_summary_ex = {
        "input": amazon_summary_inp,
        "output": amazon_summary_out,
        "metadata_dataset": "amazon_video_games",
        "metadata_task_type": "parameter_control_summary",
    }
    for vname in variant_names:
        s = amazon_summaries.get(vname, {})
        amazon_summary_ex[f"predict_{vname}"] = json.dumps({
            "rmse_mean": s.get("rmse", {}).get("mean"),
            "rmse_std": s.get("rmse", {}).get("std"),
            "mae_mean": s.get("mae", {}).get("mean"),
            "param_count": s.get("param_count"),
        })
    amazon_examples.append(amazon_summary_ex)

    # Build examples for F1 dataset
    f1_examples = []
    f1_variant_names = list(f1_results.keys())
    for vname in f1_variant_names:
        for seed in SEEDS:
            if seed not in f1_results[vname]:
                continue
            metrics = f1_results[vname][seed]
            inp = json.dumps({
                "dataset": "rel-f1",
                "variant": vname,
                "seed": seed,
                "hidden_dim": HIDDEN_DIM if "wide" not in vname else wide_hidden_dim_f1,
            })
            out = json.dumps({
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "best_epoch": metrics["best_epoch"],
                "param_count": metrics["param_count"],
            })

            ex = {"input": inp, "output": out}
            for other_vname in f1_variant_names:
                if seed in f1_results.get(other_vname, {}):
                    other_m = f1_results[other_vname][seed]
                    ex[f"predict_{other_vname}"] = json.dumps({
                        "rmse": other_m["rmse"],
                        "mae": other_m["mae"],
                        "r2": other_m["r2"],
                    })
                else:
                    ex[f"predict_{other_vname}"] = json.dumps({})

            ex["metadata_dataset"] = "rel-f1"
            ex["metadata_variant"] = vname
            ex["metadata_seed"] = seed
            ex["metadata_task_type"] = "parameter_control"
            f1_examples.append(ex)

    # F1 summary example
    f1_summary_inp = json.dumps({
        "dataset": "rel-f1",
        "analysis_type": "parameter_matched_control_summary",
    })
    f1_summary_out = json.dumps({
        "summaries": f1_summaries,
        "cohens_d": {k: v for k, v in f1_analysis.items() if k != "interpretation"},
        "interpretation": f1_analysis.get("interpretation", ""),
    })
    f1_summary_ex = {
        "input": f1_summary_inp,
        "output": f1_summary_out,
        "metadata_dataset": "rel-f1",
        "metadata_task_type": "parameter_control_summary",
    }
    for vname in f1_variant_names:
        s = f1_summaries.get(vname, {})
        f1_summary_ex[f"predict_{vname}"] = json.dumps({
            "rmse_mean": s.get("rmse", {}).get("mean"),
            "mae_mean": s.get("mae", {}).get("mean"),
            "param_count": s.get("param_count"),
        })
    f1_examples.append(f1_summary_ex)

    output = {
        "metadata": {
            "experiment_type": "parameter_matched_control",
            "datasets": ["amazon_video_games", "rel-f1"],
            "variants": {
                "A_standard_sage": "Standard SAGEConv-like mean aggregation",
                "B_prmp": "Predictive Residual Message Passing (predict-subtract)",
                "C_wide_sage": "Wider SAGEConv to match PRMP param count",
                "D_aux_mlp": "Standard agg + auxiliary MLP (same params, no subtraction)",
                "E_skip_residual": "Standard agg + parent skip connection",
            },
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "wide_hidden_dim_amazon": wide_hidden_dim_amazon,
                "wide_hidden_dim_f1": wide_hidden_dim_f1,
                "num_layers": NUM_LAYERS,
                "seeds": SEEDS,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
            },
            "parameter_counts": param_counts,
            "amazon_summaries": amazon_summaries,
            "f1_summaries": f1_summaries,
            "amazon_analysis": amazon_analysis,
            "f1_analysis": f1_analysis,
            "runtime_seconds": round(total_runtime, 1),
        },
        "datasets": [
            {
                "dataset": "amazon_parameter_controls",
                "examples": amazon_examples,
            },
            {
                "dataset": "f1_parameter_controls",
                "examples": f1_examples,
            },
        ],
    }
    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_global = time.time()
    logger.info("=" * 70)
    logger.info("PARAMETER-MATCHED CONTROL EXPERIMENT")
    logger.info("Isolating PRMP's Predict-Subtract Mechanism")
    logger.info("=" * 70)

    all_param_counts = {}

    # ══════════════════════════════════════════════════════════════════════
    # AMAZON DATASET
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("DATASET 1: Amazon Video Games")
    logger.info("=" * 70)

    # Load data
    amazon_examples = load_amazon_data(AMAZON_DATA_PATH)
    features, targets, product_ids, customer_ids, folds = parse_amazon_examples(amazon_examples)
    logger.info(f"Features: {features.shape}, targets range: [{targets.min():.1f}, {targets.max():.1f}]")

    # Build graph
    (az_x_dict, az_edge_dict, az_targets, az_train, az_val, az_test,
     az_node_dims, az_edge_types) = build_amazon_graph(
        features, targets, product_ids, customer_ids, folds, DEVICE)

    # Verify graph integrity
    for etype, ei in az_edge_dict.items():
        src_type, _, dst_type = etype
        assert ei[0].max() < az_x_dict[src_type].size(0), f"Invalid src in {etype}"
        assert ei[1].max() < az_x_dict[dst_type].size(0), f"Invalid dst in {etype}"
    logger.info("Amazon graph integrity verified")

    # Phase 5: Compute parameter counts & find wide_hidden_dim
    logger.info("\n--- Computing parameter counts ---")
    std_model = HeteroGNN(az_node_dims, az_edge_types, HIDDEN_DIM, NUM_LAYERS,
                          StandardConv, "review").to('cpu')
    prmp_model = HeteroGNN(az_node_dims, az_edge_types, HIDDEN_DIM, NUM_LAYERS,
                           PRMPConv, "review").to('cpu')
    aux_model = HeteroGNN(az_node_dims, az_edge_types, HIDDEN_DIM, NUM_LAYERS,
                          AuxMLPConv, "review").to('cpu')
    skip_model = HeteroGNN(az_node_dims, az_edge_types, HIDDEN_DIM, NUM_LAYERS,
                           SkipResidualConv, "review").to('cpu')

    std_params = count_params(std_model)
    prmp_params = count_params(prmp_model)
    aux_params = count_params(aux_model)
    skip_params = count_params(skip_model)

    logger.info(f"  Standard:     {std_params:>8,d} params")
    logger.info(f"  PRMP:         {prmp_params:>8,d} params")
    logger.info(f"  AuxMLP:       {aux_params:>8,d} params")
    logger.info(f"  SkipResidual: {skip_params:>8,d} params")
    logger.info(f"  PRMP extra:   {prmp_params - std_params:>8,d} params vs Standard")
    logger.info(f"  AuxMLP match: {abs(aux_params - prmp_params):>8,d} param diff vs PRMP")

    del std_model, prmp_model, aux_model, skip_model
    gc.collect()

    # Find wide hidden dim for Amazon
    wide_dim_az = find_wide_hidden_dim(az_node_dims, az_edge_types, prmp_params, "review")
    wide_model = HeteroGNN(az_node_dims, az_edge_types, wide_dim_az, NUM_LAYERS,
                           WideStandardConv, "review").to('cpu')
    wide_params = count_params(wide_model)
    del wide_model
    gc.collect()

    logger.info(f"  Wide SAGEConv: {wide_params:>8,d} params (hidden_dim={wide_dim_az})")

    all_param_counts["amazon"] = {
        "A_standard_sage": std_params,
        "B_prmp": prmp_params,
        "C_wide_sage": wide_params,
        "D_aux_mlp": aux_params,
        "E_skip_residual": skip_params,
        "wide_hidden_dim": wide_dim_az,
    }

    # Train all 5 variants on Amazon
    logger.info("\n--- Training Amazon variants ---")
    loss_fn_amazon = nn.MSELoss()
    amazon_results = {}

    variants_amazon = [
        ("A_standard_sage", StandardConv, HIDDEN_DIM),
        ("B_prmp", PRMPConv, HIDDEN_DIM),
        ("C_wide_sage", WideStandardConv, wide_dim_az),
        ("D_aux_mlp", AuxMLPConv, HIDDEN_DIM),
        ("E_skip_residual", SkipResidualConv, HIDDEN_DIM),
    ]

    for vname, conv_cls, hdim in variants_amazon:
        amazon_results[vname] = train_and_evaluate_variant(
            variant_name=vname, conv_cls=conv_cls, hidden_dim=hdim,
            node_dims=az_node_dims, edge_types=az_edge_types,
            target_node_type="review", x_dict=az_x_dict,
            edge_index_dict=az_edge_dict, targets=az_targets,
            train_mask=az_train, val_mask=az_val, test_mask=az_test,
            loss_fn=loss_fn_amazon, primary_metric_name="rmse",
            device=DEVICE)

    # Statistical analysis for Amazon
    amazon_analysis = analyze_results(amazon_results, "rmse", "Amazon Video Games")

    # Free Amazon graph memory
    del az_x_dict, az_edge_dict, az_targets, az_train, az_val, az_test
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # F1 DATASET
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("DATASET 2: Formula 1 (driver-position task)")
    logger.info("=" * 70)

    f1_results = {}
    f1_analysis = {}
    wide_dim_f1 = HIDDEN_DIM + 10  # default fallback

    try:
        # Load F1 tables
        tables = load_f1_tables()
        if not tables:
            raise FileNotFoundError("No F1 tables loaded")

        # Load F1 dependency data (FK link metadata)
        dep_data = json.loads(F1_DEP_DATA_PATH.read_text())

        # Build F1 graph
        (f1_x_dict, f1_edge_dict, f1_targets, f1_train, f1_val, f1_test,
         f1_node_dims, f1_edge_types, f1_target_type) = build_f1_graph(
            tables, dep_data, DEVICE)

        del tables  # Free table memory
        gc.collect()

        # Verify graph integrity
        for etype, ei in f1_edge_dict.items():
            src_type, _, dst_type = etype
            assert ei[0].max() < f1_x_dict[src_type].size(0), f"Invalid src in {etype}"
            assert ei[1].max() < f1_x_dict[dst_type].size(0), f"Invalid dst in {etype}"
        logger.info("F1 graph integrity verified")

        # Compute F1 param counts
        logger.info("\n--- F1 parameter counts ---")
        f1_std_model = HeteroGNN(f1_node_dims, f1_edge_types, HIDDEN_DIM, NUM_LAYERS,
                                  StandardConv, f1_target_type).to('cpu')
        f1_prmp_model = HeteroGNN(f1_node_dims, f1_edge_types, HIDDEN_DIM, NUM_LAYERS,
                                   PRMPConv, f1_target_type).to('cpu')
        f1_std_params = count_params(f1_std_model)
        f1_prmp_params = count_params(f1_prmp_model)
        logger.info(f"  Standard: {f1_std_params:,d}, PRMP: {f1_prmp_params:,d}")
        del f1_std_model, f1_prmp_model
        gc.collect()

        # Find wide dim for F1
        wide_dim_f1 = find_wide_hidden_dim(f1_node_dims, f1_edge_types,
                                           f1_prmp_params, f1_target_type)

        # Verify F1 param counts
        f1_aux_model = HeteroGNN(f1_node_dims, f1_edge_types, HIDDEN_DIM, NUM_LAYERS,
                                  AuxMLPConv, f1_target_type).to('cpu')
        f1_skip_model = HeteroGNN(f1_node_dims, f1_edge_types, HIDDEN_DIM, NUM_LAYERS,
                                   SkipResidualConv, f1_target_type).to('cpu')
        f1_wide_model = HeteroGNN(f1_node_dims, f1_edge_types, wide_dim_f1, NUM_LAYERS,
                                   WideStandardConv, f1_target_type).to('cpu')

        f1_aux_params = count_params(f1_aux_model)
        f1_skip_params = count_params(f1_skip_model)
        f1_wide_params = count_params(f1_wide_model)
        del f1_aux_model, f1_skip_model, f1_wide_model
        gc.collect()

        logger.info(f"  AuxMLP: {f1_aux_params:,d}, SkipRes: {f1_skip_params:,d}, "
                     f"Wide (dim={wide_dim_f1}): {f1_wide_params:,d}")

        all_param_counts["f1"] = {
            "A_standard_sage": f1_std_params,
            "B_prmp": f1_prmp_params,
            "C_wide_sage": f1_wide_params,
            "D_aux_mlp": f1_aux_params,
            "E_skip_residual": f1_skip_params,
            "wide_hidden_dim": wide_dim_f1,
        }

        # Train all 5 variants on F1
        logger.info("\n--- Training F1 variants ---")
        loss_fn_f1 = nn.L1Loss()  # MAE for driver-position regression

        variants_f1 = [
            ("A_standard_sage", StandardConv, HIDDEN_DIM),
            ("B_prmp", PRMPConv, HIDDEN_DIM),
            ("C_wide_sage", WideStandardConv, wide_dim_f1),
            ("D_aux_mlp", AuxMLPConv, HIDDEN_DIM),
            ("E_skip_residual", SkipResidualConv, HIDDEN_DIM),
        ]

        for vname, conv_cls, hdim in variants_f1:
            f1_results[vname] = train_and_evaluate_variant(
                variant_name=vname, conv_cls=conv_cls, hidden_dim=hdim,
                node_dims=f1_node_dims, edge_types=f1_edge_types,
                target_node_type=f1_target_type, x_dict=f1_x_dict,
                edge_index_dict=f1_edge_dict, targets=f1_targets,
                train_mask=f1_train, val_mask=f1_val, test_mask=f1_test,
                loss_fn=loss_fn_f1, primary_metric_name="mae",
                device=DEVICE)

        # Statistical analysis for F1
        f1_analysis = analyze_results(f1_results, "mae", "F1 driver-position")

        del f1_x_dict, f1_edge_dict, f1_targets, f1_train, f1_val, f1_test
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    except Exception:
        logger.exception("F1 dataset failed — falling back to Amazon-only results")
        f1_analysis = {"interpretation": "F1 experiment failed, see logs for details"}

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════
    total_runtime = time.time() - t_global
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING OUTPUT")
    logger.info("=" * 70)

    output = build_output(
        amazon_results=amazon_results,
        f1_results=f1_results,
        amazon_analysis=amazon_analysis,
        f1_analysis=f1_analysis,
        param_counts=all_param_counts,
        wide_hidden_dim_amazon=wide_dim_az,
        wide_hidden_dim_f1=wide_dim_f1,
        total_runtime=total_runtime,
    )

    # Write output
    out_path = OUTPUT_DIR / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Output written to {out_path} ({file_size_mb:.1f} MB)")

    # ── Summary Tables ──────────────────────────────────────────────────
    logger.info(f"\nTotal runtime: {total_runtime:.1f} seconds ({total_runtime / 60:.1f} minutes)")

    for dataset_name, results_dict, metric in [
        ("Amazon", amazon_results, "rmse"),
        ("F1", f1_results, "mae"),
    ]:
        if not results_dict:
            continue
        logger.info(f"\n=== {dataset_name} RESULTS ({metric.upper()}) ===")
        logger.info(f"{'Variant':<22} {'Params':>10} {metric.upper():>14} {'MAE':>14} {'R2':>14}")
        logger.info("-" * 78)
        for vname in results_dict:
            vals_m = [results_dict[vname][s][metric] for s in SEEDS if s in results_dict[vname]]
            vals_mae = [results_dict[vname][s]["mae"] for s in SEEDS if s in results_dict[vname]]
            vals_r2 = [results_dict[vname][s]["r2"] for s in SEEDS if s in results_dict[vname]]
            pc = results_dict[vname][SEEDS[0]]["param_count"] if SEEDS[0] in results_dict[vname] else 0
            logger.info(
                f"{vname:<22} {pc:>10,d} "
                f"{np.mean(vals_m):>7.4f}+/-{np.std(vals_m):.4f} "
                f"{np.mean(vals_mae):>7.4f}+/-{np.std(vals_mae):.4f} "
                f"{np.mean(vals_r2):>7.4f}+/-{np.std(vals_r2):.4f}"
            )

    # Print analysis
    for name, analysis in [("Amazon", amazon_analysis), ("F1", f1_analysis)]:
        interp = analysis.get("interpretation", "N/A")
        logger.info(f"\n{name} interpretation: {interp}")
        for key, val in analysis.items():
            if key == "interpretation":
                continue
            if isinstance(val, dict):
                d = val.get("cohens_d", "?")
                imp = val.get("improvement_pct", "?")
                better = val.get("prmp_better", "?")
                logger.info(f"  {key}: d={d}, improvement={imp}%, PRMP_better={better}")


if __name__ == "__main__":
    main()
