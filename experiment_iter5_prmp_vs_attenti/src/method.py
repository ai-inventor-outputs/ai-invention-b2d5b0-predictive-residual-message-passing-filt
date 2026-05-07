#!/usr/bin/env python3
"""PRMP vs Attention-Based Aggregation (GATConv) Comparison on Amazon Video Games.

Compare 4 heterogeneous GNN variants on Amazon Video Games review-rating regression:
  (A) HeteroSAGEConv — mean aggregation baseline
  (B) HeteroGATConv — attention-weighted aggregation
  (C) PRMP+SAGE — predict-subtract with mean aggregation
  (D) PRMP+GAT — predict-subtract with attention aggregation

All GNN layers implemented in PURE PyTorch (no torch-geometric) to avoid
torch-scatter/torch-sparse compilation issues.

3 seeds x 200 epochs with early stopping. Reports RMSE, MAE, R2, parameter counts,
and pairwise Cohen's d.
"""

import copy
import gc
import itertools
import json
import math
import os
import resource
import sys
import time
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(str(log_dir / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
# ---------------------------------------------------------------------------

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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Set memory limits
RAM_BUDGET = int(12 * 1024**3)  # 12 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(16 * 1024**3)  # 16 GB of 20 GB
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'} ({VRAM_GB:.1f} GB VRAM), device={DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_5/gen_art/exp_id4_it5__opus")
DEP_WS = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
RAW_AMAZON_PATH = DEP_WS / "temp" / "datasets" / "full_LoganKells_amazon_product_reviews_video_games_train.json"
DEP_DATA_OUT = DEP_WS / "full_data_out.json"
OUTPUT_FILE = WS / "method_out.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 20
SEEDS = [42, 123, 456]
GAT_HEADS = 4
GAT_DIM_PER_HEAD = HIDDEN_DIM // GAT_HEADS  # 32
TEXT_HASH_DIM = 16


# ===========================================================================
# PHASE 1: Data Loading & Graph Construction
# ===========================================================================

def parse_helpful(val) -> tuple:
    """Parse helpful string like '[8, 12]' into (up, total)."""
    try:
        if isinstance(val, str):
            v = eval(val)
            return int(v[0]), int(v[1])
        elif isinstance(val, list):
            return int(val[0]), int(val[1])
    except Exception:
        pass
    return 0, 0


def encode_text_hash(series: pd.Series, n_features: int = TEXT_HASH_DIM) -> np.ndarray:
    """Simple hash-based text encoding."""
    result = np.zeros((len(series), n_features), dtype=np.float32)
    for i, text in enumerate(series.fillna("").astype(str)):
        if text:
            words = text.lower().split()[:50]
            for w in words:
                h = int(md5(w.encode()).hexdigest(), 16) % n_features
                result[i, h] += 1.0
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result /= norms
    return result


def build_graph_data() -> dict:
    """Load raw Amazon data and build heterogeneous graph tensors."""
    logger.info(f"Loading raw Amazon data from {RAW_AMAZON_PATH}")
    raw = pd.read_json(RAW_AMAZON_PATH)
    logger.info(f"Loaded {len(raw)} reviews, columns: {list(raw.columns)}")

    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])

    # --- Entity ID mappings ---
    unique_products = raw["asin"].unique()
    unique_customers = raw["reviewerID"].unique()
    product_id_map = {pid: i for i, pid in enumerate(unique_products)}
    customer_id_map = {cid: i for i, cid in enumerate(unique_customers)}
    n_products = len(unique_products)
    n_customers = len(unique_customers)
    n_reviews = len(raw)
    logger.info(f"Entities: {n_products} products, {n_customers} customers, {n_reviews} reviews")

    # --- Review (child) node features --- 21 dimensions ---
    feature_arrays = []
    feature_names = []

    # Time features (3-dim)
    ts = pd.to_datetime(raw["unixReviewTime"], unit="s", errors="coerce")
    time_feats = np.column_stack([
        ts.dt.year.fillna(2000).values,
        ts.dt.month.fillna(1).values,
        ts.dt.dayofweek.fillna(0).values,
    ]).astype(np.float32)
    time_feats = StandardScaler().fit_transform(time_feats)
    feature_arrays.append(time_feats)
    feature_names.extend(["time_year", "time_month", "time_dayofweek"])

    # Summary hash (16-dim)
    summary_feats = encode_text_hash(raw["summary"], n_features=TEXT_HASH_DIM)
    feature_arrays.append(summary_feats)
    feature_names.extend([f"summary_h{i}" for i in range(TEXT_HASH_DIM)])

    # Helpful votes (2-dim)
    helpful_parsed = raw["helpful"].apply(parse_helpful)
    helpful_up = helpful_parsed.apply(lambda x: x[0]).values.astype(np.float32).reshape(-1, 1)
    helpful_total = helpful_parsed.apply(lambda x: x[1]).values.astype(np.float32).reshape(-1, 1)
    feature_arrays.append(StandardScaler().fit_transform(helpful_up))
    feature_arrays.append(StandardScaler().fit_transform(helpful_total))
    feature_names.extend(["helpful_up", "helpful_total"])

    X_review = np.hstack(feature_arrays).astype(np.float32)
    logger.info(f"Review features: {X_review.shape} ({len(feature_names)} dims)")
    del feature_arrays, time_feats, summary_feats, helpful_parsed, helpful_up, helpful_total
    gc.collect()

    # --- Product node features --- 5 dimensions ---
    hp = raw["helpful"].apply(parse_helpful)
    agg_df = raw[["asin", "reviewerID", "overall"]].copy()
    agg_df["h_up"] = hp.apply(lambda x: x[0]).astype(np.float32)
    agg_df["h_total"] = hp.apply(lambda x: x[1]).astype(np.float32)

    prod_agg = agg_df.groupby("asin").agg(
        mean_r=("overall", "mean"), std_r=("overall", "std"),
        cnt=("overall", "count"), mean_hu=("h_up", "mean"), mean_ht=("h_total", "mean")
    ).astype(np.float32)
    prod_agg["std_r"] = prod_agg["std_r"].fillna(0)

    # Map to ordered array matching product_id_map
    X_product_raw = np.zeros((n_products, 5), dtype=np.float32)
    for pid, idx in product_id_map.items():
        if pid in prod_agg.index:
            X_product_raw[idx] = prod_agg.loc[pid].values
    X_product = StandardScaler().fit_transform(X_product_raw).astype(np.float32)
    logger.info(f"Product features: {X_product.shape}")

    # --- Customer node features --- 5 dimensions ---
    cust_agg = agg_df.groupby("reviewerID").agg(
        mean_r=("overall", "mean"), std_r=("overall", "std"),
        cnt=("overall", "count"), mean_hu=("h_up", "mean"), mean_ht=("h_total", "mean")
    ).astype(np.float32)
    cust_agg["std_r"] = cust_agg["std_r"].fillna(0)

    X_customer_raw = np.zeros((n_customers, 5), dtype=np.float32)
    for cid, idx in customer_id_map.items():
        if cid in cust_agg.index:
            X_customer_raw[idx] = cust_agg.loc[cid].values
    X_customer = StandardScaler().fit_transform(X_customer_raw).astype(np.float32)
    logger.info(f"Customer features: {X_customer.shape}")

    del agg_df, hp, prod_agg, cust_agg, X_product_raw, X_customer_raw
    gc.collect()

    # --- Edge indices ---
    review_product_dst = np.array([product_id_map[a] for a in raw["asin"].values], dtype=np.int64)
    review_customer_dst = np.array([customer_id_map[c] for c in raw["reviewerID"].values], dtype=np.int64)
    review_src = np.arange(n_reviews, dtype=np.int64)

    # Ratings target
    ratings = raw["overall"].values.astype(np.float32)

    # --- Train/Val/Test split --- 80/10/10 on review nodes ---
    n = n_reviews
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    train_mask = perm[:int(0.8 * n)]
    val_mask = perm[int(0.8 * n):int(0.9 * n)]
    test_mask = perm[int(0.9 * n):]

    logger.info(f"Split: train={len(train_mask)}, val={len(val_mask)}, test={len(test_mask)}")

    # --- Cardinality stats ---
    prod_card = raw.groupby("asin").size()
    cust_card = raw.groupby("reviewerID").size()
    cardinality = {
        "product_to_review_mean": round(float(prod_card.mean()), 4),
        "customer_to_review_mean": round(float(cust_card.mean()), 4),
    }

    # --- Build graph dict ---
    graph = {
        "x_product": torch.tensor(X_product, dtype=torch.float32),
        "x_customer": torch.tensor(X_customer, dtype=torch.float32),
        "x_review": torch.tensor(X_review, dtype=torch.float32),
        "y": torch.tensor(ratings, dtype=torch.float32),
        # Edge: review -> product (child -> parent)
        "edge_review_to_product_src": torch.tensor(review_src, dtype=torch.long),
        "edge_review_to_product_dst": torch.tensor(review_product_dst, dtype=torch.long),
        # Edge: product -> review (parent -> child) = reverse
        "edge_product_to_review_src": torch.tensor(review_product_dst, dtype=torch.long),
        "edge_product_to_review_dst": torch.tensor(review_src, dtype=torch.long),
        # Edge: review -> customer (child -> parent)
        "edge_review_to_customer_src": torch.tensor(review_src, dtype=torch.long),
        "edge_review_to_customer_dst": torch.tensor(review_customer_dst, dtype=torch.long),
        # Edge: customer -> review (parent -> child) = reverse
        "edge_customer_to_review_src": torch.tensor(review_customer_dst, dtype=torch.long),
        "edge_customer_to_review_dst": torch.tensor(review_src, dtype=torch.long),
        # Masks
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        # Counts
        "n_products": n_products,
        "n_customers": n_customers,
        "n_reviews": n_reviews,
        # Metadata
        "feature_names": feature_names,
        "cardinality": cardinality,
    }

    del raw, X_review, X_product, X_customer, ratings
    gc.collect()

    return graph


def move_graph_to_device(graph: dict, device: torch.device) -> dict:
    """Move all tensors in graph dict to device."""
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            graph[k] = v.to(device)
    return graph


# ===========================================================================
# PHASE 2: Pure PyTorch GNN Layers (no torch-geometric)
# ===========================================================================

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter mean aggregation: aggregate src by index, returning mean per target."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    count = count.clamp(min=1)
    return out / count


def scatter_add_fn(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter add aggregation."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


# --- SAGEConv (bipartite) ---
class BipartiteSAGEConv(nn.Module):
    """SAGEConv for bipartite edges: aggregates src features to dst nodes."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        neigh_feats = x_src[edge_src]  # [E, src_dim]
        agg = scatter_mean(neigh_feats, edge_dst, num_dst)  # [num_dst, src_dim]
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# --- GATConv (bipartite) ---
class BipartiteGATConv(nn.Module):
    """GATConv for bipartite edges with multi-head attention."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int,
                 heads: int = 4, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_per_head = out_dim // heads if concat else out_dim
        self.concat = concat

        self.lin_src = nn.Linear(src_dim, heads * self.out_per_head, bias=False)
        self.lin_dst = nn.Linear(dst_dim, heads * self.out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.randn(1, heads, self.out_per_head) * 0.01)
        self.att_dst = nn.Parameter(torch.randn(1, heads, self.out_per_head) * 0.01)
        self.lin_self = nn.Linear(dst_dim, heads * self.out_per_head if concat else out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        H = self.heads
        D = self.out_per_head

        src_feat = self.lin_src(x_src).view(-1, H, D)  # [N_src, H, D]
        dst_feat = self.lin_dst(x_dst).view(-1, H, D)  # [N_dst, H, D]

        src_edge = src_feat[edge_src]  # [E, H, D]
        dst_edge = dst_feat[edge_dst]  # [E, H, D]

        alpha_src = (src_edge * self.att_src).sum(dim=-1)  # [E, H]
        alpha_dst = (dst_edge * self.att_dst).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)

        alpha = self._edge_softmax(alpha, edge_dst, num_dst)

        msg = src_edge * alpha.unsqueeze(-1)  # [E, H, D]
        msg_flat = msg.view(-1, H * D)
        agg = scatter_add_fn(msg_flat, edge_dst, num_dst)

        out = agg + self.lin_self(x_dst)
        return out

    def _edge_softmax(self, alpha: torch.Tensor, edge_dst: torch.Tensor,
                      num_dst: int) -> torch.Tensor:
        # Max for numerical stability
        alpha_max = torch.full((num_dst, alpha.size(1)), float('-inf'),
                               device=alpha.device, dtype=alpha.dtype)
        alpha_max.scatter_reduce_(0, edge_dst.unsqueeze(1).expand_as(alpha),
                                  alpha, reduce="amax", include_self=True)
        alpha = alpha - alpha_max[edge_dst]
        alpha = alpha.exp()

        alpha_sum = torch.zeros(num_dst, alpha.size(1), device=alpha.device,
                                dtype=alpha.dtype)
        alpha_sum.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[edge_dst] + 1e-16)
        return alpha


# --- PRMPSAGEConv (predict-subtract + mean aggregation) ---
class PRMPSAGEConv(nn.Module):
    """PRMP: parent predicts child, subtract prediction, aggregate residuals."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        # Initialize near-zero for stability (residuals ~ raw features at start)
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        pred_src = self.pred_mlp(x_dst)  # [num_dst, src_dim]

        src_feats = x_src[edge_src]       # [E, src_dim]
        pred_feats = pred_src[edge_dst]   # [E, src_dim]
        residual = src_feats - pred_feats  # THE CORE PRMP OPERATION

        agg = scatter_mean(residual, edge_dst, num_dst)
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# --- PRMPGATConv (predict-subtract + attention aggregation) ---
class PRMPGATConv(nn.Module):
    """PRMP with attention-weighted aggregation of residuals."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int,
                 heads: int = 4, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_per_head = out_dim // heads if concat else out_dim
        self.concat = concat

        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.lin_res = nn.Linear(src_dim, heads * self.out_per_head, bias=False)
        self.lin_src = nn.Linear(src_dim, heads * self.out_per_head, bias=False)
        self.lin_dst = nn.Linear(dst_dim, heads * self.out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.randn(1, heads, self.out_per_head) * 0.01)
        self.att_dst = nn.Parameter(torch.randn(1, heads, self.out_per_head) * 0.01)
        self.lin_self = nn.Linear(dst_dim, heads * self.out_per_head if concat else out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        H = self.heads
        D = self.out_per_head

        # Predict child features from parent
        pred_src = self.pred_mlp(x_dst)  # [num_dst, src_dim]

        # Per-edge residual
        src_feats = x_src[edge_src]       # [E, src_dim]
        pred_feats = pred_src[edge_dst]   # [E, src_dim]
        residual = src_feats - pred_feats  # [E, src_dim]

        # Attention on original features
        src_feat = self.lin_src(x_src).view(-1, H, D)
        dst_feat = self.lin_dst(x_dst).view(-1, H, D)

        src_edge = src_feat[edge_src]
        dst_edge = dst_feat[edge_dst]
        alpha_src = (src_edge * self.att_src).sum(dim=-1)
        alpha_dst = (dst_edge * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
        alpha = self._edge_softmax(alpha, edge_dst, num_dst)

        # Transform residuals and apply attention
        res_transformed = self.lin_res(residual).view(-1, H, D)
        msg = res_transformed * alpha.unsqueeze(-1)
        msg_flat = msg.view(-1, H * D)
        agg = scatter_add_fn(msg_flat, edge_dst, num_dst)

        out = agg + self.lin_self(x_dst)
        return out

    def _edge_softmax(self, alpha: torch.Tensor, edge_dst: torch.Tensor,
                      num_dst: int) -> torch.Tensor:
        alpha_max = torch.full((num_dst, alpha.size(1)), float('-inf'),
                               device=alpha.device, dtype=alpha.dtype)
        alpha_max.scatter_reduce_(0, edge_dst.unsqueeze(1).expand_as(alpha),
                                  alpha, reduce="amax", include_self=True)
        alpha = alpha - alpha_max[edge_dst]
        alpha = alpha.exp()
        alpha_sum = torch.zeros(num_dst, alpha.size(1), device=alpha.device,
                                dtype=alpha.dtype)
        alpha_sum.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[edge_dst] + 1e-16)
        return alpha


# ===========================================================================
# PHASE 2b: Heterogeneous GNN Models (4 variants)
# ===========================================================================

# Edge type definitions used by all models
EDGE_TYPES = [
    ("review__rev_of__product", "review", "product",
     "edge_review_to_product_src", "edge_review_to_product_dst", "n_products"),
    ("product__has_review__review", "product", "review",
     "edge_product_to_review_src", "edge_product_to_review_dst", "n_reviews"),
    ("review__written_by__customer", "review", "customer",
     "edge_review_to_customer_src", "edge_review_to_customer_dst", "n_customers"),
    ("customer__wrote__review", "customer", "review",
     "edge_customer_to_review_src", "edge_customer_to_review_dst", "n_reviews"),
]

# Child->parent edges (get PRMP treatment)
PRMP_EDGE_KEYS = {"review__rev_of__product", "review__written_by__customer"}
# Parent->child edges (standard conv)
STD_EDGE_KEYS = {"product__has_review__review", "customer__wrote__review"}


class HeteroGNNLayer(nn.Module):
    """One layer of heterogeneous message passing across all edge types."""
    def __init__(self, conv_dict: dict):
        super().__init__()
        self.convs = nn.ModuleDict()
        for key, conv in conv_dict.items():
            self.convs[key] = conv

    def forward(self, x_dict: dict, graph: dict) -> dict:
        out_dict = {}
        for key, src_type, dst_type, esrc_key, edst_key, ndst_key in EDGE_TYPES:
            if key not in self.convs:
                continue
            conv = self.convs[key]
            result = conv(x_dict[src_type], x_dict[dst_type],
                          graph[esrc_key], graph[edst_key], graph[ndst_key])
            if dst_type not in out_dict:
                out_dict[dst_type] = result
            else:
                out_dict[dst_type] = out_dict[dst_type] + result
        return out_dict


def _make_hetero_model(conv_factory, hidden: int = HIDDEN_DIM):
    """Create a 2-layer heterogeneous GNN with given conv factory."""
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_lins = nn.ModuleDict({
                "product": nn.Linear(5, hidden),
                "customer": nn.Linear(5, hidden),
                "review": nn.Linear(21, hidden),
            })
            self.conv1 = HeteroGNNLayer({
                k: conv_factory(hidden, hidden, hidden)
                for k, *_ in EDGE_TYPES
            })
            self.conv2 = HeteroGNNLayer({
                k: conv_factory(hidden, hidden, hidden)
                for k, *_ in EDGE_TYPES
            })
            self.head = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, graph: dict) -> torch.Tensor:
            x_dict = {
                "product": F.relu(self.input_lins["product"](graph["x_product"])),
                "customer": F.relu(self.input_lins["customer"](graph["x_customer"])),
                "review": F.relu(self.input_lins["review"](graph["x_review"])),
            }
            out = self.conv1(x_dict, graph)
            x_dict = {k: F.relu(self.dropout(v)) for k, v in out.items()}
            out = self.conv2(x_dict, graph)
            x_dict = {k: F.relu(v) for k, v in out.items()}
            return self.head(x_dict["review"]).squeeze(-1)
    return _Model


def _make_prmp_hetero_model(prmp_factory, std_factory, hidden: int = HIDDEN_DIM):
    """Create a 2-layer hetero GNN with PRMP on child->parent and std on parent->child."""
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_lins = nn.ModuleDict({
                "product": nn.Linear(5, hidden),
                "customer": nn.Linear(5, hidden),
                "review": nn.Linear(21, hidden),
            })
            conv1_dict = {}
            conv2_dict = {}
            for k, *_ in EDGE_TYPES:
                if k in PRMP_EDGE_KEYS:
                    conv1_dict[k] = prmp_factory(hidden, hidden, hidden)
                    conv2_dict[k] = prmp_factory(hidden, hidden, hidden)
                else:
                    conv1_dict[k] = std_factory(hidden, hidden, hidden)
                    conv2_dict[k] = std_factory(hidden, hidden, hidden)
            self.conv1 = HeteroGNNLayer(conv1_dict)
            self.conv2 = HeteroGNNLayer(conv2_dict)
            self.head = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, graph: dict) -> torch.Tensor:
            x_dict = {
                "product": F.relu(self.input_lins["product"](graph["x_product"])),
                "customer": F.relu(self.input_lins["customer"](graph["x_customer"])),
                "review": F.relu(self.input_lins["review"](graph["x_review"])),
            }
            out = self.conv1(x_dict, graph)
            x_dict = {k: F.relu(self.dropout(v)) for k, v in out.items()}
            out = self.conv2(x_dict, graph)
            x_dict = {k: F.relu(v) for k, v in out.items()}
            return self.head(x_dict["review"]).squeeze(-1)
    return _Model


# Build model classes
def _sage_factory(s, d, o):
    return BipartiteSAGEConv(s, d, o)

def _gat_factory(s, d, o):
    return BipartiteGATConv(s, d, o, heads=GAT_HEADS)

def _prmp_sage_factory(s, d, o):
    return PRMPSAGEConv(s, d, o)

def _prmp_gat_factory(s, d, o):
    return PRMPGATConv(s, d, o, heads=GAT_HEADS)

HeteroSAGE = _make_hetero_model(_sage_factory)
HeteroGAT = _make_hetero_model(_gat_factory)
HeteroPRMPSAGE = _make_prmp_hetero_model(_prmp_sage_factory, _sage_factory)
HeteroPRMPGAT = _make_prmp_hetero_model(_prmp_gat_factory, _gat_factory)


# ===========================================================================
# PHASE 3: Training Loop
# ===========================================================================

def count_parameters(model: nn.Module) -> dict:
    """Count model parameters with breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
    return {"total": total, "trainable": trainable, "breakdown": breakdown}


def train_variant(variant_name: str, ModelClass, graph: dict,
                  device: torch.device) -> list:
    """Train one model variant across all seeds."""
    variant_results = []

    for seed in SEEDS:
        logger.info(f"--- Training {variant_name} | seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if HAS_GPU:
            torch.cuda.manual_seed(seed)

        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_epoch = 0

        train_mask = graph["train_mask"]
        val_mask = graph["val_mask"]
        test_mask = graph["test_mask"]
        y = graph["y"]

        t_start = time.time()

        for epoch in range(MAX_EPOCHS):
            # --- Train ---
            model.train()
            optimizer.zero_grad()
            pred = model(graph)
            loss = loss_fn(pred[train_mask], y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                pred = model(graph)
                val_loss = loss_fn(pred[val_mask], y[val_mask]).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"[{variant_name}|seed={seed}] Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"[{variant_name}|seed={seed}] Epoch {epoch}: "
                            f"train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

        elapsed = time.time() - t_start
        logger.info(f"[{variant_name}|seed={seed}] Training done in {elapsed:.1f}s, "
                     f"best_epoch={best_epoch}")

        # --- Test ---
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(graph)
            test_pred = pred[test_mask].cpu().numpy()
            test_true = y[test_mask].cpu().numpy()

        rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
        mae = float(np.mean(np.abs(test_pred - test_true)))
        ss_res = float(np.sum((test_true - test_pred) ** 2))
        ss_tot = float(np.sum((test_true - test_true.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        logger.info(f"[{variant_name}|seed={seed}] Test: RMSE={rmse:.4f}, "
                     f"MAE={mae:.4f}, R2={r2:.4f}")

        variant_results.append({
            "seed": seed,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
            "best_epoch": best_epoch,
            "train_time_s": round(elapsed, 1),
        })

        del model, optimizer, best_state
        if HAS_GPU:
            torch.cuda.empty_cache()
        gc.collect()

    return variant_results


# ===========================================================================
# PHASE 4-5: Parameter Counts & Cohen's d
# ===========================================================================

def compute_param_counts() -> dict:
    """Compute parameter counts for all 4 variants."""
    param_counts = {}
    for name, cls in [("SAGE", HeteroSAGE), ("GAT", HeteroGAT),
                      ("PRMP_SAGE", HeteroPRMPSAGE), ("PRMP_GAT", HeteroPRMPGAT)]:
        model = cls()
        info = count_parameters(model)
        param_counts[name] = info
        logger.info(f"Params [{name}]: total={info['total']:,}, "
                     f"trainable={info['trainable']:,}")
        del model
    gc.collect()
    return param_counts


def compute_cohens_d(results: dict) -> dict:
    """Pairwise Cohen's d on test RMSE distributions."""
    variant_names = list(results.keys())
    cohens_d = {}
    for v1, v2 in itertools.combinations(variant_names, 2):
        rmse1 = [r["rmse"] for r in results[v1]]
        rmse2 = [r["rmse"] for r in results[v2]]
        mean_diff = np.mean(rmse1) - np.mean(rmse2)
        pooled_std = np.sqrt((np.var(rmse1, ddof=1) + np.var(rmse2, ddof=1)) / 2)
        d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0
        cohens_d[f"{v1}_vs_{v2}"] = round(d, 4)
    return cohens_d


# ===========================================================================
# PHASE 6: Output in exp_gen_sol_out.json schema
# ===========================================================================

def build_output(results: dict, param_counts: dict, cohens_d: dict,
                 graph: dict) -> dict:
    """Build the exp_gen_sol_out.json format output.

    Schema requires: { "datasets": [ { "dataset": str, "examples": [ { "input": str, "output": str, "predict_*": str } ] } ] }
    """
    logger.info("Loading dependency data for output construction...")
    dep_data = json.loads(DEP_DATA_OUT.read_text())
    examples = dep_data["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} dependency examples")

    # Add variant mean RMSE as predictions per example
    for ex in examples:
        for variant_name, variant_results in results.items():
            mean_rmse = np.mean([r["rmse"] for r in variant_results])
            ex[f"predict_{variant_name}"] = str(round(mean_rmse, 4))

    # Build results summary for metadata
    results_summary = {}
    for variant_name, variant_results in results.items():
        rmses = [r["rmse"] for r in variant_results]
        maes = [r["mae"] for r in variant_results]
        r2s = [r["r2"] for r in variant_results]
        results_summary[variant_name] = {
            "per_seed": variant_results,
            "mean_rmse": round(float(np.mean(rmses)), 6),
            "std_rmse": round(float(np.std(rmses)), 6),
            "mean_mae": round(float(np.mean(maes)), 6),
            "std_mae": round(float(np.std(maes)), 6),
            "mean_r2": round(float(np.mean(r2s)), 6),
            "std_r2": round(float(np.std(r2s)), 6),
            "param_count": {
                "total": param_counts.get(variant_name, {}).get("total", 0),
                "trainable": param_counts.get(variant_name, {}).get("trainable", 0),
            },
        }

    method_out = {
        "metadata": {
            "description": "PRMP vs Attention (GATConv) comparison on Amazon Video Games",
            "dataset": "amazon_video_games",
            "num_reviews": graph["n_reviews"],
            "num_products": graph["n_products"],
            "num_customers": graph["n_customers"],
            "fk_cardinality": graph["cardinality"],
            "training_config": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "seeds": SEEDS,
                "split": "80/10/10",
                "loss": "MSE",
                "gat_heads": GAT_HEADS,
                "gat_dim_per_head": GAT_DIM_PER_HEAD,
            },
            "parameter_counts": {
                k: {"total": v["total"], "trainable": v["trainable"]}
                for k, v in param_counts.items()
            },
            "results_summary": results_summary,
            "pairwise_cohens_d": cohens_d,
            "interpretation": {
                "key_comparisons": {
                    "PRMP_SAGE_vs_GAT": "If PRMP_SAGE < GAT RMSE: predict-subtract is superior to attention",
                    "PRMP_GAT_vs_GAT": "If PRMP_GAT < GAT: mechanisms are complementary",
                    "PRMP_GAT_vs_PRMP_SAGE": "If PRMP_GAT < PRMP_SAGE: attention on residuals helps further",
                    "GAT_vs_SAGE": "Attention improvement over mean aggregation baseline",
                },
            },
        },
        "datasets": [
            {
                "dataset": "amazon_video_games_reviews",
                "examples": examples,
            },
        ],
    }

    return method_out


# ===========================================================================
# Main
# ===========================================================================

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("PRMP vs Attention-Based Aggregation Comparison")
    logger.info("Pure PyTorch implementation (no torch-geometric)")
    logger.info("=" * 70)

    t_total_start = time.time()

    # PHASE 1: Build graph
    logger.info("PHASE 1: Loading data and building heterogeneous graph...")
    graph = build_graph_data()
    graph = move_graph_to_device(graph, DEVICE)
    logger.info(f"Graph on {DEVICE}: review={graph['n_reviews']}, "
                f"product={graph['n_products']}, customer={graph['n_customers']}")

    # Smoke test: single forward pass for each variant
    logger.info("Smoke test: single forward pass per variant...")
    for name, cls in [("SAGE", HeteroSAGE), ("GAT", HeteroGAT),
                      ("PRMP_SAGE", HeteroPRMPSAGE), ("PRMP_GAT", HeteroPRMPGAT)]:
        model = cls().to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(graph)
        assert out.shape[0] == graph["n_reviews"], f"{name}: wrong output shape {out.shape}"
        assert not torch.isnan(out).any(), f"{name}: NaN in output"
        assert not torch.isinf(out).any(), f"{name}: Inf in output"
        logger.info(f"  {name}: output shape={out.shape}, "
                     f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        del model, out
        if HAS_GPU:
            torch.cuda.empty_cache()
    gc.collect()

    # PHASE 4: Parameter counts
    logger.info("PHASE 4: Computing parameter counts...")
    param_counts = compute_param_counts()

    # PHASE 3: Training
    logger.info("PHASE 3: Training all variants...")
    results = {}
    variants = [
        ("SAGE", HeteroSAGE),
        ("GAT", HeteroGAT),
        ("PRMP_SAGE", HeteroPRMPSAGE),
        ("PRMP_GAT", HeteroPRMPGAT),
    ]

    for variant_name, ModelClass in variants:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training variant: {variant_name}")
        logger.info(f"{'='*50}")
        variant_results = train_variant(variant_name, ModelClass, graph, DEVICE)
        results[variant_name] = variant_results

        rmses = [r["rmse"] for r in variant_results]
        logger.info(f"[{variant_name}] Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")

    # PHASE 5: Cohen's d
    logger.info("PHASE 5: Computing pairwise Cohen's d...")
    cohens_d = compute_cohens_d(results)
    for pair, d in cohens_d.items():
        logger.info(f"  Cohen's d ({pair}): {d}")

    # PHASE 6: Output
    logger.info("PHASE 6: Building output JSON...")
    method_out = build_output(results, param_counts, cohens_d, graph)

    OUTPUT_FILE.write_text(json.dumps(method_out, indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

    t_total = time.time() - t_total_start
    logger.info(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
