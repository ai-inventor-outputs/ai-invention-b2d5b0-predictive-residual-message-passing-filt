#!/usr/bin/env python3
"""Gradient-Detached Prediction Ablation: Testing Whether PRMP Needs End-to-End Gradient Flow.

Train 4 PRMP gradient-flow variants × 3 seeds on Amazon Video Games (review-rating MAE)
and F1 (driver-position MAE):
  (A) Standard     — baseline SAGEConv, no prediction
  (B) Full_PRMP    — end-to-end gradients through pred_mlp
  (C) Detached_PRMP — pred_mlp output detached (no gradient through predictions)
  (D) Frozen_PRMP  — pred_mlp weights frozen after random init

All GNN layers implemented in PURE PyTorch (no torch-geometric) to avoid
torch-scatter/torch-sparse compilation issues.

Instrumentation: per-epoch prediction R², residual/prediction variance, gradient norms.
"""

import copy
import gc
import io
import itertools
import json
import math
import os
import resource
import sys
import time
import warnings
import zipfile
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from loguru import logger

WS = Path(__file__).resolve().parent
(WS / "logs").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WS / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
logger.info(f"Detected: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(16 * 1024**3)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f} GB")
else:
    VRAM_GB = 0

# Set RAM limits
RAM_BUDGET = int(12 * 1024**3)  # 12 GB
_avail = psutil.virtual_memory().available
if RAM_BUDGET < _avail:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
else:
    logger.warning(f"RAM budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB, skipping limit")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'} ({VRAM_GB:.1f} GB VRAM), device={DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEP_AMAZON = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
                  "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
DEP_F1 = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
              "/3_invention_loop/iter_2/gen_art/data_id5_it2__opus")
RAW_AMAZON_PATH = DEP_AMAZON / "temp" / "datasets" / "full_LoganKells_amazon_product_reviews_video_games_train.json"
F1_ZIP_PATH = DEP_F1 / "temp" / "datasets" / "f1db_csv.zip"
TEMP_DIR = WS / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
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
TEXT_HASH_DIM = 16
TIME_BUDGET = 7000  # seconds (~116 min)

# ===========================================================================
# Utility: JSON sanitization
# ===========================================================================

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return obj


# ===========================================================================
# PHASE 1: Pure PyTorch GNN Layers
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


# --- VARIANT A: Standard SAGEConv (baseline) ---
class BipartiteSAGEConv(nn.Module):
    """SAGEConv for bipartite edges: aggregates src features to dst nodes."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        neigh_feats = x_src[edge_src]
        agg = scatter_mean(neigh_feats, edge_dst, num_dst)
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# --- VARIANT B: Full PRMP (end-to-end gradients through pred_mlp) ---
class PRMPConvFull(nn.Module):
    """PRMP: parent predicts child, subtract prediction, aggregate residuals.
    Full end-to-end gradient flow through pred_mlp.
    """
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        # Zero-init last layer for identity-start (residuals ~ raw features initially)
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

        # Diagnostics storage (set during forward)
        self.last_r2 = None
        self.last_pred_var = None
        self.last_residual_var = None

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        pred_src = self.pred_mlp(x_dst)           # GRADIENTS FLOW through pred_mlp

        src_feats = x_src[edge_src]               # [E, src_dim]
        pred_feats = pred_src[edge_dst]            # [E, src_dim]
        residual = src_feats - pred_feats          # THE CORE PRMP OPERATION

        # Diagnostics (no_grad to avoid affecting training)
        with torch.no_grad():
            src_var = src_feats.var().item()
            pred_v = pred_feats.var().item()
            res_v = residual.var().item()
            self.last_pred_var = pred_v
            self.last_residual_var = res_v
            if src_var > 1e-10:
                ss_res = (residual ** 2).sum().item()
                ss_tot = ((src_feats - src_feats.mean(dim=0, keepdim=True)) ** 2).sum().item()
                self.last_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            else:
                self.last_r2 = 0.0

        agg = scatter_mean(residual, edge_dst, num_dst)
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# --- VARIANT C: Detached PRMP (pred_mlp output detached) ---
class PRMPConvDetached(nn.Module):
    """PRMP with detached prediction: no gradient through pred_mlp output.
    pred_mlp weights still get *indirect* updates via their effect on x_dst
    in subsequent layers, but no direct gradient through the prediction output.
    """
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

        self.last_r2 = None
        self.last_pred_var = None
        self.last_residual_var = None

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        pred_src = self.pred_mlp(x_dst).detach()  # KEY: .detach() - no gradient

        src_feats = x_src[edge_src]
        pred_feats = pred_src[edge_dst]
        residual = src_feats - pred_feats

        with torch.no_grad():
            src_var = src_feats.var().item()
            pred_v = pred_feats.var().item()
            res_v = residual.var().item()
            self.last_pred_var = pred_v
            self.last_residual_var = res_v
            if src_var > 1e-10:
                ss_res = (residual ** 2).sum().item()
                ss_tot = ((src_feats - src_feats.mean(dim=0, keepdim=True)) ** 2).sum().item()
                self.last_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            else:
                self.last_r2 = 0.0

        agg = scatter_mean(residual, edge_dst, num_dst)
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# --- VARIANT D: Frozen PRMP (pred_mlp weights frozen after random init) ---
class PRMPConvFrozen(nn.Module):
    """PRMP with frozen random projection: pred_mlp has requires_grad=False.
    Tests whether a random (non-learned) prediction still helps via structural
    skip-connection effect.
    """
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        # Xavier init for meaningful random projection (NOT zero-init)
        nn.init.xavier_uniform_(self.pred_mlp[0].weight)
        nn.init.zeros_(self.pred_mlp[0].bias)
        nn.init.xavier_uniform_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        # Freeze all pred_mlp parameters
        for p in self.pred_mlp.parameters():
            p.requires_grad = False

        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

        self.last_r2 = None
        self.last_pred_var = None
        self.last_residual_var = None

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        with torch.no_grad():
            pred_src = self.pred_mlp(x_dst)  # Frozen random projection

        src_feats = x_src[edge_src]
        pred_feats = pred_src[edge_dst]
        residual = src_feats - pred_feats

        with torch.no_grad():
            src_var = src_feats.var().item()
            pred_v = pred_feats.var().item()
            res_v = residual.var().item()
            self.last_pred_var = pred_v
            self.last_residual_var = res_v
            if src_var > 1e-10:
                ss_res = (residual ** 2).sum().item()
                ss_tot = ((src_feats - src_feats.mean(dim=0, keepdim=True)) ** 2).sum().item()
                self.last_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            else:
                self.last_r2 = 0.0

        agg = scatter_mean(residual, edge_dst, num_dst)
        out = self.lin_neigh(agg) + self.lin_self(x_dst)
        return out


# ===========================================================================
# PHASE 2: Heterogeneous GNN Model Assembly
# ===========================================================================

# Edge type definitions for Amazon
AMAZON_EDGE_TYPES = [
    ("review__rev_of__product", "review", "product",
     "edge_review_to_product_src", "edge_review_to_product_dst", "n_products"),
    ("product__has_review__review", "product", "review",
     "edge_product_to_review_src", "edge_product_to_review_dst", "n_reviews"),
    ("review__written_by__customer", "review", "customer",
     "edge_review_to_customer_src", "edge_review_to_customer_dst", "n_customers"),
    ("customer__wrote__review", "customer", "review",
     "edge_customer_to_review_src", "edge_customer_to_review_dst", "n_reviews"),
]

# Child→parent edges get PRMP treatment
AMAZON_PRMP_EDGES = {"review__rev_of__product", "review__written_by__customer"}


class HeteroGNNLayer(nn.Module):
    """One layer of heterogeneous message passing across all edge types."""
    def __init__(self, conv_dict: dict, edge_types: list):
        super().__init__()
        self.convs = nn.ModuleDict()
        for key, conv in conv_dict.items():
            self.convs[key] = conv
        self.edge_types = edge_types

    def forward(self, x_dict: dict, graph: dict) -> dict:
        out_dict = {}
        for key, src_type, dst_type, esrc_key, edst_key, ndst_key in self.edge_types:
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


def make_model(variant_name: str, node_feat_dims: dict, edge_types: list,
               prmp_edge_keys: set, target_node_type: str,
               hidden: int = HIDDEN_DIM):
    """Create a 2-layer heterogeneous GNN for a given variant.

    Args:
        variant_name: one of A_Standard, B_Full_PRMP, C_Detached_PRMP, D_Frozen_PRMP
        node_feat_dims: {node_type: input_dim}
        edge_types: list of (key, src_type, dst_type, esrc_key, edst_key, ndst_key)
        prmp_edge_keys: set of edge keys that get PRMP treatment
        target_node_type: node type for prediction head
        hidden: hidden dimension
    """
    # Select conv factory based on variant
    if variant_name == "A_Standard":
        def prmp_factory(s, d, o):
            return BipartiteSAGEConv(s, d, o)
        def std_factory(s, d, o):
            return BipartiteSAGEConv(s, d, o)
    elif variant_name == "B_Full_PRMP":
        def prmp_factory(s, d, o):
            return PRMPConvFull(s, d, o)
        def std_factory(s, d, o):
            return BipartiteSAGEConv(s, d, o)
    elif variant_name == "C_Detached_PRMP":
        def prmp_factory(s, d, o):
            return PRMPConvDetached(s, d, o)
        def std_factory(s, d, o):
            return BipartiteSAGEConv(s, d, o)
    elif variant_name == "D_Frozen_PRMP":
        def prmp_factory(s, d, o):
            return PRMPConvFrozen(s, d, o)
        def std_factory(s, d, o):
            return BipartiteSAGEConv(s, d, o)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_lins = nn.ModuleDict({
                nt: nn.Linear(dim, hidden) for nt, dim in node_feat_dims.items()
            })

            conv1_dict = {}
            conv2_dict = {}
            for k, *_ in edge_types:
                if k in prmp_edge_keys and variant_name != "A_Standard":
                    conv1_dict[k] = prmp_factory(hidden, hidden, hidden)
                    conv2_dict[k] = prmp_factory(hidden, hidden, hidden)
                else:
                    conv1_dict[k] = std_factory(hidden, hidden, hidden)
                    conv2_dict[k] = std_factory(hidden, hidden, hidden)
            self.conv1 = HeteroGNNLayer(conv1_dict, edge_types)
            self.conv2 = HeteroGNNLayer(conv2_dict, edge_types)
            self.head = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, graph: dict) -> torch.Tensor:
            x_dict = {}
            for nt in node_feat_dims:
                x_dict[nt] = F.relu(self.input_lins[nt](graph[f"x_{nt}"]))

            out = self.conv1(x_dict, graph)
            x_dict = {k: F.relu(self.dropout(v)) for k, v in out.items()}
            out = self.conv2(x_dict, graph)
            x_dict = {k: F.relu(v) for k, v in out.items()}
            return self.head(x_dict[target_node_type]).squeeze(-1)

    return _Model()


# ===========================================================================
# PHASE 3: Amazon Video Games Data Loading
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


def build_amazon_graph() -> dict:
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
        "dataset_name": "amazon",
        "target_node_type": "review",
        "node_feat_dims": {"product": 5, "customer": 5, "review": 21},
        "edge_types": AMAZON_EDGE_TYPES,
        "prmp_edge_keys": AMAZON_PRMP_EDGES,
    }

    del raw, X_review, X_product, X_customer, ratings
    gc.collect()
    return graph


# ===========================================================================
# PHASE 4: F1 Data Loading (from Ergast CSV)
# ===========================================================================

# Tables used in F1 graph
F1_NODE_TABLES = ["drivers", "races", "constructors", "circuits", "status",
                  "results", "qualifying", "driver_standings"]

F1_ID_COLUMNS = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "drivers": ["driverId"],
    "driver_standings": ["driverStandingsId", "raceId", "driverId"],
    "qualifying": ["qualifyId", "raceId", "driverId", "constructorId"],
    "races": ["raceId", "circuitId"],
    "results": ["resultId", "raceId", "driverId", "constructorId", "statusId"],
    "status": ["statusId"],
}

F1_FK_LINKS = [
    ("results", "raceId", "races", "raceId"),
    ("results", "driverId", "drivers", "driverId"),
    ("results", "constructorId", "constructors", "constructorId"),
    ("results", "statusId", "status", "statusId"),
    ("qualifying", "raceId", "races", "raceId"),
    ("qualifying", "driverId", "drivers", "driverId"),
    ("qualifying", "constructorId", "constructors", "constructorId"),
    ("races", "circuitId", "circuits", "circuitId"),
    ("driver_standings", "raceId", "races", "raceId"),
    ("driver_standings", "driverId", "drivers", "driverId"),
]


def download_f1_csv() -> dict:
    """Download and extract F1 Ergast CSV tables."""
    import csv as csv_mod

    if F1_ZIP_PATH.exists():
        zip_path = F1_ZIP_PATH
    else:
        local_zip = TEMP_DIR / "f1db_csv.zip"
        if local_zip.exists():
            zip_path = local_zip
        else:
            import requests
            logger.info("Downloading Ergast F1 CSV...")
            resp = requests.get("https://github.com/rubenv/ergast-mrd/raw/master/f1db_csv.zip",
                                timeout=120)
            resp.raise_for_status()
            local_zip.write_bytes(resp.content)
            zip_path = local_zip

    tables = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for csv_name in sorted(zf.namelist()):
            if not csv_name.endswith(".csv"):
                continue
            tname = csv_name.replace(".csv", "")
            if tname not in F1_NODE_TABLES:
                continue
            with zf.open(csv_name) as f:
                raw_bytes = f.read().decode("utf-8")
                lines = raw_bytes.strip().split("\n")
                header_fields = next(csv_mod.reader([lines[0]]))
                if len(lines) > 1:
                    data_fields = next(csv_mod.reader([lines[1]]))
                    if len(data_fields) > len(header_fields):
                        extra = len(data_fields) - len(header_fields)
                        for ei in range(extra):
                            header_fields.append(f"_extra_{ei}")
                        lines[0] = ",".join(header_fields)
                        raw_bytes = "\n".join(lines)
                df = pd.read_csv(
                    io.StringIO(raw_bytes),
                    na_values=["\\N", "NULL", ""],
                    engine="python",
                )
            # Force Arrow types to numpy-compatible types
            for col in df.columns:
                s = df[col]
                dtype_str = str(type(s.dtype))
                if "Arrow" in dtype_str or "arrow" in dtype_str:
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().sum() > len(num) * 0.5 and s.notna().sum() > 0:
                        df[col] = num
                    else:
                        df[col] = s.astype(object)
            # Ensure ID columns are numeric
            for col in F1_ID_COLUMNS.get(tname, []):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            tables[tname] = df
    logger.info(f"Loaded {len(tables)} F1 tables: {list(tables.keys())}")
    return tables


def encode_f1_table(df: pd.DataFrame, table_name: str) -> tuple:
    """Encode F1 table to numeric features."""
    id_cols = set(F1_ID_COLUMNS.get(table_name, []))
    encoded = df.copy()
    for col in encoded.columns:
        if encoded[col].dtype == "object":
            try:
                dt = pd.to_datetime(encoded[col], format="mixed", errors="coerce")
                if dt.notna().sum() > len(dt) * 0.5:
                    encoded[col] = dt.astype("int64") // 10**9
                    encoded[col] = encoded[col].where(dt.notna(), -1)
                    continue
            except Exception:
                pass
            codes, _ = pd.factorize(encoded[col], sort=False)
            encoded[col] = codes.astype(np.float32)
            continue
        if pd.api.types.is_numeric_dtype(encoded[col]):
            encoded[col] = encoded[col].fillna(-1).astype(np.float32)
        else:
            codes, _ = pd.factorize(encoded[col].astype(str), sort=False)
            encoded[col] = codes.astype(np.float32)

    feat_cols = [c for c in encoded.columns if c not in id_cols]
    if not feat_cols:
        return np.ones((len(encoded), 1), dtype=np.float32), ["_dummy"]
    arr = encoded[feat_cols].values.astype(np.float32)
    mu = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) + 1e-8
    arr = (arr - mu) / std
    arr = np.nan_to_num(arr, nan=0.0)
    return arr, feat_cols


def build_f1_graph() -> dict:
    """Build F1 heterogeneous graph for driver-position prediction."""
    logger.info("Building F1 heterogeneous graph...")
    tables = download_f1_csv()

    # Build ID maps and feature tensors per node table
    id_maps = {}
    node_feat_dims = {}
    graph = {"dataset_name": "f1", "target_node_type": "drivers"}

    for tname in F1_NODE_TABLES:
        if tname not in tables:
            continue
        df = tables[tname]
        pk = F1_ID_COLUMNS.get(tname, [None])[0]

        # Build ID map
        if pk and pk in df.columns:
            unique_ids = df[pk].dropna().unique()
            try:
                unique_ids = unique_ids.astype(int)
            except (ValueError, TypeError):
                unique_ids = np.arange(len(df))
            imap = {int(v): i for i, v in enumerate(unique_ids)}
        else:
            imap = {i: i for i in range(len(df))}

        # For child/event tables with more rows than unique PKs, use row index
        if pk and pk in df.columns and len(df) > len(imap):
            imap = {i: i for i in range(len(df))}

        id_maps[tname] = imap

        feat_arr, feat_cols = encode_f1_table(df, tname)
        feat_t = torch.tensor(feat_arr[:len(imap)], dtype=torch.float32)
        graph[f"x_{tname}"] = feat_t
        graph[f"n_{tname}"] = len(imap)
        node_feat_dims[tname] = feat_t.shape[1]
        logger.info(f"  F1 {tname}: {len(imap)} nodes, {feat_t.shape[1]} feats")

    # Build edges
    edge_types = []
    prmp_edge_keys = set()

    for child_t, child_col, parent_t, parent_col in F1_FK_LINKS:
        if child_t not in tables or parent_t not in tables:
            continue
        child_df = tables[child_t]
        if child_col not in child_df.columns:
            continue

        child_imap = id_maps[child_t]
        parent_imap = id_maps[parent_t]

        fk_series = pd.to_numeric(child_df[child_col], errors="coerce")
        fk_vals = fk_series.values
        valid_mask = ~pd.isna(fk_vals)
        fk_valid = fk_vals[valid_mask].astype(int) if valid_mask.any() else np.array([], dtype=int)
        child_row_indices = np.arange(len(child_df))[valid_mask]

        src_list = []
        dst_list = []
        for ci, fv in zip(child_row_indices, fk_valid):
            s = child_imap.get(int(ci), -1)
            d = parent_imap.get(int(fv), -1)
            if s >= 0 and d >= 0:
                src_list.append(s)
                dst_list.append(d)

        if not src_list:
            continue

        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)

        # Forward edge (child → parent)
        fwd_key = f"{child_t}__fk_{child_col}__{parent_t}"
        esrc_key = f"edge_{fwd_key}_src"
        edst_key = f"edge_{fwd_key}_dst"
        graph[esrc_key] = src_t
        graph[edst_key] = dst_t
        edge_types.append((fwd_key, child_t, parent_t, esrc_key, edst_key, f"n_{parent_t}"))
        prmp_edge_keys.add(fwd_key)  # child→parent gets PRMP

        # Reverse edge (parent → child)
        rev_key = f"{parent_t}__rev_{child_col}__{child_t}"
        rev_esrc_key = f"edge_{rev_key}_src"
        rev_edst_key = f"edge_{rev_key}_dst"
        graph[rev_esrc_key] = dst_t
        graph[rev_edst_key] = src_t
        edge_types.append((rev_key, parent_t, child_t, rev_esrc_key, rev_edst_key, f"n_{child_t}"))

        logger.info(f"  F1 {child_t}->{parent_t} via {child_col}: {len(src_list)} edges")

    # --- Target: driver average finishing position ---
    results_df = tables["results"]
    races_df = tables["races"].copy()
    results_df = results_df.copy()

    results_df["raceId"] = pd.to_numeric(results_df["raceId"], errors="coerce").astype("float64")
    results_df["driverId"] = pd.to_numeric(results_df["driverId"], errors="coerce").astype("float64")
    races_df["raceId"] = pd.to_numeric(races_df["raceId"], errors="coerce").astype("float64")

    pos_col = "positionOrder" if "positionOrder" in results_df.columns else "position"
    results_df["_pos"] = pd.to_numeric(results_df[pos_col], errors="coerce")

    # Aggregate position per driver
    driver_pos = results_df.groupby("driverId")["_pos"].mean()
    driver_map = id_maps["drivers"]
    num_drivers = len(driver_map)

    y = torch.full((num_drivers,), float("nan"), dtype=torch.float32)
    for did, idx in driver_map.items():
        if did in driver_pos.index:
            val = driver_pos.loc[did]
            if not pd.isna(val):
                y[idx] = val

    valid_mask = ~torch.isnan(y)
    valid_indices = torch.where(valid_mask)[0]
    # Replace NaN with mean for graph computation, but only evaluate on valid
    y_mean = y[valid_mask].mean()
    y[~valid_mask] = y_mean

    logger.info(f"F1 target: {valid_mask.sum().item()}/{num_drivers} drivers with valid position, "
                f"mean={y_mean.item():.2f}")

    # --- Temporal split ---
    if "date" in races_df.columns:
        races_df["_date"] = pd.to_datetime(races_df["date"].astype(str), format="mixed", errors="coerce")
    else:
        races_df["_date"] = pd.RangeIndex(len(races_df))

    results_m = results_df.merge(races_df[["raceId", "_date"]], on="raceId", how="inner")
    race_dates = races_df[["raceId", "_date"]].dropna(subset=["_date"]).sort_values("_date")
    n_races = len(race_dates)

    if n_races > 10:
        cutoff_train = race_dates.iloc[int(n_races * 0.7)]["_date"]
        cutoff_val = race_dates.iloc[int(n_races * 0.8)]["_date"]

        driver_ids_train = set(results_m[results_m["_date"] <= cutoff_train]["driverId"].dropna().unique())
        driver_ids_val = set(results_m[(results_m["_date"] > cutoff_train) &
                                        (results_m["_date"] <= cutoff_val)]["driverId"].dropna().unique())
        driver_ids_test = set(results_m[results_m["_date"] > cutoff_val]["driverId"].dropna().unique())

        use_random = len(driver_ids_test) < 30
    else:
        use_random = True

    if use_random:
        logger.info("Using random split for F1 drivers")
        n_valid = len(valid_indices)
        perm = valid_indices[torch.randperm(n_valid, generator=torch.Generator().manual_seed(0))]
        n_train = int(0.7 * n_valid)
        n_val = int(0.1 * n_valid)
        train_mask = perm[:n_train]
        val_mask = perm[n_train:n_train + n_val]
        test_mask = perm[n_train + n_val:]
    else:
        train_idx = [driver_map[int(d)] for d in driver_ids_train
                     if int(d) in driver_map and valid_mask[driver_map[int(d)]]]
        val_idx = [driver_map[int(d)] for d in driver_ids_val
                   if int(d) in driver_map and valid_mask[driver_map[int(d)]]]
        test_idx = [driver_map[int(d)] for d in driver_ids_test
                    if int(d) in driver_map and valid_mask[driver_map[int(d)]]]
        train_mask = torch.tensor(train_idx, dtype=torch.long)
        val_mask = torch.tensor(val_idx, dtype=torch.long)
        test_mask = torch.tensor(test_idx, dtype=torch.long)

    logger.info(f"F1 split: train={len(train_mask)}, val={len(val_mask)}, test={len(test_mask)}")

    graph["y"] = y
    graph["train_mask"] = train_mask
    graph["val_mask"] = val_mask
    graph["test_mask"] = test_mask
    graph["node_feat_dims"] = node_feat_dims
    graph["edge_types"] = edge_types
    graph["prmp_edge_keys"] = prmp_edge_keys

    del tables
    gc.collect()
    return graph


# ===========================================================================
# PHASE 5: Instrumentation - Gradient Norms & R² Collection
# ===========================================================================

def collect_prmp_diagnostics(model: nn.Module) -> dict:
    """Collect R², pred_var, residual_var from all PRMP conv layers."""
    r2_vals = []
    pred_vars = []
    res_vars = []

    for name, module in model.named_modules():
        if isinstance(module, (PRMPConvFull, PRMPConvDetached, PRMPConvFrozen)):
            if module.last_r2 is not None:
                r2_vals.append(module.last_r2)
            if module.last_pred_var is not None:
                pred_vars.append(module.last_pred_var)
            if module.last_residual_var is not None:
                res_vars.append(module.last_residual_var)

    return {
        "pred_r2": float(np.mean(r2_vals)) if r2_vals else None,
        "pred_var": float(np.mean(pred_vars)) if pred_vars else None,
        "residual_var": float(np.mean(res_vars)) if res_vars else None,
    }


def compute_gradient_norms(model: nn.Module) -> dict:
    """Compute gradient norms for pred_mlp and lin_neigh parameter groups."""
    pred_mlp_norms = []
    lin_neigh_norms = []
    lin_self_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            if "pred_mlp" in name:
                pred_mlp_norms.append(norm)
            elif "lin_neigh" in name:
                lin_neigh_norms.append(norm)
            elif "lin_self" in name:
                lin_self_norms.append(norm)

    return {
        "pred_mlp_grad_norm": float(np.mean(pred_mlp_norms)) if pred_mlp_norms else 0.0,
        "lin_neigh_grad_norm": float(np.mean(lin_neigh_norms)) if lin_neigh_norms else 0.0,
        "lin_self_grad_norm": float(np.mean(lin_self_norms)) if lin_self_norms else 0.0,
    }


# ===========================================================================
# PHASE 6: Training Loop
# ===========================================================================

def count_parameters(model: nn.Module) -> dict:
    """Count model parameters with breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def move_graph_to_device(graph: dict, device: torch.device) -> dict:
    """Move all tensors in graph dict to device."""
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            graph[k] = v.to(device)
    return graph


def train_variant(variant_name: str, graph: dict, device: torch.device,
                  seed: int) -> dict:
    """Train one model variant with one seed. Returns result dict with metrics + diagnostics."""
    logger.info(f"--- Training {variant_name} | seed={seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    # Build model
    model = make_model(
        variant_name=variant_name,
        node_feat_dims=graph["node_feat_dims"],
        edge_types=graph["edge_types"],
        prmp_edge_keys=graph["prmp_edge_keys"],
        target_node_type=graph["target_node_type"],
        hidden=HIDDEN_DIM,
    )
    model = model.to(device)

    param_info = count_parameters(model)
    logger.info(f"  Params: total={param_info['total']:,}, trainable={param_info['trainable']:,}")

    # Use L1 loss for F1 (direct MAE optimization), MSE for Amazon
    if graph["dataset_name"] == "f1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    train_mask = graph["train_mask"]
    val_mask = graph["val_mask"]
    test_mask = graph["test_mask"]
    y = graph["y"]

    epoch_records = []
    t_start = time.time()

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        pred = model(graph)
        loss = loss_fn(pred[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Collect gradient norms BEFORE optimizer step
        grad_norms = {}
        if variant_name != "A_Standard":
            grad_norms = compute_gradient_norms(model)

        optimizer.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            pred = model(graph)
            val_loss = loss_fn(pred[val_mask], y[val_mask]).item()
            val_mae = float(torch.mean(torch.abs(pred[val_mask] - y[val_mask])).item())
            test_mae = float(torch.mean(torch.abs(pred[test_mask] - y[test_mask])).item())

        # Collect PRMP diagnostics
        diagnostics = {}
        if variant_name != "A_Standard":
            diagnostics = collect_prmp_diagnostics(model)
            diagnostics.update(grad_norms)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(loss.item(), 6),
            "val_loss": round(val_loss, 6),
            "val_mae": round(val_mae, 6),
            "test_mae": round(test_mae, 6),
        }
        if diagnostics:
            epoch_record.update({k: round(v, 6) if v is not None else None
                                 for k, v in diagnostics.items()})
        epoch_records.append(epoch_record)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  [{variant_name}|seed={seed}] Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            diag_str = ""
            if diagnostics and diagnostics.get("pred_r2") is not None:
                diag_str = f", R²={diagnostics['pred_r2']:.4f}"
            logger.info(f"  [{variant_name}|seed={seed}] Epoch {epoch}: "
                        f"train_loss={loss.item():.4f}, val_mae={val_mae:.4f}{diag_str}")

    elapsed = time.time() - t_start
    logger.info(f"  [{variant_name}|seed={seed}] Done in {elapsed:.1f}s, best_epoch={best_epoch}")

    # --- Test with best model ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(graph)
        test_pred = pred[test_mask].cpu().numpy()
        test_true = y[test_mask].cpu().numpy()

    test_rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred - test_true)))
    ss_res = float(np.sum((test_true - test_pred) ** 2))
    ss_tot = float(np.sum((test_true - test_true.mean()) ** 2))
    test_r2 = float(1 - ss_res / max(ss_tot, 1e-10)) if ss_tot > 1e-10 else 0.0

    logger.info(f"  [{variant_name}|seed={seed}] Test: MAE={test_mae:.4f}, "
                f"RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

    # Get final diagnostics
    final_diagnostics = {}
    if variant_name != "A_Standard":
        # Run one more forward pass to get final R²
        model.eval()
        with torch.no_grad():
            _ = model(graph)
        final_diagnostics = collect_prmp_diagnostics(model)

    result = {
        "variant": variant_name,
        "seed": seed,
        "test_mae": round(test_mae, 6),
        "test_rmse": round(test_rmse, 6),
        "test_r2": round(test_r2, 6),
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
        "param_total": param_info["total"],
        "param_trainable": param_info["trainable"],
        "final_pred_r2": final_diagnostics.get("pred_r2"),
        "final_pred_var": final_diagnostics.get("pred_var"),
        "final_residual_var": final_diagnostics.get("residual_var"),
        "epoch_records": epoch_records,
    }

    del model, optimizer, best_state
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    return result


# ===========================================================================
# PHASE 7: Analysis
# ===========================================================================

def compute_cohens_d(values_a: list, values_b: list) -> float:
    """Compute Cohen's d between two sample lists."""
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    mean_diff = np.mean(values_a) - np.mean(values_b)
    pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
    return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0


def analyze_results(all_results: list) -> dict:
    """Analyze experiment results across variants and datasets."""
    analysis = {}

    for dataset_name in ["amazon", "f1"]:
        ds_results = [r for r in all_results if r["dataset"] == dataset_name]
        if not ds_results:
            continue

        variant_metrics = {}
        for variant in ["A_Standard", "B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
            v_results = [r for r in ds_results if r["variant"] == variant]
            if not v_results:
                continue
            maes = [r["test_mae"] for r in v_results]
            rmses = [r["test_rmse"] for r in v_results]
            r2s = [r["test_r2"] for r in v_results]

            variant_metrics[variant] = {
                "mean_mae": round(float(np.mean(maes)), 6),
                "std_mae": round(float(np.std(maes)), 6),
                "seeds_mae": [round(m, 6) for m in maes],
                "mean_rmse": round(float(np.mean(rmses)), 6),
                "std_rmse": round(float(np.std(rmses)), 6),
                "mean_r2": round(float(np.mean(r2s)), 6),
                "final_pred_r2": [r.get("final_pred_r2") for r in v_results],
                "final_pred_var": [r.get("final_pred_var") for r in v_results],
                "final_residual_var": [r.get("final_residual_var") for r in v_results],
            }

        # Compute relative improvements vs baseline
        baseline_mae = variant_metrics.get("A_Standard", {}).get("mean_mae", 1.0)
        improvements = {}
        cohens_d_pairs = {}

        for variant in ["B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
            if variant in variant_metrics and "A_Standard" in variant_metrics:
                v_mae = variant_metrics[variant]["mean_mae"]
                improvements[variant] = round((baseline_mae - v_mae) / max(baseline_mae, 1e-10) * 100, 2)

                a_maes = variant_metrics["A_Standard"]["seeds_mae"]
                v_maes = variant_metrics[variant]["seeds_mae"]
                cohens_d_pairs[f"A_vs_{variant}"] = round(compute_cohens_d(a_maes, v_maes), 4)

        # Pairwise Cohen's d among PRMP variants
        for v1, v2 in itertools.combinations(
            ["B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"], 2
        ):
            if v1 in variant_metrics and v2 in variant_metrics:
                maes1 = variant_metrics[v1]["seeds_mae"]
                maes2 = variant_metrics[v2]["seeds_mae"]
                cohens_d_pairs[f"{v1}_vs_{v2}"] = round(compute_cohens_d(maes1, maes2), 4)

        # Interpretation
        b_mae = variant_metrics.get("B_Full_PRMP", {}).get("mean_mae", float("inf"))
        c_mae = variant_metrics.get("C_Detached_PRMP", {}).get("mean_mae", float("inf"))
        d_mae = variant_metrics.get("D_Frozen_PRMP", {}).get("mean_mae", float("inf"))

        if b_mae < c_mae < d_mae:
            interpretation = "B >> C >> D: End-to-end training essential → 'learned prediction' mechanism"
        elif abs(b_mae - c_mae) < 0.02 * baseline_mae and c_mae < d_mae:
            interpretation = "B ≈ C >> D: Prediction quality matters but not gradient flow → 'representation structure'"
        elif abs(b_mae - c_mae) < 0.02 * baseline_mae and abs(c_mae - d_mae) < 0.02 * baseline_mae:
            interpretation = "B ≈ C ≈ D: Architectural skip-connection effect, not prediction quality"
        else:
            interpretation = f"Mixed pattern: B={b_mae:.4f}, C={c_mae:.4f}, D={d_mae:.4f}"

        analysis[dataset_name] = {
            "variant_metrics": variant_metrics,
            "improvements_vs_baseline_pct": improvements,
            "cohens_d": cohens_d_pairs,
            "interpretation": interpretation,
        }

    return analysis


# ===========================================================================
# PHASE 8: Output in exp_gen_sol_out.json schema
# ===========================================================================

def build_output(all_results: list, analysis: dict) -> dict:
    """Build exp_gen_sol_out.json format output."""
    examples = []

    for r in all_results:
        dataset = r["dataset"]
        variant = r["variant"]
        seed = r["seed"]

        # Build input string describing the run configuration
        input_str = json.dumps({
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        })

        # Build output string with metrics
        output_str = json.dumps({
            "test_mae": r["test_mae"],
            "test_rmse": r["test_rmse"],
            "test_r2": r["test_r2"],
            "best_epoch": r["best_epoch"],
            "train_time_s": r["train_time_s"],
            "param_total": r["param_total"],
            "param_trainable": r["param_trainable"],
            "final_pred_r2": r.get("final_pred_r2"),
            "final_pred_var": r.get("final_pred_var"),
            "final_residual_var": r.get("final_residual_var"),
        })

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_dataset": dataset,
            "metadata_variant": variant,
            "metadata_seed": seed,
            "metadata_test_mae": r["test_mae"],
            "metadata_test_rmse": r["test_rmse"],
            "metadata_best_epoch": r["best_epoch"],
            "predict_A_Standard": str(
                analysis.get(dataset, {}).get("variant_metrics", {})
                .get("A_Standard", {}).get("mean_mae", "N/A")
            ),
            "predict_B_Full_PRMP": str(
                analysis.get(dataset, {}).get("variant_metrics", {})
                .get("B_Full_PRMP", {}).get("mean_mae", "N/A")
            ),
            "predict_C_Detached_PRMP": str(
                analysis.get(dataset, {}).get("variant_metrics", {})
                .get("C_Detached_PRMP", {}).get("mean_mae", "N/A")
            ),
            "predict_D_Frozen_PRMP": str(
                analysis.get(dataset, {}).get("variant_metrics", {})
                .get("D_Frozen_PRMP", {}).get("mean_mae", "N/A")
            ),
        }
        examples.append(example)

    # Also add per-epoch learning curve examples (sampled)
    for r in all_results:
        epoch_records = r.get("epoch_records", [])
        if not epoch_records:
            continue
        # Sample key epochs: first, every 10th, last
        sampled = [epoch_records[0]]
        for i in range(10, len(epoch_records), 10):
            sampled.append(epoch_records[i])
        if epoch_records[-1] not in sampled:
            sampled.append(epoch_records[-1])

        for er in sampled:
            input_str = json.dumps({
                "type": "learning_curve",
                "dataset": r["dataset"],
                "variant": r["variant"],
                "seed": r["seed"],
                "epoch": er["epoch"],
            })
            output_str = json.dumps(sanitize_for_json({
                "train_loss": er["train_loss"],
                "val_mae": er["val_mae"],
                "test_mae": er["test_mae"],
                "pred_r2": er.get("pred_r2"),
                "pred_var": er.get("pred_var"),
                "residual_var": er.get("residual_var"),
                "pred_mlp_grad_norm": er.get("pred_mlp_grad_norm"),
                "lin_neigh_grad_norm": er.get("lin_neigh_grad_norm"),
            }))
            example = {
                "input": input_str,
                "output": output_str,
                "metadata_dataset": r["dataset"],
                "metadata_variant": r["variant"],
                "metadata_seed": r["seed"],
                "metadata_epoch": er["epoch"],
                "metadata_type": "learning_curve",
            }
            examples.append(example)

    # Cross-variant analysis examples
    for ds_name, ds_analysis in analysis.items():
        input_str = json.dumps({
            "type": "cross_variant_analysis",
            "dataset": ds_name,
        })
        output_str = json.dumps(sanitize_for_json({
            "improvements_vs_baseline_pct": ds_analysis.get("improvements_vs_baseline_pct"),
            "cohens_d": ds_analysis.get("cohens_d"),
            "interpretation": ds_analysis.get("interpretation"),
        }))
        example = {
            "input": input_str,
            "output": output_str,
            "metadata_dataset": ds_name,
            "metadata_type": "cross_variant_analysis",
        }
        examples.append(example)

    variant_descriptions = {
        "A_Standard": "Baseline SAGEConv with mean aggregation, no prediction mechanism",
        "B_Full_PRMP": "Full PRMP with end-to-end gradient flow through pred_mlp",
        "C_Detached_PRMP": "PRMP with detached prediction (pred_mlp output .detach()), no direct gradient through predictions",
        "D_Frozen_PRMP": "PRMP with frozen random pred_mlp (Xavier init, requires_grad=False)",
    }

    method_out = {
        "metadata": sanitize_for_json({
            "method_name": "PRMP_Gradient_Detach_Ablation",
            "description": "Tests whether PRMP benefit requires end-to-end gradient flow "
                           "through the prediction MLP, or operates as a structural skip-connection. "
                           "4 variants × 3 seeds × 2 datasets.",
            "variants": variant_descriptions,
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "seeds": SEEDS,
            },
            "results_summary": analysis,
        }),
        "datasets": [
            {
                "dataset": "gradient_detach_ablation",
                "examples": examples,
            }
        ],
    }

    return method_out


# ===========================================================================
# Main
# ===========================================================================

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("Gradient-Detached Prediction Ablation Experiment")
    logger.info("Testing Whether PRMP Needs End-to-End Gradient Flow")
    logger.info("Pure PyTorch implementation (no torch-geometric)")
    logger.info("=" * 70)

    t_total_start = time.time()
    all_results = []

    # ================================================================
    # DATASET 1: Amazon Video Games
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DATASET 1: Amazon Video Games")
    logger.info("=" * 70)

    try:
        amazon_graph = build_amazon_graph()
        amazon_graph = move_graph_to_device(amazon_graph, DEVICE)
        logger.info(f"Amazon graph on {DEVICE}: review={amazon_graph['n_reviews']}, "
                    f"product={amazon_graph['n_products']}, customer={amazon_graph['n_customers']}")

        # Smoke test all variants
        logger.info("Smoke test: single forward+backward pass per variant...")
        for vname in ["A_Standard", "B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
            model = make_model(
                variant_name=vname,
                node_feat_dims=amazon_graph["node_feat_dims"],
                edge_types=amazon_graph["edge_types"],
                prmp_edge_keys=amazon_graph["prmp_edge_keys"],
                target_node_type=amazon_graph["target_node_type"],
            ).to(DEVICE)

            model.train()
            pred = model(amazon_graph)
            loss = F.mse_loss(pred[amazon_graph["train_mask"]],
                              amazon_graph["y"][amazon_graph["train_mask"]])
            loss.backward()

            # Verify gradient properties
            pi = count_parameters(model)
            has_pred_grad = False
            has_pred_params = False
            for name, p in model.named_parameters():
                if "pred_mlp" in name:
                    has_pred_params = True
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        has_pred_grad = True

            if vname == "A_Standard":
                assert not has_pred_params, f"{vname}: should not have pred_mlp"
            elif vname == "B_Full_PRMP":
                assert has_pred_grad, f"{vname}: pred_mlp should have gradients"
            elif vname == "C_Detached_PRMP":
                # C has pred_mlp params but no gradient through detached output
                assert has_pred_params, f"{vname}: should have pred_mlp"
                # Note: pred_mlp may still have zero grads due to detach
            elif vname == "D_Frozen_PRMP":
                # D has pred_mlp params but requires_grad=False
                for name, p in model.named_parameters():
                    if "pred_mlp" in name:
                        assert not p.requires_grad, f"{vname}: pred_mlp should be frozen"

            logger.info(f"  {vname}: params={pi['total']:,} (trainable={pi['trainable']:,}), "
                        f"loss={loss.item():.4f}, pred_grad={has_pred_grad}")
            del model, pred, loss
            if HAS_GPU:
                torch.cuda.empty_cache()
        gc.collect()

        # Train all variants
        for variant_name in ["A_Standard", "B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
            for seed in SEEDS:
                elapsed = time.time() - t_total_start
                if elapsed > TIME_BUDGET:
                    logger.warning(f"Time budget exceeded ({elapsed:.0f}s > {TIME_BUDGET}s), stopping")
                    break

                result = train_variant(variant_name, amazon_graph, DEVICE, seed)
                result["dataset"] = "amazon"
                all_results.append(result)

        # Free Amazon graph
        del amazon_graph
        if HAS_GPU:
            torch.cuda.empty_cache()
        gc.collect()

    except Exception:
        logger.exception("Amazon dataset failed")

    # ================================================================
    # DATASET 2: F1
    # ================================================================
    elapsed = time.time() - t_total_start
    if elapsed < TIME_BUDGET:
        logger.info("\n" + "=" * 70)
        logger.info("DATASET 2: F1 Ergast")
        logger.info("=" * 70)

        try:
            f1_graph = build_f1_graph()
            f1_graph = move_graph_to_device(f1_graph, DEVICE)

            # Smoke test
            logger.info("Smoke test F1 variants...")
            for vname in ["A_Standard", "B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
                model = make_model(
                    variant_name=vname,
                    node_feat_dims=f1_graph["node_feat_dims"],
                    edge_types=f1_graph["edge_types"],
                    prmp_edge_keys=f1_graph["prmp_edge_keys"],
                    target_node_type=f1_graph["target_node_type"],
                ).to(DEVICE)
                model.train()
                pred = model(f1_graph)
                loss = F.l1_loss(pred[f1_graph["train_mask"]],
                                 f1_graph["y"][f1_graph["train_mask"]])
                loss.backward()
                pi = count_parameters(model)
                logger.info(f"  F1 {vname}: params={pi['total']:,}, loss={loss.item():.4f}")
                del model, pred, loss
                if HAS_GPU:
                    torch.cuda.empty_cache()
            gc.collect()

            # Train F1 variants
            for variant_name in ["A_Standard", "B_Full_PRMP", "C_Detached_PRMP", "D_Frozen_PRMP"]:
                for seed in SEEDS:
                    elapsed = time.time() - t_total_start
                    if elapsed > TIME_BUDGET:
                        logger.warning(f"Time budget exceeded ({elapsed:.0f}s > {TIME_BUDGET}s)")
                        break

                    result = train_variant(variant_name, f1_graph, DEVICE, seed)
                    result["dataset"] = "f1"
                    all_results.append(result)

            del f1_graph
            if HAS_GPU:
                torch.cuda.empty_cache()
            gc.collect()

        except Exception:
            logger.exception("F1 dataset failed")
    else:
        logger.warning("Skipping F1 dataset due to time budget")

    # ================================================================
    # Analysis & Output
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS & OUTPUT")
    logger.info("=" * 70)

    if not all_results:
        logger.error("No results to analyze!")
        return

    analysis = analyze_results(all_results)

    # Log summary
    for ds_name, ds_analysis in analysis.items():
        logger.info(f"\n--- {ds_name.upper()} Results ---")
        for variant, metrics in ds_analysis.get("variant_metrics", {}).items():
            logger.info(f"  {variant}: MAE={metrics['mean_mae']:.4f} ± {metrics['std_mae']:.4f}")
        logger.info(f"  Improvements vs baseline: {ds_analysis.get('improvements_vs_baseline_pct')}")
        logger.info(f"  Cohen's d: {ds_analysis.get('cohens_d')}")
        logger.info(f"  Interpretation: {ds_analysis.get('interpretation')}")

    # Build and save output
    method_out = build_output(all_results, analysis)
    OUTPUT_FILE.write_text(json.dumps(sanitize_for_json(method_out), indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

    t_total = time.time() - t_total_start
    logger.info(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
