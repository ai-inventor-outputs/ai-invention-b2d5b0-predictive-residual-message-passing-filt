#!/usr/bin/env python3
"""Instrumented PRMP Mechanism Analysis: Information Filtering vs Implicit Regularization.

Per-epoch instrumented training of PRMP, standard SAGEConv, and no-subtraction variant
on Amazon Video Games relational graph, measuring prediction MLP R-squared, mutual
information, gradient norms, and weight dynamics. 100 epochs x 3 seeds x 3 variants.
Outputs time-series diagnostics to method_out.json.
"""

import gc
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
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

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
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Memory limits
RAM_BUDGET = int(20 * 1024**3)  # 20GB budget out of 57GB container
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(17 * 1024**3)  # 17GB out of 20GB
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
DATA_DEP = Path(__file__).resolve().parent.parent.parent.parent / "iter_1" / "gen_art" / "data_id5_it1__opus"

# Hyperparameters
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
SEEDS = [42, 123, 7]
VARIANTS = ["standard", "prmp", "no_subtract"]
MI_INTERVAL = 10
MI_SUBSAMPLE = 2000
TEXT_HASH_DIM = 16

# ---------------------------------------------------------------------------
# Lazy PyG imports (after CUDA setup)
# ---------------------------------------------------------------------------
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing


# ---------------------------------------------------------------------------
# Feature encoding helpers (matching data.py exactly)
# ---------------------------------------------------------------------------

def encode_timestamp(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, unit="s", errors="coerce")
    feats = np.column_stack([
        ts.dt.year.fillna(2000).values,
        ts.dt.month.fillna(1).values,
        ts.dt.dayofweek.fillna(0).values,
    ]).astype(np.float32)
    return StandardScaler().fit_transform(feats)


def encode_text_hash(series: pd.Series, n_features: int = TEXT_HASH_DIM) -> np.ndarray:
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


def parse_helpful(helpful_str) -> tuple:
    try:
        if isinstance(helpful_str, str):
            vals = eval(helpful_str)
            return int(vals[0]), int(vals[1])
        elif isinstance(helpful_str, list):
            return int(helpful_str[0]), int(helpful_str[1])
    except Exception:
        pass
    return 0, 0


# ---------------------------------------------------------------------------
# Data Loading & Graph Construction
# ---------------------------------------------------------------------------

def load_and_build_graph(seed: int = 42) -> tuple:
    """Load Amazon Video Games dataset, build heterogeneous graph, return HeteroData + metadata."""
    logger.info("Loading Amazon Video Games dataset...")

    # Try cached path first
    amazon_file = DATA_DEP / "temp" / "datasets" / "full_LoganKells_amazon_product_reviews_video_games_train.json"
    if not amazon_file.exists():
        logger.error(f"Dataset not found at {amazon_file}")
        raise FileNotFoundError(f"Dataset not found at {amazon_file}")

    raw = pd.read_json(amazon_file)
    logger.info(f"Loaded {len(raw)} rows, columns: {list(raw.columns)}")

    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])

    # Limit to 50K rows
    if len(raw) > 50000:
        raw = raw.iloc[:50000].reset_index(drop=True)
        logger.info(f"Trimmed to 50000 rows")

    n_reviews = len(raw)

    # --- Build relational tables ---
    products = raw[["asin"]].drop_duplicates().reset_index(drop=True)
    product_id_map = {asin: i for i, asin in enumerate(products["asin"])}
    n_products = len(products)
    logger.info(f"Products: {n_products}")

    customers = raw[["reviewerID"]].drop_duplicates("reviewerID").reset_index(drop=True)
    customer_id_map = {rid: i for i, rid in enumerate(customers["reviewerID"])}
    n_customers = len(customers)
    logger.info(f"Customers: {n_customers}")

    # --- Encode child (review) features ---
    feature_names = []
    feature_arrays = []

    # Time features (3 dims)
    if "unixReviewTime" in raw.columns:
        ts_feats = encode_timestamp(raw["unixReviewTime"])
        feature_names.extend(["time_year", "time_month", "time_dayofweek"])
        feature_arrays.append(ts_feats)

    # Text hash features (16 dims)
    if "summary" in raw.columns:
        summary_feats = encode_text_hash(raw["summary"], n_features=TEXT_HASH_DIM)
        feature_names.extend([f"summary_h{i}" for i in range(TEXT_HASH_DIM)])
        feature_arrays.append(summary_feats)

    # Helpful votes (2 dims)
    helpful_parsed = raw["helpful"].apply(parse_helpful)
    helpful_up = helpful_parsed.apply(lambda x: x[0]).values.astype(np.float32).reshape(-1, 1)
    helpful_total = helpful_parsed.apply(lambda x: x[1]).values.astype(np.float32).reshape(-1, 1)
    feature_arrays.extend([
        StandardScaler().fit_transform(helpful_up),
        StandardScaler().fit_transform(helpful_total),
    ])
    feature_names.extend(["helpful_up", "helpful_total"])
    del helpful_parsed, helpful_up, helpful_total

    X_child = np.hstack(feature_arrays).astype(np.float32)
    del feature_arrays
    gc.collect()
    child_dim = X_child.shape[1]
    logger.info(f"Child features: {X_child.shape} ({child_dim} dims: {feature_names})")

    # --- Build parent features via aggregation ---
    helpful_parsed = raw["helpful"].apply(parse_helpful)
    agg_df = raw[["asin", "reviewerID", "overall"]].copy()
    agg_df["helpful_up"] = helpful_parsed.apply(lambda x: x[0]).astype(np.float32)
    agg_df["helpful_total"] = helpful_parsed.apply(lambda x: x[1]).astype(np.float32)
    del helpful_parsed

    # Product features (5 dims)
    product_agg = agg_df.groupby("asin").agg(
        prod_mean_rating=("overall", "mean"),
        prod_std_rating=("overall", "std"),
        prod_review_count=("overall", "count"),
        prod_mean_helpful_up=("helpful_up", "mean"),
        prod_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    product_agg["prod_std_rating"] = product_agg["prod_std_rating"].fillna(0)
    product_feat_names = list(product_agg.columns)

    # Standardize and reorder by product_id_map
    prod_scaler = StandardScaler()
    prod_feats_np = prod_scaler.fit_transform(product_agg.values).astype(np.float32)
    product_features = np.zeros((n_products, 5), dtype=np.float32)
    for asin, idx in product_id_map.items():
        if asin in product_agg.index:
            loc = product_agg.index.get_loc(asin)
            product_features[idx] = prod_feats_np[loc]
    product_dim = 5
    logger.info(f"Product features: ({n_products}, {product_dim})")

    # Customer features (5 dims)
    customer_agg = agg_df.groupby("reviewerID").agg(
        cust_mean_rating=("overall", "mean"),
        cust_std_rating=("overall", "std"),
        cust_review_count=("overall", "count"),
        cust_mean_helpful_up=("helpful_up", "mean"),
        cust_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    customer_agg["cust_std_rating"] = customer_agg["cust_std_rating"].fillna(0)
    customer_feat_names = list(customer_agg.columns)

    cust_scaler = StandardScaler()
    cust_feats_np = cust_scaler.fit_transform(customer_agg.values).astype(np.float32)
    customer_features = np.zeros((n_customers, 5), dtype=np.float32)
    for rid, idx in customer_id_map.items():
        if rid in customer_agg.index:
            loc = customer_agg.index.get_loc(rid)
            customer_features[idx] = cust_feats_np[loc]
    customer_dim = 5
    logger.info(f"Customer features: ({n_customers}, {customer_dim})")

    del agg_df, product_agg, customer_agg, prod_feats_np, cust_feats_np
    gc.collect()

    # --- Build edge indices ---
    review_product_src = []  # review indices
    review_product_dst = []  # product indices
    review_customer_src = []
    review_customer_dst = []

    asins = raw["asin"].values
    reviewer_ids = raw["reviewerID"].values

    for i in range(n_reviews):
        pid = product_id_map[asins[i]]
        cid = customer_id_map[reviewer_ids[i]]
        review_product_src.append(i)
        review_product_dst.append(pid)
        review_customer_src.append(i)
        review_customer_dst.append(cid)

    review_product_src = np.array(review_product_src, dtype=np.int64)
    review_product_dst = np.array(review_product_dst, dtype=np.int64)
    review_customer_src = np.array(review_customer_src, dtype=np.int64)
    review_customer_dst = np.array(review_customer_dst, dtype=np.int64)

    # --- Build HeteroData ---
    data = HeteroData()
    data["product"].x = torch.tensor(product_features, dtype=torch.float32)
    data["customer"].x = torch.tensor(customer_features, dtype=torch.float32)
    data["review"].x = torch.tensor(X_child, dtype=torch.float32)

    # Target: rating (standardized for regression)
    ratings = raw["overall"].values.astype(np.float32)
    rating_scaler = StandardScaler()
    ratings_scaled = rating_scaler.fit_transform(ratings.reshape(-1, 1)).flatten().astype(np.float32)
    data["review"].y = torch.tensor(ratings_scaled, dtype=torch.float32)
    data["review"].y_raw = torch.tensor(ratings, dtype=torch.float32)

    # Edge indices: review -> product (child -> parent)
    data[("review", "of_product", "product")].edge_index = torch.stack([
        torch.tensor(review_product_src, dtype=torch.long),
        torch.tensor(review_product_dst, dtype=torch.long),
    ])
    # Reverse: product -> review (parent -> child)
    data[("product", "has_review", "review")].edge_index = torch.stack([
        torch.tensor(review_product_dst, dtype=torch.long),
        torch.tensor(review_product_src, dtype=torch.long),
    ])
    # review -> customer
    data[("review", "by_customer", "customer")].edge_index = torch.stack([
        torch.tensor(review_customer_src, dtype=torch.long),
        torch.tensor(review_customer_dst, dtype=torch.long),
    ])
    # customer -> review
    data[("customer", "wrote_review", "review")].edge_index = torch.stack([
        torch.tensor(review_customer_dst, dtype=torch.long),
        torch.tensor(review_customer_src, dtype=torch.long),
    ])

    # --- Train/val/test split ---
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_reviews)
    n_train = int(0.7 * n_reviews)
    n_val = int(0.15 * n_reviews)

    train_mask = torch.zeros(n_reviews, dtype=torch.bool)
    val_mask = torch.zeros(n_reviews, dtype=torch.bool)
    test_mask = torch.zeros(n_reviews, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data["review"].train_mask = train_mask
    data["review"].val_mask = val_mask
    data["review"].test_mask = test_mask

    del raw, X_child, product_features, customer_features
    gc.collect()

    metadata = {
        "n_products": n_products,
        "n_customers": n_customers,
        "n_reviews": n_reviews,
        "child_dim": child_dim,
        "product_dim": product_dim,
        "customer_dim": customer_dim,
        "feature_names": feature_names,
        "product_feat_names": product_feat_names,
        "customer_feat_names": customer_feat_names,
        "rating_mean": float(rating_scaler.mean_[0]),
        "rating_std": float(rating_scaler.scale_[0]),
    }

    logger.info(f"Graph built: {data}")
    logger.info(f"  Products: {n_products}, Customers: {n_customers}, Reviews: {n_reviews}")
    logger.info(f"  Train/Val/Test: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")

    return data, metadata


# ---------------------------------------------------------------------------
# Model Implementations
# ---------------------------------------------------------------------------

class PRMPConv(MessagePassing):
    """Predictive Residual Message Passing convolution.

    From research_id2_it1__opus Section 7.1:
    - 2-layer MLP prediction, zero-init final layer
    - LayerNorm on residuals (for 'prmp' mode)
    - Mean aggregation, detached parent input
    - GraphSAGE-style concat+linear update
    """

    def __init__(self, in_src: int, in_dst: int, out_channels: int, mode: str = "prmp"):
        super().__init__(aggr="mean")
        self.mode = mode
        hidden = min(in_dst, in_src)

        # Prediction MLP: parent -> predicted child
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_dst, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_src),
        )
        # Zero-init final layer (Section 5.3)
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)

        # LayerNorm on residuals (Section 5.2 Option B)
        self.norm = nn.LayerNorm(in_src) if mode == "prmp" else None

        # Update MLP (GraphSAGE-style)
        if mode == "no_subtract":
            update_in = in_dst + in_src * 2  # concat raw + predicted
        else:
            update_in = in_dst + in_src  # concat parent + residual
        self.update_mlp = nn.Linear(update_in, out_channels)

        # Instrumentation storage
        self._last_predicted = None
        self._last_residual = None
        self._last_child_h = None
        self._last_parent_h = None

    def forward(self, x, edge_index):
        if isinstance(x, tuple) or isinstance(x, list):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x
        aggr_out = self.propagate(edge_index, x=(x_src, x_dst))
        out = self.update_mlp(torch.cat([x_dst, aggr_out], dim=-1))
        return out

    def message(self, x_j, x_i):
        # x_j = source (child), x_i = destination (parent)
        predicted = self.pred_mlp(x_i.detach())

        # Store for instrumentation
        self._last_predicted = predicted.detach()
        self._last_child_h = x_j.detach()
        self._last_parent_h = x_i.detach()

        if self.mode == "no_subtract":
            return torch.cat([x_j, predicted], dim=-1)
        else:
            residual = x_j - predicted
            self._last_residual = residual.detach()
            if self.norm is not None:
                return self.norm(residual)
            return residual


class InstrumentedHeteroGNN(nn.Module):
    """Heterogeneous GNN with full instrumentation for mechanism analysis."""

    def __init__(self, product_dim: int, customer_dim: int, review_dim: int,
                 hidden_dim: int, num_layers: int, variant: str):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Feature encoders
        self.product_enc = nn.Sequential(
            nn.Linear(product_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.customer_enc = nn.Sequential(
            nn.Linear(customer_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.review_enc = nn.Sequential(
            nn.Linear(review_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN layers with HeteroConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        hd = hidden_dim

        for layer in range(num_layers):
            if variant == "standard":
                conv_dict = {
                    ("review", "of_product", "product"): SAGEConv((hd, hd), hd),
                    ("product", "has_review", "review"): SAGEConv((hd, hd), hd),
                    ("review", "by_customer", "customer"): SAGEConv((hd, hd), hd),
                    ("customer", "wrote_review", "review"): SAGEConv((hd, hd), hd),
                }
            else:
                # PRMP/no_subtract on child->parent edges, SAGEConv on parent->child
                conv_dict = {
                    ("review", "of_product", "product"): PRMPConv(hd, hd, hd, mode=variant),
                    ("product", "has_review", "review"): SAGEConv((hd, hd), hd),
                    ("review", "by_customer", "customer"): PRMPConv(hd, hd, hd, mode=variant),
                    ("customer", "wrote_review", "review"): SAGEConv((hd, hd), hd),
                }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # Per-node-type LayerNorm
            norm_dict = {
                "product": LayerNorm(hd),
                "customer": LayerNorm(hd),
                "review": LayerNorm(hd),
            }
            self.norms.append(nn.ModuleDict(norm_dict))

        # Prediction head: review node -> rating
        self.head = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hd // 2, 1)
        )

    def encode_features(self, data: HeteroData) -> dict:
        """Encode raw features into hidden representations."""
        return {
            "product": self.product_enc(data["product"].x),
            "customer": self.customer_enc(data["customer"].x),
            "review": self.review_enc(data["review"].x),
        }

    def forward(self, data: HeteroData) -> torch.Tensor:
        h_dict = self.encode_features(data)

        edge_index_dict = {
            key: data[key].edge_index for key in [
                ("review", "of_product", "product"),
                ("product", "has_review", "review"),
                ("review", "by_customer", "customer"),
                ("customer", "wrote_review", "review"),
            ]
        }

        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h_dict, edge_index_dict)
            # Apply norms, ReLU, and residual connections
            for node_type in h_new:
                h_new[node_type] = norm_dict[node_type](h_new[node_type])
                h_new[node_type] = F.relu(h_new[node_type])
                if layer_idx > 0:
                    h_new[node_type] = h_new[node_type] + h_dict[node_type]
            h_dict = h_new

        return self.head(h_dict["review"]).squeeze(-1)

    def get_prmp_convs(self) -> list:
        """Return list of (layer_idx, edge_type, prmp_conv) for instrumentation."""
        result = []
        for layer_idx, conv in enumerate(self.convs):
            for edge_type, subconv in conv.convs.items():
                if isinstance(subconv, PRMPConv):
                    result.append((layer_idx, edge_type, subconv))
        return result


# ---------------------------------------------------------------------------
# Instrumentation Functions
# ---------------------------------------------------------------------------

def param_grad_norm(module: nn.Module) -> float:
    """Compute total gradient norm for a module's parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm().item() ** 2
    return total ** 0.5


@torch.no_grad()
def measure_prediction_r2(model: InstrumentedHeteroGNN, data: HeteroData) -> dict:
    """After each epoch: R-squared of pred_mlp(parent_emb) vs actual child embeddings."""
    model.eval()
    results = {}

    h_dict = model.encode_features(data)
    edge_index_dict = {
        key: data[key].edge_index for key in [
            ("review", "of_product", "product"),
            ("product", "has_review", "review"),
            ("review", "by_customer", "customer"),
            ("customer", "wrote_review", "review"),
        ]
    }

    for layer_idx, conv in enumerate(model.convs):
        for edge_type, subconv in conv.convs.items():
            if not isinstance(subconv, PRMPConv):
                continue

            ei = edge_index_dict[edge_type]
            child_type = edge_type[0]
            parent_type = edge_type[2]

            # Subsample edges for efficiency
            n_edges = ei.shape[1]
            max_edges = 10000
            if n_edges > max_edges:
                idx = torch.randperm(n_edges, device=ei.device)[:max_edges]
                ei_sub = ei[:, idx]
            else:
                ei_sub = ei

            child_h = h_dict[child_type][ei_sub[0]]
            parent_h = h_dict[parent_type][ei_sub[1]]
            predicted = subconv.pred_mlp(parent_h)

            ss_res = ((child_h - predicted) ** 2).sum().item()
            ss_tot = ((child_h - child_h.mean(0)) ** 2).sum().item()
            r2 = 1 - ss_res / max(ss_tot, 1e-8)

            key = f"L{layer_idx}_{child_type[:3]}2{parent_type[:3]}"
            results[key] = {
                "pred_r2": round(r2, 6),
                "res_std": round((child_h - predicted).std().item(), 6),
                "pred_std": round(predicted.std().item(), 6),
            }

        # Pass through conv layer
        h_new = conv(h_dict, edge_index_dict)
        for node_type in h_new:
            h_new[node_type] = model.norms[layer_idx][node_type](h_new[node_type])
            h_new[node_type] = F.relu(h_new[node_type])
            if layer_idx > 0:
                h_new[node_type] = h_new[node_type] + h_dict[node_type]
        h_dict = h_new

    return results


def measure_mutual_information(model: InstrumentedHeteroGNN, data: HeteroData,
                                target: torch.Tensor) -> dict:
    """Compute MI between features and target rating."""
    model.eval()
    results = {}

    # Subsample review nodes
    n = len(target)
    n_sub = min(MI_SUBSAMPLE, n)
    idx = np.random.choice(n, n_sub, replace=False)
    y = target[idx].cpu().numpy()

    with torch.no_grad():
        h_dict = model.encode_features(data)

    edge_index_dict = {
        key: data[key].edge_index for key in [
            ("review", "of_product", "product"),
            ("product", "has_review", "review"),
            ("review", "by_customer", "customer"),
            ("customer", "wrote_review", "review"),
        ]
    }

    for layer_idx, conv in enumerate(model.convs):
        for edge_type, subconv in conv.convs.items():
            if not isinstance(subconv, PRMPConv):
                continue

            ei = edge_index_dict[edge_type]
            child_type = edge_type[0]
            parent_type = edge_type[2]

            # Find edges from subsampled review nodes
            idx_tensor = torch.tensor(idx, device=ei.device, dtype=torch.long)
            mask = torch.isin(ei[0], idx_tensor)

            if mask.sum() < 50:
                continue

            # Limit to prevent MI being too slow
            masked_indices = torch.where(mask)[0]
            if len(masked_indices) > 3000:
                sub_idx = torch.randperm(len(masked_indices))[:3000]
                masked_indices = masked_indices[sub_idx]

            with torch.no_grad():
                child_h = h_dict[child_type][ei[0][masked_indices]].cpu().numpy()
                parent_h = h_dict[parent_type][ei[1][masked_indices]].cpu().numpy()
                predicted = subconv.pred_mlp(
                    torch.tensor(parent_h, device=DEVICE)
                ).cpu().numpy()

            residual = child_h - predicted
            edge_targets = target[ei[0][masked_indices]].cpu().numpy()

            try:
                mi_raw = mutual_info_regression(child_h, edge_targets, n_neighbors=5, random_state=42).mean()
                mi_residual = mutual_info_regression(residual, edge_targets, n_neighbors=5, random_state=42).mean()
                mi_predicted = mutual_info_regression(predicted, edge_targets, n_neighbors=5, random_state=42).mean()

                key = f"L{layer_idx}_{child_type[:3]}2{parent_type[:3]}"
                results[key] = {
                    "mi_raw": round(float(mi_raw), 6),
                    "mi_res": round(float(mi_residual), 6),
                    "mi_pred": round(float(mi_predicted), 6),
                    "mi_ratio": round(float(mi_residual / max(mi_raw, 1e-8)), 6),
                }
            except Exception as e:
                logger.warning(f"MI computation failed for {edge_type}: {e}")

        # Pass through conv layer
        with torch.no_grad():
            h_new = conv(h_dict, edge_index_dict)
            for node_type in h_new:
                h_new[node_type] = model.norms[layer_idx][node_type](h_new[node_type])
                h_new[node_type] = F.relu(h_new[node_type])
                if layer_idx > 0:
                    h_new[node_type] = h_new[node_type] + h_dict[node_type]
            h_dict = h_new

    return results


@torch.no_grad()
def measure_weight_dynamics(model: InstrumentedHeteroGNN) -> dict:
    """Track prediction MLP weight norms and evolution."""
    results = {}
    for layer_idx, edge_type, subconv in model.get_prmp_convs():
        key = f"L{layer_idx}_{edge_type[0][:3]}2{edge_type[2][:3]}"
        total_norm_sq = 0.0
        for name, param in subconv.pred_mlp.named_parameters():
            pnorm = param.data.norm().item()
            results[f"{key}_{name}_norm"] = round(pnorm, 6)
            total_norm_sq += pnorm ** 2
        results[f"{key}_total_norm"] = round(total_norm_sq ** 0.5, 6)
        # Final layer norm (prediction strength indicator)
        final_layer = subconv.pred_mlp[-1]
        results[f"{key}_final_w_norm"] = round(final_layer.weight.data.norm().item(), 6)
    return results


def measure_gradient_flow(model: InstrumentedHeteroGNN, data: HeteroData,
                          target: torch.Tensor, criterion: nn.Module) -> dict:
    """One forward+backward pass with gradient measurement."""
    model.train()
    model.zero_grad()

    # Register hooks on pred_mlp params
    grad_stats = {}
    hooks = []

    for layer_idx, edge_type, subconv in model.get_prmp_convs():
        key = f"L{layer_idx}_{edge_type[0][:3]}2{edge_type[2][:3]}"
        for name, param in subconv.pred_mlp.named_parameters():
            def make_hook(k, n):
                def hook_fn(grad):
                    grad_stats[f"{k}_{n}_gnorm"] = round(grad.norm().item(), 6)
                    return grad
                return hook_fn
            hooks.append(param.register_hook(make_hook(key, name)))

    out = model(data)
    mask = data["review"].train_mask
    loss = criterion(out[mask], target[mask])
    loss.backward()

    results = dict(grad_stats)

    # Encoder gradient norms
    results["rev_enc_gnorm"] = round(param_grad_norm(model.review_enc), 6)
    results["prod_enc_gnorm"] = round(param_grad_norm(model.product_enc), 6)
    results["cust_enc_gnorm"] = round(param_grad_norm(model.customer_enc), 6)
    results["total_gnorm"] = round(
        sum(p.grad.data.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5, 6
    )

    for h in hooks:
        h.remove()

    return results


def measure_gradient_angle(model: InstrumentedHeteroGNN, data: HeteroData,
                           target: torch.Tensor, criterion: nn.Module) -> dict:
    """Compare direction of task gradient vs prediction accuracy gradient on pred_mlp params."""
    results = {}

    try:
        # Step 1: Task gradient
        model.train()
        model.zero_grad()
        out = model(data)
        mask = data["review"].train_mask
        task_loss = criterion(out[mask], target[mask])
        task_loss.backward(retain_graph=False)
        task_grads = {}
        for n, p in model.named_parameters():
            if "pred_mlp" in n and p.grad is not None:
                task_grads[n] = p.grad.clone()

        # Step 2: Prediction accuracy gradient
        model.zero_grad()
        # Forward to get embeddings and compute prediction MSE
        h_dict = model.encode_features(data)
        edge_index_dict = {
            key: data[key].edge_index for key in [
                ("review", "of_product", "product"),
                ("product", "has_review", "review"),
                ("review", "by_customer", "customer"),
                ("customer", "wrote_review", "review"),
            ]
        }

        pred_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for layer_idx, conv in enumerate(model.convs):
            for edge_type, subconv in conv.convs.items():
                if not isinstance(subconv, PRMPConv):
                    continue
                ei = edge_index_dict[edge_type]
                child_type, parent_type = edge_type[0], edge_type[2]
                # Subsample
                n_edges = ei.shape[1]
                max_e = 5000
                if n_edges > max_e:
                    sub = torch.randperm(n_edges, device=ei.device)[:max_e]
                    ei_s = ei[:, sub]
                else:
                    ei_s = ei
                child_h = h_dict[child_type][ei_s[0]].detach()
                parent_h = h_dict[parent_type][ei_s[1]].detach()
                predicted = subconv.pred_mlp(parent_h)
                pred_loss = pred_loss + F.mse_loss(predicted, child_h)

            # Pass through layer
            with torch.no_grad():
                h_new = conv(h_dict, edge_index_dict)
                for nt in h_new:
                    h_new[nt] = model.norms[layer_idx][nt](h_new[nt])
                    h_new[nt] = F.relu(h_new[nt])
                    if layer_idx > 0:
                        h_new[nt] = h_new[nt] + h_dict[nt]
                h_dict = h_new

        pred_loss.backward()
        pred_grads = {}
        for n, p in model.named_parameters():
            if "pred_mlp" in n and p.grad is not None:
                pred_grads[n] = p.grad.clone()

        # Step 3: Cosine similarity
        cosines = []
        for name in task_grads:
            if name in pred_grads:
                cos_sim = F.cosine_similarity(
                    task_grads[name].flatten().unsqueeze(0),
                    pred_grads[name].flatten().unsqueeze(0),
                ).item()
                cosines.append(cos_sim)

        if cosines:
            results["mean_cos"] = round(float(np.mean(cosines)), 6)
            results["min_cos"] = round(float(np.min(cosines)), 6)
            results["max_cos"] = round(float(np.max(cosines)), 6)
    except Exception as e:
        logger.warning(f"Gradient angle measurement failed: {e}")
        results["error"] = str(e)[:100]

    return results


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def run_instrumented_training(data: HeteroData, variant: str, seed: int,
                              epochs: int = EPOCHS) -> dict:
    """Full instrumented training for one variant + seed."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    metadata = {
        "n_products": data["product"].x.shape[0],
        "n_customers": data["customer"].x.shape[0],
        "n_reviews": data["review"].x.shape[0],
    }

    model = InstrumentedHeteroGNN(
        product_dim=data["product"].x.shape[1],
        customer_dim=data["customer"].x.shape[1],
        review_dim=data["review"].x.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,
        variant=variant,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    data_dev = data.to(DEVICE)
    target = data_dev["review"].y

    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()

        # --- TRAIN STEP ---
        model.train()
        optimizer.zero_grad()
        out = model(data_dev)
        mask = data_dev["review"].train_mask
        loss = criterion(out[mask], target[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- EVAL ---
        model.eval()
        with torch.no_grad():
            out_eval = model(data_dev)
            val_loss = criterion(out_eval[data_dev["review"].val_mask],
                                 target[data_dev["review"].val_mask]).item()
            test_loss = criterion(out_eval[data_dev["review"].test_mask],
                                  target[data_dev["review"].test_mask]).item()
            val_rmse = val_loss ** 0.5
            test_rmse = test_loss ** 0.5

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(loss.item(), 6),
            "val_rmse": round(val_rmse, 6),
            "test_rmse": round(test_rmse, 6),
        }

        # --- Prediction R-squared (every epoch for PRMP variants, fast) ---
        if variant in ("prmp", "no_subtract"):
            try:
                pred_r2 = measure_prediction_r2(model, data_dev)
                epoch_data["pred_r2"] = pred_r2
            except Exception as e:
                logger.warning(f"pred_r2 failed epoch {epoch}: {e}")

        # --- Mutual Information (every MI_INTERVAL epochs, slow) ---
        if epoch % MI_INTERVAL == 0 or epoch == 1:
            try:
                mi_data = measure_mutual_information(model, data_dev, target)
                epoch_data["mi"] = mi_data
            except Exception as e:
                logger.warning(f"MI failed epoch {epoch}: {e}")

        # --- Gradient Flow (every 5 epochs) ---
        if epoch % 5 == 0 or epoch <= 5:
            try:
                grad_data = measure_gradient_flow(model, data_dev, target, criterion)
                epoch_data["grad"] = grad_data
            except Exception as e:
                logger.warning(f"Gradient flow failed epoch {epoch}: {e}")

        # --- Weight Dynamics (every epoch for PRMP variants) ---
        if variant in ("prmp", "no_subtract"):
            try:
                weight_data = measure_weight_dynamics(model)
                epoch_data["weights"] = weight_data
            except Exception as e:
                logger.warning(f"Weight dynamics failed epoch {epoch}: {e}")

        # --- Gradient Angle (every MI_INTERVAL for PRMP only) ---
        if variant == "prmp" and (epoch % MI_INTERVAL == 0 or epoch == 1):
            try:
                angle_data = measure_gradient_angle(model, data_dev, target, criterion)
                epoch_data["grad_angle"] = angle_data
            except Exception as e:
                logger.warning(f"Gradient angle failed epoch {epoch}: {e}")

        epoch_metrics.append(epoch_data)
        epoch_time = time.time() - epoch_t0

        if epoch <= 5 or epoch % 10 == 0:
            logger.info(
                f"[{variant}/s{seed}] E{epoch}: "
                f"train={loss.item():.4f} val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} "
                f"({epoch_time:.1f}s)"
            )

    total_time = time.time() - t0
    logger.info(f"[{variant}/s{seed}] Done in {total_time:.1f}s")

    # Collect test set predictions for per-review output examples
    model.eval()
    with torch.no_grad():
        out_final = model(data_dev)
        test_mask_bool = data_dev["review"].test_mask
        test_preds_np = out_final[test_mask_bool].cpu().numpy().astype(np.float64)
        test_targets_np = data_dev["review"].y_raw[test_mask_bool].cpu().numpy().astype(np.float64)
        test_features_np = data_dev["review"].x[test_mask_bool].cpu().numpy().astype(np.float64)

    # Clean up
    del data_dev
    if HAS_GPU:
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "seed": seed,
        "epochs": epoch_metrics,
        "final_test_rmse": epoch_metrics[-1]["test_rmse"],
        "best_val_rmse": min(e["val_rmse"] for e in epoch_metrics),
        "best_val_epoch": min(range(len(epoch_metrics)), key=lambda i: epoch_metrics[i]["val_rmse"]) + 1,
        "num_params": sum(p.numel() for p in model.parameters()),
        "total_time_s": round(total_time, 1),
        "test_preds": test_preds_np,
        "test_targets": test_targets_np,
        "test_features": test_features_np,
    }


def thin_epoch_data(epoch_metrics: list) -> list:
    """Subsample epoch data to keep file manageable. Keep first 5 + every 10th."""
    thinned = []
    for e in epoch_metrics:
        ep = e["epoch"]
        if ep <= 5 or ep % 10 == 0:
            thinned.append(e)
        else:
            # Keep only scalar metrics
            thinned.append({
                "epoch": ep,
                "train_loss": e["train_loss"],
                "val_rmse": e["val_rmse"],
                "test_rmse": e["test_rmse"],
            })
    return thinned


def analyze_mechanism(all_runs: list) -> dict:
    """Extract mechanism diagnosis from PRMP runs."""
    diagnosis = {
        "information_filtering_supported": False,
        "mi_residual_vs_raw_ratio": None,
        "prediction_r2_final": None,
        "gradient_angle_trend": "unknown",
        "weight_growth_rate": None,
    }

    prmp_runs = [r for r in all_runs if r["variant"] == "prmp"]
    if not prmp_runs:
        return diagnosis

    # Gather MI ratios from final MI measurement
    mi_ratios = []
    pred_r2s = []
    cos_means = []

    for run in prmp_runs:
        epochs = run["epochs"]

        # Find last epoch with MI data
        for e in reversed(epochs):
            if "mi" in e:
                for key, vals in e["mi"].items():
                    if "mi_ratio" in vals:
                        mi_ratios.append(vals["mi_ratio"])
                break

        # Final prediction R2
        if epochs and "pred_r2" in epochs[-1]:
            for key, vals in epochs[-1]["pred_r2"].items():
                if "pred_r2" in vals:
                    pred_r2s.append(vals["pred_r2"])

        # Gradient angle from last measurement
        for e in reversed(epochs):
            if "grad_angle" in e and "mean_cos" in e["grad_angle"]:
                cos_means.append(e["grad_angle"]["mean_cos"])
                break

    if mi_ratios:
        mean_ratio = float(np.mean(mi_ratios))
        diagnosis["mi_residual_vs_raw_ratio"] = round(mean_ratio, 4)
        diagnosis["information_filtering_supported"] = mean_ratio > 1.0

    if pred_r2s:
        diagnosis["prediction_r2_final"] = round(float(np.mean(pred_r2s)), 4)

    if cos_means:
        mean_cos = float(np.mean(cos_means))
        if mean_cos > 0.3:
            diagnosis["gradient_angle_trend"] = "aligned"
        elif mean_cos < -0.3:
            diagnosis["gradient_angle_trend"] = "opposing"
        else:
            diagnosis["gradient_angle_trend"] = "orthogonal"

    # Weight growth: compare epoch 1 vs final weight norms
    weight_growths = []
    for run in prmp_runs:
        epochs = run["epochs"]
        if len(epochs) >= 2 and "weights" in epochs[0] and "weights" in epochs[-1]:
            for key in epochs[0]["weights"]:
                if "total_norm" in key:
                    w0 = epochs[0]["weights"].get(key, 0)
                    wf = epochs[-1]["weights"].get(key, 0)
                    if w0 > 1e-8:
                        weight_growths.append(wf / w0)
                    elif wf > 1e-8:
                        weight_growths.append(float("inf"))

    if weight_growths:
        finite_growths = [g for g in weight_growths if g != float("inf")]
        if finite_growths:
            diagnosis["weight_growth_rate"] = round(float(np.mean(finite_growths)), 4)

    return diagnosis


def generate_conclusion(diagnosis: dict, aggregated: dict) -> str:
    """Generate a text conclusion from mechanism diagnosis."""
    parts = []

    # Performance comparison
    std_rmse = aggregated.get("standard", {}).get("mean_test_rmse", 0)
    prmp_rmse = aggregated.get("prmp", {}).get("mean_test_rmse", 0)
    nosub_rmse = aggregated.get("no_subtract", {}).get("mean_test_rmse", 0)

    if std_rmse > 0:
        prmp_improve = (std_rmse - prmp_rmse) / std_rmse * 100
        nosub_improve = (std_rmse - nosub_rmse) / std_rmse * 100
        parts.append(
            f"PRMP achieves {prmp_improve:+.2f}% relative RMSE change vs standard. "
            f"No-subtract variant: {nosub_improve:+.2f}%."
        )

    # Mechanism analysis
    if diagnosis.get("information_filtering_supported"):
        parts.append(
            f"Information filtering IS supported: MI(residual,target)/MI(raw,target) = "
            f"{diagnosis.get('mi_residual_vs_raw_ratio', 'N/A')} > 1.0, "
            f"meaning residuals carry more task-relevant information than raw features."
        )
    else:
        ratio = diagnosis.get("mi_residual_vs_raw_ratio", "N/A")
        parts.append(
            f"Information filtering NOT supported: MI ratio = {ratio}. "
            f"PRMP's benefit may come from implicit regularization or gradient dynamics."
        )

    if diagnosis.get("prediction_r2_final") is not None:
        parts.append(
            f"Prediction MLP final R2 = {diagnosis['prediction_r2_final']:.4f}."
        )

    if diagnosis.get("gradient_angle_trend") != "unknown":
        parts.append(
            f"Task vs prediction gradient angle: {diagnosis['gradient_angle_trend']}."
        )

    if diagnosis.get("weight_growth_rate") is not None:
        parts.append(
            f"Prediction MLP weight growth: {diagnosis['weight_growth_rate']:.2f}x from init to final."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("INSTRUMENTED PRMP MECHANISM ANALYSIS")
    logger.info("=" * 70)

    # Step 1: Load data and build graph
    data, data_meta = load_and_build_graph(seed=42)

    # Step 2: Run all experiments
    all_runs = []
    n_reviews = data["review"].x.shape[0]
    for seed in SEEDS:
        # Only change the train/val/test split mask for different seeds
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_reviews)
        n_train = int(0.7 * n_reviews)
        n_val = int(0.15 * n_reviews)
        train_mask = torch.zeros(n_reviews, dtype=torch.bool)
        val_mask = torch.zeros(n_reviews, dtype=torch.bool)
        test_mask = torch.zeros(n_reviews, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True
        data["review"].train_mask = train_mask
        data["review"].val_mask = val_mask
        data["review"].test_mask = test_mask

        for variant in VARIANTS:
            elapsed = time.time() - start_time
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running {variant} seed={seed} (elapsed: {elapsed:.0f}s)")
            logger.info(f"{'=' * 60}")

            result = run_instrumented_training(data, variant, seed, EPOCHS)
            all_runs.append(result)

            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

            # Time check: if running too long, reduce scope
            elapsed = time.time() - start_time
            if elapsed > 2700:  # 45 minutes
                logger.warning(f"Time budget getting tight ({elapsed:.0f}s). Continuing with remaining runs.")

    # Step 3: Aggregate results
    aggregated = {}
    for variant in VARIANTS:
        variant_runs = [r for r in all_runs if r["variant"] == variant]
        if not variant_runs:
            continue
        test_rmses = [r["final_test_rmse"] for r in variant_runs]
        aggregated[variant] = {
            "mean_test_rmse": round(float(np.mean(test_rmses)), 6),
            "std_test_rmse": round(float(np.std(test_rmses)), 6),
            "mean_best_val_rmse": round(float(np.mean([r["best_val_rmse"] for r in variant_runs])), 6),
            "num_params": variant_runs[0]["num_params"],
        }

    # Relative improvements
    if "standard" in aggregated:
        std_rmse = aggregated["standard"]["mean_test_rmse"]
        for v in ["prmp", "no_subtract"]:
            if v in aggregated:
                v_rmse = aggregated[v]["mean_test_rmse"]
                aggregated[v]["rel_improve_pct"] = round((std_rmse - v_rmse) / std_rmse * 100, 4)

    # Step 4: Mechanism diagnosis
    mechanism_diagnosis = analyze_mechanism(all_runs)
    conclusion = generate_conclusion(mechanism_diagnosis, aggregated)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for v, agg in aggregated.items():
        logger.info(f"  {v}: test_rmse={agg['mean_test_rmse']:.4f} +/- {agg['std_test_rmse']:.4f}")
    logger.info(f"Conclusion: {conclusion}")

    # Step 5: Build method_out.json with per-review examples + predict_* fields
    examples = []

    # Group runs by seed for aligned predictions
    runs_by_seed: dict[int, dict[str, dict]] = {}
    for run in all_runs:
        s = run["seed"]
        if s not in runs_by_seed:
            runs_by_seed[s] = {}
        runs_by_seed[s][run["variant"]] = run

    # Rating scale factors to convert standardised predictions back to raw
    r_mean = data_meta["rating_mean"]
    r_std = data_meta["rating_std"]

    # Per-review examples: sample 20 test reviews per seed (60 total)
    n_per_seed = 20
    for seed_val in SEEDS:
        seed_runs = runs_by_seed.get(seed_val, {})
        if len(seed_runs) < len(VARIANTS):
            logger.warning(f"Seed {seed_val} has only {len(seed_runs)} variants, skipping review examples")
            continue

        std_run = seed_runs["standard"]
        prmp_run = seed_runs["prmp"]
        nosub_run = seed_runs["no_subtract"]

        n_test = len(std_run["test_preds"])
        rng_sample = np.random.RandomState(seed_val + 999)
        sample_idx = rng_sample.choice(n_test, min(n_per_seed, n_test), replace=False)

        for idx in sample_idx:
            # Build feature dict from review features
            feat_dict = {}
            feats = std_run["test_features"][idx]
            fnames = data_meta["feature_names"]
            for fi in range(len(feats)):
                fname = fnames[fi] if fi < len(fnames) else f"feat_{fi}"
                feat_dict[fname] = round(float(feats[fi]), 4)

            true_rating = round(float(std_run["test_targets"][idx]), 1)

            # Convert standardised predictions to raw rating scale
            pred_std_raw = round(float(std_run["test_preds"][idx]) * r_std + r_mean, 4)
            pred_prmp_raw = round(float(prmp_run["test_preds"][idx]) * r_std + r_mean, 4)
            pred_nosub_raw = round(float(nosub_run["test_preds"][idx]) * r_std + r_mean, 4)

            example = {
                "input": json.dumps(feat_dict),
                "output": str(true_rating),
                "predict_standard": str(pred_std_raw),
                "predict_prmp": str(pred_prmp_raw),
                "predict_no_subtract": str(pred_nosub_raw),
                "metadata_seed": seed_val,
                "metadata_task_type": "regression",
                "metadata_review_index": int(idx),
            }
            examples.append(example)

    # Per-variant run summary examples (one per variant with epoch diagnostics)
    for variant in VARIANTS:
        variant_runs = [r for r in all_runs if r["variant"] == variant]
        if not variant_runs:
            continue
        all_epochs_thinned = []
        for run in variant_runs:
            all_epochs_thinned.append({
                "seed": run["seed"],
                "epochs": thin_epoch_data(run["epochs"]),
            })
        example = {
            "input": json.dumps({
                "variant": variant,
                "dataset": "amazon_video_games",
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_GNN_LAYERS,
                "epochs": EPOCHS,
                "seeds": SEEDS,
            }),
            "output": json.dumps({
                "aggregated": aggregated.get(variant, {}),
                "per_seed_epochs": all_epochs_thinned,
            }),
            "predict_standard": str(aggregated.get("standard", {}).get("mean_test_rmse", "")),
            "predict_prmp": str(aggregated.get("prmp", {}).get("mean_test_rmse", "")),
            "predict_no_subtract": str(aggregated.get("no_subtract", {}).get("mean_test_rmse", "")),
            "metadata_variant": variant,
            "metadata_seed": 0,
        }
        examples.append(example)

    # Mechanism diagnosis summary example
    summary_example = {
        "input": json.dumps({
            "analysis": "mechanism_diagnosis",
            "dataset_stats": data_meta,
        }),
        "output": json.dumps({
            "aggregated_results": aggregated,
            "mechanism_diagnosis": mechanism_diagnosis,
            "conclusion": conclusion,
        }),
        "predict_standard": str(aggregated.get("standard", {}).get("mean_test_rmse", "")),
        "predict_prmp": str(aggregated.get("prmp", {}).get("mean_test_rmse", "")),
        "predict_no_subtract": str(aggregated.get("no_subtract", {}).get("mean_test_rmse", "")),
        "metadata_variant": "summary",
        "metadata_seed": 0,
    }
    examples.append(summary_example)

    logger.info(f"Generated {len(examples)} examples ({len(examples) - 4} review-level + 4 summaries)")

    output = {
        "metadata": {
            "method_name": "PRMP Mechanism Analysis (Instrumented)",
            "description": "Per-epoch instrumented training measuring information-theoretic mechanism of PRMP",
            "dataset": "amazon_video_games",
            "variants": VARIANTS,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "hidden_dim": HIDDEN_DIM,
        },
        "datasets": [{
            "dataset": "amazon_video_games_instrumented",
            "examples": examples,
        }],
    }

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output: {out_path} ({size_mb:.1f} MB)")

    total_time = time.time() - start_time
    logger.info(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
