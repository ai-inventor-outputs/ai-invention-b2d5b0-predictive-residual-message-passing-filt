#!/usr/bin/env python3
"""Embedding-Space Cross-Table Predictability Trajectories:
Standard vs PRMP HeteroGNN on Amazon Video Games Reviews.

Trains Standard HeteroSAGE and PRMP HeteroSAGE GNNs (pure PyTorch, no PyG needed).
Every CHECKPOINT_EVERY epochs, extracts per-layer node embeddings and computes
Ridge R², Random Forest R², and Mutual Information between parent and child
embeddings for each FK link. Compares trajectories to test whether:
  (1) embedding-space R² emerges during training (unlike raw-feature R² ≈ 0),
  (2) PRMP produces lower embedding-space R² (confirming it filters predictable info),
  (3) the predictability gap explains why raw-feature diagnostics failed.
"""

import json
import sys
import os
import time
import math
import gc
import resource
from pathlib import Path
from collections import defaultdict

# Add torch/sklearn libs from /tmp if needed
if os.path.isdir("/tmp/torchlibs"):
    sys.path.insert(0, "/tmp/torchlibs")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from loguru import logger
import psutil

# ============================================================
# Logging
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# Hardware Detection (cgroup-aware)
# ============================================================
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
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# Memory limits — dataset is small (~20MB), models are tiny
RAM_BUDGET = int(12 * 1024**3)  # 12 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(6 * 1024**3)  # 6 GB (graph is tiny)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# ============================================================
# Configuration
# ============================================================
FEATURE_NAMES = [
    "time_year", "time_month", "time_dayofweek",
    "summary_h0", "summary_h1", "summary_h2", "summary_h3", "summary_h4",
    "summary_h5", "summary_h6", "summary_h7", "summary_h8", "summary_h9",
    "summary_h10", "summary_h11", "summary_h12", "summary_h13", "summary_h14",
    "summary_h15", "helpful_up", "helpful_total",
]

HIDDEN_DIM = 64
LR = 0.005
WEIGHT_DECAY = 1e-4
CHECKPOINT_EVERY = 10
R2_SUBSAMPLE = 5000

# Adapt epochs based on GPU
if HAS_GPU:
    NUM_EPOCHS = 100
else:
    NUM_EPOCHS = 60
    HIDDEN_DIM = 32

DATA_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_1/gen_art/data_id5_it1__opus/full_data_out.json"
)
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_4/gen_art/exp_id2_it4__opus"
)


# ============================================================
# Scatter operations (pure PyTorch — no PyG dependency)
# ============================================================
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-aggregate src vectors by index. src: [E, D], index: [E] -> [dim_size, D]."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx_exp = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_exp, src)

    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1),
                       torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype))
    count = count.clamp(min=1)
    return out / count


# ============================================================
# Model components
# ============================================================
class ManualSAGEConv(nn.Module):
    """GraphSAGE-style convolution using pure PyTorch scatter ops."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        messages = x_src[src_idx]  # [E, D]
        agg = scatter_mean(messages, dst_idx, x_dst.size(0))  # [N_dst, D]
        return self.lin_neigh(agg) + self.lin_self(x_dst)


class PRMPConv(nn.Module):
    """Predictive Residual Message Passing convolution.
    For child→parent edges: predict child embedding from parent, aggregate residuals.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
        )
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]        # child features per edge [E, D]
        x_dst_i = x_dst[dst_idx]    # parent features per edge [E, D]

        pred = self.pred_mlp(x_dst_i)   # parent predicts child [E, D]
        residual = x_j - pred           # surprise = actual - predicted
        messages = self.lin(residual)    # transform residual [E, D_out]
        return scatter_mean(messages, dst_idx, x_dst.size(0))


class HeteroConvLayer(nn.Module):
    """Manual heterogeneous convolution layer — aggregates messages from
    multiple edge types that update each node type."""

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
            out = conv(x_dict[src_type], x_dict[dst_type], edge_index_dict[key])
            updates[dst_type].append(out)

        result = {}
        for nt in x_dict:
            if updates[nt]:
                result[nt] = sum(updates[nt])
            else:
                result[nt] = x_dict[nt]
        return result


class HeteroSAGE(nn.Module):
    """Heterogeneous GraphSAGE for review rating prediction.
    With use_prmp=True, child→parent edges use PRMPConv instead of SAGEConv.
    """

    def __init__(self, product_in: int = 5, customer_in: int = 5,
                 review_in: int = 21, hidden: int = 64, use_prmp: bool = False):
        super().__init__()
        self.proj_product = nn.Linear(product_in, hidden)
        self.proj_customer = nn.Linear(customer_in, hidden)
        self.proj_review = nn.Linear(review_in, hidden)

        self.conv1 = self._make_layer(hidden, use_prmp)
        self.conv2 = self._make_layer(hidden, use_prmp)

        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.cached_embeddings: dict = {}

    @staticmethod
    def _make_layer(hidden: int, use_prmp: bool) -> HeteroConvLayer:
        ChildConv = PRMPConv if use_prmp else ManualSAGEConv
        return HeteroConvLayer({
            ("review", "of", "product"): ChildConv(hidden, hidden),        # child→parent
            ("product", "has", "review"): ManualSAGEConv(hidden, hidden),  # parent→child
            ("review", "by", "customer"): ChildConv(hidden, hidden),       # child→parent
            ("customer", "wrote", "review"): ManualSAGEConv(hidden, hidden),  # parent→child
        })

    def forward(self, x_dict: dict, edge_index_dict: dict,
                cache: bool = False) -> torch.Tensor:
        # Layer 0: project inputs
        h = {
            "product": F.relu(self.proj_product(x_dict["product"])),
            "customer": F.relu(self.proj_customer(x_dict["customer"])),
            "review": F.relu(self.proj_review(x_dict["review"])),
        }
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

        return self.head(h["review"]).squeeze(-1)


# ============================================================
# R² / MI computation
# ============================================================
def compute_embedding_predictability(
    parent_emb: np.ndarray,
    child_emb: np.ndarray,
    n_subsample: int = 5000,
    num_cpus: int = 1,
) -> dict:
    """Compute Ridge R², RF R², and MI between aligned parent/child embeddings."""
    if len(parent_emb) > n_subsample:
        idx = np.random.choice(len(parent_emb), n_subsample, replace=False)
        parent_emb, child_emb = parent_emb[idx], child_emb[idx]

    # Ensure no NaN/Inf
    if np.any(~np.isfinite(parent_emb)) or np.any(~np.isfinite(child_emb)):
        logger.warning("Non-finite values in embeddings, replacing with 0")
        parent_emb = np.nan_to_num(parent_emb, nan=0.0, posinf=0.0, neginf=0.0)
        child_emb = np.nan_to_num(child_emb, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA reduce child to manageable dims
    n_child_components = min(10, child_emb.shape[1])
    if child_emb.shape[1] > 10:
        pca = PCA(n_components=n_child_components)
        child_reduced = pca.fit_transform(child_emb)
    else:
        child_reduced = child_emb

    # Multi-output Ridge R² (5-fold CV)
    ridge_r2 = 0.0
    try:
        ridge = MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ridge_scores = cross_val_score(
            ridge, parent_emb, child_reduced, cv=5, scoring="r2"
        )
        ridge_r2 = float(np.mean(ridge_scores))
    except Exception as e:
        logger.warning(f"Ridge R² failed: {e}")

    # Random Forest R² (3-fold CV for speed)
    rf_r2 = 0.0
    try:
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=8, n_jobs=min(num_cpus, 4)
        )
        rf_mo = MultiOutputRegressor(rf)
        rf_scores = cross_val_score(
            rf_mo, parent_emb, child_reduced, cv=3, scoring="r2"
        )
        rf_r2 = float(np.mean(rf_scores))
    except Exception as e:
        logger.warning(f"RF R² failed: {e}")

    # Mutual information (average across reduced child dims)
    mi = 0.0
    try:
        mi_vals = []
        for d in range(child_reduced.shape[1]):
            mi_vals.append(
                float(np.mean(
                    mutual_info_regression(parent_emb, child_reduced[:, d], n_neighbors=5)
                ))
            )
        mi = float(np.mean(mi_vals))
    except Exception as e:
        logger.warning(f"MI failed: {e}")

    return {"ridge_r2": ridge_r2, "rf_r2": rf_r2, "mi": mi}


def extract_and_measure(
    model: HeteroSAGE,
    x_dict: dict,
    edge_index_dict: dict,
    fk_edge_map: dict,
    n_subsample: int = 5000,
    num_cpus: int = 1,
) -> dict:
    """Run instrumented forward pass, compute R² for each FK link at each layer."""
    model.eval()
    with torch.no_grad():
        _ = model(x_dict, edge_index_dict, cache=True)

    results = {}
    for fk_name, (child_type, parent_type, edge_key) in fk_edge_map.items():
        edge_idx = edge_index_dict[edge_key]
        child_indices = edge_idx[0].cpu().numpy()
        parent_indices = edge_idx[1].cpu().numpy()

        results[fk_name] = {}
        for layer_name in ["layer_0", "layer_1", "layer_2"]:
            parent_emb = model.cached_embeddings[layer_name][parent_type].numpy()
            child_emb = model.cached_embeddings[layer_name][child_type].numpy()

            aligned_parent = parent_emb[parent_indices]
            aligned_child = child_emb[child_indices]

            metrics = compute_embedding_predictability(
                aligned_parent, aligned_child,
                n_subsample=n_subsample, num_cpus=num_cpus,
            )
            results[fk_name][layer_name] = metrics

    model.train()
    return results


# ============================================================
# Data loading & graph construction
# ============================================================
def load_and_build_graph(data_path: Path, max_examples: int | None = None) -> dict:
    """Load data and construct heterogeneous graph tensors."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())

    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]
    N = len(examples)
    logger.info(f"Using {N} examples")

    fk_links = raw["metadata"]["datasets_info"]["amazon_video_games"]["fk_links"]

    # Parse examples
    product_ids = [ex["metadata_product_id"] for ex in examples]
    customer_ids = [ex["metadata_customer_id"] for ex in examples]
    features_list = [json.loads(ex["input"]) for ex in examples]
    ratings = [float(ex["output"]) for ex in examples]
    folds = [ex["metadata_fold"] for ex in examples]

    # ID → index mappings
    unique_products = sorted(set(product_ids))
    unique_customers = sorted(set(customer_ids))
    prod_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    cust_to_idx = {cid: i for i, cid in enumerate(unique_customers)}

    N_prod = len(unique_products)
    N_cust = len(unique_customers)
    logger.info(f"Nodes: {N} reviews, {N_prod} products, {N_cust} customers")

    # Review feature tensor [N, 21]
    review_feat = torch.tensor(
        [[features_list[i][fn] for fn in FEATURE_NAMES] for i in range(N)],
        dtype=torch.float32,
    )

    # Aggregated product features [N_prod, 5]
    prod_rating_lists: dict[str, list] = defaultdict(list)
    prod_helpful_up: dict[str, list] = defaultdict(list)
    prod_helpful_total: dict[str, list] = defaultdict(list)
    for i in range(N):
        pid = product_ids[i]
        prod_rating_lists[pid].append(ratings[i])
        prod_helpful_up[pid].append(features_list[i]["helpful_up"])
        prod_helpful_total[pid].append(features_list[i]["helpful_total"])

    product_feat = torch.zeros(N_prod, 5, dtype=torch.float32)
    for pid, idx in prod_to_idx.items():
        r = np.array(prod_rating_lists.get(pid, [0.0]))
        product_feat[idx, 0] = float(np.mean(r))
        product_feat[idx, 1] = float(np.std(r)) if len(r) > 1 else 0.0
        product_feat[idx, 2] = float(len(r))
        hu = prod_helpful_up.get(pid, [0.0])
        ht = prod_helpful_total.get(pid, [0.0])
        product_feat[idx, 3] = float(np.mean(hu))
        product_feat[idx, 4] = float(np.mean(ht))

    # Aggregated customer features [N_cust, 5]
    cust_rating_lists: dict[str, list] = defaultdict(list)
    cust_helpful_up: dict[str, list] = defaultdict(list)
    cust_helpful_total: dict[str, list] = defaultdict(list)
    for i in range(N):
        cid = customer_ids[i]
        cust_rating_lists[cid].append(ratings[i])
        cust_helpful_up[cid].append(features_list[i]["helpful_up"])
        cust_helpful_total[cid].append(features_list[i]["helpful_total"])

    customer_feat = torch.zeros(N_cust, 5, dtype=torch.float32)
    for cid, idx in cust_to_idx.items():
        r = np.array(cust_rating_lists.get(cid, [0.0]))
        customer_feat[idx, 0] = float(np.mean(r))
        customer_feat[idx, 1] = float(np.std(r)) if len(r) > 1 else 0.0
        customer_feat[idx, 2] = float(len(r))
        hu = cust_helpful_up.get(cid, [0.0])
        ht = cust_helpful_total.get(cid, [0.0])
        customer_feat[idx, 3] = float(np.mean(hu))
        customer_feat[idx, 4] = float(np.mean(ht))

    # Z-score normalize product & customer features (per column)
    for feat_tensor in [product_feat, customer_feat]:
        for col in range(feat_tensor.size(1)):
            col_data = feat_tensor[:, col]
            mu = col_data.mean()
            std = col_data.std()
            if std > 1e-8:
                feat_tensor[:, col] = (col_data - mu) / std
            else:
                feat_tensor[:, col] = 0.0

    # Edge indices — bidirectional
    review_indices = torch.arange(N, dtype=torch.long)
    prod_indices = torch.tensor(
        [prod_to_idx[pid] for pid in product_ids], dtype=torch.long
    )
    cust_indices = torch.tensor(
        [cust_to_idx[cid] for cid in customer_ids], dtype=torch.long
    )

    edge_index_dict = {
        ("review", "of", "product"): torch.stack([review_indices, prod_indices]),
        ("product", "has", "review"): torch.stack([prod_indices, review_indices]),
        ("review", "by", "customer"): torch.stack([review_indices, cust_indices]),
        ("customer", "wrote", "review"): torch.stack([cust_indices, review_indices]),
    }

    x_dict = {
        "product": product_feat,
        "customer": customer_feat,
        "review": review_feat,
    }

    rating_tensor = torch.tensor(ratings, dtype=torch.float32)
    fold_tensor = torch.tensor(folds, dtype=torch.long)

    # Split: folds 0,1,2 = train, 3 = val, 4 = test
    train_mask = fold_tensor <= 2
    val_mask = fold_tensor == 3
    test_mask = fold_tensor == 4

    logger.info(f"Split: train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
                f"test={int(test_mask.sum())}")
    logger.info(f"Edges: {N} per type, {4 * N} total")

    # Sanity checks
    assert review_feat.shape == (N, 21), f"review_feat shape {review_feat.shape}"
    assert product_feat.shape == (N_prod, 5), f"product_feat shape {product_feat.shape}"
    assert customer_feat.shape == (N_cust, 5), f"customer_feat shape {customer_feat.shape}"
    assert not torch.isnan(review_feat).any(), "NaN in review features"
    assert not torch.isnan(product_feat).any(), "NaN in product features"
    assert not torch.isnan(customer_feat).any(), "NaN in customer features"
    assert (prod_indices >= 0).all() and (prod_indices < N_prod).all()
    assert (cust_indices >= 0).all() and (cust_indices < N_cust).all()

    return {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "rating_tensor": rating_tensor,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "examples": examples,
        "fk_links": fk_links,
        "N": N,
        "N_prod": N_prod,
        "N_cust": N_cust,
    }


# ============================================================
# Training
# ============================================================
def train_model(
    model: HeteroSAGE,
    model_name: str,
    x_dict_dev: dict,
    edge_dict_dev: dict,
    y: torch.Tensor,
    train_mask_dev: torch.Tensor,
    val_mask_dev: torch.Tensor,
    fk_edge_map: dict,
    use_prmp: bool = False,
) -> dict:
    """Train a model and collect R² trajectories."""
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    r2_timeseries = {
        fk: {
            layer: {"epochs": [], "ridge_r2": [], "rf_r2": [], "mi": []}
            for layer in ["layer_0", "layer_1", "layer_2"]
        }
        for fk in fk_edge_map
    }

    for epoch in range(NUM_EPOCHS):
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict_dev, edge_dict_dev, cache=False)
        loss = loss_fn(pred[train_mask_dev], y[train_mask_dev])

        # Check for NaN loss
        if torch.isnan(loss):
            logger.error(f"[{model_name}] NaN loss at epoch {epoch}, stopping")
            break

        loss.backward()

        # Gradient clipping for PRMP stability
        if use_prmp:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_losses.append(float(loss.item()))

        # --- Val step ---
        model.eval()
        with torch.no_grad():
            val_pred = model(x_dict_dev, edge_dict_dev, cache=False)
            val_loss = loss_fn(val_pred[val_mask_dev], y[val_mask_dev])
        val_losses.append(float(val_loss.item()))

        # --- R² instrumentation at checkpoints ---
        is_checkpoint = (epoch % CHECKPOINT_EVERY == 0) or (epoch == NUM_EPOCHS - 1)
        if is_checkpoint:
            logger.info(
                f"[{model_name}] Epoch {epoch}: "
                f"train_loss={loss:.4f}, val_loss={val_loss:.4f}"
            )
            r2_results = extract_and_measure(
                model, x_dict_dev, edge_dict_dev, fk_edge_map,
                n_subsample=R2_SUBSAMPLE, num_cpus=NUM_CPUS,
            )
            for fk_name, layers in r2_results.items():
                for layer_name, metrics in layers.items():
                    ts = r2_timeseries[fk_name][layer_name]
                    ts["epochs"].append(epoch)
                    ts["ridge_r2"].append(metrics["ridge_r2"])
                    ts["rf_r2"].append(metrics["rf_r2"])
                    ts["mi"].append(metrics["mi"])
                    logger.info(
                        f"  {fk_name}/{layer_name}: "
                        f"ridge_r2={metrics['ridge_r2']:.4f}, "
                        f"rf_r2={metrics['rf_r2']:.4f}, "
                        f"mi={metrics['mi']:.4f}"
                    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "r2_timeseries": r2_timeseries,
    }


# ============================================================
# Main
# ============================================================
@logger.catch
def main():
    start_time = time.time()

    # Load and build graph
    graph_data = load_and_build_graph(DATA_PATH)

    x_dict = graph_data["x_dict"]
    edge_index_dict = graph_data["edge_index_dict"]
    rating_tensor = graph_data["rating_tensor"]
    train_mask = graph_data["train_mask"]
    val_mask = graph_data["val_mask"]
    test_mask = graph_data["test_mask"]
    examples = graph_data["examples"]
    fk_links = graph_data["fk_links"]
    N = graph_data["N"]
    N_prod = graph_data["N_prod"]
    N_cust = graph_data["N_cust"]

    # ============================================================
    # STEP 3: Raw-feature R² baseline
    # ============================================================
    logger.info("=" * 60)
    logger.info("Computing raw-feature R² baseline")
    logger.info("=" * 60)
    raw_baseline = {}
    for fk_name in ["product_to_review", "customer_to_review"]:
        fk = fk_links[fk_name]
        parent_mat = np.array(fk["aligned_parent_features_sample"])
        child_mat = np.array(fk["aligned_child_features_sample"])
        logger.info(f"Raw {fk_name}: parent={parent_mat.shape}, child={child_mat.shape}")

        metrics = compute_embedding_predictability(
            parent_mat, child_mat, n_subsample=2000, num_cpus=NUM_CPUS
        )
        raw_baseline[fk_name] = metrics
        logger.info(
            f"Raw {fk_name}: ridge_r2={metrics['ridge_r2']:.4f}, "
            f"rf_r2={metrics['rf_r2']:.4f}, mi={metrics['mi']:.4f}"
        )

    # ============================================================
    # STEP 5: Train both models
    # ============================================================
    fk_edge_map = {
        "product_to_review": ("review", "product", ("review", "of", "product")),
        "customer_to_review": ("review", "customer", ("review", "by", "customer")),
    }

    # Move to device
    x_dict_dev = {k: v.to(DEVICE) for k, v in x_dict.items()}
    edge_dict_dev = {k: v.to(DEVICE) for k, v in edge_index_dict.items()}
    y = rating_tensor.to(DEVICE)
    train_mask_dev = train_mask.to(DEVICE)
    val_mask_dev = val_mask.to(DEVICE)
    test_mask_dev = test_mask.to(DEVICE)

    all_results = {}
    all_predictions = {}

    for model_name, use_prmp in [("standard", False), ("prmp", True)]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_name} HeteroSAGE ({'PRMP' if use_prmp else 'Standard'})")
        logger.info(f"{'=' * 60}")

        model = HeteroSAGE(
            product_in=5, customer_in=5, review_in=21,
            hidden=HIDDEN_DIM, use_prmp=use_prmp,
        ).to(DEVICE)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {param_count:,}")

        train_result = train_model(
            model, model_name,
            x_dict_dev, edge_dict_dev, y,
            train_mask_dev, val_mask_dev,
            fk_edge_map, use_prmp=use_prmp,
        )

        # Final predictions
        model.eval()
        with torch.no_grad():
            final_pred = model(x_dict_dev, edge_dict_dev, cache=False).cpu().numpy()

        # Compute MAE on val and test
        val_mae = float(mean_absolute_error(
            rating_tensor[val_mask].numpy(),
            final_pred[val_mask.numpy()],
        ))
        test_mae = float(mean_absolute_error(
            rating_tensor[test_mask].numpy(),
            final_pred[test_mask.numpy()],
        ))

        all_predictions[model_name] = final_pred
        all_results[model_name] = {
            **train_result,
            "final_val_loss": train_result["val_losses"][-1] if train_result["val_losses"] else float("nan"),
            "final_val_mae": val_mae,
            "final_test_mae": test_mae,
        }

        logger.info(
            f"[{model_name}] Final: val_loss={all_results[model_name]['final_val_loss']:.4f}, "
            f"val_mae={val_mae:.4f}, test_mae={test_mae:.4f}"
        )

        # Free GPU cache between models
        del model
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    # ============================================================
    # STEP 6: Analysis
    # ============================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Analysis & Output")
    logger.info(f"{'=' * 60}")

    # Predictability gap at convergence
    predictability_gap = {}
    for fk_name in fk_edge_map:
        raw_r2 = raw_baseline[fk_name]["ridge_r2"]
        for mn in ["standard", "prmp"]:
            ts = all_results[mn]["r2_timeseries"][fk_name]["layer_2"]
            emb_r2 = ts["ridge_r2"][-1] if ts["ridge_r2"] else 0.0
            gap = emb_r2 - raw_r2
            predictability_gap[f"{mn}_{fk_name}"] = {
                "raw_r2": raw_r2,
                "embedding_r2": emb_r2,
                "gap": gap,
            }
            logger.info(
                f"Gap {mn}/{fk_name}: raw_r2={raw_r2:.4f}, "
                f"emb_r2={emb_r2:.4f}, gap={gap:.4f}"
            )

    # Key findings
    findings = {}
    for fk_name in fk_edge_map:
        # Does embedding R² increase during training?
        std_ts = all_results["standard"]["r2_timeseries"][fk_name]["layer_2"]["ridge_r2"]
        if len(std_ts) >= 2:
            findings[f"embedding_r2_increases_{fk_name}"] = bool(std_ts[-1] > std_ts[0])
        else:
            findings[f"embedding_r2_increases_{fk_name}"] = False

        # Does PRMP have lower embedding R² than standard?
        prmp_ts = all_results["prmp"]["r2_timeseries"][fk_name]["layer_2"]["ridge_r2"]
        if std_ts and prmp_ts:
            findings[f"prmp_lower_r2_{fk_name}"] = bool(prmp_ts[-1] < std_ts[-1])
        else:
            findings[f"prmp_lower_r2_{fk_name}"] = False

    findings["prmp_improves_task_performance"] = bool(
        all_results["prmp"]["final_val_loss"] < all_results["standard"]["final_val_loss"]
    )
    findings["predictability_gap_is_large"] = bool(
        any(v["gap"] > 0.05 for v in predictability_gap.values())
    )

    for k, v in findings.items():
        logger.info(f"Finding: {k} = {v}")

    # ============================================================
    # Build output in exp_gen_sol_out schema
    # ============================================================
    output_examples = []
    for i, ex in enumerate(examples):
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "predict_standard": str(round(float(all_predictions["standard"][i]), 4)),
            "predict_prmp": str(round(float(all_predictions["prmp"][i]), 4)),
            "metadata_fold": ex["metadata_fold"],
            "metadata_row_index": ex["metadata_row_index"],
            "metadata_task_type": ex["metadata_task_type"],
            "metadata_feature_names": ex["metadata_feature_names"],
            "metadata_product_id": ex["metadata_product_id"],
            "metadata_customer_id": ex["metadata_customer_id"],
        }
        output_examples.append(out_ex)

    output = {
        "metadata": {
            "description": (
                "Embedding-space cross-table predictability trajectories comparing "
                "Standard vs PRMP HeteroGNN on Amazon Video Games reviews. "
                "Tests whether embedding-space R² emerges during training, "
                "whether PRMP produces lower R², and whether the gap explains "
                "why raw-feature diagnostics failed."
            ),
            "dataset": "amazon_video_games",
            "num_reviews": N,
            "num_products": N_prod,
            "num_customers": N_cust,
            "model_config": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": 2,
                "epochs": NUM_EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "checkpoint_every": CHECKPOINT_EVERY,
            },
            "gpu_used": HAS_GPU,
            "device": str(DEVICE),
            "raw_feature_baseline": raw_baseline,
            "training_curves": {
                mn: {
                    "train_losses": all_results[mn]["train_losses"],
                    "val_losses": all_results[mn]["val_losses"],
                }
                for mn in ["standard", "prmp"]
            },
            "embedding_predictability": {
                mn: all_results[mn]["r2_timeseries"]
                for mn in ["standard", "prmp"]
            },
            "predictability_gap": predictability_gap,
            "task_performance": {
                mn: {
                    "final_val_loss": all_results[mn]["final_val_loss"],
                    "final_val_mae": all_results[mn]["final_val_mae"],
                    "final_test_mae": all_results[mn]["final_test_mae"],
                }
                for mn in ["standard", "prmp"]
            },
            "key_findings": findings,
            "runtime_seconds": time.time() - start_time,
        },
        "datasets": [
            {
                "dataset": "amazon_video_games_reviews",
                "examples": output_examples,
            }
        ],
    }

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    file_size_kb = out_path.stat().st_size / 1024
    logger.info(f"Output saved to {out_path} ({file_size_kb:.1f} KB)")
    logger.info(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
