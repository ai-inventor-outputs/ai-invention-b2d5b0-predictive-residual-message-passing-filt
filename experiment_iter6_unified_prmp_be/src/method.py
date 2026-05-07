#!/usr/bin/env python3
"""Unified PRMP Benchmark: 5 Variants x 5 Seeds x 3 Tasks on Amazon & F1.

Tests the Predictive Residual Message Passing (PRMP) hypothesis:
In heterogeneous GNNs over FK-linked tables, predicting child features from
parent features and passing RESIDUALS instead of raw features improves
learning, especially for predictable FK links.

5 model variants:
  1. Standard HeteroSAGEConv (baseline)
  2. PRMP (predict-subtract-aggregate)
  3. Wide (parameter-matched standard)
  4. Auxiliary MLP (extra capacity, no subtraction)
  5. Random Frozen (random predictions, tests if any subtraction helps)

3 tasks:
  1. rel-amazon/review-rating  (regression, MAE)
  2. rel-f1/result-position    (regression, MAE)
  3. rel-f1/result-dnf         (classification, Average Precision)
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from torch_scatter import scatter_mean
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 0: LOGGING & HARDWARE DETECTION
# ============================================================
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
(WORKSPACE / "checkpoints").mkdir(exist_ok=True)
(WORKSPACE / "cache").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")


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
TOTAL_RAM_GB = _container_ram_gb() or 42.0

# Set memory limits
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)  # 70% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.85)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f}GB")
else:
    VRAM_GB = 0
logger.info(f"CPUs: {NUM_CPUS}, RAM: {TOTAL_RAM_GB:.1f}GB, GPU: {HAS_GPU}")

# ============================================================
# PHASE 1: CONFIGURATION
# ============================================================
CONFIG = {
    "hidden_dim": 128,
    "num_gnn_layers": 2,
    "lr": 0.001,
    "max_epochs": 200,
    "patience": 20,
    "seeds": [42, 123, 456, 789, 1024],
    "gradient_clip": 1.0,
    "weight_decay": 1e-5,
    "prediction_mlp_hidden": 64,
}

TASKS = [
    {
        "dataset": "rel-amazon",
        "task": "review-rating",
        "type": "regression",
        "primary_metric": "mae",
        "lower_better": True,
        "loss": "mse",
    },
    {
        "dataset": "rel-f1",
        "task": "result-position",
        "type": "regression",
        "primary_metric": "mae",
        "lower_better": True,
        "loss": "mse",
    },
    {
        "dataset": "rel-f1",
        "task": "result-dnf",
        "type": "classification",
        "primary_metric": "average_precision",
        "lower_better": False,
        "loss": "bce",
    },
]

VARIANTS = ["standard", "prmp", "wide", "auxiliary_mlp", "random_frozen"]

# Dependency data paths
DEP_AMAZON = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
DEP_F1 = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_2/gen_art/data_id5_it2__opus"
)


# ============================================================
# PHASE 2: HETEROGENEOUS GRAPH DATA STRUCTURE
# ============================================================
class HeteroGraphData:
    """Lightweight heterogeneous graph container."""

    def __init__(self):
        self.node_features: dict[str, torch.Tensor] = {}
        self.edge_index: dict[tuple, torch.Tensor] = {}
        self.target: torch.Tensor | None = None
        self.target_node_type: str = ""
        self.train_mask: torch.Tensor | None = None
        self.val_mask: torch.Tensor | None = None
        self.test_mask: torch.Tensor | None = None
        self.num_nodes: dict[str, int] = {}
        self.fk_edges: list[tuple] = []
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def to(self, device: torch.device) -> "HeteroGraphData":
        g = HeteroGraphData()
        g.node_features = {k: v.to(device) for k, v in self.node_features.items()}
        g.edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        if self.target is not None:
            g.target = self.target.to(device)
        if self.train_mask is not None:
            g.train_mask = self.train_mask.to(device)
        if self.val_mask is not None:
            g.val_mask = self.val_mask.to(device)
        if self.test_mask is not None:
            g.test_mask = self.test_mask.to(device)
        g.num_nodes = dict(self.num_nodes)
        g.target_node_type = self.target_node_type
        g.fk_edges = list(self.fk_edges)
        g.target_mean = self.target_mean
        g.target_std = self.target_std
        return g

    @property
    def node_types(self) -> list[str]:
        return list(self.node_features.keys())

    @property
    def edge_types(self) -> list[tuple]:
        return list(self.edge_index.keys())

    @property
    def in_channels_dict(self) -> dict[str, int]:
        return {k: v.shape[1] for k, v in self.node_features.items()}


# ============================================================
# PHASE 3: DATA LOADING & GRAPH CONSTRUCTION
# ============================================================
def load_amazon_graph() -> HeteroGraphData:
    """Build heterogeneous graph from Amazon dependency data."""
    logger.info("Loading Amazon review data...")
    data_path = DEP_AMAZON / "full_data_out.json"
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} Amazon review examples")

    # Parse examples
    feature_keys = sorted(json.loads(examples[0]["input"]).keys())
    review_features_list = []
    ratings = []
    product_ids = []
    customer_ids = []
    folds = []

    for ex in examples:
        inp = json.loads(ex["input"])
        review_features_list.append([float(inp[k]) for k in feature_keys])
        ratings.append(float(ex["output"]))
        product_ids.append(str(ex["metadata_product_id"]))
        customer_ids.append(str(ex["metadata_customer_id"]))
        folds.append(int(ex["metadata_fold"]))

    # Node index mappings
    unique_products = sorted(set(product_ids))
    unique_customers = sorted(set(customer_ids))
    prod_to_idx = {p: i for i, p in enumerate(unique_products)}
    cust_to_idx = {c: i for i, c in enumerate(unique_customers)}

    n_reviews = len(examples)
    n_products = len(unique_products)
    n_customers = len(unique_customers)
    feat_dim = len(feature_keys)
    logger.info(
        f"  Nodes: {n_reviews} reviews, {n_products} products, "
        f"{n_customers} customers, feat_dim={feat_dim}"
    )

    # Review features tensor
    review_features = torch.tensor(review_features_list, dtype=torch.float32)

    # Aggregate review features to create product/customer features
    # Use only training data (folds 0,1,2) to avoid leakage
    train_mask_np = np.isin(folds, [0, 1, 2])
    product_feat_acc = defaultdict(lambda: np.zeros(feat_dim, dtype=np.float64))
    product_feat_cnt = defaultdict(int)
    customer_feat_acc = defaultdict(lambda: np.zeros(feat_dim, dtype=np.float64))
    customer_feat_cnt = defaultdict(int)

    for i in range(n_reviews):
        if train_mask_np[i]:
            pid = product_ids[i]
            cid = customer_ids[i]
            feats = review_features_list[i]
            product_feat_acc[pid] += feats
            product_feat_cnt[pid] += 1
            customer_feat_acc[cid] += feats
            customer_feat_cnt[cid] += 1

    # Build product feature matrix
    product_features = torch.zeros(n_products, feat_dim, dtype=torch.float32)
    for pid, idx in prod_to_idx.items():
        if product_feat_cnt[pid] > 0:
            product_features[idx] = torch.tensor(
                product_feat_acc[pid] / product_feat_cnt[pid], dtype=torch.float32
            )

    # Build customer feature matrix
    customer_features = torch.zeros(n_customers, feat_dim, dtype=torch.float32)
    for cid, idx in cust_to_idx.items():
        if customer_feat_cnt[cid] > 0:
            customer_features[idx] = torch.tensor(
                customer_feat_acc[cid] / customer_feat_cnt[cid], dtype=torch.float32
            )

    # Build edge indices: FK direction is child -> parent
    # review -> product, review -> customer
    rev_idx = torch.arange(n_reviews, dtype=torch.long)
    prod_dst = torch.tensor([prod_to_idx[p] for p in product_ids], dtype=torch.long)
    cust_dst = torch.tensor([cust_to_idx[c] for c in customer_ids], dtype=torch.long)

    graph = HeteroGraphData()
    graph.node_features = {
        "review": review_features,
        "product": product_features,
        "customer": customer_features,
    }
    graph.num_nodes = {
        "review": n_reviews,
        "product": n_products,
        "customer": n_customers,
    }
    # FK edges: child(review) -> parent(product/customer)
    graph.edge_index = {
        ("review", "belongs_to", "product"): torch.stack([rev_idx, prod_dst]),
        ("review", "written_by", "customer"): torch.stack([rev_idx, cust_dst]),
        # Reverse: parent -> child
        ("product", "has_review", "review"): torch.stack([prod_dst, rev_idx]),
        ("customer", "wrote", "review"): torch.stack([cust_dst, rev_idx]),
    }
    graph.fk_edges = [
        ("review", "belongs_to", "product"),
        ("review", "written_by", "customer"),
    ]

    # Target: review rating (normalize)
    target = torch.tensor(ratings, dtype=torch.float32)
    graph.target_mean = target.mean().item()
    graph.target_std = target.std().item()
    graph.target = (target - graph.target_mean) / max(graph.target_std, 1e-6)
    graph.target_node_type = "review"

    # Normalize node features (per-type, zero mean unit variance)
    for ntype in graph.node_features:
        x = graph.node_features[ntype]
        std = x.std(dim=0).clamp(min=1e-6)
        mean = x.mean(dim=0)
        graph.node_features[ntype] = (x - mean) / std

    # Train/val/test split
    folds_arr = np.array(folds)
    graph.train_mask = torch.tensor(np.isin(folds_arr, [0, 1, 2]), dtype=torch.bool)
    graph.val_mask = torch.tensor(folds_arr == 3, dtype=torch.bool)
    graph.test_mask = torch.tensor(folds_arr == 4, dtype=torch.bool)
    logger.info(
        f"  Split: train={graph.train_mask.sum().item()}, "
        f"val={graph.val_mask.sum().item()}, test={graph.test_mask.sum().item()}"
    )
    return graph


def _extract_unique_parents(
    df: pd.DataFrame, parent_prefix: str = "parent__"
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract unique parent feature vectors and return (features, mapping, n_unique)."""
    parent_cols = [c for c in df.columns if c.startswith(parent_prefix)]
    raw = df[parent_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    # Find unique rows by hashing
    # Use a dict for deduplication
    unique_map = {}
    mapping = np.zeros(len(raw), dtype=np.int64)
    for i, row in enumerate(raw):
        key = row.tobytes()
        if key not in unique_map:
            unique_map[key] = len(unique_map)
        mapping[i] = unique_map[key]

    n_unique = len(unique_map)
    features = np.zeros((n_unique, len(parent_cols)), dtype=np.float32)
    seen = set()
    for i, row in enumerate(raw):
        idx = mapping[i]
        if idx not in seen:
            features[idx] = row
            seen.add(idx)

    return features, mapping, n_unique


def load_f1_graph(task_name: str) -> HeteroGraphData:
    """Build heterogeneous graph from F1 parquet features for a task."""
    logger.info(f"Loading F1 graph for task: {task_name}...")
    pq_dir = DEP_F1 / "parquet_features"

    # Load FK link parquets centered on 'results'
    # Only use parquets with same row count as primary (drivers=25420)
    df_rd = pd.read_parquet(pq_dir / "results__drivers__driverId_features.parquet")
    df_rc = pd.read_parquet(pq_dir / "results__constructors__constructorId_features.parquet")
    n_results = len(df_rd)
    logger.info(f"  Results rows: {n_results}")

    # Also try to load driver_standings for extra context
    ds_path = pq_dir / "driver_standings__drivers__driverId_features.parquet"
    df_ds = pd.read_parquet(ds_path) if ds_path.exists() else None
    if df_ds is not None:
        logger.info(f"  Driver standings rows: {len(df_ds)}")

    # ----- Determine input features and target -----
    child_cols_all = [c for c in df_rd.columns if c.startswith("child__")]

    if task_name == "result-position":
        target_col = "child__positionOrder"
        # Exclude leaky columns
        exclude = {
            "child__position", "child__positionText",
            "child__positionOrder", "child__points",
        }
        input_cols = [c for c in child_cols_all if c not in exclude]
        target_raw = pd.to_numeric(df_rd[target_col], errors="coerce").fillna(20.0)
        target_values = target_raw.values.astype(np.float32)

    elif task_name == "result-dnf":
        # DNF = position == -1 (encoded as -1 for did-not-finish)
        exclude = {"child__position", "child__positionText", "child__positionOrder"}
        input_cols = [c for c in child_cols_all if c not in exclude]
        positions = pd.to_numeric(df_rd["child__position"], errors="coerce").fillna(-1)
        target_values = (positions < 0).astype(np.float32).values
        logger.info(f"  DNF rate: {target_values.mean():.3f} ({int(target_values.sum())}/{len(target_values)})")
    else:
        raise ValueError(f"Unknown F1 task: {task_name}")

    # Result node features
    result_feats = (
        df_rd[input_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float32)
    )

    # Extract unique parent nodes for each FK link (drivers & constructors)
    drv_feats, drv_map, n_drivers = _extract_unique_parents(df_rd)
    con_feats, con_map, n_constructors = _extract_unique_parents(df_rc)

    logger.info(
        f"  Nodes: {n_results} results, {n_drivers} drivers, "
        f"{n_constructors} constructors"
    )

    # Build graph
    result_idx = np.arange(n_results)
    graph = HeteroGraphData()
    graph.node_features = {
        "result": torch.tensor(result_feats, dtype=torch.float32),
        "driver": torch.tensor(drv_feats, dtype=torch.float32),
        "constructor": torch.tensor(con_feats, dtype=torch.float32),
    }
    graph.num_nodes = {
        "result": n_results,
        "driver": n_drivers,
        "constructor": n_constructors,
    }

    # FK edges: result(child) -> parent
    graph.edge_index = {
        ("result", "by_driver", "driver"): torch.stack([
            torch.from_numpy(result_idx.astype(np.int64)),
            torch.from_numpy(drv_map),
        ]),
        ("result", "by_constructor", "constructor"): torch.stack([
            torch.from_numpy(result_idx.astype(np.int64)),
            torch.from_numpy(con_map),
        ]),
        # Reverse edges
        ("driver", "has_result", "result"): torch.stack([
            torch.from_numpy(drv_map),
            torch.from_numpy(result_idx.astype(np.int64)),
        ]),
        ("constructor", "has_result", "result"): torch.stack([
            torch.from_numpy(con_map),
            torch.from_numpy(result_idx.astype(np.int64)),
        ]),
    }
    graph.fk_edges = [
        ("result", "by_driver", "driver"),
        ("result", "by_constructor", "constructor"),
    ]

    # Target
    target_t = torch.tensor(target_values, dtype=torch.float32)
    if task_name == "result-position":
        graph.target_mean = float(target_t.mean())
        graph.target_std = float(target_t.std())
        graph.target = (target_t - graph.target_mean) / max(graph.target_std, 1e-6)
    else:
        graph.target_mean = 0.0
        graph.target_std = 1.0
        graph.target = target_t
    graph.target_node_type = "result"

    # Normalize features
    for ntype in graph.node_features:
        x = graph.node_features[ntype]
        std = x.std(dim=0).clamp(min=1e-6)
        mean = x.mean(dim=0)
        graph.node_features[ntype] = (x - mean) / std

    # Temporal split (80/10/10 by row order)
    train_end = int(n_results * 0.8)
    val_end = int(n_results * 0.9)
    graph.train_mask = torch.zeros(n_results, dtype=torch.bool)
    graph.val_mask = torch.zeros(n_results, dtype=torch.bool)
    graph.test_mask = torch.zeros(n_results, dtype=torch.bool)
    graph.train_mask[:train_end] = True
    graph.val_mask[train_end:val_end] = True
    graph.test_mask[val_end:] = True

    logger.info(
        f"  Split: train={graph.train_mask.sum().item()}, "
        f"val={graph.val_mask.sum().item()}, test={graph.test_mask.sum().item()}"
    )
    return graph


def load_task_data(task_config: dict) -> HeteroGraphData:
    """Load and cache graph data for a task."""
    cache_key = f"{task_config['dataset']}_{task_config['task']}"
    cache_path = WORKSPACE / "cache" / f"{cache_key}.pt"

    if cache_path.exists():
        logger.info(f"Loading cached graph: {cache_key}")
        return torch.load(cache_path, weights_only=False)

    if task_config["dataset"] == "rel-amazon":
        graph = load_amazon_graph()
    else:
        graph = load_f1_graph(task_config["task"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, cache_path)
    logger.info(f"Cached graph: {cache_key}")
    return graph


# ============================================================
# PHASE 4: MODEL ARCHITECTURES
# ============================================================
class TabularEncoder(nn.Module):
    """Per-node-type MLP: raw features -> hidden_dim embeddings."""

    def __init__(self, in_channels_dict: dict[str, int], hidden_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.encoders[ntype] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: self.encoders[nt](x) for nt, x in x_dict.items()}


class HeteroSAGEConvLayer(nn.Module):
    """Standard heterogeneous SAGEConv layer with scatter aggregation."""

    def __init__(
        self, hidden_dim: int, edge_types: list[tuple], node_types: list[str]
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-edge-type message transform
        self.msg_linears = nn.ModuleDict()
        for et in edge_types:
            key = "__".join(et)
            self.msg_linears[key] = nn.Linear(hidden_dim, hidden_dim)

        # Per-node-type self-loop + combine
        self.self_linears = nn.ModuleDict()
        self.combine_linears = nn.ModuleDict()
        for nt in node_types:
            self.self_linears[nt] = nn.Linear(hidden_dim, hidden_dim)
            self.combine_linears[nt] = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Aggregate messages per destination node type
        agg_dict: dict[str, list[torch.Tensor]] = {nt: [] for nt in x_dict}

        for et, ei in edge_index_dict.items():
            key = "__".join(et)
            if key not in self.msg_linears:
                continue
            src_type, _, dst_type = et
            src_x = x_dict[src_type]
            num_dst = x_dict[dst_type].shape[0]

            msg = self.msg_linears[key](src_x[ei[0]])
            agg = scatter_mean(msg, ei[1], dim=0, dim_size=num_dst)
            agg_dict[dst_type].append(agg)

        out = {}
        for nt, x in x_dict.items():
            self_out = self.self_linears[nt](x)
            if agg_dict[nt]:
                neigh = torch.stack(agg_dict[nt]).mean(dim=0)
                combined = self.combine_linears[nt](
                    torch.cat([self_out, neigh], dim=-1)
                )
                out[nt] = F.relu(combined)
            else:
                out[nt] = F.relu(self_out)
        return out


class PRMPConvLayer(nn.Module):
    """PRMP convolution: predict-subtract-aggregate on FK edges + standard on all.

    For FK edge (child -> parent):
      1. Standard message aggregation (same as HeteroSAGEConv)
      2. ADDITIONALLY: parent predicts child features via MLP
      3. Residual = actual_child - predicted_child
      4. Aggregate residuals to parent
      5. Both signals combined in the update

    The prediction MLPs are EXTRA capacity on top of standard message passing.
    This is what the wide/auxiliary_mlp controls test against.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_types: list[tuple],
        node_types: list[str],
        fk_edges: list[tuple],
        pred_mlp_hidden: int = 64,
        freeze_predictors: bool = False,
        no_subtraction: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fk_edge_set = {tuple(e) for e in fk_edges}
        self.no_subtraction = no_subtraction

        # Standard message components for ALL edges
        self.msg_linears = nn.ModuleDict()
        for et in edge_types:
            key = "__".join(et)
            self.msg_linears[key] = nn.Linear(hidden_dim, hidden_dim)

        # EXTRA prediction MLPs for FK edges only
        self.pred_mlps = nn.ModuleDict()
        for et in edge_types:
            if tuple(et) in self.fk_edge_set:
                key = "__".join(et)
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, pred_mlp_hidden),
                    nn.ReLU(),
                    nn.Linear(pred_mlp_hidden, hidden_dim),
                )
                if freeze_predictors:
                    for p in mlp.parameters():
                        p.requires_grad = False
                self.pred_mlps[key] = mlp

        # Per-node-type update
        self.self_linears = nn.ModuleDict()
        self.combine_linears = nn.ModuleDict()
        for nt in node_types:
            self.self_linears[nt] = nn.Linear(hidden_dim, hidden_dim)
            self.combine_linears[nt] = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        agg_dict: dict[str, list[torch.Tensor]] = {nt: [] for nt in x_dict}

        for et, ei in edge_index_dict.items():
            src_type, rel, dst_type = et
            key = "__".join(et)

            # Standard message aggregation for ALL edges
            if key in self.msg_linears:
                src_x = x_dict[src_type]
                num_dst = x_dict[dst_type].shape[0]
                msg = self.msg_linears[key](src_x[ei[0]])
                agg = scatter_mean(msg, ei[1], dim=0, dim_size=num_dst)
                agg_dict[dst_type].append(agg)

            # ADDITIONALLY for FK edges: predict-subtract mechanism
            if tuple(et) in self.fk_edge_set and key in self.pred_mlps:
                # FK edge: src=child, dst=parent
                child_x = x_dict[src_type]
                parent_x = x_dict[dst_type]
                num_parent = parent_x.shape[0]

                # Parent predicts child features (via edge mapping)
                predicted_child = self.pred_mlps[key](parent_x[ei[1]])
                actual_child = child_x[ei[0]]

                if self.no_subtraction:
                    # Auxiliary MLP: add prediction as extra signal
                    enriched = actual_child + predicted_child
                    prmp_agg = scatter_mean(
                        enriched, ei[1], dim=0, dim_size=num_parent
                    )
                else:
                    # PRMP: residual = actual - predicted
                    residuals = actual_child - predicted_child
                    prmp_agg = scatter_mean(
                        residuals, ei[1], dim=0, dim_size=num_parent
                    )

                agg_dict[dst_type].append(prmp_agg)

        out = {}
        for nt, x in x_dict.items():
            self_out = self.self_linears[nt](x)
            if agg_dict[nt]:
                neigh = torch.stack(agg_dict[nt]).mean(dim=0)
                combined = self.combine_linears[nt](
                    torch.cat([self_out, neigh], dim=-1)
                )
                out[nt] = F.relu(combined)
            else:
                out[nt] = F.relu(self_out)
        return out


class PredictionHead(nn.Module):
    """MLP prediction head: hidden_dim -> 1."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class HeteroGNN(nn.Module):
    """Complete heterogeneous GNN with configurable conv layers."""

    def __init__(
        self,
        in_channels_dict: dict[str, int],
        hidden_dim: int,
        num_layers: int,
        edge_types: list[tuple],
        node_types: list[str],
        variant: str = "standard",
        fk_edges: list[tuple] | None = None,
        pred_mlp_hidden: int = 64,
    ):
        super().__init__()
        self.variant = variant
        self.encoder = TabularEncoder(in_channels_dict, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if variant in ("standard", "wide"):
                layer = HeteroSAGEConvLayer(hidden_dim, edge_types, node_types)
            elif variant == "prmp":
                layer = PRMPConvLayer(
                    hidden_dim, edge_types, node_types,
                    fk_edges or [], pred_mlp_hidden,
                )
            elif variant == "auxiliary_mlp":
                layer = PRMPConvLayer(
                    hidden_dim, edge_types, node_types,
                    fk_edges or [], pred_mlp_hidden,
                    no_subtraction=True,
                )
            elif variant == "random_frozen":
                layer = PRMPConvLayer(
                    hidden_dim, edge_types, node_types,
                    fk_edges or [], pred_mlp_hidden,
                    freeze_predictors=True,
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")
            self.convs.append(layer)

        self.head = PredictionHead(hidden_dim)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(x_dict)
        for conv in self.convs:
            h = conv(h, edge_index_dict)
        return h

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        target_node_type: str,
    ) -> torch.Tensor:
        h = self.encode(x_dict, edge_index_dict)
        return self.head(h[target_node_type])


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _find_wide_dim(
    in_channels_dict: dict[str, int],
    edge_types: list[tuple],
    node_types: list[str],
    num_layers: int,
    target_params: int,
) -> int:
    """Binary search for hidden_dim that matches target param count."""
    lo, hi = 64, 512
    best_dim = 128
    best_diff = float("inf")

    while lo <= hi:
        mid = (lo + hi) // 2
        m = HeteroGNN(
            in_channels_dict, mid, num_layers,
            edge_types, node_types, "standard",
        )
        p = _count_params(m)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = mid
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
        del m

    return best_dim


def build_model(
    variant: str,
    graph: HeteroGraphData,
    config: dict,
    task_config: dict,
) -> tuple[HeteroGNN, int]:
    """Build model for a variant. Returns (model, num_params)."""
    hidden_dim = config["hidden_dim"]
    edge_types = [list(et) for et in graph.edge_types]
    node_types = graph.node_types
    fk_edges = [list(e) for e in graph.fk_edges]
    num_layers = config["num_gnn_layers"]
    pred_mlp_hidden = config["prediction_mlp_hidden"]

    if variant == "wide":
        # Get PRMP param count as target
        prmp = HeteroGNN(
            graph.in_channels_dict, hidden_dim, num_layers,
            edge_types, node_types, "prmp", fk_edges, pred_mlp_hidden,
        )
        target_params = _count_params(prmp)
        del prmp

        wide_dim = _find_wide_dim(
            graph.in_channels_dict, edge_types, node_types,
            num_layers, target_params,
        )
        model = HeteroGNN(
            graph.in_channels_dict, wide_dim, num_layers,
            edge_types, node_types, "standard",
        )
        logger.debug(
            f"Wide dim={wide_dim}, params={_count_params(model)} "
            f"(target={target_params})"
        )
    else:
        model = HeteroGNN(
            graph.in_channels_dict, hidden_dim, num_layers,
            edge_types, node_types, variant, fk_edges, pred_mlp_hidden,
        )

    return model, _count_params(model)


# ============================================================
# PHASE 5: TRAINING & EVALUATION
# ============================================================
def evaluate_model(
    model: HeteroGNN,
    graph: HeteroGraphData,
    mask: torch.Tensor,
    task_config: dict,
    return_preds: bool = False,
):
    """Evaluate model on masked subset."""
    model.eval()
    with torch.no_grad():
        pred = model(graph.node_features, graph.edge_index, graph.target_node_type)
        p = pred[mask]
        t = graph.target[mask]

        if task_config["type"] == "regression":
            p_np = (p * graph.target_std + graph.target_mean).cpu().numpy()
            t_np = (t * graph.target_std + graph.target_mean).cpu().numpy()
            mae = float(mean_absolute_error(t_np, p_np))
            metrics = {"mae": mae}
        else:
            probs = torch.sigmoid(p).cpu().numpy()
            t_np = t.cpu().numpy()
            if len(np.unique(t_np)) < 2:
                ap = 0.5
            else:
                ap = float(average_precision_score(t_np, probs))
            metrics = {"average_precision": ap}
            p_np = probs

    if return_preds:
        return metrics, p_np, t_np
    return metrics


def train_single_run(
    task_config: dict,
    variant: str,
    seed: int,
    config: dict,
    graph: HeteroGraphData,
) -> dict:
    """Train one model variant on one task with one seed. Returns result dict."""
    run_id = f"{task_config['dataset']}_{task_config['task']}_{variant}_s{seed}"

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Move graph to device
    g = graph.to(DEVICE)

    # Build model
    model, num_params = build_model(variant, g, config, task_config)
    model = model.to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    criterion = nn.MSELoss() if task_config["loss"] == "mse" else nn.BCEWithLogitsLoss()

    lower = task_config["lower_better"]
    best_val = float("inf") if lower else float("-inf")
    best_epoch = 0
    patience_ctr = 0
    best_state = None
    t0 = time.time()

    for epoch in range(config["max_epochs"]):
        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        pred = model(g.node_features, g.edge_index, g.target_node_type)
        loss = criterion(pred[g.train_mask], g.target[g.train_mask])

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{run_id}: NaN/Inf loss @ epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()

        # ---- Validate ----
        val_m = evaluate_model(model, g, g.val_mask, task_config)
        val_metric = val_m[task_config["primary_metric"]]

        improved = (val_metric < best_val) if lower else (val_metric > best_val)
        if improved:
            best_val = val_metric
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                break

    train_time = time.time() - t0

    # ---- Test with best model ----
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_m, test_preds, test_targets = evaluate_model(
        model, g, g.test_mask, task_config, return_preds=True
    )

    # ---- Embeddings for Ridge R^2 ----
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(g.node_features, g.edge_index)
        embeddings = {k: v.cpu().numpy() for k, v in embeddings.items()}

    result = {
        "variant": variant,
        "seed": seed,
        "task": task_config["task"],
        "dataset": task_config["dataset"],
        "best_epoch": best_epoch,
        "best_val_metric": float(best_val),
        "test_metrics": {k: float(v) for k, v in test_m.items()},
        "num_params": num_params,
        "trainable_params": trainable_params,
        "train_time_seconds": round(train_time, 2),
    }

    logger.info(
        f"  {run_id}: "
        f"{task_config['primary_metric']}={test_m[task_config['primary_metric']]:.4f}, "
        f"epoch={best_epoch}/{epoch}, params={num_params}, "
        f"time={train_time:.1f}s"
    )

    # Cleanup
    del model, best_state
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    return result, embeddings, test_preds, test_targets


# ============================================================
# PHASE 6: RIDGE R^2 DIAGNOSTIC
# ============================================================
def compute_ridge_r2(embeddings: dict, graph: HeteroGraphData) -> dict:
    """Ridge R^2: how well parent embeddings predict child embeddings per FK link."""
    results = {}
    for fk in graph.fk_edges:
        src_type, rel, dst_type = fk  # child, rel, parent
        et = tuple(fk)
        if et not in graph.edge_index:
            continue

        ei = graph.edge_index[et]
        child_idx = ei[0].numpy()
        parent_idx = ei[1].numpy()

        if src_type not in embeddings or dst_type not in embeddings:
            continue

        child_emb = embeddings[src_type][child_idx]
        parent_emb = embeddings[dst_type][parent_idx]

        # Subsample if needed
        if len(parent_emb) > 50000:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(parent_emb), 50000, replace=False)
            parent_emb, child_emb = parent_emb[sel], child_emb[sel]

        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(parent_emb, child_emb)
            pred = ridge.predict(parent_emb)
            r2 = float(r2_score(child_emb, pred, multioutput="uniform_average"))
        except Exception:
            r2 = 0.0

        link = f"{src_type}->{dst_type}"
        results[link] = {"ridge_r2": r2, "num_edges": int(ei.shape[1])}

    return results


# ============================================================
# PHASE 7: STATISTICAL ANALYSIS
# ============================================================
def compute_statistics(
    all_results: list[dict], tasks: list[dict], variants: list[str]
) -> dict:
    """Mean +/- std, Cohen's d, paired t-tests."""
    analysis = {}
    for tc in tasks:
        task_key = f"{tc['dataset']}/{tc['task']}"
        metric = tc["primary_metric"]
        lower = tc["lower_better"]

        ok = [
            r for r in all_results
            if r["task"] == tc["task"]
            and r["dataset"] == tc["dataset"]
            and "error" not in r
        ]

        vm: dict[str, list[float]] = {}
        for v in variants:
            vm[v] = [
                r["test_metrics"][metric] for r in ok if r["variant"] == v
            ]

        summary = {}
        for v, vals in vm.items():
            if vals:
                summary[v] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                    "values": vals,
                }
            else:
                summary[v] = {"mean": float("nan"), "std": 0.0, "n": 0, "values": []}

        pairwise = {}
        for v1, v2 in combinations(variants, 2):
            m1, m2 = vm.get(v1, []), vm.get(v2, [])
            if len(m1) >= 2 and len(m2) >= 2 and len(m1) == len(m2):
                try:
                    t_stat, p_val = stats.ttest_rel(m1, m2)
                    pooled = np.sqrt(
                        (np.var(m1, ddof=1) + np.var(m2, ddof=1)) / 2
                    )
                    d = (np.mean(m1) - np.mean(m2)) / pooled if pooled > 0 else 0
                    pairwise[f"{v1}_vs_{v2}"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_val),
                        "cohens_d": float(d),
                    }
                except Exception:
                    pass

        best_v, best_m = None, float("inf") if lower else float("-inf")
        for v, s in summary.items():
            if s["n"] > 0:
                if (lower and s["mean"] < best_m) or (
                    not lower and s["mean"] > best_m
                ):
                    best_m = s["mean"]
                    best_v = v

        analysis[task_key] = {
            "summary": summary,
            "pairwise": pairwise,
            "metric": metric,
            "lower_better": lower,
            "best_variant": best_v,
        }
    return analysis


# ============================================================
# PHASE 8: MAIN EXPERIMENT
# ============================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("PRMP Unified Benchmark")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"Device: {DEVICE}")
    total_runs = len(TASKS) * len(VARIANTS) * len(CONFIG["seeds"])
    logger.info(
        f"Tasks={len(TASKS)}, Variants={len(VARIANTS)}, "
        f"Seeds={len(CONFIG['seeds'])}, Total runs={total_runs}"
    )
    logger.info("=" * 60)

    t_start = time.time()
    all_results: list[dict] = []
    all_diag: list[dict] = []
    all_preds: dict[tuple, tuple] = {}  # (task, variant, seed) -> (preds, targets)
    param_counts: dict[str, int] = {}

    # Pre-load data
    cached: dict[str, HeteroGraphData] = {}
    for tc in TASKS:
        ck = f"{tc['dataset']}_{tc['task']}"
        try:
            cached[ck] = load_task_data(tc)
            g = cached[ck]
            logger.info(
                f"Graph {ck}: nodes={g.num_nodes}, "
                f"edges={len(g.edge_types)}, fk={len(g.fk_edges)}"
            )
        except Exception:
            logger.exception(f"Failed to load: {ck}")

    # Main loop
    run_i = 0
    for tc in TASKS:
        ck = f"{tc['dataset']}_{tc['task']}"
        if ck not in cached:
            continue
        graph = cached[ck]
        tk = f"{tc['dataset']}/{tc['task']}"

        logger.info(f"\n{'='*50}\nTask: {tk}\n{'='*50}")

        for variant in VARIANTS:
            for seed in CONFIG["seeds"]:
                run_i += 1
                rid = f"{tc['dataset']}_{tc['task']}_{variant}_s{seed}"
                logger.info(f"\n--- Run {run_i}/{total_runs}: {rid} ---")

                try:
                    res, embs, preds, targets = train_single_run(
                        tc, variant, seed, CONFIG, graph
                    )
                    all_results.append(res)
                    all_preds[(tc["task"], variant, seed)] = (preds, targets)

                    pk = f"{variant}_{ck}"
                    if pk not in param_counts:
                        param_counts[pk] = res["num_params"]

                    # Ridge R^2 on first seed only
                    if seed == CONFIG["seeds"][0]:
                        ridge = compute_ridge_r2(embs, graph)
                        all_diag.append({
                            "variant": variant,
                            "task": tc["task"],
                            "dataset": tc["dataset"],
                            "fk_ridge_r2": ridge,
                        })

                    del embs
                    if HAS_GPU:
                        torch.cuda.empty_cache()
                    gc.collect()

                except Exception:
                    logger.exception(f"FAILED: {rid}")
                    all_results.append({
                        "variant": variant,
                        "seed": seed,
                        "task": tc["task"],
                        "dataset": tc["dataset"],
                        "error": "training_failed",
                    })

        del graph
        if HAS_GPU:
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t_start
    logger.info(f"\nAll runs completed in {total_time:.1f}s")

    # ============================================================
    # PHASE 9: STATISTICAL ANALYSIS & OUTPUT
    # ============================================================
    stat_analysis = compute_statistics(all_results, TASKS, VARIANTS)

    # Print summary
    logger.info("\n=== RESULTS SUMMARY ===")
    for tk, an in stat_analysis.items():
        logger.info(f"\n{tk} ({an['metric']}, lower={an['lower_better']}):")
        for v in VARIANTS:
            s = an["summary"].get(v, {})
            if s.get("n", 0) > 0:
                logger.info(f"  {v:20s}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})")
        comp = "standard_vs_prmp"
        if comp in an.get("pairwise", {}):
            pw = an["pairwise"][comp]
            logger.info(
                f"  >> Standard vs PRMP: d={pw['cohens_d']:.3f}, p={pw['p_value']:.4f}"
            )

    # Build exp_gen_sol_out.json format
    output_datasets = []
    first_seed = CONFIG["seeds"][0]

    for tc in TASKS:
        tk = f"{tc['dataset']}/{tc['task']}"

        # Find a valid prediction set to get test targets
        ref_key = None
        for v in VARIANTS:
            k = (tc["task"], v, first_seed)
            if k in all_preds:
                ref_key = k
                break
        if ref_key is None:
            continue

        _, targets = all_preds[ref_key]
        n_test = len(targets)

        examples = []
        for i in range(n_test):
            ex: dict = {
                "input": json.dumps({
                    "task": tk,
                    "index": i,
                    "type": tc["type"],
                    "metric": tc["primary_metric"],
                }),
                "output": str(float(targets[i])),
                "metadata_task": tk,
                "metadata_type": tc["type"],
            }
            for v in VARIANTS:
                k = (tc["task"], v, first_seed)
                if k in all_preds:
                    ex[f"predict_{v}"] = str(float(all_preds[k][0][i]))
            examples.append(ex)

        output_datasets.append({"dataset": tk, "examples": examples})

    output = {
        "metadata": {
            "method_name": "PRMP_Benchmark",
            "description": (
                "Unified PRMP benchmark comparing 5 GNN variants across "
                "3 relational tasks (Amazon review-rating, F1 result-position, "
                "F1 result-dnf)"
            ),
            "config": CONFIG,
            "tasks": [f"{t['dataset']}/{t['task']}" for t in TASKS],
            "variants": VARIANTS,
            "total_runs_attempted": len(all_results),
            "total_runs_successful": len([r for r in all_results if "error" not in r]),
            "total_time_seconds": round(total_time, 1),
            "statistical_analysis": {
                k: {
                    "summary": {
                        vv: {kk: vvv for kk, vvv in ss.items() if kk != "values"}
                        for vv, ss in v["summary"].items()
                    },
                    "pairwise": v["pairwise"],
                    "metric": v["metric"],
                    "lower_better": v["lower_better"],
                    "best_variant": v["best_variant"],
                }
                for k, v in stat_analysis.items()
            },
            "embedding_diagnostics": all_diag,
            "parameter_counts": param_counts,
            "per_run_results": all_results,
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    sz = out_path.stat().st_size
    logger.info(f"Output: {out_path} ({sz / 1024 / 1024:.1f} MB)")

    return output


if __name__ == "__main__":
    main()
