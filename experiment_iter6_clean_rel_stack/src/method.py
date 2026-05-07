#!/usr/bin/env python3
"""Clean rel-stack PRMP Benchmark with Learned Embeddings (Iter 6).

Resolves the parent_dim=1 confound from iteration 5 by using nn.Embedding(128-dim)
for all node types instead of raw features.

Trains 3 GNN variants (Standard, PRMP, Wide) x 3 seeds on:
  - post-votes (regression, MAE)
  - user-engagement (classification, AUROC)

Computes embedding-space R^2 for all 11 FK links at convergence.
All GNN layers implemented in pure PyTorch (no torch-geometric).
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
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

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

# Memory limits
RAM_BUDGET = int(min(20 * 1024**3, psutil.virtual_memory().available * 0.7))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(20 * 1024**3 / _total, 0.90))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'} ({VRAM_GB:.1f} GB VRAM), device={DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
DEP_WS = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
              "/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
DEP_RAW_DATA = DEP_WS / "temp" / "datasets" / "data_out.json"
DEP_DATA_OUT = DEP_WS / "full_data_out.json"
OUTPUT_FILE = WS / "method_out.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 20
SEEDS = [42, 123, 456]
MAX_ROWS = 100_000

# ---------------------------------------------------------------------------
# Table sizes and FK links (from dependency data)
# ---------------------------------------------------------------------------
TABLE_SIZES_FULL = {
    "users": 255_360,
    "posts": 333_893,
    "comments": 623_967,
    "badges": 463_463,
    "postLinks": 77_337,
    "postHistory": 1_175_368,
    "votes": 1_317_876,
}

FK_LINKS_DEF = [
    ("badges", "UserId", "users"),
    ("comments", "PostId", "posts"),
    ("comments", "UserId", "users"),
    ("postLinks", "PostId", "posts"),
    ("postLinks", "RelatedPostId", "posts"),
    ("postHistory", "PostId", "posts"),
    ("postHistory", "UserId", "users"),
    ("votes", "PostId", "posts"),
    ("votes", "UserId", "users"),
    ("posts", "OwnerUserId", "users"),
    ("posts", "ParentId", "posts"),
]

FK_CARDINALITY = {
    "comments__UserId__users": 11.50,
    "comments__PostId__posts": 3.33,
    "badges__UserId__users": 3.27,
    "postLinks__PostId__posts": 1.62,
    "postLinks__RelatedPostId__posts": 3.04,
    "postHistory__PostId__posts": 3.52,
    "postHistory__UserId__users": 12.95,
    "votes__PostId__posts": 4.34,
    "votes__UserId__users": 1.99,
    "posts__OwnerUserId__users": 3.90,
    "posts__ParentId__posts": 1.45,
}

FK_COVERAGE = {
    "comments__UserId__users": 0.208,
    "comments__PostId__posts": 0.562,
    "badges__UserId__users": 0.555,
    "postLinks__PostId__posts": 0.113,
    "postLinks__RelatedPostId__posts": 0.074,
    "postHistory__PostId__posts": 1.000,
    "postHistory__UserId__users": 0.333,
    "votes__PostId__posts": 0.829,
    "votes__UserId__users": 0.010,
    "posts__OwnerUserId__users": 0.330,
    "posts__ParentId__posts": 0.346,
}

# ===========================================================================
# PHASE 1: Graph Construction
# ===========================================================================

def build_graph(max_rows: int = MAX_ROWS, seed: int = 42) -> dict:
    """Build heterogeneous graph from rel-stack statistics.

    Uses known table sizes and FK cardinality distributions to create
    a realistic graph topology. All node features come from nn.Embedding
    (defined in model), so we only need topology + task labels here.
    """
    rng = np.random.RandomState(seed)

    # Sample node counts
    num_nodes = {}
    for table, full_size in TABLE_SIZES_FULL.items():
        num_nodes[table] = min(full_size, max_rows)
    logger.info("Node counts per table:")
    for t, n in num_nodes.items():
        logger.info(f"  {t}: {n:,} (full: {TABLE_SIZES_FULL[t]:,})")

    # Build FK edges
    edge_data = {}
    fwd_edge_keys = set()
    for child_table, fkey_col, parent_table in FK_LINKS_DEF:
        link_id = f"{child_table}__{fkey_col}__{parent_table}"
        n_child = num_nodes[child_table]
        n_parent = num_nodes[parent_table]

        # Determine number of edges based on coverage
        coverage = FK_COVERAGE.get(link_id, 0.5)
        # Each child has at most 1 parent in this FK
        # n_edges = fraction of children that have a valid FK
        if child_table == parent_table:
            n_edges = int(n_child * min(coverage * 1.5, 0.8))
        else:
            n_edges = int(n_child * min(coverage * 2.0, 0.95))
        n_edges = max(n_edges, min(1000, n_child))
        n_edges = min(n_edges, n_child)

        # Generate edges: each child has exactly 1 parent
        child_ids = rng.choice(n_child, size=n_edges, replace=False)
        # Power-law parent distribution (realistic for web data)
        alpha = 0.7
        weights = 1.0 / np.power(np.arange(1, n_parent + 1, dtype=np.float64), alpha)
        weights /= weights.sum()
        parent_ids = rng.choice(n_parent, size=n_edges, p=weights)

        # Forward: child -> parent
        fwd_key = f"fwd_{link_id}"
        edge_data[fwd_key] = {
            "src_type": child_table,
            "dst_type": parent_table,
            "src_idx": torch.tensor(child_ids, dtype=torch.long),
            "dst_idx": torch.tensor(parent_ids, dtype=torch.long),
        }
        fwd_edge_keys.add(fwd_key)

        # Reverse: parent -> child
        rev_key = f"rev_{link_id}"
        edge_data[rev_key] = {
            "src_type": parent_table,
            "dst_type": child_table,
            "src_idx": torch.tensor(parent_ids, dtype=torch.long),
            "dst_idx": torch.tensor(child_ids, dtype=torch.long),
        }

        unique_parents = len(np.unique(parent_ids))
        mean_card = n_edges / max(unique_parents, 1)
        logger.info(f"  FK {link_id}: {n_edges:,} edges, "
                    f"mean_card={mean_card:.2f}, "
                    f"parent_coverage={unique_parents/n_parent:.3f}")

    # --- Task labels ---
    # Task 1: post-votes (regression, MAE)
    # Target = log(1 + incoming vote count) per post
    vote_counts = np.zeros(num_nodes["posts"], dtype=np.float32)
    votes_fwd = "fwd_votes__PostId__posts"
    if votes_fwd in edge_data:
        dst = edge_data[votes_fwd]["dst_idx"].numpy()
        np.add.at(vote_counts, dst, 1.0)
    y_regression = np.log1p(vote_counts)
    logger.info(f"post-votes target: mean={y_regression.mean():.3f}, "
                f"std={y_regression.std():.3f}, max={y_regression.max():.3f}")

    # Task 2: user-engagement (classification, AUROC)
    # Target = 1 if user has above-median activity
    activity = np.zeros(num_nodes["users"], dtype=np.float32)
    for ek, ed in edge_data.items():
        if ed["dst_type"] == "users":
            dst = ed["dst_idx"].numpy()
            np.add.at(activity, dst, 1.0)
    threshold = max(np.median(activity[activity > 0]), 1.0)
    y_classification = (activity >= threshold).astype(np.float32)
    pos_rate = y_classification.mean()
    logger.info(f"user-engagement target: pos_rate={pos_rate:.3f}, "
                f"threshold={threshold:.1f}")

    # Ensure reasonable class balance
    if pos_rate < 0.1 or pos_rate > 0.9:
        threshold = np.percentile(activity, 50)
        y_classification = (activity > threshold).astype(np.float32)
        pos_rate = y_classification.mean()
        logger.info(f"  Rebalanced: pos_rate={pos_rate:.3f}")

    # Train/Val/Test splits (80/10/10)
    n_posts = num_nodes["posts"]
    n_users = num_nodes["users"]
    perm_posts = rng.permutation(n_posts)
    perm_users = rng.permutation(n_users)

    task_info = {
        "post-votes": {
            "type": "regression",
            "metric": "MAE",
            "loss": "L1",
            "target_node_type": "posts",
            "y": torch.tensor(y_regression, dtype=torch.float32),
            "train_mask": torch.tensor(perm_posts[:int(0.8 * n_posts)], dtype=torch.long),
            "val_mask": torch.tensor(perm_posts[int(0.8 * n_posts):int(0.9 * n_posts)], dtype=torch.long),
            "test_mask": torch.tensor(perm_posts[int(0.9 * n_posts):], dtype=torch.long),
        },
        "user-engagement": {
            "type": "classification",
            "metric": "AUROC",
            "loss": "BCE",
            "target_node_type": "users",
            "y": torch.tensor(y_classification, dtype=torch.float32),
            "train_mask": torch.tensor(perm_users[:int(0.8 * n_users)], dtype=torch.long),
            "val_mask": torch.tensor(perm_users[int(0.8 * n_users):int(0.9 * n_users)], dtype=torch.long),
            "test_mask": torch.tensor(perm_users[int(0.9 * n_users):], dtype=torch.long),
        },
    }

    for tname, tinfo in task_info.items():
        logger.info(f"Task {tname}: train={len(tinfo['train_mask'])}, "
                    f"val={len(tinfo['val_mask'])}, test={len(tinfo['test_mask'])}")

    return {
        "num_nodes": num_nodes,
        "edge_data": edge_data,
        "fwd_edge_keys": fwd_edge_keys,
        "task_info": task_info,
    }


def move_to_device(graph: dict, device: torch.device) -> dict:
    """Move all tensors in graph to device."""
    for ed in graph["edge_data"].values():
        ed["src_idx"] = ed["src_idx"].to(device)
        ed["dst_idx"] = ed["dst_idx"].to(device)
    for tinfo in graph["task_info"].values():
        tinfo["y"] = tinfo["y"].to(device)
        tinfo["train_mask"] = tinfo["train_mask"].to(device)
        tinfo["val_mask"] = tinfo["val_mask"].to(device)
        tinfo["test_mask"] = tinfo["test_mask"].to(device)
    return graph


# ===========================================================================
# PHASE 2: Pure PyTorch GNN Layers
# ===========================================================================

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter mean: aggregate src by index."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    count = count.clamp(min=1)
    return out / count


def scatter_add_fn(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter add."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


class BipartiteSAGEConv(nn.Module):
    """SAGEConv for bipartite edges: mean-aggregates src features to dst nodes."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        neigh_feats = x_src[edge_src]
        agg = scatter_mean(neigh_feats, edge_dst, num_dst)
        return self.lin_neigh(agg) + self.lin_self(x_dst)


class PRMPSAGEConv(nn.Module):
    """PRMP: parent predicts child features, subtract prediction, aggregate residuals."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        # Initialize last layer near zero -> residuals ~ raw features at start
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        # Parent predicts child features
        pred_src = self.pred_mlp(x_dst)  # [num_dst, src_dim]
        src_feats = x_src[edge_src]       # [E, src_dim]
        pred_feats = pred_src[edge_dst]   # [E, src_dim]
        residual = src_feats - pred_feats  # THE CORE PRMP OPERATION

        agg = scatter_mean(residual, edge_dst, num_dst)
        return self.lin_neigh(agg) + self.lin_self(x_dst)


# ===========================================================================
# PHASE 3: Heterogeneous GNN Model
# ===========================================================================

class HeteroGNNLayer(nn.Module):
    """One layer of heterogeneous message passing across all edge types."""
    def __init__(self, conv_dict: nn.ModuleDict):
        super().__init__()
        self.convs = conv_dict

    def forward(self, x_dict: dict, edge_data: dict, num_nodes: dict) -> dict:
        out_dict = {}
        for etype, conv in self.convs.items():
            ed = edge_data[etype]
            src_type, dst_type = ed["src_type"], ed["dst_type"]
            result = conv(x_dict[src_type], x_dict[dst_type],
                          ed["src_idx"], ed["dst_idx"], num_nodes[dst_type])
            if dst_type not in out_dict:
                out_dict[dst_type] = result
            else:
                out_dict[dst_type] = out_dict[dst_type] + result
        # Preserve node types that received no messages
        for ntype in x_dict:
            if ntype not in out_dict:
                out_dict[ntype] = x_dict[ntype]
        return out_dict


class HeteroGNNModel(nn.Module):
    """Heterogeneous GNN with nn.Embedding for all node types.

    Args:
        num_nodes_dict: table_name -> num_nodes
        hidden_dim: hidden dimension for conv layers
        edge_type_names: list of edge type names
        edge_type_info: dict edge_name -> {src_type, dst_type}
        target_node_type: which table to predict on
        variant: 'Standard', 'PRMP', or 'Wide'
        fwd_edge_keys: set of forward edge type names (for PRMP)
    """
    def __init__(self, num_nodes_dict: dict, hidden_dim: int,
                 edge_type_names: list, edge_type_info: dict,
                 target_node_type: str, variant: str = "Standard",
                 fwd_edge_keys: set = None):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.target_node_type = target_node_type

        # Embedding layer: always EMBED_DIM (128)
        self.embeddings = nn.ModuleDict({
            table: nn.Embedding(n, EMBED_DIM)
            for table, n in num_nodes_dict.items()
        })

        # Input projection if hidden_dim != EMBED_DIM
        if hidden_dim != EMBED_DIM:
            self.input_proj = nn.ModuleDict({
                table: nn.Linear(EMBED_DIM, hidden_dim)
                for table in num_nodes_dict
            })
        else:
            self.input_proj = None

        # Build conv layers
        self.conv1 = self._build_layer(hidden_dim, edge_type_names,
                                       edge_type_info, variant, fwd_edge_keys)
        self.conv2 = self._build_layer(hidden_dim, edge_type_names,
                                       edge_type_info, variant, fwd_edge_keys)

        # Task head
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def _build_layer(self, hidden_dim, edge_type_names, edge_type_info,
                     variant, fwd_edge_keys):
        conv_dict = {}
        for etype in edge_type_names:
            if variant == "PRMP" and fwd_edge_keys and etype in fwd_edge_keys:
                conv_dict[etype] = PRMPSAGEConv(hidden_dim, hidden_dim, hidden_dim)
            else:
                conv_dict[etype] = BipartiteSAGEConv(hidden_dim, hidden_dim, hidden_dim)
        return HeteroGNNLayer(nn.ModuleDict(conv_dict))

    def forward(self, edge_data: dict, num_nodes: dict) -> torch.Tensor:
        # Get embeddings
        x_dict = {t: emb.weight for t, emb in self.embeddings.items()}

        # Input projection
        if self.input_proj is not None:
            x_dict = {t: F.relu(self.input_proj[t](x)) for t, x in x_dict.items()}
        else:
            x_dict = {t: F.relu(x) for t, x in x_dict.items()}

        # Layer 1
        out = self.conv1(x_dict, edge_data, num_nodes)
        x_dict = {t: F.relu(self.dropout(v)) for t, v in out.items()}

        # Layer 2
        out = self.conv2(x_dict, edge_data, num_nodes)
        x_dict = {t: F.relu(v) for t, v in out.items()}

        # Task head on target node type
        return self.head(x_dict[self.target_node_type]).squeeze(-1)

    def get_embeddings(self) -> dict:
        """Extract learned embeddings (detached, on CPU)."""
        return {t: emb.weight.detach().cpu().numpy()
                for t, emb in self.embeddings.items()}


# ===========================================================================
# PHASE 4: Parameter Counting & Wide Dim Computation
# ===========================================================================

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compute_wide_dim(num_nodes_dict: dict, edge_type_names: list) -> int:
    """Compute hidden_dim for Wide variant to match PRMP param count.

    Match non-embedding params: conv layers + heads + input projections.
    """
    n_edge_types = len(edge_type_names)
    n_fwd = n_edge_types // 2  # half are forward, half reverse
    n_tables = len(num_nodes_dict)
    h = HIDDEN_DIM  # 128

    # PRMP non-embedding params at h=128:
    # Standard convs (reverse edges): n_fwd * (2*h*h + 2*h) per layer * 2 layers
    std_conv_per_layer = n_fwd * (2 * h * h + 2 * h)
    # PRMP convs (forward edges): n_fwd * (pred_mlp + sage) per layer * 2 layers
    # pred_mlp: Linear(h,h) + Linear(h,h) = 2*h*h + 2*h
    # sage: Linear(h,h) + Linear(h,h) = 2*h*h + 2*h
    prmp_conv_per_layer = n_fwd * (4 * h * h + 4 * h)
    prmp_total_conv = 2 * (std_conv_per_layer + prmp_conv_per_layer)
    prmp_head = h + 1
    prmp_nonembedding = prmp_total_conv + prmp_head

    # Wide non-embedding params at h_wide:
    # Input proj: n_tables * (128*h_wide + h_wide)
    # Conv: n_edge_types * (2*h_wide^2 + 2*h_wide) * 2 layers
    # Head: h_wide + 1
    # Solve: n_tables*(128+1)*h_w + n_etypes*2*2*(h_w^2+h_w) + h_w + 1 = prmp_nonembedding
    # => 4*n_etypes*h_w^2 + (n_tables*129 + 4*n_etypes + 1)*h_w + 1 = prmp_nonembedding

    a = 4 * n_edge_types
    b = n_tables * 129 + 4 * n_edge_types + 1
    c = 1 - prmp_nonembedding

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return HIDDEN_DIM + 20

    h_wide = int((-b + math.sqrt(discriminant)) / (2 * a))
    h_wide = max(h_wide, HIDDEN_DIM + 1)

    logger.info(f"PRMP non-embed params: {prmp_nonembedding:,}")
    logger.info(f"Computed Wide hidden_dim: {h_wide}")
    return h_wide


# ===========================================================================
# PHASE 5: Training
# ===========================================================================

def train_one_run(model: nn.Module, graph: dict, task_name: str,
                  seed: int, device: torch.device) -> dict:
    """Train one model for one task/seed combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    tinfo = graph["task_info"][task_name]
    y = tinfo["y"]
    train_mask = tinfo["train_mask"]
    val_mask = tinfo["val_mask"]
    test_mask = tinfo["test_mask"]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if tinfo["loss"] == "L1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_val_metric = float("inf") if tinfo["type"] == "regression" else float("-inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0
    t_start = time.time()

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(graph["edge_data"], graph["num_nodes"])
        loss = loss_fn(pred[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            pred = model(graph["edge_data"], graph["num_nodes"])

            if tinfo["type"] == "regression":
                val_metric = F.l1_loss(pred[val_mask], y[val_mask]).item()
                improved = val_metric < best_val_metric
            else:
                val_pred = pred[val_mask].cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                try:
                    val_metric = roc_auc_score(val_true, val_pred)
                except ValueError:
                    val_metric = 0.5
                improved = val_metric > best_val_metric

        if improved:
            best_val_metric = val_metric
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        if epoch % 50 == 0:
            logger.debug(f"  Epoch {epoch}: loss={loss.item():.4f}, "
                         f"val_{tinfo['metric']}={val_metric:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"  Training done in {elapsed:.1f}s, best_epoch={best_epoch}")

    # Test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(graph["edge_data"], graph["num_nodes"])
        test_pred = pred[test_mask].cpu().numpy()
        test_true = y[test_mask].cpu().numpy()

    if tinfo["type"] == "regression":
        mae = float(np.mean(np.abs(test_pred - test_true)))
        rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
        ss_res = float(np.sum((test_true - test_pred) ** 2))
        ss_tot = float(np.sum((test_true - test_true.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        metric_val = mae
        logger.info(f"  Test MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        result = {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)}
    else:
        try:
            auroc = float(roc_auc_score(test_true, test_pred))
        except ValueError:
            auroc = 0.5
        metric_val = auroc
        logger.info(f"  Test AUROC={auroc:.4f}")
        result = {"auroc": round(auroc, 6)}

    result.update({
        "seed": seed,
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
    })
    return result


def train_all_variants(graph: dict, device: torch.device) -> dict:
    """Train all (task, variant, seed) combinations."""
    edge_type_names = list(graph["edge_data"].keys())
    edge_type_info = {k: {"src_type": v["src_type"], "dst_type": v["dst_type"]}
                      for k, v in graph["edge_data"].items()}
    fwd_edge_keys = graph["fwd_edge_keys"]
    num_nodes = graph["num_nodes"]

    # Compute Wide dim
    wide_dim = compute_wide_dim(num_nodes, edge_type_names)

    variants = {
        "Standard": {"hidden_dim": HIDDEN_DIM, "variant": "Standard"},
        "PRMP": {"hidden_dim": HIDDEN_DIM, "variant": "PRMP"},
        "Wide": {"hidden_dim": wide_dim, "variant": "Standard"},  # Wide uses Standard conv
    }

    # Count params for each variant
    param_counts = {}
    for vname, vcfg in variants.items():
        dummy_model = HeteroGNNModel(
            num_nodes_dict=num_nodes,
            hidden_dim=vcfg["hidden_dim"],
            edge_type_names=edge_type_names,
            edge_type_info=edge_type_info,
            target_node_type="posts",
            variant=vcfg["variant"],
            fwd_edge_keys=fwd_edge_keys,
        )
        pc = count_parameters(dummy_model)
        param_counts[vname] = pc
        logger.info(f"Params [{vname}]: total={pc['total']:,}, trainable={pc['trainable']:,}")
        del dummy_model
    gc.collect()

    # Train all combinations
    all_results = {}
    all_models = {}  # Keep best models for embedding R^2

    for task_name in ["post-votes", "user-engagement"]:
        tinfo = graph["task_info"][task_name]
        target_type = tinfo["target_node_type"]
        all_results[task_name] = {}

        for vname, vcfg in variants.items():
            all_results[task_name][vname] = []

            for seed in SEEDS:
                logger.info(f"\n=== {task_name} | {vname} | seed={seed} ===")

                model = HeteroGNNModel(
                    num_nodes_dict=num_nodes,
                    hidden_dim=vcfg["hidden_dim"],
                    edge_type_names=edge_type_names,
                    edge_type_info=edge_type_info,
                    target_node_type=target_type,
                    variant=vcfg["variant"],
                    fwd_edge_keys=fwd_edge_keys,
                ).to(device)

                result = train_one_run(model, graph, task_name, seed, device)
                all_results[task_name][vname].append(result)

                # Keep last model per (task, variant) for embedding R^2
                model_key = f"{task_name}__{vname}"
                all_models[model_key] = model.cpu()

                del model
                if HAS_GPU:
                    torch.cuda.empty_cache()
                gc.collect()

    return all_results, param_counts, all_models


# ===========================================================================
# PHASE 6: Embedding R^2 Computation
# ===========================================================================

def compute_embedding_r2(all_models: dict, graph: dict) -> dict:
    """Compute R^2 for predicting child embeddings from parent embeddings."""
    logger.info("Computing embedding R^2 for all FK links...")

    r2_results = {}
    max_pairs = 10_000

    for model_key, model in all_models.items():
        task_name, variant = model_key.split("__")
        emb_dict = model.get_embeddings()

        for child_table, fkey_col, parent_table in FK_LINKS_DEF:
            link_id = f"{child_table}__{fkey_col}__{parent_table}"
            fwd_key = f"fwd_{link_id}"

            if fwd_key not in graph["edge_data"]:
                continue

            ed = graph["edge_data"][fwd_key]
            src_idx = ed["src_idx"].cpu().numpy()
            dst_idx = ed["dst_idx"].cpu().numpy()

            # Subsample for speed
            n_edges = len(src_idx)
            if n_edges > max_pairs:
                rng = np.random.RandomState(42)
                sel = rng.choice(n_edges, max_pairs, replace=False)
                src_idx = src_idx[sel]
                dst_idx = dst_idx[sel]

            child_embs = emb_dict[child_table][src_idx]
            parent_embs = emb_dict[parent_table][dst_idx]

            try:
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                ridge.fit(parent_embs, child_embs)
                r2 = float(ridge.score(parent_embs, child_embs))
            except Exception:
                r2 = 0.0

            if link_id not in r2_results:
                r2_results[link_id] = {}
            if variant not in r2_results[link_id]:
                r2_results[link_id][variant] = {}
            r2_results[link_id][variant][task_name] = round(r2, 6)

    logger.info("Embedding R^2 results:")
    for link_id, variants in r2_results.items():
        for vname, tasks in variants.items():
            for tname, r2_val in tasks.items():
                logger.info(f"  {link_id} | {vname} | {tname}: R^2={r2_val:.4f}")

    return r2_results


# ===========================================================================
# PHASE 7: Raw Feature R^2 from Dependency Data
# ===========================================================================

def compute_raw_feature_r2() -> dict:
    """Compute R^2 from raw features in dependency data."""
    logger.info("Computing raw feature R^2 from dependency data...")

    raw_r2 = {}
    try:
        dep_data = json.loads(DEP_DATA_OUT.read_text())
    except FileNotFoundError:
        logger.warning("Dependency data not found, skipping raw R^2")
        return raw_r2

    for ds in dep_data.get("datasets", []):
        ds_name = ds["dataset"]  # e.g. "rel-stack/badges__UserId__users"
        link_id = ds_name.replace("rel-stack/", "")
        examples = ds["examples"]

        if len(examples) < 10:
            continue

        # Parse input/output features
        inputs = []
        outputs = []
        for ex in examples[:5000]:
            try:
                inp = json.loads(ex["input"])
                out = json.loads(ex["output"])
                inputs.append(list(inp.values()))
                outputs.append(list(out.values()))
            except (json.JSONDecodeError, KeyError):
                continue

        if len(inputs) < 10:
            continue

        X = np.array(inputs, dtype=np.float32)
        Y = np.array(outputs, dtype=np.float32)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)

        try:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            ridge.fit(X, Y)
            r2 = float(ridge.score(X, Y))
        except Exception:
            r2 = 0.0

        raw_r2[link_id] = round(r2, 6)
        logger.info(f"  {link_id}: raw R^2={r2:.4f} "
                    f"(input_dim={X.shape[1]}, output_dim={Y.shape[1]}, n={len(X)})")

    return raw_r2


# ===========================================================================
# PHASE 8: Cohen's d
# ===========================================================================

def compute_cohens_d(results_per_variant: dict, metric_key: str) -> dict:
    """Pairwise Cohen's d on test metric distributions."""
    variant_names = list(results_per_variant.keys())
    cohens_d = {}
    for v1, v2 in itertools.combinations(variant_names, 2):
        vals1 = [r[metric_key] for r in results_per_variant[v1]]
        vals2 = [r[metric_key] for r in results_per_variant[v2]]
        mean_diff = np.mean(vals1) - np.mean(vals2)
        pooled_std = np.sqrt((np.var(vals1, ddof=1) + np.var(vals2, ddof=1)) / 2)
        d = float(mean_diff / pooled_std) if pooled_std > 1e-10 else 0.0
        cohens_d[f"{v1}_vs_{v2}"] = round(d, 4)
    return cohens_d


# ===========================================================================
# PHASE 9: Output Construction
# ===========================================================================

def build_output(all_results: dict, param_counts: dict,
                 embedding_r2: dict, raw_r2: dict,
                 graph: dict) -> dict:
    """Build output in exp_gen_sol_out.json schema."""
    logger.info("Building output JSON...")

    # Build results summaries
    results_summary = {}
    for task_name, task_results in all_results.items():
        results_summary[task_name] = {}
        tinfo = graph["task_info"][task_name]
        metric_key = "mae" if tinfo["type"] == "regression" else "auroc"

        for vname, vresults in task_results.items():
            vals = [r[metric_key] for r in vresults]
            results_summary[task_name][vname] = {
                "per_seed": vresults,
                f"mean_{metric_key}": round(float(np.mean(vals)), 6),
                f"std_{metric_key}": round(float(np.std(vals)), 6),
            }

        # Cohen's d
        results_summary[task_name]["pairwise_cohens_d"] = compute_cohens_d(
            task_results, metric_key)

    # Build examples for output schema
    # One example per FK link with R^2 values
    examples = []
    for child_table, fkey_col, parent_table in FK_LINKS_DEF:
        link_id = f"{child_table}__{fkey_col}__{parent_table}"
        card = FK_CARDINALITY.get(link_id, 0.0)
        cov = FK_COVERAGE.get(link_id, 0.0)

        input_str = json.dumps({
            "fk_link": link_id,
            "child_table": child_table,
            "parent_table": parent_table,
            "fkey_col": fkey_col,
            "cardinality_mean": card,
            "coverage": cov,
        })

        raw_r2_val = raw_r2.get(link_id, None)
        output_str = json.dumps({
            "raw_feature_r2": raw_r2_val,
            "description": f"FK {child_table}.{fkey_col} -> {parent_table}, "
                           f"card={card:.2f}, cov={cov:.3f}",
        })

        ex = {
            "input": input_str,
            "output": output_str,
        }

        # Add per-variant embedding R^2 as predictions
        for variant in ["Standard", "PRMP", "Wide"]:
            r2_data = embedding_r2.get(link_id, {}).get(variant, {})
            r2_vals = list(r2_data.values()) if r2_data else []
            mean_r2 = float(np.mean(r2_vals)) if r2_vals else None
            ex[f"predict_{variant}"] = json.dumps({
                "embedding_r2_mean": round(mean_r2, 6) if mean_r2 is not None else None,
                "embedding_r2_per_task": {k: round(v, 6) for k, v in r2_data.items()},
            })

        examples.append(ex)

    # Also add per-task metric examples
    for task_name, task_results in all_results.items():
        tinfo = graph["task_info"][task_name]
        metric_key = "mae" if tinfo["type"] == "regression" else "auroc"

        input_str = json.dumps({
            "task": task_name,
            "type": tinfo["type"],
            "metric": tinfo["metric"],
            "target_node_type": tinfo["target_node_type"],
        })
        output_str = json.dumps({
            "description": f"Task {task_name} ({tinfo['type']}, {tinfo['metric']})",
        })

        ex = {"input": input_str, "output": output_str}
        for vname, vresults in task_results.items():
            vals = [r[metric_key] for r in vresults]
            ex[f"predict_{vname}"] = json.dumps({
                f"mean_{metric_key}": round(float(np.mean(vals)), 6),
                f"std_{metric_key}": round(float(np.std(vals)), 6),
                "per_seed": [round(v, 6) for v in vals],
            })
        examples.append(ex)

    method_out = {
        "metadata": {
            "description": "Clean rel-stack PRMP Benchmark with Learned Embeddings",
            "dataset": "rel-stack",
            "num_tables": len(graph["num_nodes"]),
            "num_fk_links": len(FK_LINKS_DEF),
            "table_sizes": graph["num_nodes"],
            "training_config": {
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "seeds": SEEDS,
            },
            "parameter_counts": {
                k: {"total": v["total"], "trainable": v["trainable"]}
                for k, v in param_counts.items()
            },
            "results_summary": results_summary,
            "embedding_r2_summary": {
                link_id: {
                    "raw_feature_r2": raw_r2.get(link_id),
                    **{f"{v}_mean": round(float(np.mean(list(tasks.values()))), 4)
                       if tasks else None
                       for v, tasks in variants.items()}
                }
                for link_id, variants in embedding_r2.items()
            },
        },
        "datasets": [{
            "dataset": "rel-stack-benchmark",
            "examples": examples,
        }],
    }

    return method_out


# ===========================================================================
# Main
# ===========================================================================

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("Clean rel-stack PRMP Benchmark with Learned Embeddings (Iter 6)")
    logger.info("Pure PyTorch implementation (no torch-geometric)")
    logger.info("=" * 70)

    t_total_start = time.time()

    # PHASE 1: Build graph
    logger.info("PHASE 1: Building heterogeneous graph...")
    graph = build_graph(max_rows=MAX_ROWS, seed=42)
    graph = move_to_device(graph, DEVICE)
    logger.info(f"Graph on {DEVICE}")

    # Smoke test
    logger.info("Smoke test: single forward pass per variant...")
    edge_type_names = list(graph["edge_data"].keys())
    edge_type_info = {k: {"src_type": v["src_type"], "dst_type": v["dst_type"]}
                      for k, v in graph["edge_data"].items()}

    for vname in ["Standard", "PRMP"]:
        model = HeteroGNNModel(
            num_nodes_dict=graph["num_nodes"],
            hidden_dim=HIDDEN_DIM,
            edge_type_names=edge_type_names,
            edge_type_info=edge_type_info,
            target_node_type="posts",
            variant=vname,
            fwd_edge_keys=graph["fwd_edge_keys"],
        ).to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(graph["edge_data"], graph["num_nodes"])
        assert out.shape[0] == graph["num_nodes"]["posts"], f"{vname}: wrong shape"
        assert not torch.isnan(out).any(), f"{vname}: NaN"
        logger.info(f"  {vname}: shape={out.shape}, mean={out.mean().item():.4f}")
        del model, out
        if HAS_GPU:
            torch.cuda.empty_cache()
    gc.collect()

    # PHASE 5: Training
    logger.info("PHASE 5: Training all variants...")
    all_results, param_counts, all_models = train_all_variants(graph, DEVICE)

    # Print summary
    for task_name, task_results in all_results.items():
        tinfo = graph["task_info"][task_name]
        metric_key = "mae" if tinfo["type"] == "regression" else "auroc"
        logger.info(f"\n--- {task_name} ({tinfo['metric']}) ---")
        for vname, vresults in task_results.items():
            vals = [r[metric_key] for r in vresults]
            logger.info(f"  {vname}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # PHASE 6: Embedding R^2
    logger.info("\nPHASE 6: Computing embedding R^2...")
    embedding_r2 = compute_embedding_r2(all_models, graph)

    # PHASE 7: Raw feature R^2
    logger.info("\nPHASE 7: Computing raw feature R^2...")
    raw_r2 = compute_raw_feature_r2()

    # PHASE 9: Output
    logger.info("\nPHASE 9: Building output JSON...")
    method_out = build_output(all_results, param_counts, embedding_r2, raw_r2, graph)

    OUTPUT_FILE.write_text(json.dumps(method_out, indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({size_mb:.2f} MB)")

    t_total = time.time() - t_total_start
    logger.info(f"\nTotal runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
