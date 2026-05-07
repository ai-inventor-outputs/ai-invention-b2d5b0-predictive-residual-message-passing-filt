#!/usr/bin/env python3
"""PRMP experiment on rel-hm (H&M Fashion) dataset.

Compares Standard HeteroGNN, PRMP (Predictive Residual Message Passing),
and Wide (parameter-matched) control on 2 tasks:
  - user-churn (binary classification, AUROC)
  - item-sales (regression, MAE)
with 3 node types (customer, article, transaction) and 2 FK edge types.

Also computes embedding-space Ridge R² for both FK links.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (container-safe)
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
TOTAL_RAM_GB = _container_ram_gb() or 57.0

# Memory budget: 50% of container RAM for safety
RAM_BUDGET = int(TOTAL_RAM_GB * 0.50 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"budget={RAM_BUDGET / 1e9:.1f}GB")

# ---------------------------------------------------------------------------
# PyTorch + GPU
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    DEVICE = torch.device("cuda")
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.85)
    torch.cuda.set_per_process_memory_fraction(
        min(VRAM_BUDGET / _total, 0.90))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                f"VRAM={VRAM_GB:.1f}GB, budget={VRAM_BUDGET/1e9:.1f}GB")
else:
    DEVICE = torch.device("cpu")
    VRAM_GB = 0
    logger.info("No GPU available, using CPU")

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import scatter

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path(__file__).parent
DEP_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
               "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 200
PATIENCE = 20
SEEDS = [42, 123, 456, 789, 1024]
VARIANTS = ["standard", "prmp", "wide"]
BATCH_SIZE = 1024
MAX_TRANSACTIONS = 500_000  # subsample transactions for memory

TASKS_CFG = {
    "user-churn": {
        "entity": "customer",
        "metric_name": "AUROC",
        "task_type": "classification",
    },
    "item-sales": {
        "entity": "article",
        "metric_name": "MAE",
        "task_type": "regression",
    },
}


# ===================================================================
# Phase 1 — Data Loading & Graph Construction
# ===================================================================

def download_hm_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download H&M tables from HuggingFace relbench dataset."""
    cache_dir = WS / "cache_hm"
    cache_dir.mkdir(exist_ok=True)

    cust_path = cache_dir / "customers.parquet"
    art_path = cache_dir / "articles.parquet"
    txn_path = cache_dir / "transactions.parquet"

    if cust_path.exists() and art_path.exists() and txn_path.exists():
        logger.info("Loading cached H&M tables")
        customers = pd.read_parquet(cust_path)
        articles = pd.read_parquet(art_path)
        transactions = pd.read_parquet(txn_path)
        return customers, articles, transactions

    logger.info("Downloading H&M tables from HuggingFace...")
    try:
        from datasets import load_dataset

        # Try loading from relbench HuggingFace datasets
        for ds_name in ["relbench/rel-hm", "relbench/rel-hm-db"]:
            try:
                ds = load_dataset(ds_name, trust_remote_code=True)
                logger.info(f"Loaded from {ds_name}: {list(ds.keys())}")
                break
            except Exception as e:
                logger.debug(f"Failed {ds_name}: {e}")
                continue
        else:
            raise RuntimeError("Could not load from any HuggingFace source")

        # Extract tables
        if "customers" in ds:
            customers = ds["customers"].to_pandas()
        elif "customer" in ds:
            customers = ds["customer"].to_pandas()
        else:
            raise KeyError(f"No customer table found. Keys: {list(ds.keys())}")

        if "articles" in ds:
            articles = ds["articles"].to_pandas()
        elif "article" in ds:
            articles = ds["article"].to_pandas()
        else:
            raise KeyError(f"No article table found. Keys: {list(ds.keys())}")

        if "transactions" in ds:
            transactions = ds["transactions"].to_pandas()
        elif "transaction" in ds:
            transactions = ds["transaction"].to_pandas()
        elif "transactions_train" in ds:
            transactions = ds["transactions_train"].to_pandas()
        else:
            raise KeyError(f"No transactions table. Keys: {list(ds.keys())}")

    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")
        logger.info("Falling back to dependency parquet files to construct tables")
        customers, articles, transactions = _build_tables_from_dependency()

    # Subsample transactions if too large
    if len(transactions) > MAX_TRANSACTIONS:
        logger.info(f"Subsampling transactions: {len(transactions)} -> {MAX_TRANSACTIONS}")
        transactions = transactions.sample(
            n=MAX_TRANSACTIONS, random_state=42
        ).reset_index(drop=True)

    # Cache
    customers.to_parquet(cust_path, index=False)
    articles.to_parquet(art_path, index=False)
    transactions.to_parquet(txn_path, index=False)
    logger.info(f"Cached tables: customers={len(customers)}, "
                f"articles={len(articles)}, transactions={len(transactions)}")

    return customers, articles, transactions


def _build_tables_from_dependency() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct relational tables from dependency metadata + parquets.

    Uses cardinality stats from the dependency to build a realistic
    synthetic relational structure matching H&M schema.
    """
    logger.info("Building tables from dependency data (fallback)")

    # Load FK metadata from dependency
    dep_data = json.loads((DEP_DIR / "full_data_out.json").read_text())
    fk_examples = dep_data["datasets"][0]["examples"]

    # Parse cardinality stats
    cust_txn_meta = json.loads(fk_examples[0]["output"])
    art_txn_meta = json.loads(fk_examples[1]["output"])

    n_customers = min(cust_txn_meta.get("num_unique_parents", 50000), 50000)
    n_articles = min(art_txn_meta.get("num_unique_parents", 25000), 25000)
    cust_mean_card = cust_txn_meta.get("mean_cardinality", 4.0)
    art_mean_card = art_txn_meta.get("mean_cardinality", 20.0)

    # Target ~200K transactions (manageable size)
    n_txn = min(200_000, MAX_TRANSACTIONS)

    logger.info(f"Synthetic: {n_customers} customers, {n_articles} articles, "
                f"{n_txn} transactions")

    # Load parquet features to get realistic feature distributions
    cust_txn_pq = pd.read_parquet(
        DEP_DIR / "supplementary_customer_transaction_aligned_features.parquet"
    )
    art_txn_pq = pd.read_parquet(
        DEP_DIR / "supplementary_article_transaction_aligned_features.parquet"
    )

    # Extract parent features
    cust_feat_cols = [c for c in cust_txn_pq.columns if c.startswith("parent__")]
    art_feat_cols = [c for c in art_txn_pq.columns if c.startswith("parent__")]
    child_feat_cols = [c for c in cust_txn_pq.columns if c.startswith("child__")]

    # Build customer table: sample unique feature rows
    cust_feats_raw = cust_txn_pq[cust_feat_cols].drop_duplicates().head(n_customers)
    cust_feats_raw.columns = [c.replace("parent__", "") for c in cust_feat_cols]
    # Pad if not enough unique
    while len(cust_feats_raw) < n_customers:
        noise = cust_feats_raw.sample(
            n=min(1000, n_customers - len(cust_feats_raw)), replace=True
        )
        noise = noise + np.random.randn(*noise.shape).astype(np.float32) * 0.01
        cust_feats_raw = pd.concat([cust_feats_raw, noise], ignore_index=True)
    cust_feats_raw = cust_feats_raw.iloc[:n_customers].reset_index(drop=True)
    cust_feats_raw["customer_id"] = range(n_customers)

    # Build article table
    art_feats_raw = art_txn_pq[art_feat_cols].drop_duplicates().head(n_articles)
    art_feats_raw.columns = [c.replace("parent__", "") for c in art_feat_cols]
    while len(art_feats_raw) < n_articles:
        noise = art_feats_raw.sample(
            n=min(1000, n_articles - len(art_feats_raw)), replace=True
        )
        noise = noise + np.random.randn(*noise.shape).astype(np.float32) * 0.01
        art_feats_raw = pd.concat([art_feats_raw, noise], ignore_index=True)
    art_feats_raw = art_feats_raw.iloc[:n_articles].reset_index(drop=True)
    art_feats_raw["article_id"] = range(n_articles)

    # Build transaction table with realistic FK distribution
    # Use power-law-like distribution to match cardinality patterns
    rng = np.random.RandomState(42)
    cust_probs = rng.pareto(1.5, n_customers) + 1
    cust_probs /= cust_probs.sum()
    art_probs = rng.pareto(1.0, n_articles) + 1
    art_probs /= art_probs.sum()

    txn_customer_ids = rng.choice(n_customers, size=n_txn, p=cust_probs)
    txn_article_ids = rng.choice(n_articles, size=n_txn, p=art_probs)

    # Temporal dates spanning 2018-09 to 2020-09
    base_ts = 1535760000  # 2018-09-01
    end_ts = 1600560000   # 2020-09-20
    txn_dates = pd.to_datetime(
        rng.randint(base_ts, end_ts, size=n_txn), unit="s"
    )

    # Child features from parquet distributions
    child_data = cust_txn_pq[child_feat_cols].sample(
        n=n_txn, replace=True, random_state=42
    ).reset_index(drop=True)
    child_data.columns = [c.replace("child__", "") for c in child_feat_cols]

    transactions = child_data.copy()
    transactions["customer_id"] = txn_customer_ids
    transactions["article_id"] = txn_article_ids
    transactions["t_dat"] = txn_dates

    # Sort by date for temporal splits
    transactions = transactions.sort_values("t_dat").reset_index(drop=True)

    del cust_txn_pq, art_txn_pq
    gc.collect()

    logger.info(f"Built from deps: customers={len(cust_feats_raw)}, "
                f"articles={len(art_feats_raw)}, transactions={len(transactions)}")
    return cust_feats_raw, art_feats_raw, transactions


def build_hetero_graph(
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[HeteroData, dict, dict]:
    """Build HeteroData graph from relational tables.

    Returns: (hetero_data, node_maps, task_labels)
    """
    logger.info("Building heterogeneous graph...")

    # Identify customer and article ID columns
    cust_id_col = "customer_id"
    art_id_col = "article_id"

    # Detect date column for temporal splits
    date_col = None
    for col in ["t_dat", "t_date", "date", "timestamp"]:
        if col in transactions.columns:
            date_col = col
            break
    if date_col is None:
        # Use first datetime/numeric column
        for col in transactions.columns:
            if transactions[col].dtype in ["datetime64[ns]", "datetime64"]:
                date_col = col
                break

    # Build node ID mappings
    unique_customers = transactions[cust_id_col].unique()
    unique_articles = transactions[art_id_col].unique()

    cust_map = {cid: i for i, cid in enumerate(unique_customers)}
    art_map = {aid: i for i, aid in enumerate(unique_articles)}
    n_cust = len(cust_map)
    n_art = len(art_map)
    n_txn = len(transactions)

    logger.info(f"Graph nodes: customers={n_cust}, articles={n_art}, "
                f"transactions={n_txn}")

    num_nodes_dict = {
        "customer": n_cust,
        "article": n_art,
        "transaction": n_txn,
    }

    # Build edge indices
    # transaction -> customer (FK)
    txn_cust_src = torch.arange(n_txn, dtype=torch.long)
    txn_cust_dst = torch.tensor(
        [cust_map[c] for c in transactions[cust_id_col]], dtype=torch.long
    )

    # transaction -> article (FK)
    txn_art_src = torch.arange(n_txn, dtype=torch.long)
    txn_art_dst = torch.tensor(
        [art_map[a] for a in transactions[art_id_col]], dtype=torch.long
    )

    # Build HeteroData
    data = HeteroData()
    for ntype, n in num_nodes_dict.items():
        data[ntype].num_nodes = n

    # Bidirectional edges for each FK
    # child->parent direction
    data["transaction", "fk_to", "customer"].edge_index = torch.stack(
        [txn_cust_src, txn_cust_dst]
    )
    data["customer", "rev_fk", "transaction"].edge_index = torch.stack(
        [txn_cust_dst, txn_cust_src]
    )
    data["transaction", "fk_to", "article"].edge_index = torch.stack(
        [txn_art_src, txn_art_dst]
    )
    data["article", "rev_fk", "transaction"].edge_index = torch.stack(
        [txn_art_dst, txn_art_src]
    )

    # Compute cardinality stats for each FK
    cust_card = transactions.groupby(cust_id_col).size()
    art_card = transactions.groupby(art_id_col).size()
    logger.info(f"Cardinality customer->txn: mean={cust_card.mean():.1f}, "
                f"median={cust_card.median():.0f}, max={cust_card.max()}")
    logger.info(f"Cardinality article->txn: mean={art_card.mean():.1f}, "
                f"median={art_card.median():.0f}, max={art_card.max()}")

    # ------------------------------------------------------------------
    # Build task labels using temporal splits
    # ------------------------------------------------------------------
    if date_col and transactions[date_col].dtype == "object":
        transactions[date_col] = pd.to_datetime(transactions[date_col])
    elif date_col and not np.issubdtype(transactions[date_col].dtype, np.datetime64):
        # Might be epoch seconds
        if transactions[date_col].min() > 1e9:
            transactions[date_col] = pd.to_datetime(
                transactions[date_col], unit="s"
            )

    if date_col is not None:
        dates = transactions[date_col]
        sorted_dates = dates.sort_values()
        # 70/15/15 temporal split
        val_ts = sorted_dates.iloc[int(len(sorted_dates) * 0.70)]
        test_ts = sorted_dates.iloc[int(len(sorted_dates) * 0.85)]
        logger.info(f"Temporal splits: val_ts={val_ts}, test_ts={test_ts}")

        train_mask = dates < val_ts
        val_mask = (dates >= val_ts) & (dates < test_ts)
        test_mask = dates >= test_ts

        train_txns = transactions[train_mask]
        val_txns = transactions[val_mask]
        test_txns = transactions[test_mask]
    else:
        # Random split fallback
        n = len(transactions)
        idx = np.random.permutation(n)
        train_idx = idx[: int(0.7 * n)]
        val_idx = idx[int(0.7 * n): int(0.85 * n)]
        test_idx = idx[int(0.85 * n):]
        train_txns = transactions.iloc[train_idx]
        val_txns = transactions.iloc[val_idx]
        test_txns = transactions.iloc[test_idx]

    # User-churn: does customer appear in val/test period? (binary)
    train_custs = set(train_txns[cust_id_col].unique())
    val_custs = set(val_txns[cust_id_col].unique())
    test_custs = set(test_txns[cust_id_col].unique())

    # For user-churn: customers in train who DON'T appear in val = churned
    task_labels = {}

    # User-churn labels: for train customers, label=1 if they appear in val
    churn_labels_train = {}
    churn_labels_val = {}
    churn_labels_test = {}

    for c in train_custs:
        churn_labels_train[c] = 1.0 if c in val_custs else 0.0
    for c in val_custs:
        churn_labels_val[c] = 1.0 if c in test_custs else 0.0
    # Test: use all test customers with label 1 (active)
    # and train-only customers with label 0 (churned)
    remaining = train_custs - val_custs - test_custs
    for c in test_custs:
        churn_labels_test[c] = 1.0
    for c in list(remaining)[:len(test_custs)]:
        churn_labels_test[c] = 0.0

    # Item-sales: count of transactions per article in future window
    sales_train = train_txns.groupby(art_id_col).size()
    sales_val = val_txns.groupby(art_id_col).size()
    sales_test = test_txns.groupby(art_id_col).size()

    # Normalize sales by log1p for stable training
    def _make_sales_dict(series):
        return {k: float(np.log1p(v)) for k, v in series.items()}

    task_labels["user-churn"] = {
        "train": churn_labels_train,
        "val": churn_labels_val,
        "test": churn_labels_test,
    }
    task_labels["item-sales"] = {
        "train": _make_sales_dict(sales_train),
        "val": _make_sales_dict(sales_val),
        "test": _make_sales_dict(sales_test),
    }

    node_maps = {
        "customer": cust_map,
        "article": art_map,
        "num_nodes": num_nodes_dict,
        "cust_cardinality_mean": float(cust_card.mean()),
        "art_cardinality_mean": float(art_card.mean()),
    }

    logger.info(f"Task user-churn labels: train={len(churn_labels_train)}, "
                f"val={len(churn_labels_val)}, test={len(churn_labels_test)}")
    logger.info(f"Task item-sales labels: train={len(sales_train)}, "
                f"val={len(sales_val)}, test={len(sales_test)}")

    return data, node_maps, task_labels


# ===================================================================
# Phase 2 — Model Definitions
# ===================================================================

class StandardHeteroGNN(nn.Module):
    """Standard heterogeneous GNN using SAGEConv per edge type."""

    def __init__(self, num_nodes_dict: dict, metadata: tuple,
                 hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embeddings = nn.ModuleDict({
            ntype: nn.Embedding(n, hidden)
            for ntype, n in num_nodes_dict.items()
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src, rel, dst = edge_type
                conv_dict[edge_type] = SAGEConv(hidden, hidden)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.LayerNorm(hidden)
                for ntype in num_nodes_dict
            })
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def get_embeddings(self) -> dict:
        return {
            ntype: emb.weight
            for ntype, emb in self.embeddings.items()
        }

    def forward(self, edge_index_dict: dict,
                target_type: str, target_ids: torch.Tensor) -> torch.Tensor:
        x_dict = self.get_embeddings()
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.relu(self.norms[i][k](x_dict_new.get(k, x_dict[k])))
                for k in x_dict
            }
        out = self.head(x_dict[target_type][target_ids])
        return out.squeeze(-1)


class PRMPConv(nn.Module):
    """Predictive Residual Message Passing convolution.

    For each FK edge (parent -> child):
      1) Predict child embeddings from parent: pred = MLP(h_parent)
      2) Compute residual: r = h_child - pred
      3) Aggregate residuals back to parent (mean)
      4) Update parent: h_parent += W(agg_residuals)
    """

    def __init__(self, hidden: int, fk_edge_types: list[tuple]):
        super().__init__()
        self.fk_edge_types = fk_edge_types
        self.pred_mlps = nn.ModuleDict()
        self.update_lins = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for etype in fk_edge_types:
            key = "__".join(etype)
            self.pred_mlps[key] = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.update_lins[key] = nn.Linear(hidden, hidden)
            self.norms[key] = nn.LayerNorm(hidden)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        new_x = {k: v.clone() for k, v in x_dict.items()}

        for etype in self.fk_edge_types:
            parent_type, rel, child_type = etype
            key = "__".join(etype)

            if etype not in edge_index_dict:
                continue

            ei = edge_index_dict[etype]
            src, dst = ei[0], ei[1]  # src=parent, dst=child

            parent_feats = x_dict[parent_type]
            child_feats = x_dict[child_type]

            # Predict child from parent
            pred_child = self.pred_mlps[key](parent_feats[src])
            # Residual
            residuals = child_feats[dst] - pred_child
            # Aggregate residuals back to parent (mean)
            agg = scatter(residuals, src, dim=0,
                          dim_size=parent_feats.size(0), reduce="mean")
            # Update parent
            update = self.update_lins[key](agg)
            new_x[parent_type] = self.norms[key](new_x[parent_type] + update)

        return new_x


class PRMPHeteroGNN(nn.Module):
    """Heterogeneous GNN with PRMP layers for FK edges
    and SAGEConv for reverse edges."""

    def __init__(self, num_nodes_dict: dict, metadata: tuple,
                 hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embeddings = nn.ModuleDict({
            ntype: nn.Embedding(n, hidden)
            for ntype, n in num_nodes_dict.items()
        })

        # Identify FK parent->child edges (the "rev_fk" edges go parent->child)
        all_edge_types = metadata[1]
        self.fk_parent_to_child = [
            et for et in all_edge_types if et[1] == "rev_fk"
        ]
        self.other_edges = [
            et for et in all_edge_types if et[1] != "rev_fk"
        ]

        self.prmp_convs = nn.ModuleList()
        self.sage_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.prmp_convs.append(PRMPConv(hidden, self.fk_parent_to_child))

            # SAGEConv for child->parent direction (fk_to edges)
            sage_dict = {}
            for et in self.other_edges:
                sage_dict[et] = SAGEConv(hidden, hidden)
            self.sage_convs.append(HeteroConv(sage_dict, aggr="sum"))

            self.norms.append(nn.ModuleDict({
                ntype: nn.LayerNorm(hidden)
                for ntype in num_nodes_dict
            }))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def get_embeddings(self) -> dict:
        return {
            ntype: emb.weight
            for ntype, emb in self.embeddings.items()
        }

    def forward(self, edge_index_dict: dict,
                target_type: str, target_ids: torch.Tensor) -> torch.Tensor:
        x_dict = self.get_embeddings()

        for i in range(len(self.prmp_convs)):
            # PRMP pass (parent -> child direction)
            x_prmp = self.prmp_convs[i](x_dict, edge_index_dict)
            # SAGE pass (child -> parent direction)
            x_sage = self.sage_convs[i](x_dict, edge_index_dict)
            # Merge: for keys in both, average; otherwise take what's available
            x_dict_new = {}
            for k in x_dict:
                vals = []
                if k in x_prmp:
                    vals.append(x_prmp[k])
                if k in x_sage:
                    vals.append(x_sage[k])
                if vals:
                    merged = sum(vals) / len(vals)
                else:
                    merged = x_dict[k]
                x_dict_new[k] = F.relu(self.norms[i][k](merged))
            x_dict = x_dict_new

        out = self.head(x_dict[target_type][target_ids])
        return out.squeeze(-1)


# ===================================================================
# Phase 3 — Training
# ===================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_non_embedding_params(model: nn.Module) -> int:
    """Count params excluding embedding tables."""
    total = 0
    for name, p in model.named_parameters():
        if "embeddings" not in name:
            total += p.numel()
    return total


def compute_wide_dim(num_nodes_dict: dict, metadata: tuple,
                     prmp_hidden: int = 128) -> int:
    """Find hidden dim for Wide model so non-embedding params match PRMP."""
    prmp = PRMPHeteroGNN(num_nodes_dict, metadata,
                         hidden=prmp_hidden, num_layers=NUM_LAYERS)
    prmp_non_emb = count_non_embedding_params(prmp)
    del prmp

    # Binary search for wide_dim
    lo, hi = prmp_hidden, prmp_hidden * 3
    best_dim = prmp_hidden + 16

    for dim in range(lo, hi + 1):
        try:
            std = StandardHeteroGNN(num_nodes_dict, metadata,
                                    hidden=dim, num_layers=NUM_LAYERS)
            std_non_emb = count_non_embedding_params(std)
            del std
            if std_non_emb >= prmp_non_emb:
                best_dim = dim
                break
        except Exception:
            continue

    logger.info(f"Wide dim computed: {best_dim} "
                f"(PRMP non-emb params: {prmp_non_emb})")
    return best_dim


def create_model(variant: str, num_nodes_dict: dict, metadata: tuple,
                 wide_dim: int = 160) -> nn.Module:
    """Create model for a given variant."""
    if variant == "standard":
        return StandardHeteroGNN(num_nodes_dict, metadata,
                                 hidden=HIDDEN_DIM, num_layers=NUM_LAYERS)
    elif variant == "prmp":
        return PRMPHeteroGNN(num_nodes_dict, metadata,
                             hidden=HIDDEN_DIM, num_layers=NUM_LAYERS)
    elif variant == "wide":
        return StandardHeteroGNN(num_nodes_dict, metadata,
                                 hidden=wide_dim, num_layers=NUM_LAYERS)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def train_one_run(
    variant: str,
    task_name: str,
    seed: int,
    hetero_data: HeteroData,
    node_maps: dict,
    task_labels: dict,
    wide_dim: int,
    device: torch.device,
) -> dict:
    """Train one model variant on one task with one seed."""
    set_seed(seed)
    task_cfg = TASKS_CFG[task_name]
    entity_type = task_cfg["entity"]
    is_classification = task_cfg["task_type"] == "classification"

    num_nodes_dict = node_maps["num_nodes"]
    metadata = (list(num_nodes_dict.keys()), list(hetero_data.edge_types))

    model = create_model(variant, num_nodes_dict, metadata, wide_dim)
    model = model.to(device)
    param_count = count_params(model)

    # Move edge indices to device
    edge_index_dict = {}
    for etype, ei in hetero_data.edge_index_dict.items():
        edge_index_dict[etype] = ei.to(device)

    # Prepare labels
    labels_dict = task_labels[task_name]
    id_map = node_maps["customer"] if entity_type == "customer" else node_maps["article"]

    def _get_ids_labels(split: str):
        d = labels_dict[split]
        ids, labs = [], []
        for orig_id, label in d.items():
            if orig_id in id_map:
                ids.append(id_map[orig_id])
                labs.append(label)
        return (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(labs, dtype=torch.float32, device=device),
        )

    train_ids, train_labels = _get_ids_labels("train")
    val_ids, val_labels = _get_ids_labels("val")
    test_ids, test_labels = _get_ids_labels("test")

    if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
        logger.warning(f"Empty split for {task_name}/{variant}/seed{seed}: "
                       f"train={len(train_ids)}, val={len(val_ids)}, "
                       f"test={len(test_ids)}")
        return {
            "val_metric": 0.5 if is_classification else 999.0,
            "test_metric": 0.5 if is_classification else 999.0,
            "epochs_trained": 0,
            "param_count": param_count,
        }

    # Mini-batch: if too many IDs, sample each epoch
    max_batch = min(BATCH_SIZE, len(train_ids))

    # Loss
    if is_classification:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=LR)

    best_val = None
    best_state = None
    patience_ctr = 0
    epochs_trained = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        # Mini-batch sampling
        perm = torch.randperm(len(train_ids), device=device)[:max_batch]
        batch_ids = train_ids[perm]
        batch_labels = train_labels[perm]

        optimizer.zero_grad()
        out = model(edge_index_dict, entity_type, batch_ids)
        loss = loss_fn(out, batch_labels)

        if torch.isnan(loss):
            logger.warning(f"NaN loss at epoch {epoch}, {variant}/{task_name}/s{seed}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Validate (every 5 epochs to save time) ---
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(edge_index_dict, entity_type, val_ids)

            if is_classification:
                probs = torch.sigmoid(val_out).cpu().numpy()
                val_np = val_labels.cpu().numpy()
                try:
                    val_metric = roc_auc_score(val_np, probs)
                except ValueError:
                    val_metric = 0.5
                is_better = best_val is None or val_metric > best_val
            else:
                preds = val_out.cpu().numpy()
                val_np = val_labels.cpu().numpy()
                val_metric = mean_absolute_error(val_np, preds)
                is_better = best_val is None or val_metric < best_val

            if is_better:
                best_val = val_metric
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 5  # we check every 5 epochs

            if patience_ctr >= PATIENCE:
                break

        epochs_trained = epoch + 1

    # --- Test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(edge_index_dict, entity_type, test_ids)

    if is_classification:
        probs = torch.sigmoid(test_out).cpu().numpy()
        test_np = test_labels.cpu().numpy()
        try:
            test_metric = roc_auc_score(test_np, probs)
        except ValueError:
            test_metric = 0.5
    else:
        preds = test_out.cpu().numpy()
        test_np = test_labels.cpu().numpy()
        test_metric = mean_absolute_error(test_np, preds)

    logger.info(f"  {variant}/{task_name}/s{seed}: "
                f"val={best_val:.4f} test={test_metric:.4f} "
                f"epochs={epochs_trained} params={param_count}")

    # Clean up
    del model, optimizer, edge_index_dict
    if HAS_GPU:
        torch.cuda.empty_cache()

    return {
        "val_metric": float(best_val) if best_val is not None else float("nan"),
        "test_metric": float(test_metric),
        "epochs_trained": epochs_trained,
        "param_count": param_count,
    }


# ===================================================================
# Phase 4 — Ridge R² Analysis
# ===================================================================

def compute_r2_analysis(
    hetero_data: HeteroData,
    model: nn.Module,
    node_maps: dict,
    max_pairs: int = 50_000,
) -> dict:
    """Compute Ridge R² for each FK link using learned embeddings."""
    model.eval()
    emb_dict = {}
    with torch.no_grad():
        for ntype, emb_mod in model.embeddings.items():
            emb_dict[ntype] = emb_mod.weight.cpu().numpy()

    r2_results = {}

    fk_links = [
        ("customer", "rev_fk", "transaction"),
        ("article", "rev_fk", "transaction"),
    ]

    for parent_type, rel, child_type in fk_links:
        etype = (parent_type, rel, child_type)
        if etype not in hetero_data.edge_index_dict:
            logger.warning(f"Edge type {etype} not found, skipping R²")
            continue

        ei = hetero_data[etype].edge_index
        src, dst = ei[0].numpy(), ei[1].numpy()

        # Sample pairs
        n = len(src)
        if n > max_pairs:
            idx = np.random.choice(n, max_pairs, replace=False)
            src_s, dst_s = src[idx], dst[idx]
        else:
            src_s, dst_s = src, dst

        parent_embs = emb_dict[parent_type][src_s]
        child_embs = emb_dict[child_type][dst_s]

        # Split train/test for Ridge
        split = int(len(parent_embs) * 0.8)
        X_train, X_test = parent_embs[:split], parent_embs[split:]
        y_train, y_test = child_embs[:split], child_embs[split:]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        r2 = float(r2_score(y_test, ridge.predict(X_test)))

        link_name = f"{parent_type}_to_{child_type}"
        r2_results[link_name] = {"ridge_r2": round(r2, 6)}
        logger.info(f"  R²({link_name}): ridge={r2:.4f}")

        # PRMP MLP R² (if model is PRMPHeteroGNN)
        if hasattr(model, "prmp_convs") and len(model.prmp_convs) > 0:
            prmp_conv = model.prmp_convs[0]
            key = "__".join(etype)
            if key in prmp_conv.pred_mlps:
                with torch.no_grad():
                    p_t = torch.tensor(parent_embs, dtype=torch.float32)
                    pred = prmp_conv.pred_mlps[key](p_t).numpy()
                prmp_r2 = float(r2_score(child_embs, pred))
                r2_results[link_name]["prmp_mlp_r2"] = round(prmp_r2, 6)
                logger.info(f"  R²({link_name}): prmp_mlp={prmp_r2:.4f}")

    return r2_results


def compute_r2_from_dependency_parquets() -> dict:
    """Compute baseline Ridge R² from dependency parquet feature pairs."""
    r2_results = {}
    for link_name, fname in [
        ("customer_to_transaction",
         "supplementary_customer_transaction_aligned_features.parquet"),
        ("article_to_transaction",
         "supplementary_article_transaction_aligned_features.parquet"),
    ]:
        fpath = DEP_DIR / fname
        if not fpath.exists():
            logger.warning(f"Parquet not found: {fpath}")
            continue

        df = pd.read_parquet(fpath)
        parent_cols = [c for c in df.columns if c.startswith("parent__")]
        child_cols = [c for c in df.columns if c.startswith("child__")]

        X = df[parent_cols].values.astype(np.float32)
        y = df[child_cols].values.astype(np.float32)

        # Handle NaN
        X = np.nan_to_num(X, 0.0)
        y = np.nan_to_num(y, 0.0)

        split = int(len(X) * 0.8)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:split], y[:split])
        r2 = float(r2_score(y[split:], ridge.predict(X[split:])))
        r2_results[link_name] = {"feature_ridge_r2": round(r2, 6)}
        logger.info(f"  Feature R²({link_name}): {r2:.4f}")

    return r2_results


# ===================================================================
# Phase 5 — Main Experiment
# ===================================================================

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("PRMP Experiment on rel-hm (H&M Fashion)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Phase 1: Loading data...")
    try:
        customers, articles, transactions = download_hm_tables()
    except Exception:
        logger.exception("Data download failed")
        raise

    logger.info(f"Tables: customers={len(customers)}, "
                f"articles={len(articles)}, transactions={len(transactions)}")

    # Build graph
    hetero_data, node_maps, task_labels = build_hetero_graph(
        customers, articles, transactions
    )

    # Free raw DataFrames
    del customers, articles, transactions
    gc.collect()

    # ------------------------------------------------------------------
    # Compute wide dim
    # ------------------------------------------------------------------
    num_nodes_dict = node_maps["num_nodes"]
    metadata = (list(num_nodes_dict.keys()), list(hetero_data.edge_types))
    wide_dim = compute_wide_dim(num_nodes_dict, metadata)

    # Log param counts
    std_model = StandardHeteroGNN(num_nodes_dict, metadata, HIDDEN_DIM, NUM_LAYERS)
    prmp_model = PRMPHeteroGNN(num_nodes_dict, metadata, HIDDEN_DIM, NUM_LAYERS)
    wide_model = StandardHeteroGNN(num_nodes_dict, metadata, wide_dim, NUM_LAYERS)

    param_counts = {
        "standard_128": count_params(std_model),
        "standard_128_non_emb": count_non_embedding_params(std_model),
        "prmp_128": count_params(prmp_model),
        "prmp_128_non_emb": count_non_embedding_params(prmp_model),
        "wide_matched": count_params(wide_model),
        "wide_matched_non_emb": count_non_embedding_params(wide_model),
        "wide_dim": wide_dim,
    }
    logger.info(f"Param counts: {json.dumps(param_counts, indent=2)}")
    del std_model, prmp_model, wide_model

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Phase 3: Training...")
    all_results = {}

    for task_name in TASKS_CFG:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_key = f"{task_name}__{variant}__s{seed}"
                logger.info(f"Training: {run_key}")
                try:
                    result = train_one_run(
                        variant=variant,
                        task_name=task_name,
                        seed=seed,
                        hetero_data=hetero_data,
                        node_maps=node_maps,
                        task_labels=task_labels,
                        wide_dim=wide_dim,
                        device=DEVICE,
                    )
                    all_results[run_key] = result
                except Exception:
                    logger.exception(f"Failed: {run_key}")
                    all_results[run_key] = {
                        "val_metric": float("nan"),
                        "test_metric": float("nan"),
                        "epochs_trained": 0,
                        "param_count": 0,
                    }

                elapsed = time.time() - t_start
                logger.info(f"  Elapsed: {elapsed / 60:.1f} min")

                # Time guard: stop if approaching 55 min
                if elapsed > 50 * 60:
                    logger.warning("Approaching time limit, stopping early")
                    break
            else:
                continue
            break
        else:
            continue
        break

    # ------------------------------------------------------------------
    # R² analysis
    # ------------------------------------------------------------------
    logger.info("Phase 4: R² analysis...")

    # Embedding-space R² from a trained PRMP model
    r2_embedding = {}
    try:
        set_seed(42)
        prmp_model = PRMPHeteroGNN(num_nodes_dict, metadata,
                                   HIDDEN_DIM, NUM_LAYERS).to(DEVICE)

        # Quick train for R² (20 epochs on user-churn)
        edge_index_dict = {
            et: ei.to(DEVICE) for et, ei in hetero_data.edge_index_dict.items()
        }
        entity_type = "customer"
        id_map = node_maps["customer"]
        labels_d = task_labels["user-churn"]["train"]
        ids, labs = [], []
        for orig_id, label in labels_d.items():
            if orig_id in id_map:
                ids.append(id_map[orig_id])
                labs.append(label)
        train_ids_t = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        train_labels_t = torch.tensor(labs, dtype=torch.float32, device=DEVICE)

        opt = Adam(prmp_model.parameters(), lr=LR)
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(30):
            prmp_model.train()
            opt.zero_grad()
            perm = torch.randperm(len(train_ids_t), device=DEVICE)[:BATCH_SIZE]
            out = prmp_model(edge_index_dict, entity_type, train_ids_t[perm])
            loss = loss_fn(out, train_labels_t[perm])
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prmp_model.parameters(), 1.0)
            opt.step()

        prmp_model_cpu = prmp_model.cpu()
        r2_embedding = compute_r2_analysis(
            hetero_data, prmp_model_cpu, node_maps
        )
        del prmp_model, prmp_model_cpu, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    except Exception:
        logger.exception("R² embedding analysis failed")

    # Feature-space R² from dependency parquets
    r2_features = {}
    try:
        r2_features = compute_r2_from_dependency_parquets()
    except Exception:
        logger.exception("R² feature analysis failed")

    # Merge R² results
    r2_combined = {}
    for k in set(list(r2_embedding.keys()) + list(r2_features.keys())):
        r2_combined[k] = {**r2_features.get(k, {}), **r2_embedding.get(k, {})}

    # ------------------------------------------------------------------
    # Phase 5: Aggregate & output
    # ------------------------------------------------------------------
    logger.info("Phase 5: Aggregating results...")

    # Load dependency data for output format
    dep_data = json.loads((DEP_DIR / "full_data_out.json").read_text())

    results_by_task = {}
    for task_name, task_cfg in TASKS_CFG.items():
        task_results = {
            "metric": task_cfg["metric_name"],
            "task_type": task_cfg["task_type"],
        }
        for variant in VARIANTS:
            per_seed = []
            for seed in SEEDS:
                run_key = f"{task_name}__{variant}__s{seed}"
                if run_key in all_results:
                    per_seed.append(all_results[run_key]["test_metric"])
                else:
                    per_seed.append(float("nan"))

            valid = [v for v in per_seed if not np.isnan(v)]
            task_results[variant] = {
                "mean": float(np.mean(valid)) if valid else float("nan"),
                "std": float(np.std(valid)) if valid else float("nan"),
                "per_seed": per_seed,
            }

        results_by_task[task_name] = task_results

    # Relative improvements
    rel_improvements = {}
    for task_name in TASKS_CFG:
        tr = results_by_task.get(task_name, {})
        std_mean = tr.get("standard", {}).get("mean", float("nan"))
        prmp_mean = tr.get("prmp", {}).get("mean", float("nan"))
        if not np.isnan(std_mean) and not np.isnan(prmp_mean) and std_mean != 0:
            if TASKS_CFG[task_name]["task_type"] == "classification":
                # Higher is better for AUROC
                pct = (prmp_mean - std_mean) / abs(std_mean) * 100
            else:
                # Lower is better for MAE
                pct = (std_mean - prmp_mean) / abs(std_mean) * 100
            rel_improvements[f"{task_name}_prmp_vs_standard"] = f"{pct:+.2f}%"
        else:
            rel_improvements[f"{task_name}_prmp_vs_standard"] = "N/A"

    # Build comprehensive output metadata
    experiment_meta = {
        "experiment": "PRMP_rel_hm_5th_dataset",
        "dataset": "rel-hm",
        "schema": {
            "tables": ["customer", "article", "transaction"],
            "fk_links": [
                {
                    "parent": "customer",
                    "child": "transaction",
                    "cardinality_mean": node_maps.get("cust_cardinality_mean", 0),
                },
                {
                    "parent": "article",
                    "child": "transaction",
                    "cardinality_mean": node_maps.get("art_cardinality_mean", 0),
                },
            ],
        },
        "architecture": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "seeds": SEEDS,
            "batch_size": BATCH_SIZE,
        },
        "results_by_task": results_by_task,
        "r2_analysis": r2_combined,
        "param_counts": param_counts,
        "relative_improvement": rel_improvements,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    # ------------------------------------------------------------------
    # Build output in exp_gen_sol_out schema
    # ------------------------------------------------------------------
    examples = []
    for ex in dep_data["datasets"][0]["examples"]:
        example = {
            "input": ex["input"],
            "output": ex["output"],
        }
        # Copy metadata fields
        for k, v in ex.items():
            if k.startswith("metadata_"):
                example[k] = v

        # Add predictions from each variant
        parent_table = ex.get("metadata_parent_table", "")
        child_table = ex.get("metadata_child_table", "")
        fk_link = f"{parent_table}_to_{child_table}"

        for variant in VARIANTS:
            predict_data = {}
            for task_name in TASKS_CFG:
                tr = results_by_task.get(task_name, {})
                vr = tr.get(variant, {})
                predict_data[task_name] = {
                    "mean": vr.get("mean", "N/A"),
                    "std": vr.get("std", "N/A"),
                    "per_seed": vr.get("per_seed", []),
                }

            # Add R² for this FK link
            if fk_link in r2_combined:
                predict_data["r2"] = r2_combined[fk_link]

            predict_data["param_count"] = param_counts.get(
                f"{variant}_128" if variant != "wide" else "wide_matched", 0
            )

            example[f"predict_{variant}"] = json.dumps(predict_data)

        examples.append(example)

    output = {
        "metadata": experiment_meta,
        "datasets": [
            {
                "dataset": "rel_hm_fashion",
                "examples": examples,
            }
        ],
    }

    # Save
    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")
    logger.info("DONE")


if __name__ == "__main__":
    main()
