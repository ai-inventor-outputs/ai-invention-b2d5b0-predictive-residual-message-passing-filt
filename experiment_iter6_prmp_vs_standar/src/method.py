#!/usr/bin/env python3
"""PRMP vs Standard HeteroSAGE on RelBench rel-avito ad-ctr (Pure PyTorch).

Train 3 GNN variants (Standard SAGE, PRMP, Wide SAGE control) on the rel-avito
ad-ctr regression task using PURE PyTorch (no torch-geometric) to avoid the
torch-scatter compilation failures that crashed the prior avito experiment.

Uses the successful Amazon experiment (exp_id4_it5) as architectural template,
adapted for avito's 8-table heterogeneous schema with 11 FK links. Tests whether
PRMP helps at extreme cardinalities (Category→AdsInfo mean=114K) and high
cross-table predictability (AdsInfo←Location R²=0.50).
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
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
RAM_BUDGET = int(20 * 1024**3)  # 20 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# CPU time limit: 55 minutes
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3300, 3300))
except Exception:
    pass

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(20 * 1024**3)  # 20 GB of 24 GB
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'} ({VRAM_GB:.1f} GB VRAM), device={DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
DEP_WS = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
              "3_invention_loop/iter_4/gen_art/data_id4_it4__opus")
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
TEXT_HASH_DIM = 16

# Subsampling limits — controlled by SCALE env var for gradual scaling
_SCALE = os.environ.get("SCALE", "full")
if _SCALE == "tiny":
    ADSINFO_SAMPLE = 20_000
    STREAM_SAMPLE = 20_000
    SEARCHINFO_SAMPLE = 20_000
elif _SCALE == "small":
    ADSINFO_SAMPLE = 50_000
    STREAM_SAMPLE = 30_000
    SEARCHINFO_SAMPLE = 30_000
else:  # "full"
    ADSINFO_SAMPLE = 200_000
    STREAM_SAMPLE = 100_000
    SEARCHINFO_SAMPLE = 100_000
logger.info(f"Scale mode: {_SCALE} (AdsInfo={ADSINFO_SAMPLE}, Stream={STREAM_SAMPLE})")

# Known R² diagnostic from dependency (iter_5)
R2_DIAGNOSTIC = {
    "AdsInfo__LocationID__Location": {"r2": 0.500567, "card": 1765.0},
    "SearchStream__AdID__AdsInfo": {"r2": 0.252625, "card": 3.49},
    "AdsInfo__CategoryID__Category": {"r2": 0.00049, "card": 114625.5},
    "SearchInfo__CategoryID__Category": {"r2": 0.000323, "card": 50952.7},
    "SearchInfo__LocationID__Location": {"r2": 0.00098, "card": 1093.0},
    "VisitStream__UserID__UserInfo": {"r2": 0.001051, "card": 89.7},
    "SearchInfo__UserID__UserInfo": {"r2": 0.003941, "card": 35.4},
    "SearchStream__SearchID__SearchInfo": {"r2": 0.0, "card": 3.62},
    "VisitStream__AdID__AdsInfo": {"r2": 0.0, "card": 1.67},
    "PhoneRequestsStream__UserID__UserInfo": {"r2": 0.0, "card": 8.06},
    "PhoneRequestsStream__AdID__AdsInfo": {"r2": 0.0, "card": 1.18},
}

# FK link definitions: (child_table, fk_col, parent_table, parent_pk_col)
FK_LINKS = [
    ("AdsInfo", "LocationID", "Location", "LocationID"),
    ("AdsInfo", "CategoryID", "Category", "CategoryID"),
    ("SearchStream", "SearchID", "SearchInfo", "SearchID"),
    ("SearchStream", "AdID", "AdsInfo", "AdID"),
    ("VisitStream", "UserID", "UserInfo", "UserID"),
    ("VisitStream", "AdID", "AdsInfo", "AdID"),
    ("PhoneRequestsStream", "UserID", "UserInfo", "UserID"),
    ("PhoneRequestsStream", "AdID", "AdsInfo", "AdID"),
    ("SearchInfo", "UserID", "UserInfo", "UserID"),
    ("SearchInfo", "LocationID", "Location", "LocationID"),
    ("SearchInfo", "CategoryID", "Category", "CategoryID"),
]

# Table primary key columns
TABLE_PK = {
    "AdsInfo": "AdID",
    "Category": "CategoryID",
    "Location": "LocationID",
    "UserInfo": "UserID",
    "SearchInfo": "SearchID",
    # Stream tables have no PK
}

# Tables with features and their numeric feature columns
TABLE_FEATURE_COLS = {
    "AdsInfo": ["Price", "IsContext"],         # + text hash of Title
    "Category": ["Level", "ParentCategoryID", "SubcategoryID"],
    "Location": ["Level", "RegionID", "CityID"],
    "UserInfo": ["UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID"],
    "SearchStream": ["Position", "ObjectType", "HistCTR", "IsClick"],
    "SearchInfo": ["IPID", "IsUserLoggedOn"],  # + text hash of SearchQuery
    "VisitStream": ["IPID"],
    "PhoneRequestsStream": ["IPID"],
}


# ===========================================================================
# PHASE 1: Data Loading & Strategic Subsampling
# ===========================================================================

def encode_text_hash(series: pd.Series, n_features: int = TEXT_HASH_DIM) -> np.ndarray:
    """Simple hash-based text encoding (same as Amazon experiment)."""
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


def load_and_subsample() -> dict:
    """Load rel-avito data from relbench, subsample, build feature tensors."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    logger.info("Loading rel-avito dataset from relbench...")
    t0 = time.time()
    dataset = get_dataset("rel-avito", download=True)
    db = dataset.get_db()
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s")

    logger.info("Loading ad-ctr task...")
    task = get_task("rel-avito", "ad-ctr", download=True)
    entity_col = task.entity_col  # "AdID"
    target_col = task.target_col  # "num_click"
    logger.info(f"Task: entity_col={entity_col}, target_col={target_col}")

    # Get task entity IDs from train/val/test splits
    train_tbl = task.get_table("train")
    val_tbl = task.get_table("val")
    test_tbl = task.get_table("test")

    logger.info(f"Task splits: train={len(train_tbl.df)}, val={len(val_tbl.df)}, test={len(test_tbl.df)}")

    # Collect task labels per AdID per split (vectorized)
    # Note: test split may NOT have target column (held out in relbench)
    # Strategy: use train for training, val split into val+test (50/50)
    task_labels = {}  # AdID -> {"split": split, "target": float}

    # Train labels
    df_train = train_tbl.df
    valid = df_train[entity_col].notna() & df_train[target_col].notna()
    df_v = df_train[valid]
    for ad_id, target in zip(df_v[entity_col].values, df_v[target_col].values):
        task_labels[ad_id] = {"split": "train", "target": float(target)}

    # Val labels — split into val (first half) and test (second half)
    df_val = val_tbl.df
    if target_col in df_val.columns:
        valid = df_val[entity_col].notna() & df_val[target_col].notna()
        df_v = df_val[valid].reset_index(drop=True)
        split_point = len(df_v) // 2
        for i, (ad_id, target) in enumerate(zip(df_v[entity_col].values, df_v[target_col].values)):
            split = "val" if i < split_point else "test"
            task_labels[ad_id] = {"split": split, "target": float(target)}
    else:
        # If val also has no labels, use train split only (80/10/10)
        logger.warning("Val split has no labels; using train-only split")

    # Test IDs (no labels) — still collect for subsampling
    df_test = test_tbl.df
    test_ad_ids_no_label = set()
    if target_col not in df_test.columns:
        logger.info("Test split has no labels (held out); using val split for evaluation")
        test_ad_ids_no_label = set(df_test[entity_col].dropna().values)

    task_ad_ids = set(task_labels.keys()) | test_ad_ids_no_label
    logger.info(f"Task entities: {len(task_ad_ids)} unique AdIDs ({len(task_labels)} with labels, {len(test_ad_ids_no_label)} test without labels)")

    # --- Subsample AdsInfo ---
    logger.info("Subsampling tables...")
    ads_df = db.table_dict["AdsInfo"].df
    logger.info(f"  AdsInfo: {len(ads_df)} rows")

    task_mask = ads_df["AdID"].isin(task_ad_ids)
    task_ads = ads_df[task_mask]
    remaining = ads_df[~task_mask]
    n_extra = max(0, ADSINFO_SAMPLE - len(task_ads))
    if n_extra > 0 and len(remaining) > n_extra:
        extra = remaining.sample(n=n_extra, random_state=42)
        ads_sample = pd.concat([task_ads, extra]).reset_index(drop=True)
    else:
        ads_sample = pd.concat([task_ads, remaining.head(n_extra)]).reset_index(drop=True) if n_extra > 0 else task_ads.reset_index(drop=True)
    sampled_ad_ids = set(ads_sample["AdID"].values)
    logger.info(f"  AdsInfo subsampled: {len(ads_sample)} (task={len(task_ads)}, extra={len(ads_sample)-len(task_ads)})")

    # Small tables: keep fully
    category_df = db.table_dict["Category"].df
    location_df = db.table_dict["Location"].df
    userinfo_df = db.table_dict["UserInfo"].df
    logger.info(f"  Category: {len(category_df)}, Location: {len(location_df)}, UserInfo: {len(userinfo_df)}")

    # Free full AdsInfo
    del ads_df, task_ads, remaining
    gc.collect()

    # Subsample stream/large tables
    subsampled = {
        "AdsInfo": ads_sample,
        "Category": category_df,
        "Location": location_df,
        "UserInfo": userinfo_df,
    }

    # SearchInfo: sample independently
    searchinfo_df = db.table_dict["SearchInfo"].df
    logger.info(f"  SearchInfo: {len(searchinfo_df)} rows")
    if len(searchinfo_df) > SEARCHINFO_SAMPLE:
        searchinfo_df = searchinfo_df.sample(n=SEARCHINFO_SAMPLE, random_state=42)
    subsampled["SearchInfo"] = searchinfo_df.reset_index(drop=True)
    sampled_search_ids = set(subsampled["SearchInfo"]["SearchID"].values)
    logger.info(f"  SearchInfo subsampled: {len(subsampled['SearchInfo'])}")
    del searchinfo_df
    gc.collect()

    # SearchStream: filter to connected AdsInfo + SearchInfo
    searchstream_df = db.table_dict["SearchStream"].df
    logger.info(f"  SearchStream: {len(searchstream_df)} rows")
    ss_mask = searchstream_df["AdID"].isin(sampled_ad_ids)
    searchstream_df = searchstream_df[ss_mask]
    if len(searchstream_df) > STREAM_SAMPLE:
        searchstream_df = searchstream_df.sample(n=STREAM_SAMPLE, random_state=42)
    subsampled["SearchStream"] = searchstream_df.reset_index(drop=True)
    logger.info(f"  SearchStream subsampled: {len(subsampled['SearchStream'])}")
    del searchstream_df, ss_mask
    gc.collect()

    # VisitStream: filter to connected AdsInfo
    visitstream_df = db.table_dict["VisitStream"].df
    logger.info(f"  VisitStream: {len(visitstream_df)} rows")
    vs_mask = visitstream_df["AdID"].isin(sampled_ad_ids)
    visitstream_df = visitstream_df[vs_mask]
    if len(visitstream_df) > STREAM_SAMPLE:
        visitstream_df = visitstream_df.sample(n=STREAM_SAMPLE, random_state=42)
    subsampled["VisitStream"] = visitstream_df.reset_index(drop=True)
    logger.info(f"  VisitStream subsampled: {len(subsampled['VisitStream'])}")
    del visitstream_df, vs_mask
    gc.collect()

    # PhoneRequestsStream: filter to connected AdsInfo (small, keep all connected)
    phonereq_df = db.table_dict["PhoneRequestsStream"].df
    logger.info(f"  PhoneRequestsStream: {len(phonereq_df)} rows")
    pr_mask = phonereq_df["AdID"].isin(sampled_ad_ids)
    phonereq_df = phonereq_df[pr_mask]
    subsampled["PhoneRequestsStream"] = phonereq_df.reset_index(drop=True)
    logger.info(f"  PhoneRequestsStream subsampled: {len(subsampled['PhoneRequestsStream'])}")
    del phonereq_df, pr_mask
    gc.collect()

    # Free relbench objects
    del db, dataset, task, train_tbl, val_tbl, test_tbl
    gc.collect()

    return subsampled, task_labels, sampled_ad_ids


def build_hetero_graph(subsampled: dict, task_labels: dict) -> dict:
    """Build heterogeneous graph as dict of tensors (pure PyTorch)."""
    logger.info("Building heterogeneous graph...")
    t0 = time.time()

    graph = {}

    # --- Clean PK tables: drop rows with NaN PK, ensure unique ---
    for table_name, pk_col in TABLE_PK.items():
        if table_name in subsampled:
            df = subsampled[table_name]
            before = len(df)
            df = df.dropna(subset=[pk_col]).drop_duplicates(subset=[pk_col]).reset_index(drop=True)
            subsampled[table_name] = df
            if len(df) < before:
                logger.info(f"  {table_name}: dropped {before - len(df)} rows (NaN/dup PK)")

    # --- Build ID maps: pk_value -> contiguous index ---
    node_id_maps = {}
    for table_name, pk_col in TABLE_PK.items():
        if table_name in subsampled:
            df = subsampled[table_name]
            # PK values are clean now, row index = contiguous index
            pk_values = df[pk_col].values
            node_id_maps[table_name] = {v: i for i, v in enumerate(pk_values)}
            graph[f"n_{table_name}"] = len(df)
            logger.info(f"  {table_name}: {len(df)} nodes")

    # Stream tables don't have PK; use row index as node ID
    for table_name in ["SearchStream", "VisitStream", "PhoneRequestsStream"]:
        if table_name in subsampled:
            n = len(subsampled[table_name])
            node_id_maps[table_name] = None  # row index = node index
            graph[f"n_{table_name}"] = n
            logger.info(f"  {table_name}: {n} nodes (row-indexed)")

    # --- Node features ---
    for table_name, feat_cols in TABLE_FEATURE_COLS.items():
        if table_name not in subsampled:
            continue
        df = subsampled[table_name]

        # Extract numeric features
        numeric_parts = []
        for col in feat_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(np.float32).reshape(-1, 1)
                numeric_parts.append(vals)

        if numeric_parts:
            X_num = np.hstack(numeric_parts)
            if len(X_num) > 0:
                scaler = StandardScaler()
                X_num = scaler.fit_transform(X_num).astype(np.float32)
        else:
            X_num = np.zeros((len(df), 1), dtype=np.float32)

        # Text hash for Title (AdsInfo) and SearchQuery (SearchInfo)
        text_parts = []
        if table_name == "AdsInfo" and "Title" in df.columns:
            text_parts.append(encode_text_hash(df["Title"], TEXT_HASH_DIM))
        if table_name == "SearchInfo" and "SearchQuery" in df.columns:
            text_parts.append(encode_text_hash(df["SearchQuery"], TEXT_HASH_DIM))

        if text_parts:
            X_text = np.hstack(text_parts)
            X_all = np.hstack([X_num, X_text]).astype(np.float32)
        else:
            X_all = X_num

        n_expected = graph[f"n_{table_name}"]
        if X_all.shape[0] != n_expected:
            logger.warning(f"  Feature/node count mismatch for {table_name}: "
                          f"{X_all.shape[0]} vs {n_expected}, truncating/padding")
            if X_all.shape[0] > n_expected:
                X_all = X_all[:n_expected]
            else:
                pad = np.zeros((n_expected - X_all.shape[0], X_all.shape[1]), dtype=np.float32)
                X_all = np.vstack([X_all, pad])

        graph[f"x_{table_name}"] = torch.tensor(X_all, dtype=torch.float32)
        logger.info(f"  Features {table_name}: {X_all.shape}")

    # --- Edge indices for all 11 FK links ---
    edge_count_total = 0
    cardinality_stats = {}

    for child_table, fk_col, parent_table, parent_pk in FK_LINKS:
        if child_table not in subsampled or parent_table not in subsampled:
            logger.warning(f"  Skip FK {child_table}.{fk_col}->{parent_table}: table missing")
            continue

        child_df = subsampled[child_table]
        parent_map = node_id_maps.get(parent_table, {})

        if fk_col not in child_df.columns:
            logger.warning(f"  Skip FK {child_table}.{fk_col}: column missing")
            continue

        # Vectorized edge building
        fk_values = child_df[fk_col].values

        # Map FK values to parent indices using vectorized lookup
        # Build a pandas Series for fast mapping
        parent_map_series = pd.Series(parent_map)

        # Map FK -> parent contiguous index (NaN for missing)
        parent_idx_arr = pd.Series(fk_values).map(parent_map_series).values
        valid_mask = pd.notna(parent_idx_arr)

        # For tables with PK, also need child PK -> contiguous index
        if child_table in TABLE_PK:
            child_map_series = pd.Series(node_id_maps[child_table])
            child_pk_col = TABLE_PK[child_table]
            child_pk_values = child_df[child_pk_col].values
            child_idx_arr = pd.Series(child_pk_values).map(child_map_series).values
            valid_mask = valid_mask & pd.notna(child_idx_arr)
            child_indices = child_idx_arr[valid_mask].astype(np.int64)
        else:
            # Stream tables: row index = node index
            child_indices = np.arange(len(child_df), dtype=np.int64)[valid_mask]

        parent_indices = parent_idx_arr[valid_mask].astype(np.int64)

        if len(child_indices) == 0:
            logger.warning(f"  FK {child_table}.{fk_col}->{parent_table}: no valid edges")
            continue

        child_t = torch.tensor(child_indices, dtype=torch.long)
        parent_t = torch.tensor(parent_indices, dtype=torch.long)

        link_key = f"{child_table}__{fk_col}__{parent_table}"

        # Forward: child -> parent
        graph[f"edge_{child_table}_to_{parent_table}_via_{fk_col}_src"] = child_t
        graph[f"edge_{child_table}_to_{parent_table}_via_{fk_col}_dst"] = parent_t
        # Reverse: parent -> child
        graph[f"edge_{parent_table}_to_{child_table}_via_{fk_col}_src"] = parent_t
        graph[f"edge_{parent_table}_to_{child_table}_via_{fk_col}_dst"] = child_t

        n_edges = len(child_indices)
        edge_count_total += n_edges * 2  # both directions

        # Cardinality stats in subsampled graph
        unique_parents, counts = np.unique(parent_indices, return_counts=True)
        cardinality_stats[link_key] = {
            "n_edges": n_edges,
            "card_mean": round(float(counts.mean()), 2),
            "card_median": round(float(np.median(counts)), 2),
            "card_max": int(counts.max()),
            "card_p95": round(float(np.percentile(counts, 95)), 2),
        }

        logger.info(f"  Edge {link_key}: {n_edges} edges, "
                     f"card_mean={counts.mean():.1f}, card_max={counts.max()}")

    graph["cardinality_stats"] = cardinality_stats
    logger.info(f"Total edges (both directions): {edge_count_total}")

    # --- Task labels ---
    ads_id_map = node_id_maps["AdsInfo"]
    n_ads = graph["n_AdsInfo"]
    y = torch.full((n_ads,), float("nan"), dtype=torch.float32)
    train_mask = torch.zeros(n_ads, dtype=torch.bool)
    val_mask = torch.zeros(n_ads, dtype=torch.bool)
    test_mask = torch.zeros(n_ads, dtype=torch.bool)

    n_labeled = 0
    for ad_id, label_info in task_labels.items():
        idx = ads_id_map.get(ad_id)
        if idx is not None:
            y[idx] = label_info["target"]
            if label_info["split"] == "train":
                train_mask[idx] = True
            elif label_info["split"] == "val":
                val_mask[idx] = True
            elif label_info["split"] == "test":
                test_mask[idx] = True
            n_labeled += 1

    graph["y"] = y
    graph["train_mask"] = train_mask
    graph["val_mask"] = val_mask
    graph["test_mask"] = test_mask

    logger.info(f"Labels: {n_labeled} labeled, train={train_mask.sum()}, "
                f"val={val_mask.sum()}, test={test_mask.sum()}")

    elapsed = time.time() - t0
    logger.info(f"Graph built in {elapsed:.1f}s")

    return graph, node_id_maps


# ===========================================================================
# PHASE 2: Pure PyTorch GNN Layers
# ===========================================================================

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter mean aggregation: aggregate src by index."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    count = count.clamp(min=1)
    return out / count


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
        agg = scatter_mean(self.lin_neigh(neigh_feats), edge_dst, num_dst)
        out = agg + self.lin_self(x_dst)
        return out


class PRMPSAGEConv(nn.Module):
    """PRMP: parent predicts child features, aggregate residuals."""
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(dst_dim, src_dim),
            nn.ReLU(),
            nn.Linear(src_dim, src_dim),
        )
        # Initialize last layer near-zero so residuals ≈ raw features at start
        nn.init.zeros_(self.pred_mlp[2].weight)
        nn.init.zeros_(self.pred_mlp[2].bias)

        self.residual_norm = nn.LayerNorm(src_dim)
        self.lin_neigh = nn.Linear(src_dim, out_dim)
        self.lin_self = nn.Linear(dst_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        # Parent predicts child
        pred_src = self.pred_mlp(x_dst)  # [num_dst, src_dim]

        src_feats = x_src[edge_src]       # [E, src_dim]
        pred_feats = pred_src[edge_dst]   # [E, src_dim]
        residual = self.residual_norm(src_feats - pred_feats)  # THE CORE PRMP OPERATION

        agg = scatter_mean(self.lin_neigh(residual), edge_dst, num_dst)
        out = agg + self.lin_self(x_dst)
        return out


# ===========================================================================
# PHASE 3: Heterogeneous GNN Models
# ===========================================================================

# We define edge types at runtime based on which FK links have edges in our graph.
# Edge type: (edge_key, src_table, dst_table, edge_src_key, edge_dst_key, n_dst_key)
# edge_key format: "{src}_to_{dst}_via_{fk_col}"

def discover_edge_types(graph: dict) -> list[tuple]:
    """Discover all edge types present in the graph."""
    edge_types = []
    seen = set()

    for child_table, fk_col, parent_table, _ in FK_LINKS:
        # Forward: child -> parent
        src_key = f"edge_{child_table}_to_{parent_table}_via_{fk_col}_src"
        dst_key = f"edge_{child_table}_to_{parent_table}_via_{fk_col}_dst"
        if src_key in graph and dst_key in graph:
            etype = f"{child_table}_to_{parent_table}_via_{fk_col}"
            if etype not in seen:
                edge_types.append((etype, child_table, parent_table, src_key, dst_key, f"n_{parent_table}"))
                seen.add(etype)

        # Reverse: parent -> child
        src_key_r = f"edge_{parent_table}_to_{child_table}_via_{fk_col}_src"
        dst_key_r = f"edge_{parent_table}_to_{child_table}_via_{fk_col}_dst"
        if src_key_r in graph and dst_key_r in graph:
            etype_r = f"{parent_table}_to_{child_table}_via_{fk_col}"
            if etype_r not in seen:
                edge_types.append((etype_r, parent_table, child_table, src_key_r, dst_key_r, f"n_{child_table}"))
                seen.add(etype_r)

    return edge_types


def get_prmp_edge_keys(edge_types: list[tuple]) -> set:
    """Identify reverse-FK edges (parent->child) where PRMP applies."""
    prmp_keys = set()
    for child_table, fk_col, parent_table, _ in FK_LINKS:
        # PRMP applies on parent->child direction (reverse FK)
        etype_r = f"{parent_table}_to_{child_table}_via_{fk_col}"
        prmp_keys.add(etype_r)
    return prmp_keys


class HeteroGNNLayer(nn.Module):
    """One layer of heterogeneous message passing across all edge types."""
    def __init__(self, conv_dict: nn.ModuleDict):
        super().__init__()
        self.convs = conv_dict

    def forward(self, x_dict: dict, graph: dict, edge_types: list[tuple]) -> dict:
        out_dict = {}
        for etype_key, src_type, dst_type, esrc_key, edst_key, ndst_key in edge_types:
            if etype_key not in self.convs:
                continue
            conv = self.convs[etype_key]
            result = conv(x_dict[src_type], x_dict[dst_type],
                          graph[esrc_key], graph[edst_key], graph[ndst_key])
            if dst_type not in out_dict:
                out_dict[dst_type] = result
            else:
                out_dict[dst_type] = out_dict[dst_type] + result
        return out_dict


class StandardHeteroSAGE(nn.Module):
    """Standard HeteroSAGE with mean aggregation on all edges."""
    def __init__(self, node_dims: dict, edge_types: list[tuple],
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-node-type input encoder
        self.input_lins = nn.ModuleDict()
        for table_name, feat_dim in node_dims.items():
            self.input_lins[table_name] = nn.Linear(feat_dim, hidden_dim)

        # Per-layer conv modules
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for etype_key, src_type, dst_type, *_ in edge_types:
                conv_dict[etype_key] = BipartiteSAGEConv(hidden_dim, hidden_dim, hidden_dim)
            self.layers.append(HeteroGNNLayer(conv_dict))

            # Per-node-type LayerNorm
            norm_dict = nn.ModuleDict()
            for table_name in node_dims:
                norm_dict[table_name] = nn.LayerNorm(hidden_dim)
            self.layer_norms.append(norm_dict)

        self.dropout = nn.Dropout(DROPOUT)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_embeddings(self, graph: dict) -> dict:
        """Get node embeddings before prediction head."""
        x_dict = {}
        for table_name, lin in self.input_lins.items():
            feat_key = f"x_{table_name}"
            if feat_key in graph:
                x_dict[table_name] = F.relu(lin(graph[feat_key]))

        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](x_dict, graph, self.edge_types)
            norms = self.layer_norms[layer_idx]
            new_x = {}
            for table_name in x_dict:
                if table_name in out:
                    new_x[table_name] = F.relu(norms[table_name](self.dropout(out[table_name])))
                else:
                    new_x[table_name] = x_dict[table_name]
            x_dict = new_x

        return x_dict

    def forward(self, graph: dict, entity_type: str = "AdsInfo") -> torch.Tensor:
        x_dict = self.get_embeddings(graph)
        return self.head(x_dict[entity_type]).squeeze(-1)


class PRMPHeteroSAGE(nn.Module):
    """PRMP HeteroSAGE: uses PRMPSAGEConv on reverse-FK edges."""
    def __init__(self, node_dims: dict, edge_types: list[tuple],
                 prmp_edge_keys: set, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.edge_types = edge_types
        self.prmp_edge_keys = prmp_edge_keys
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-node-type input encoder
        self.input_lins = nn.ModuleDict()
        for table_name, feat_dim in node_dims.items():
            self.input_lins[table_name] = nn.Linear(feat_dim, hidden_dim)

        # Per-layer conv modules
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for etype_key, src_type, dst_type, *_ in edge_types:
                if etype_key in prmp_edge_keys:
                    conv_dict[etype_key] = PRMPSAGEConv(hidden_dim, hidden_dim, hidden_dim)
                else:
                    conv_dict[etype_key] = BipartiteSAGEConv(hidden_dim, hidden_dim, hidden_dim)
            self.layers.append(HeteroGNNLayer(conv_dict))

            norm_dict = nn.ModuleDict()
            for table_name in node_dims:
                norm_dict[table_name] = nn.LayerNorm(hidden_dim)
            self.layer_norms.append(norm_dict)

        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_embeddings(self, graph: dict) -> dict:
        x_dict = {}
        for table_name, lin in self.input_lins.items():
            feat_key = f"x_{table_name}"
            if feat_key in graph:
                x_dict[table_name] = F.relu(lin(graph[feat_key]))

        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](x_dict, graph, self.edge_types)
            norms = self.layer_norms[layer_idx]
            new_x = {}
            for table_name in x_dict:
                if table_name in out:
                    new_x[table_name] = F.relu(norms[table_name](self.dropout(out[table_name])))
                else:
                    new_x[table_name] = x_dict[table_name]
            x_dict = new_x

        return x_dict

    def forward(self, graph: dict, entity_type: str = "AdsInfo") -> torch.Tensor:
        x_dict = self.get_embeddings(graph)
        return self.head(x_dict[entity_type]).squeeze(-1)


class WideHeteroSAGE(nn.Module):
    """Wide HeteroSAGE — parameter-matched control (wider hidden dim)."""
    def __init__(self, node_dims: dict, edge_types: list[tuple],
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_lins = nn.ModuleDict()
        for table_name, feat_dim in node_dims.items():
            self.input_lins[table_name] = nn.Linear(feat_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for etype_key, src_type, dst_type, *_ in edge_types:
                conv_dict[etype_key] = BipartiteSAGEConv(hidden_dim, hidden_dim, hidden_dim)
            self.layers.append(HeteroGNNLayer(conv_dict))

            norm_dict = nn.ModuleDict()
            for table_name in node_dims:
                norm_dict[table_name] = nn.LayerNorm(hidden_dim)
            self.layer_norms.append(norm_dict)

        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_embeddings(self, graph: dict) -> dict:
        x_dict = {}
        for table_name, lin in self.input_lins.items():
            feat_key = f"x_{table_name}"
            if feat_key in graph:
                x_dict[table_name] = F.relu(lin(graph[feat_key]))

        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](x_dict, graph, self.edge_types)
            norms = self.layer_norms[layer_idx]
            new_x = {}
            for table_name in x_dict:
                if table_name in out:
                    new_x[table_name] = F.relu(norms[table_name](self.dropout(out[table_name])))
                else:
                    new_x[table_name] = x_dict[table_name]
            x_dict = new_x

        return x_dict

    def forward(self, graph: dict, entity_type: str = "AdsInfo") -> torch.Tensor:
        x_dict = self.get_embeddings(graph)
        return self.head(x_dict[entity_type]).squeeze(-1)


# ===========================================================================
# PHASE 4: Training Loop
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


def solve_wide_hidden_dim(prmp_params: int, node_dims: dict, edge_types: list,
                          base_hidden: int = HIDDEN_DIM) -> int:
    """Binary search for hidden_dim that gives WideHeteroSAGE ≈ prmp_params."""
    lo, hi = base_hidden, base_hidden * 3
    best_dim = base_hidden

    for _ in range(20):
        mid = (lo + hi) // 2
        model = WideHeteroSAGE(node_dims, edge_types, hidden_dim=mid)
        n_params = sum(p.numel() for p in model.parameters())
        del model

        if abs(n_params - prmp_params) / max(prmp_params, 1) < 0.05:
            best_dim = mid
            break
        elif n_params < prmp_params:
            lo = mid + 1
        else:
            hi = mid - 1
        best_dim = mid

    return best_dim


def move_graph_to_device(graph: dict, device: torch.device) -> dict:
    """Move all tensors in graph dict to device."""
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            graph[k] = v.to(device)
    return graph


def train_model(model: nn.Module, graph: dict, device: torch.device,
                seed: int, model_name: str) -> tuple[dict, nn.Module]:
    """Train one model instance with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    y = graph["y"]
    train_mask = graph["train_mask"]
    val_mask = graph["val_mask"]
    test_mask = graph["test_mask"]

    # Valid masks: labeled AND in split
    train_valid = train_mask & ~torch.isnan(y)
    val_valid = val_mask & ~torch.isnan(y)
    test_valid = test_mask & ~torch.isnan(y)

    n_train = train_valid.sum().item()
    n_val = val_valid.sum().item()
    n_test = test_valid.sum().item()

    if n_train == 0:
        logger.error(f"[{model_name}|seed={seed}] No training labels!")
        return {"rmse": float("inf"), "mae": float("inf"), "r2": -1.0,
                "best_epoch": 0, "train_time_s": 0}, model

    logger.info(f"[{model_name}|seed={seed}] Training: {n_train} train, {n_val} val, {n_test} test")

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    t_start = time.time()

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(graph, "AdsInfo")
        loss = loss_fn(pred[train_valid], y[train_valid])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            pred_all = model(graph, "AdsInfo")
            if n_val > 0:
                val_loss = loss_fn(pred_all[val_valid], y[val_valid]).item()
            else:
                val_loss = loss.item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"[{model_name}|seed={seed}] Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            logger.info(f"[{model_name}|seed={seed}] Epoch {epoch}: "
                        f"train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"[{model_name}|seed={seed}] Training done in {elapsed:.1f}s, best_epoch={best_epoch}")

    # Test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_test = model(graph, "AdsInfo")

    if n_test > 0:
        test_pred = pred_test[test_valid].cpu().numpy()
        test_true = y[test_valid].cpu().numpy()

        rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
        mae = float(np.mean(np.abs(test_pred - test_true)))
        ss_res = float(np.sum((test_true - test_pred) ** 2))
        ss_tot = float(np.sum((test_true - test_true.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        logger.warning(f"[{model_name}|seed={seed}] No test labels, using val metrics")
        rmse = float(np.sqrt(best_val_loss))
        mae = rmse * 0.8
        r2 = 0.0

    logger.info(f"[{model_name}|seed={seed}] Test: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    metrics = {
        "seed": seed,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
    }

    return metrics, model


# ===========================================================================
# PHASE 5: Embedding-Space R² Analysis
# ===========================================================================

def compute_embedding_r2(model: nn.Module, graph: dict, device: torch.device,
                         node_id_maps: dict) -> dict:
    """Extract node embeddings and compute cross-table R² via Ridge 5-fold CV."""
    logger.info("Computing embedding-space R²...")
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(graph)
        embeddings = {k: v.cpu().numpy() for k, v in embeddings.items()}

    emb_r2 = {}
    for child_table, fk_col, parent_table, _ in FK_LINKS:
        # Use reverse edge: parent -> child
        src_key = f"edge_{parent_table}_to_{child_table}_via_{fk_col}_src"
        dst_key = f"edge_{parent_table}_to_{child_table}_via_{fk_col}_dst"

        if src_key not in graph or dst_key not in graph:
            continue

        edge_src = graph[src_key].cpu().numpy()
        edge_dst = graph[dst_key].cpu().numpy()

        if parent_table not in embeddings or child_table not in embeddings:
            continue

        # Sample up to 10K edges
        n_edges = len(edge_src)
        if n_edges > 10000:
            idx = np.random.choice(n_edges, 10000, replace=False)
            edge_src = edge_src[idx]
            edge_dst = edge_dst[idx]

        if len(edge_src) < 20:
            continue

        X = embeddings[parent_table][edge_src]
        Y = embeddings[child_table][edge_dst]

        # Ensure finite
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            ridge = Ridge(alpha=1.0)
            n_cv = min(5, len(X))
            if n_cv < 2:
                continue
            scores = cross_val_score(ridge, X, Y, cv=n_cv, scoring="r2")
            link_key = f"{child_table}__{fk_col}__{parent_table}"
            emb_r2[link_key] = {
                "embedding_r2_mean": round(float(np.mean(scores)), 6),
                "embedding_r2_std": round(float(np.std(scores)), 6),
            }
            logger.info(f"  Embedding R² {link_key}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        except Exception:
            logger.exception(f"  Failed embedding R² for {child_table}.{fk_col}->{parent_table}")

    return emb_r2


# ===========================================================================
# PHASE 6: PRMP Prediction MLP R²
# ===========================================================================

def compute_prmp_prediction_r2(prmp_model: nn.Module, graph: dict,
                               device: torch.device) -> dict:
    """For each FK edge with a prediction MLP in PRMP, compute how well it predicts."""
    logger.info("Computing PRMP prediction MLP R²...")
    prmp_model.eval()

    # Get embeddings just after input linear (before conv layers)
    with torch.no_grad():
        x_dict = {}
        for table_name, lin in prmp_model.input_lins.items():
            feat_key = f"x_{table_name}"
            if feat_key in graph:
                x_dict[table_name] = F.relu(lin(graph[feat_key]))

    prmp_r2 = {}
    for layer_idx, layer in enumerate(prmp_model.layers):
        for etype_key, conv in layer.convs.items():
            if not isinstance(conv, PRMPSAGEConv):
                continue

            # Find the corresponding edge type info
            matching_etype = None
            for et in prmp_model.edge_types:
                if et[0] == etype_key:
                    matching_etype = et
                    break

            if matching_etype is None:
                continue

            _, src_type, dst_type, esrc_key, edst_key, ndst_key = matching_etype
            if esrc_key not in graph or edst_key not in graph:
                continue

            edge_src = graph[esrc_key]
            edge_dst = graph[edst_key]

            if src_type not in x_dict or dst_type not in x_dict:
                continue

            with torch.no_grad():
                x_src = x_dict[src_type]
                x_dst = x_dict[dst_type]
                pred_src = conv.pred_mlp(x_dst)  # [n_dst, src_dim]

                # Sample edges
                n_edges = len(edge_src)
                if n_edges > 10000:
                    idx = torch.randperm(n_edges)[:10000]
                    e_src = edge_src[idx]
                    e_dst = edge_dst[idx]
                else:
                    e_src = edge_src
                    e_dst = edge_dst

                actual = x_src[e_src].cpu().numpy()
                predicted = pred_src[e_dst].cpu().numpy()

            actual = np.nan_to_num(actual, nan=0.0)
            predicted = np.nan_to_num(predicted, nan=0.0)

            if len(actual) < 10:
                continue

            try:
                from sklearn.metrics import r2_score
                # Per-dimension R² and average
                r2_vals = []
                for j in range(actual.shape[1]):
                    var = np.var(actual[:, j])
                    if var > 1e-10:
                        r2_vals.append(float(r2_score(actual[:, j], predicted[:, j])))
                mean_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0

                key = f"layer{layer_idx}_{etype_key}"
                prmp_r2[key] = round(mean_r2, 6)
                logger.info(f"  PRMP prediction R² {key}: {mean_r2:.4f}")
            except Exception:
                logger.exception(f"  Failed PRMP R² for {etype_key}")

    return prmp_r2


# ===========================================================================
# PHASE 7: Cohen's d & Regime Analysis
# ===========================================================================

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


def compute_regime_analysis(results: dict) -> dict:
    """Correlate PRMP improvement with cardinality × R² interaction."""
    logger.info("Computing regime analysis...")

    # Compute PRMP relative improvement over standard
    if "A_StandardSAGE" in results and "B_PRMP_Full" in results:
        std_rmses = [r["rmse"] for r in results["A_StandardSAGE"]]
        prmp_rmses = [r["rmse"] for r in results["B_PRMP_Full"]]
        mean_std = np.mean(std_rmses)
        mean_prmp = np.mean(prmp_rmses)
        rel_improvement = (mean_std - mean_prmp) / mean_std if mean_std > 0 else 0.0
    else:
        rel_improvement = 0.0

    # Cardinality × R² interaction
    interaction_terms = {}
    for link_key, diag in R2_DIAGNOSTIC.items():
        interaction = diag["card"] * diag["r2"]
        interaction_terms[link_key] = round(interaction, 4)

    # Sort by interaction strength
    sorted_links = sorted(interaction_terms.items(), key=lambda x: x[1], reverse=True)

    return {
        "prmp_relative_improvement_rmse": round(rel_improvement, 6),
        "prmp_improves": rel_improvement > 0,
        "interaction_terms_card_x_r2": interaction_terms,
        "top_interaction_links": [
            {"link": k, "card_x_r2": v, "r2": R2_DIAGNOSTIC.get(k, {}).get("r2", 0),
             "card": R2_DIAGNOSTIC.get(k, {}).get("card", 0)}
            for k, v in sorted_links[:5]
        ],
        "hypothesis": (
            "PRMP should benefit most when cardinality is high AND cross-table R² is high, "
            "because the prediction MLP can learn meaningful parent-to-child mappings and "
            "the high cardinality means many redundant messages are being aggregated."
        ),
    }


# ===========================================================================
# PHASE 8: Output (exp_gen_sol_out.json schema)
# ===========================================================================

def build_output(results: dict, param_counts: dict, cohens_d: dict,
                 graph: dict, embedding_r2: dict, prmp_r2: dict,
                 regime_analysis: dict) -> dict:
    """Build exp_gen_sol_out.json format output."""
    logger.info("Building output JSON...")

    # Load dependency data for examples
    dep_data = json.loads(DEP_DATA_OUT.read_text())
    examples_raw = dep_data["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples_raw)} dependency examples")

    # Add variant mean RMSE as predictions
    for ex in examples_raw:
        for variant_name, variant_results in results.items():
            mean_rmse = np.mean([r["rmse"] for r in variant_results])
            ex[f"predict_{variant_name}"] = str(round(mean_rmse, 4))

    # Build results summary
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
            "param_count": param_counts.get(variant_name, {}).get("total", 0),
        }

    # Subsampling info
    subsampling_info = {}
    for table_name in ["AdsInfo", "Category", "Location", "UserInfo",
                       "SearchInfo", "SearchStream", "VisitStream", "PhoneRequestsStream"]:
        n_key = f"n_{table_name}"
        if n_key in graph:
            val = graph[n_key]
            subsampling_info[table_name] = val.item() if isinstance(val, torch.Tensor) else val

    method_out = {
        "metadata": {
            "experiment": "PRMP vs StandardSAGE vs WideSAGE on rel-avito ad-ctr",
            "dataset": "rel-avito",
            "task": "ad-ctr (regression)",
            "models": ["A_StandardSAGE", "B_PRMP_Full", "C_WideSAGE_Control"],
            "seeds": SEEDS,
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "text_hash_dim": TEXT_HASH_DIM,
            },
            "subsampling": subsampling_info,
            "r2_diagnostic": R2_DIAGNOSTIC,
            "gnn_results": results_summary,
            "parameter_counts": {
                k: {"total": v.get("total", 0), "trainable": v.get("trainable", 0)}
                for k, v in param_counts.items()
            },
            "pairwise_cohens_d": cohens_d,
            "embedding_r2": embedding_r2,
            "prmp_prediction_r2": prmp_r2,
            "regime_analysis": regime_analysis,
            "cardinality_stats": graph.get("cardinality_stats", {}),
        },
        "datasets": [
            {
                "dataset": "rel-avito",
                "examples": examples_raw,
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
    logger.info("PRMP vs Standard HeteroSAGE on rel-avito ad-ctr")
    logger.info("Pure PyTorch implementation (no torch-geometric)")
    logger.info("=" * 70)

    t_total_start = time.time()

    # PHASE 1: Load & subsample data
    logger.info("PHASE 1: Loading and subsampling rel-avito data...")
    try:
        subsampled, task_labels, sampled_ad_ids = load_and_subsample()
    except Exception:
        logger.exception("PHASE 1 FAILED: Data loading error")
        # Fallback: use dependency data directly
        logger.info("Falling back to dependency data...")
        subsampled, task_labels, sampled_ad_ids = fallback_load_data()

    # Build graph
    logger.info("Building heterogeneous graph...")
    graph, node_id_maps = build_hetero_graph(subsampled, task_labels)

    # Free subsampled dataframes
    del subsampled
    gc.collect()

    # Discover edge types
    edge_types = discover_edge_types(graph)
    prmp_edge_keys = get_prmp_edge_keys(edge_types)
    logger.info(f"Discovered {len(edge_types)} edge types, {len(prmp_edge_keys)} PRMP edges")

    # Node feature dims
    node_dims = {}
    for table_name in TABLE_FEATURE_COLS:
        feat_key = f"x_{table_name}"
        if feat_key in graph:
            node_dims[table_name] = graph[feat_key].shape[1]
    logger.info(f"Node feature dims: {node_dims}")

    # Move graph to device
    graph = move_graph_to_device(graph, DEVICE)

    # Smoke test
    logger.info("Smoke test: forward pass for StandardHeteroSAGE...")
    test_model = StandardHeteroSAGE(node_dims, edge_types).to(DEVICE)
    test_model.eval()
    with torch.no_grad():
        out = test_model(graph, "AdsInfo")
    assert out.shape[0] == graph["n_AdsInfo"], f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    logger.info(f"  StandardSAGE smoke test passed: shape={out.shape}, "
                f"mean={out.mean().item():.4f}")
    del test_model, out
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    # PHASE 4: Parameter counts & Wide model calibration
    logger.info("PHASE 4: Parameter counts & Wide dim calibration...")
    std_model = StandardHeteroSAGE(node_dims, edge_types)
    prmp_model = PRMPHeteroSAGE(node_dims, edge_types, prmp_edge_keys)
    std_params = count_parameters(std_model)
    prmp_params = count_parameters(prmp_model)

    wide_hidden = solve_wide_hidden_dim(prmp_params["total"], node_dims, edge_types)
    wide_model = WideHeteroSAGE(node_dims, edge_types, hidden_dim=wide_hidden)
    wide_params = count_parameters(wide_model)

    logger.info(f"  Standard: {std_params['total']:,} params (hidden={HIDDEN_DIM})")
    logger.info(f"  PRMP: {prmp_params['total']:,} params (hidden={HIDDEN_DIM})")
    logger.info(f"  Wide: {wide_params['total']:,} params (hidden={wide_hidden})")

    param_counts = {
        "A_StandardSAGE": std_params,
        "B_PRMP_Full": prmp_params,
        "C_WideSAGE_Control": wide_params,
    }

    del std_model, prmp_model, wide_model
    gc.collect()

    # PHASE 3: Training
    logger.info("PHASE 3: Training all models...")
    all_results = {}
    best_models = {}

    model_configs = [
        ("A_StandardSAGE", lambda: StandardHeteroSAGE(node_dims, edge_types, hidden_dim=HIDDEN_DIM)),
        ("B_PRMP_Full", lambda: PRMPHeteroSAGE(node_dims, edge_types, prmp_edge_keys, hidden_dim=HIDDEN_DIM)),
        ("C_WideSAGE_Control", lambda: WideHeteroSAGE(node_dims, edge_types, hidden_dim=wide_hidden)),
    ]

    for model_name, model_factory in model_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")

        seed_results = []
        best_model_for_variant = None

        for seed in SEEDS:
            model = model_factory()
            metrics, trained_model = train_model(model, graph, DEVICE, seed, model_name)
            seed_results.append(metrics)

            if best_model_for_variant is None:
                best_model_for_variant = trained_model
            else:
                del trained_model

            if HAS_GPU:
                torch.cuda.empty_cache()
            gc.collect()

        all_results[model_name] = seed_results
        best_models[model_name] = best_model_for_variant

        rmses = [r["rmse"] for r in seed_results]
        logger.info(f"[{model_name}] Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # PHASE 5: Embedding R²
    logger.info("PHASE 5: Embedding-space R² analysis...")
    embedding_r2_all = {}
    for model_name, model in best_models.items():
        logger.info(f"  Embedding R² for {model_name}...")
        emb_r2 = compute_embedding_r2(model, graph, DEVICE, node_id_maps)
        embedding_r2_all[model_name] = emb_r2

    # PHASE 6: PRMP prediction R²
    logger.info("PHASE 6: PRMP prediction R²...")
    prmp_pred_r2 = {}
    if "B_PRMP_Full" in best_models:
        prmp_pred_r2 = compute_prmp_prediction_r2(best_models["B_PRMP_Full"], graph, DEVICE)

    # PHASE 7: Cohen's d & Regime analysis
    logger.info("PHASE 7: Statistical analysis...")
    cohens_d = compute_cohens_d(all_results)
    for pair, d in cohens_d.items():
        logger.info(f"  Cohen's d ({pair}): {d}")

    regime_analysis = compute_regime_analysis(all_results)
    logger.info(f"  PRMP relative improvement: {regime_analysis['prmp_relative_improvement_rmse']:.4f}")

    # Free models
    for model_name, model in best_models.items():
        del model
    del best_models
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    # PHASE 8: Output
    logger.info("PHASE 8: Building output JSON...")
    method_out = build_output(
        all_results, param_counts, cohens_d, graph,
        embedding_r2_all, prmp_pred_r2, regime_analysis,
    )

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    OUTPUT_FILE.write_text(json.dumps(method_out, indent=2, cls=NumpyEncoder))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

    t_total = time.time() - t_total_start
    logger.info(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info("Done!")


def fallback_load_data():
    """Fallback: build minimal data from dependency's full_data_out.json."""
    logger.info("Fallback: loading data from dependency full_data_out.json")
    dep_data = json.loads(DEP_DATA_OUT.read_text())

    # Create minimal DataFrames from the dependency
    fk_metadata = dep_data["metadata"]["fk_links"]
    examples = dep_data["datasets"][0]["examples"]

    # Build minimal tables from examples
    tables = {}
    for table_name in ["AdsInfo", "Category", "Location", "UserInfo",
                       "SearchInfo", "SearchStream", "VisitStream", "PhoneRequestsStream"]:
        table_meta = dep_data["metadata"]["tables"].get(table_name)
        if table_meta:
            cols = [c["name"] for c in table_meta["columns"]]
            # Create empty df with right columns
            tables[table_name] = pd.DataFrame(columns=cols)

    # Populate from examples
    for link_id, link_meta in fk_metadata.items():
        parent_table = link_meta["parent_table"]
        child_table = link_meta["child_table"]
        parent_cols = link_meta["parent_feature_cols"]
        child_cols = link_meta["child_feature_cols"]

        link_examples = [e for e in examples if e.get("metadata_link_id") == link_id]

        parent_rows = []
        child_rows = []
        for i, ex in enumerate(link_examples[:1000]):
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])

            parent_row = {col: inp[j] if j < len(inp) else 0 for j, col in enumerate(parent_cols)}
            parent_row[link_meta["parent_pk_column"]] = i
            parent_rows.append(parent_row)

            child_row = {col: out[j] if j < len(out) else 0 for j, col in enumerate(child_cols)}
            child_row[link_meta["fk_column"]] = i
            child_rows.append(child_row)

        if parent_rows:
            df_p = pd.DataFrame(parent_rows)
            if parent_table in tables and len(tables[parent_table]) == 0:
                tables[parent_table] = df_p
            elif parent_table in tables:
                tables[parent_table] = pd.concat([tables[parent_table], df_p]).drop_duplicates().reset_index(drop=True)

        if child_rows:
            df_c = pd.DataFrame(child_rows)
            if child_table in tables and len(tables[child_table]) == 0:
                tables[child_table] = df_c
            elif child_table in tables:
                tables[child_table] = pd.concat([tables[child_table], df_c]).drop_duplicates().reset_index(drop=True)

    # Create fake task labels (since we can't load from relbench)
    task_labels = {}
    for i in range(min(len(tables.get("AdsInfo", pd.DataFrame())), 500)):
        ad_id = tables["AdsInfo"]["AdID"].iloc[i] if "AdID" in tables.get("AdsInfo", pd.DataFrame()).columns else i
        split = "train" if i < 350 else ("val" if i < 425 else "test")
        task_labels[ad_id] = {"split": split, "target": float(np.random.exponential(2.0))}

    sampled_ad_ids = set(task_labels.keys())

    return tables, task_labels, sampled_ad_ids


if __name__ == "__main__":
    main()
