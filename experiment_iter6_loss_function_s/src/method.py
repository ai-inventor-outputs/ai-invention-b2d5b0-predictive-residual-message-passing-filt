#!/usr/bin/env python3
"""Loss Function Swap Experiment: Isolating Task-Type Effect on PRMP via rel-f1.

Trains Standard and PRMP GNN models on 4 task configurations
(natural regression, binned-classification, natural classification, softened-regression)
x 3 seeds = 24 runs on the SAME rel-f1 heterogeneous graph, with gradient norm
instrumentation through PRMP prediction pathway vs main aggregation pathway.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import math
import gc
import time
import resource
import zipfile
import warnings
import copy

import numpy as np
import pandas as pd
import psutil

# ── Logging ──────────────────────────────────────────────────────────────────
WS = Path(__file__).parent
(WS / "logs").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WS / "logs" / "run.log", rotation="30 MB", level="DEBUG")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Hardware detection ───────────────────────────────────────────────────────
def _detect_cpus() -> int:
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
# NOTE: Do NOT set RLIMIT_AS before torch import — CUDA needs large virtual address space
logger.info(f"Detected: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, LayerNorm
from sklearn.metrics import (
    mean_absolute_error, roc_auc_score, average_precision_score,
    accuracy_score, r2_score,
)
from scipy import stats

HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(0.85, 20e9 / _total))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f} GB")
else:
    VRAM_GB = 0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, GPU={HAS_GPU}")

# ── Constants ────────────────────────────────────────────────────────────────
DEP_WS = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
              "/3_invention_loop/iter_2/gen_art/data_id5_it2__opus")
ERGAST_CSV_URL = "https://github.com/rubenv/ergast-mrd/raw/master/f1db_csv.zip"
TEMP_DIR = WS / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


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

HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 80
SEEDS = [42, 123, 7, 0, 1, 2, 3, 10, 21, 55, 77, 99, 256]
PATIENCE = 15

# Tables used in the graph
NODE_TABLES = ["drivers", "races", "constructors", "circuits", "status",
               "results", "qualifying", "driver_standings"]

ID_COLUMNS = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "drivers": ["driverId"],
    "driver_standings": ["driverStandingsId", "raceId", "driverId"],
    "qualifying": ["qualifyId", "raceId", "driverId", "constructorId"],
    "races": ["raceId", "circuitId"],
    "results": ["resultId", "raceId", "driverId", "constructorId", "statusId"],
    "status": ["statusId"],
}

# FK links for building graph edges (child_table, child_fk_col, parent_table, parent_pk_col)
GRAPH_FK_LINKS = [
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def download_ergast_csv() -> dict[str, pd.DataFrame]:
    """Download and extract CSV tables from Ergast F1 database."""
    dep_zip = DEP_WS / "temp" / "datasets" / "f1db_csv.zip"
    local_zip = TEMP_DIR / "f1db_csv.zip"
    if local_zip.exists():
        zip_path = local_zip
    elif dep_zip.exists():
        zip_path = dep_zip
    else:
        import requests
        logger.info(f"Downloading Ergast F1 CSV...")
        resp = requests.get(ERGAST_CSV_URL, timeout=120)
        resp.raise_for_status()
        local_zip.write_bytes(resp.content)
        zip_path = local_zip

    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for csv_name in sorted(zf.namelist()):
            if not csv_name.endswith(".csv"):
                continue
            tname = csv_name.replace(".csv", "")
            if tname not in NODE_TABLES:
                continue
            with zf.open(csv_name) as f:
                import io, csv as csv_mod
                raw = f.read().decode("utf-8")
                lines = raw.strip().split("\n")
                # Fix mismatched header/data column counts (races.csv has extra cols)
                header_fields = next(csv_mod.reader([lines[0]]))
                if len(lines) > 1:
                    data_fields = next(csv_mod.reader([lines[1]]))
                    if len(data_fields) > len(header_fields):
                        extra = len(data_fields) - len(header_fields)
                        for ei in range(extra):
                            header_fields.append(f"_extra_{ei}")
                        lines[0] = ",".join(header_fields)
                        raw = "\n".join(lines)
                df = pd.read_csv(
                    io.StringIO(raw),
                    na_values=["\\N", "NULL", ""],
                    engine="python",
                )
            # Force Arrow types to numpy-compatible types (pandas 3.0 compat)
            for col in df.columns:
                s = df[col]
                dtype_str = str(type(s.dtype))
                if "Arrow" in dtype_str or "arrow" in dtype_str:
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().sum() > len(num) * 0.5 and s.notna().sum() > 0:
                        df[col] = num
                    else:
                        df[col] = s.astype(object)
            # Ensure ID columns are always numeric
            for col in ID_COLUMNS.get(tname, []):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            tables[tname] = df
            logger.debug(f"  {tname}: cols={list(df.columns)}, dtypes={dict(df.dtypes)}")
    logger.info(f"Loaded {len(tables)} tables: {list(tables.keys())}")
    return tables


def encode_table(df: pd.DataFrame, table_name: str) -> tuple[np.ndarray, list[str]]:
    """Encode table to numeric features. Returns (feature_array, feature_col_names)."""
    id_cols = set(ID_COLUMNS.get(table_name, []))
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Heterogeneous Graph Construction (VECTORIZED)
# ══════════════════════════════════════════════════════════════════════════════
def build_hetero_graph(tables: dict[str, pd.DataFrame]):
    """Build HeteroData graph using vectorized operations. Returns (data, id_maps, stats)."""
    data = HeteroData()
    id_maps: dict[str, dict[int, int]] = {}
    graph_stats = {}

    for tname in NODE_TABLES:
        if tname not in tables:
            continue
        df = tables[tname]
        pk = ID_COLUMNS.get(tname, [None])[0]

        # Build ID map: original_id -> sequential node index
        if pk and pk in df.columns:
            unique_ids = df[pk].dropna().unique().astype(int)
            imap = {int(v): i for i, v in enumerate(unique_ids)}
        else:
            imap = {i: i for i in range(len(df))}
        id_maps[tname] = imap

        # Encode features
        feat_arr, feat_cols = encode_table(df, tname)
        # If table has more rows than unique IDs (e.g. results), aggregate by PK
        if pk and pk in df.columns and len(df) > len(imap):
            # This is a child/event table - use all rows as nodes
            # Actually for child tables, each row IS a node
            imap = {i: i for i in range(len(df))}
            id_maps[tname] = imap
            data[tname].x = torch.tensor(feat_arr, dtype=torch.float32)
        else:
            data[tname].x = torch.tensor(feat_arr[:len(imap)], dtype=torch.float32)

        data[tname].num_nodes = len(imap)
        graph_stats[tname] = len(imap)
        logger.info(f"  {tname}: {len(imap)} nodes, {data[tname].x.shape[1]} feats")

    # Build edges vectorized
    edge_count = 0
    for child_t, child_col, parent_t, parent_col in GRAPH_FK_LINKS:
        if child_t not in tables or parent_t not in tables:
            continue
        child_df = tables[child_t]
        if child_col not in child_df.columns:
            continue

        child_imap = id_maps[child_t]
        parent_imap = id_maps[parent_t]

        # Vectorized edge building using numpy
        fk_series = pd.to_numeric(child_df[child_col], errors="coerce")
        fk_vals = fk_series.values
        valid_mask = ~pd.isna(fk_vals)
        fk_valid = fk_vals[valid_mask].astype(int)
        child_row_indices = np.arange(len(child_df))[valid_mask]

        # Build lookup arrays for fast mapping
        child_keys = np.array(list(child_imap.keys()))
        child_vals = np.array(list(child_imap.values()))
        parent_keys = np.array(list(parent_imap.keys()))
        parent_vals = np.array(list(parent_imap.values()))

        # Map child rows → node indices
        child_idx_map = dict(zip(child_keys, child_vals))
        parent_idx_map = dict(zip(parent_keys, parent_vals))

        src_arr = np.array([child_idx_map.get(int(c), -1) for c in child_row_indices])
        dst_arr = np.array([parent_idx_map.get(int(f), -1) for f in fk_valid])
        both_valid = (src_arr >= 0) & (dst_arr >= 0)

        if not both_valid.any():
            continue

        src_t = torch.tensor(src_arr[both_valid], dtype=torch.long)
        dst_t = torch.tensor(dst_arr[both_valid], dtype=torch.long)

        et_fwd = (child_t, f"fk_{child_col}", parent_t)
        data[et_fwd].edge_index = torch.stack([src_t, dst_t], dim=0)

        et_rev = (parent_t, f"rev_{child_col}", child_t)
        data[et_rev].edge_index = torch.stack([dst_t, src_t], dim=0)

        n_edges = int(both_valid.sum())
        edge_count += n_edges
        logger.info(f"  {child_t}->{parent_t} via {child_col}: {n_edges} edges")

    graph_stats["total_edges"] = edge_count
    logger.info(f"Total edges: {edge_count}")
    return data, id_maps, graph_stats


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Target Label Construction
# ══════════════════════════════════════════════════════════════════════════════
def build_targets(tables: dict[str, pd.DataFrame], id_maps: dict):
    """Build target labels for 4 task configurations."""
    results = tables["results"]
    races = tables["races"]
    drivers_map = id_maps["drivers"]
    num_drivers = len(drivers_map)

    # Parse race dates for temporal split
    races = races.copy()
    if "date" in races.columns:
        races["_date"] = pd.to_datetime(races["date"].astype(str), format="mixed", errors="coerce")
    else:
        races["_date"] = pd.RangeIndex(len(races))

    # Ensure merge columns are compatible types (cast all FKs to float64)
    results = results.copy()
    results["raceId"] = pd.to_numeric(results["raceId"], errors="coerce").astype("float64")
    results["driverId"] = pd.to_numeric(results["driverId"], errors="coerce").astype("float64")
    if "statusId" in results.columns:
        results["statusId"] = pd.to_numeric(results["statusId"], errors="coerce").astype("float64")
    races["raceId"] = pd.to_numeric(races["raceId"], errors="coerce").astype("float64")

    logger.info(f"  results.raceId dtype={results['raceId'].dtype}, races.raceId dtype={races['raceId'].dtype}")
    logger.info(f"  races._date valid={races['_date'].notna().sum()}/{len(races)}")

    results_m = results.merge(races[["raceId", "_date"]], on="raceId", how="inner")
    results_m = results_m.sort_values("_date")
    logger.info(f"  Merged results: {len(results_m)} rows (from {len(results)} results)")

    # Position column
    pos_col = "positionOrder" if "positionOrder" in results_m.columns else "position"
    results_m["_pos"] = pd.to_numeric(results_m[pos_col], errors="coerce")
    logger.info(f"  Position col={pos_col}, valid={results_m['_pos'].notna().sum()}")

    # Temporal split on races
    race_dates = races[["raceId", "_date"]].dropna(subset=["_date"]).sort_values("_date")
    n_races = len(race_dates)
    cutoff_train = race_dates.iloc[int(n_races * 0.7)]["_date"]
    cutoff_val = race_dates.iloc[int(n_races * 0.8)]["_date"]

    # Aggregate per driver across ALL their races (graph has all data)
    # But split masks based on which drivers appear in each temporal split
    driver_ids_train = set(results_m[results_m["_date"] <= cutoff_train]["driverId"].dropna().unique())
    driver_ids_val = set(results_m[(results_m["_date"] > cutoff_train) & (results_m["_date"] <= cutoff_val)]["driverId"].dropna().unique())
    driver_ids_test = set(results_m[results_m["_date"] > cutoff_val]["driverId"].dropna().unique())

    logger.info(f"Driver split: train={len(driver_ids_train)}, val={len(driver_ids_val)}, test={len(driver_ids_test)}")

    use_random = len(driver_ids_test) < 50
    if use_random:
        logger.warning("Too few test drivers, using random split")

    def make_masks(valid_mask_tensor):
        """Make train/val/test masks."""
        if use_random:
            valid_idx = valid_mask_tensor.nonzero(as_tuple=True)[0].numpy()
            rng = np.random.RandomState(42)
            rng.shuffle(valid_idx)
            n = len(valid_idx)
            nt, nv = int(n * 0.7), int(n * 0.15)
            tr = torch.zeros(num_drivers, dtype=torch.bool)
            va = torch.zeros(num_drivers, dtype=torch.bool)
            te = torch.zeros(num_drivers, dtype=torch.bool)
            tr[valid_idx[:nt]] = True
            va[valid_idx[nt:nt+nv]] = True
            te[valid_idx[nt+nv:]] = True
            return tr, va, te
        else:
            tr = torch.zeros(num_drivers, dtype=torch.bool)
            va = torch.zeros(num_drivers, dtype=torch.bool)
            te = torch.zeros(num_drivers, dtype=torch.bool)
            for did, idx in drivers_map.items():
                if not valid_mask_tensor[idx]:
                    continue
                if did in driver_ids_train:
                    tr[idx] = True
                if did in driver_ids_val:
                    va[idx] = True
                if did in driver_ids_test:
                    te[idx] = True
            return tr, va, te

    configs = {}

    # ── CONFIG 1: Natural Regression (position, MAE) ──────────────────────
    pos_agg = results_m.dropna(subset=["_pos"]).groupby("driverId")["_pos"].mean()
    pos_labels = torch.full((num_drivers,), float("nan"))
    valid_pos = torch.zeros(num_drivers, dtype=torch.bool)
    for did, idx in drivers_map.items():
        if did in pos_agg.index:
            pos_labels[idx] = float(pos_agg.loc[did])
            valid_pos[idx] = True

    c1_tr, c1_va, c1_te = make_masks(valid_pos)
    pos_clean = pos_labels.clone()
    pos_clean[torch.isnan(pos_clean)] = 0.0

    configs["config1_natural_regression"] = dict(
        labels=pos_clean, train_mask=c1_tr, val_mask=c1_va, test_mask=c1_te,
        loss_type="MAE", target_type="position", num_classes=1,
        description="Natural regression: mean position with L1Loss",
    )
    logger.info(f"C1: train={c1_tr.sum()}, val={c1_va.sum()}, test={c1_te.sum()}")

    # ── CONFIG 2: Binned Classification (position→bins, CE) ───────────────
    train_vals = pos_clean[c1_tr].numpy()
    try:
        _, bin_edges = pd.qcut(train_vals, q=5, retbins=True, duplicates="drop")
    except ValueError:
        bin_edges = np.array([0, 5, 10, 15, 20, 30])
    num_bins = len(bin_edges) - 1

    bin_labels = torch.full((num_drivers,), 0, dtype=torch.long)
    for i in range(num_drivers):
        if valid_pos[i]:
            v = pos_labels[i].item()
            bi = min(np.searchsorted(bin_edges[1:], v, side="right"), num_bins - 1)
            bin_labels[i] = bi

    bin_centers = torch.tensor([(bin_edges[i]+bin_edges[i+1])/2 for i in range(num_bins)], dtype=torch.float32)

    configs["config2_binned_classification"] = dict(
        labels=bin_labels, train_mask=c1_tr, val_mask=c1_va, test_mask=c1_te,
        loss_type="CrossEntropy", target_type="position", num_classes=num_bins,
        bin_centers=bin_centers, bin_edges=bin_edges.tolist(),
        description=f"Binned classification: {num_bins}-class with CrossEntropyLoss",
    )
    logger.info(f"C2: {num_bins} bins, edges={[round(e,1) for e in bin_edges]}")

    # ── CONFIG 3: Natural Classification (DNF, BCE) ───────────────────────
    # Vectorized: compute DNF stats per driver using groupby
    dnf_labels = torch.full((num_drivers,), float("nan"))
    valid_dnf = torch.zeros(num_drivers, dtype=torch.bool)
    soft_labels = torch.full((num_drivers,), float("nan"))
    valid_soft = torch.zeros(num_drivers, dtype=torch.bool)

    if "statusId" in results_m.columns:
        results_m["_is_dnf"] = (results_m["statusId"] != 1).astype(float)
    else:
        results_m["_is_dnf"] = results_m["_pos"].isna().astype(float)

    dnf_agg = results_m.dropna(subset=["driverId"]).groupby("driverId")["_is_dnf"].agg(["max", "mean"])
    for did, idx in drivers_map.items():
        if did in dnf_agg.index:
            dnf_labels[idx] = 1.0 if dnf_agg.loc[did, "max"] > 0 else 0.0
            valid_dnf[idx] = True
            soft_labels[idx] = float(dnf_agg.loc[did, "mean"])
            valid_soft[idx] = True

    c3_tr, c3_va, c3_te = make_masks(valid_dnf)
    dnf_clean = dnf_labels.clone()
    dnf_clean[torch.isnan(dnf_clean)] = 0.0

    configs["config3_natural_classification"] = dict(
        labels=dnf_clean, train_mask=c3_tr, val_mask=c3_va, test_mask=c3_te,
        loss_type="BCE", target_type="binary", num_classes=1,
        description="Natural classification: DNF binary with BCEWithLogitsLoss",
    )
    pos_rate = dnf_clean[c3_tr].mean().item()
    logger.info(f"C3: train={c3_tr.sum()}, pos_rate={pos_rate:.3f}")

    # ── CONFIG 4: Softened Regression (DNF fraction, MSE) ─────────────────
    c4_tr, c4_va, c4_te = make_masks(valid_soft)
    soft_clean = soft_labels.clone()
    soft_clean[torch.isnan(soft_clean)] = 0.0

    configs["config4_softened_regression"] = dict(
        labels=soft_clean, train_mask=c4_tr, val_mask=c4_va, test_mask=c4_te,
        loss_type="MSE", target_type="binary", num_classes=1,
        description="Softened regression: DNF fraction with MSELoss",
    )
    logger.info(f"C4: mean={soft_clean[c4_tr].mean():.3f}, std={soft_clean[c4_tr].std():.3f}")

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
class PRMPConv(nn.Module):
    """Predictive Residual Message Passing convolution for one edge type."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, in_dim),
        )
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)
        self.ln = nn.LayerNorm(in_dim)
        self.update_lin = nn.Linear(in_dim * 2, out_dim)
        self.grad_pred: list[float] = []
        self.grad_main: list[float] = []

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_j = x_src[src_idx]
        predicted = self.pred_mlp(x_dst[dst_idx].detach())
        residual = self.ln(x_j - predicted)

        if self.training and residual.requires_grad:
            predicted.register_hook(lambda g: self.grad_pred.append(g.norm().item()))
            x_j.register_hook(lambda g: self.grad_main.append(g.norm().item()))

        # Mean aggregation
        num_dst = x_dst.shape[0]
        aggr = torch.zeros(num_dst, residual.shape[1], device=residual.device)
        cnt = torch.zeros(num_dst, 1, device=residual.device)
        aggr.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(residual), residual)
        cnt.scatter_add_(0, dst_idx.unsqueeze(1), torch.ones(len(dst_idx), 1, device=residual.device))
        aggr = aggr / cnt.clamp(min=1)

        return self.update_lin(torch.cat([x_dst, aggr], dim=-1))


class F1HeteroGNN(nn.Module):
    """Heterogeneous GNN with Standard (SAGEConv) or PRMP convolutions."""

    def __init__(self, feat_dims: dict[str, int], hidden: int, n_layers: int,
                 edge_types: list[tuple], conv_type: str = "standard",
                 num_classes: int = 1):
        super().__init__()
        self.conv_type = conv_type
        self.n_layers = n_layers

        # Per-type encoders
        self.encs = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(fd, hidden), nn.ReLU())
            for nt, fd in feat_dims.items()
        })

        # Conv layers
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(n_layers):
            cd = {}
            for et in edge_types:
                k = "__".join(et)
                if conv_type == "prmp":
                    cd[et] = PRMPConv(hidden, hidden)
                else:
                    cd[et] = SAGEConv((hidden, hidden), hidden)
            self.convs.append(HeteroConv(cd, aggr="sum"))
            self.lns.append(nn.ModuleDict({
                nt: LayerNorm(hidden) for nt in feat_dims
            }))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.encs[nt](x) if nt in self.encs else x
             for nt, x in x_dict.items()}

        for i in range(self.n_layers):
            if self.conv_type == "prmp":
                new_h = self._prmp_fwd(h, edge_index_dict, i)
            else:
                new_h = self.convs[i](h, edge_index_dict)
            for nt in h:
                if nt in new_h and nt in self.lns[i]:
                    new_h[nt] = F.relu(self.lns[i][nt](new_h[nt])) + h[nt]
            h = {k: new_h.get(k, h[k]) for k in h}

        return self.head(h["drivers"])

    def _prmp_fwd(self, h, eid, layer_i):
        new_h = {}
        cl = self.convs[layer_i]
        for et, ei in eid.items():
            src_t, _, dst_t = et
            if src_t not in h or dst_t not in h:
                continue
            k = "__".join(et)
            if k in cl.convs:
                out = cl.convs[k](h[src_t], h[dst_t], ei)
                new_h[dst_t] = new_h.get(dst_t, torch.zeros_like(h[dst_t])) + out
        return new_h

    def collect_grad_norms(self) -> dict:
        pn, mn = [], []
        for cl in self.convs:
            for c in cl.convs.values():
                if isinstance(c, PRMPConv):
                    pn.extend(c.grad_pred)
                    mn.extend(c.grad_main)
        return {"pred": pn, "main": mn}

    def clear_grad_norms(self):
        for cl in self.convs:
            for c in cl.convs.values():
                if isinstance(c, PRMPConv):
                    c.grad_pred.clear()
                    c.grad_main.clear()

    def prmp_diagnostics(self, x_dict, eid):
        if self.conv_type != "prmp":
            return {}
        self.eval()
        with torch.no_grad():
            h = {nt: self.encs[nt](x) if nt in self.encs else x
                 for nt, x in x_dict.items()}
            r2s, rmeans, rstds = [], [], []
            cl = self.convs[0]
            for et, ei in eid.items():
                src_t, _, dst_t = et
                if src_t not in h or dst_t not in h:
                    continue
                k = "__".join(et)
                if k not in cl.convs:
                    continue
                pc = cl.convs[k]
                if not isinstance(pc, PRMPConv):
                    continue
                si, di = ei[0], ei[1]
                xj = h[src_t][si]
                pred = pc.pred_mlp(h[dst_t][di])
                res = xj - pred
                ss_res = ((xj - pred)**2).sum().item()
                ss_tot = ((xj - xj.mean(0))**2).sum().item()
                r2s.append(1 - ss_res / (ss_tot + 1e-8))
                rmeans.append(res.mean().item())
                rstds.append(res.std().item())
            if r2s:
                return {"prediction_r2": float(np.mean(r2s)),
                        "residual_mean": float(np.mean(rmeans)),
                        "residual_std": float(np.mean(rstds))}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Training
# ══════════════════════════════════════════════════════════════════════════════
def train_run(data: HeteroData, config: dict, conv_type: str, seed: int,
              epochs: int = EPOCHS, patience: int = PATIENCE) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    dd = data.to(DEVICE)
    labels = config["labels"].to(DEVICE)
    tr_m = config["train_mask"].to(DEVICE)
    va_m = config["val_mask"].to(DEVICE)
    te_m = config["test_mask"].to(DEVICE)
    lt = config["loss_type"]
    nc = config["num_classes"]

    fd = {nt: dd[nt].x.shape[1] for nt in dd.node_types}
    ets = list(dd.edge_types)

    model = F1HeteroGNN(fd, HIDDEN_DIM, NUM_LAYERS, ets, conv_type, nc).to(DEVICE)
    crit = {"MAE": nn.L1Loss(), "CrossEntropy": nn.CrossEntropyLoss(),
            "BCE": nn.BCEWithLogitsLoss(), "MSE": nn.MSELoss()}[lt]
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    xd = {nt: dd[nt].x for nt in dd.node_types}
    eid = {et: dd[et].edge_index for et in dd.edge_types}

    best_val = float("inf") if lt in ["MAE", "MSE"] else float("-inf")
    best_st = None
    no_imp = 0
    losses = []
    grad_ratios = []

    for ep in range(epochs):
        model.train()
        if conv_type == "prmp":
            model.clear_grad_norms()
        opt.zero_grad()
        out = model(xd, eid)

        if lt == "CrossEntropy":
            loss = crit(out[tr_m], labels[tr_m])
        else:
            loss = crit(out[tr_m].squeeze(-1), labels[tr_m])

        if torch.isnan(loss):
            logger.warning(f"NaN loss at epoch {ep}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if conv_type == "prmp":
            gn = model.collect_grad_norms()
            if gn["pred"] and gn["main"]:
                r = np.mean(gn["pred"]) / (np.mean(gn["main"]) + 1e-8)
                grad_ratios.append(r)

        model.eval()
        with torch.no_grad():
            vo = model(xd, eid)
        vm = _metric(vo, labels, va_m, lt, nc, config)

        improved = (vm < best_val) if lt in ["MAE", "MSE"] else (vm > best_val)
        if improved:
            best_val = vm
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_st:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_st.items()})

    model.eval()
    with torch.no_grad():
        to = model(xd, eid)
    tm = _metric(to, labels, te_m, lt, nc, config)
    tmf = _all_metrics(to, labels, te_m, lt, nc, config)

    diag = model.prmp_diagnostics(xd, eid) if conv_type == "prmp" else {}

    ga = {}
    if conv_type == "prmp" and grad_ratios:
        last10 = grad_ratios[-10:]
        ga = {"ratio_last_10_mean": float(np.mean(last10)),
              "ratio_last_10_std": float(np.std(last10))}

    del model, dd
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    return {"test_metric": tm, "test_metrics_full": tmf,
            "best_val_metric": best_val, "train_losses": losses,
            "gradient_analysis": ga, "prmp_diagnostics": diag,
            "num_epochs": len(losses)}


def _metric(out, labels, mask, lt, nc, cfg) -> float:
    if mask.sum() == 0:
        return 0.0
    p = out[mask].cpu()
    t = labels[mask].cpu()
    if lt == "MAE":
        return float(mean_absolute_error(t.numpy(), p.squeeze(-1).numpy()))
    if lt == "MSE":
        return float(mean_absolute_error(t.numpy(), p.squeeze(-1).numpy()))
    if lt == "CrossEntropy":
        return float(accuracy_score(t.numpy(), p.argmax(-1).numpy()))
    if lt == "BCE":
        pr = torch.sigmoid(p.squeeze(-1)).numpy()
        tg = t.numpy()
        if len(np.unique(tg)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(tg, pr))
        except ValueError:
            return 0.5
    return 0.0


def _all_metrics(out, labels, mask, lt, nc, cfg) -> dict:
    if mask.sum() == 0:
        return {}
    p = out[mask].cpu()
    t = labels[mask].cpu()
    m = {}
    if lt == "MAE":
        pv = p.squeeze(-1).numpy()
        tv = t.numpy()
        m["mae"] = float(mean_absolute_error(tv, pv))
        if len(tv) > 1:
            m["r2"] = float(r2_score(tv, pv))
    elif lt == "MSE":
        pv = p.squeeze(-1).numpy()
        tv = t.numpy()
        m["mae"] = float(mean_absolute_error(tv, pv))
        m["mse"] = float(np.mean((tv - pv)**2))
        bint = (tv > 0.5).astype(int)
        if len(np.unique(bint)) >= 2:
            try:
                m["auroc"] = float(roc_auc_score(bint, pv))
            except ValueError:
                pass
    elif lt == "CrossEntropy":
        pc = p.argmax(-1).numpy()
        tc = t.numpy()
        m["accuracy"] = float(accuracy_score(tc, pc))
        if "bin_centers" in cfg:
            bc = cfg["bin_centers"].numpy()
            m["mae_bin_centers"] = float(mean_absolute_error(bc[tc.astype(int)], bc[pc.astype(int)]))
    elif lt == "BCE":
        pr = torch.sigmoid(p.squeeze(-1)).numpy()
        tg = t.numpy()
        if len(np.unique(tg)) >= 2:
            try:
                m["auroc"] = float(roc_auc_score(tg, pr))
                m["avg_precision"] = float(average_precision_score(tg, pr))
            except ValueError:
                m["auroc"] = 0.5
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Analysis & Output
# ══════════════════════════════════════════════════════════════════════════════
def analyze(all_res: dict, configs: dict) -> dict:
    deltas = {}
    for cn in all_res:
        lt = configs[cn]["loss_type"]
        lower = lt in ["MAE", "MSE"]
        ds = []
        for s in SEEDS:
            sk = f"seed_{s}"
            sm = all_res[cn]["standard"].get(sk, {}).get("test_metric", float("nan"))
            pm = all_res[cn]["prmp"].get(sk, {}).get("test_metric", float("nan"))
            if not (np.isnan(sm) or np.isnan(pm)):
                ds.append((sm - pm) if lower else (pm - sm))
        deltas[cn] = {"mean": float(np.mean(ds)) if ds else 0.0,
                      "std": float(np.std(ds)) if ds else 0.0, "values": ds}

    # Loss-function hypothesis: {c1,c4} regression vs {c2,c3} classification
    reg_d = deltas.get("config1_natural_regression", {}).get("values", []) + \
            deltas.get("config4_softened_regression", {}).get("values", [])
    cls_d = deltas.get("config2_binned_classification", {}).get("values", []) + \
            deltas.get("config3_natural_classification", {}).get("values", [])

    rm = float(np.mean(reg_d)) if reg_d else 0.0
    cm = float(np.mean(cls_d)) if cls_d else 0.0
    if len(reg_d) >= 2 and len(cls_d) >= 2:
        ts, pv = stats.ttest_ind(reg_d, cls_d)
        ls = pv < 0.1
    else:
        ts, pv, ls = 0.0, 1.0, False

    loss_hyp = {
        "regression_loss_configs_delta_mean": rm,
        "classification_loss_configs_delta_mean": cm,
        "t_statistic": float(ts), "p_value": float(pv),
        "effect_significant": bool(ls),
        "interpretation": f"Reg-loss delta={rm:.4f}, cls-loss delta={cm:.4f}, p={pv:.4f}",
    }

    # Target-nature hypothesis: {c1,c2} position vs {c3,c4} binary
    pos_d = deltas.get("config1_natural_regression", {}).get("values", []) + \
            deltas.get("config2_binned_classification", {}).get("values", [])
    bin_d = deltas.get("config3_natural_classification", {}).get("values", []) + \
            deltas.get("config4_softened_regression", {}).get("values", [])

    pm2 = float(np.mean(pos_d)) if pos_d else 0.0
    bm2 = float(np.mean(bin_d)) if bin_d else 0.0
    if len(pos_d) >= 2 and len(bin_d) >= 2:
        ts2, pv2 = stats.ttest_ind(pos_d, bin_d)
        ts2_sig = pv2 < 0.1
    else:
        ts2, pv2, ts2_sig = 0.0, 1.0, False

    tgt_hyp = {
        "position_target_configs_delta_mean": pm2,
        "binary_target_configs_delta_mean": bm2,
        "t_statistic": float(ts2), "p_value": float(pv2),
        "effect_significant": bool(ts2_sig),
        "interpretation": f"Position delta={pm2:.4f}, binary delta={bm2:.4f}, p={pv2:.4f}",
    }

    # Gradient routing
    gr = {}
    for cn in all_res:
        rs = []
        for s in SEEDS:
            ga = all_res[cn]["prmp"].get(f"seed_{s}", {}).get("gradient_analysis", {})
            if "ratio_last_10_mean" in ga:
                rs.append(ga["ratio_last_10_mean"])
        if rs:
            gr[cn] = float(np.mean(rs))

    # Conclusion
    le = abs(rm - cm)
    te = abs(pm2 - bm2)
    if ls and not ts2_sig:
        conclusion = "Loss-driven"
    elif ts2_sig and not ls:
        conclusion = "Target-driven"
    elif ls and ts2_sig:
        conclusion = "Mixed"
    elif le > te * 1.5:
        conclusion = "Weak evidence for loss-driven"
    elif te > le * 1.5:
        conclusion = "Weak evidence for target-driven"
    else:
        conclusion = "Inconclusive"

    return {
        "deltas_per_config": {k: {"mean": v["mean"], "std": v["std"]} for k, v in deltas.items()},
        "loss_function_hypothesis": loss_hyp,
        "target_nature_hypothesis": tgt_hyp,
        "gradient_routing_analysis": {"per_config_grad_ratios": gr},
        "conclusion": conclusion,
    }


def build_output(all_res, configs, gstats, analysis):
    """Build output with one example per (config, seed), predict_standard and predict_prmp."""
    examples = []
    for cn, cfg in configs.items():
        d = analysis["deltas_per_config"].get(cn, {})
        for seed in SEEDS:
            sk = f"seed_{seed}"
            sr = all_res[cn]["standard"].get(sk, {})
            pr = all_res[cn]["prmp"].get(sk, {})

            inp = {"config_name": cn, "loss_type": cfg["loss_type"],
                   "target_type": cfg["target_type"], "num_classes": cfg["num_classes"],
                   "seed": seed, "description": cfg["description"]}

            # Ground truth output: combined metrics for this config+seed
            out_data = {
                "standard_test_metric": sr.get("test_metric", 0.0),
                "prmp_test_metric": pr.get("test_metric", 0.0),
                "standard_metrics": sr.get("test_metrics_full", {}),
                "prmp_metrics": pr.get("test_metrics_full", {}),
                "prmp_delta": (sr.get("test_metric", 0.0) - pr.get("test_metric", 0.0))
                    if cfg["loss_type"] in ["MAE", "MSE"]
                    else (pr.get("test_metric", 0.0) - sr.get("test_metric", 0.0)),
                "gradient_analysis": pr.get("gradient_analysis", {}),
                "prmp_diagnostics": pr.get("prmp_diagnostics", {}),
            }

            # predict_* fields: string predictions from each method
            std_pred = sr.get("test_metrics_full", {}).copy()
            std_pred["primary_metric"] = sr.get("test_metric", 0.0)
            std_pred["best_val"] = sr.get("best_val_metric", 0.0)
            std_pred["num_epochs"] = sr.get("num_epochs", 0)

            prmp_pred = pr.get("test_metrics_full", {}).copy()
            prmp_pred["primary_metric"] = pr.get("test_metric", 0.0)
            prmp_pred["best_val"] = pr.get("best_val_metric", 0.0)
            prmp_pred["num_epochs"] = pr.get("num_epochs", 0)
            prmp_pred["gradient_analysis"] = pr.get("gradient_analysis", {})
            prmp_pred["prmp_diagnostics"] = pr.get("prmp_diagnostics", {})

            examples.append({
                "input": json.dumps(inp, separators=(",", ":"), default=str),
                "output": json.dumps(out_data, separators=(",", ":"), default=str),
                "predict_standard": json.dumps(std_pred, separators=(",", ":"), default=str),
                "predict_prmp": json.dumps(prmp_pred, separators=(",", ":"), default=str),
                "metadata_config": cn,
                "metadata_loss_type": cfg["loss_type"],
                "metadata_target_type": cfg["target_type"],
                "metadata_seed": seed,
                "metadata_prmp_delta_mean": d.get("mean", 0.0),
            })

    return {
        "metadata": {
            "method_name": "PRMP Loss Function Swap Experiment on rel-f1",
            "description": f"Controlled experiment: 4 configs x 2 methods x {len(SEEDS)} seeds = {4*2*len(SEEDS)} runs",
            "configs": list(configs.keys()),
            "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "seeds": SEEDS,
            "graph_stats": {k: int(v) if isinstance(v, (int, np.integer)) else v
                           for k, v in gstats.items()},
            "analysis": analysis,
        },
        "datasets": [{"dataset": "rel-f1-loss-swap", "examples": examples}],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PRMP Loss Function Swap Experiment on rel-f1")
    logger.info("=" * 60)

    # Phase 0-1: Data + Graph
    tables = download_ergast_csv()
    data, id_maps, gstats = build_hetero_graph(tables)

    # Phase 2: Targets
    configs = build_targets(tables, id_maps)
    del tables
    gc.collect()

    # Smoke test
    logger.info("SMOKE TEST: 5 epochs, config1, standard, seed=42")
    sr = train_run(data, configs["config1_natural_regression"], "standard", 42, 5, 5)
    logger.info(f"Smoke: metric={sr['test_metric']:.4f}, losses={sr['train_losses']}")
    if len(sr["train_losses"]) >= 2 and sr["train_losses"][-1] < sr["train_losses"][0]:
        logger.info("Smoke test PASSED")
    else:
        logger.warning("Smoke test: loss did not decrease")

    # Full experiment
    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT: 24 runs")
    all_res = {}
    methods = ["standard", "prmp"]
    ri = 0
    total = len(configs) * len(methods) * len(SEEDS)

    for cn, cfg in configs.items():
        all_res[cn] = {"standard": {}, "prmp": {}}
        for meth in methods:
            for seed in SEEDS:
                ri += 1
                t1 = time.time()
                logger.info(f"[{ri}/{total}] {cn}|{meth}|seed={seed}")
                try:
                    res = train_run(data, cfg, meth, seed)
                    logger.info(f"  {time.time()-t1:.1f}s metric={res['test_metric']:.4f} ep={res['num_epochs']}")
                    all_res[cn][meth][f"seed_{seed}"] = res
                except Exception:
                    logger.exception(f"  FAILED")
                    all_res[cn][meth][f"seed_{seed}"] = {
                        "test_metric": float("nan"), "test_metrics_full": {},
                        "gradient_analysis": {}, "prmp_diagnostics": {},
                    }

    # Analysis
    analysis = analyze(all_res, configs)
    logger.info(f"Conclusion: {analysis['conclusion']}")

    # Output
    output = build_output(all_res, configs, gstats, analysis)
    output = sanitize_for_json(output)
    op = WS / "method_out.json"
    op.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {op} ({op.stat().st_size/1e6:.2f} MB)")
    logger.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
