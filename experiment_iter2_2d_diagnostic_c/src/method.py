#!/usr/bin/env python3
"""2D Diagnostic: Cross-Table Predictability (R²) and Cardinality for All FK Links.

Computes R² cross-table predictability (Ridge, 5-fold CV) and mutual information
for every FK link across 3 datasets (rel-hm: 2, rel-stack: 9, rel-amazon: 2 = 13 total).
Produces a structured results table and 2D diagnostic scatter plots.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)  # 70% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, RAM budget={RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
PIPELINE_ROOT = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_1/gen_art")
HM_WS = PIPELINE_ROOT / "data_id3_it1__opus"
STACK_WS = PIPELINE_ROOT / "data_id4_it1__opus"
AMZN_WS = PIPELINE_ROOT / "data_id5_it1__opus"

# ---------------------------------------------------------------------------
# FK Link Data structure
# ---------------------------------------------------------------------------

def make_fk_link(
    dataset: str,
    parent_table: str,
    child_table: str,
    fk_column: str,
    X_parent: np.ndarray,
    X_child: np.ndarray,
    mean_cardinality: float,
    median_cardinality: float,
    max_cardinality: float | None,
    parent_feature_names: list[str],
    child_feature_names: list[str],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "parent_table": parent_table,
        "child_table": child_table,
        "fk_column": fk_column,
        "X_parent": X_parent,
        "X_child": X_child,
        "mean_cardinality": mean_cardinality,
        "median_cardinality": median_cardinality,
        "max_cardinality": max_cardinality,
        "parent_feature_names": parent_feature_names,
        "child_feature_names": child_feature_names,
        "n_samples": X_parent.shape[0],
    }


# ---------------------------------------------------------------------------
# STEP 1a: LOADER FOR rel-hm (parquet format)
# ---------------------------------------------------------------------------

def load_rel_hm() -> list[dict]:
    logger.info("Loading rel-hm FK links from parquet + JSON...")
    t0 = time.time()

    # Load cardinality stats from full_data_out.json (small file, 1.6MB)
    data = json.loads((HM_WS / "full_data_out.json").read_text())
    examples = data["datasets"][0]["examples"]

    fk_links = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        parent_table = ex["metadata_parent_table"]
        child_table = ex["metadata_child_table"]
        fk_col = ex["metadata_fk_column"]
        supp_file = ex["metadata_supplementary_file"]

        parent_feats = inp["parent_feature_columns"]
        child_feats = inp["child_feature_columns"]
        mean_card = out["mean_cardinality"]
        median_card = out["median_cardinality"]
        max_card = out["max_cardinality"]

        # Load parquet
        parquet_path = HM_WS / supp_file
        logger.info(f"  Loading parquet: {parquet_path.name}")
        df = pd.read_parquet(parquet_path)

        parent_cols = [c for c in df.columns if c.startswith("parent__")]
        child_cols = [c for c in df.columns if c.startswith("child__")]

        X_parent = df[parent_cols].values.astype(np.float64)
        X_child = df[child_cols].values.astype(np.float64)

        # Subsample to 20K for memory efficiency
        MAX_ROWS = 20000
        if X_parent.shape[0] > MAX_ROWS:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_parent.shape[0], MAX_ROWS, replace=False)
            X_parent = X_parent[idx]
            X_child = X_child[idx]
            logger.info(f"    Subsampled {parent_table}->{child_table} to {MAX_ROWS} rows")

        del df
        gc.collect()

        fk_links.append(make_fk_link(
            dataset="rel-hm",
            parent_table=parent_table,
            child_table=child_table,
            fk_column=fk_col,
            X_parent=X_parent,
            X_child=X_child,
            mean_cardinality=mean_card,
            median_cardinality=median_card,
            max_cardinality=max_card,
            parent_feature_names=[c.replace("parent__", "") for c in parent_cols],
            child_feature_names=[c.replace("child__", "") for c in child_cols],
        ))
        logger.info(f"    {parent_table}->{child_table}: {X_parent.shape} parent, {X_child.shape} child, card_mean={mean_card}")

    logger.info(f"rel-hm loaded {len(fk_links)} FK links in {time.time()-t0:.1f}s")
    return fk_links


# ---------------------------------------------------------------------------
# STEP 1b: LOADER FOR rel-stack (JSON row-per-example format)
# ---------------------------------------------------------------------------

def load_rel_stack() -> list[dict]:
    logger.info("Loading rel-stack FK links from JSON...")
    t0 = time.time()

    data = json.loads((STACK_WS / "full_data_out.json").read_text())
    fk_links = []

    for ds_entry in data["datasets"]:
        ds_name = ds_entry["dataset"]  # e.g. "rel-stack/badges__UserId__users"
        examples = ds_entry["examples"]
        ex0 = examples[0]

        parent_table = ex0["metadata_parent_table"]
        child_table = ex0["metadata_child_table"]
        fk_col = ex0["metadata_fkey_col"]
        parent_feat_names = ex0["metadata_parent_feature_names"]
        child_feat_names = ex0["metadata_child_feature_names"]
        mean_card = ex0["metadata_cardinality_mean"]
        median_card = ex0["metadata_cardinality_median"]

        # Build X_parent and X_child from examples
        X_parent_list = []
        X_child_list = []
        for example in examples:
            try:
                parent_feats = json.loads(example["input"])
                child_feats = json.loads(example["output"])
                X_parent_list.append([float(v) for v in parent_feats.values()])
                X_child_list.append([float(v) for v in child_feats.values()])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        if len(X_parent_list) < 10:
            logger.warning(f"  Skipping {ds_name}: only {len(X_parent_list)} usable rows")
            continue

        X_parent = np.array(X_parent_list, dtype=np.float64)
        X_child = np.array(X_child_list, dtype=np.float64)

        fk_links.append(make_fk_link(
            dataset="rel-stack",
            parent_table=parent_table,
            child_table=child_table,
            fk_column=fk_col,
            X_parent=X_parent,
            X_child=X_child,
            mean_cardinality=mean_card,
            median_cardinality=median_card,
            max_cardinality=None,  # Not directly available
            parent_feature_names=parent_feat_names,
            child_feature_names=child_feat_names,
        ))
        logger.info(f"  {ds_name}: {X_parent.shape} parent, {X_child.shape} child, card_mean={mean_card:.2f}")

    del data
    gc.collect()
    logger.info(f"rel-stack loaded {len(fk_links)} FK links in {time.time()-t0:.1f}s")
    return fk_links


# ---------------------------------------------------------------------------
# STEP 1c: LOADER FOR rel-amazon (aligned arrays in metadata)
# ---------------------------------------------------------------------------

def load_rel_amazon() -> list[dict]:
    logger.info("Loading rel-amazon FK links from JSON metadata...")
    t0 = time.time()

    data = json.loads((AMZN_WS / "full_data_out.json").read_text())
    meta = data["metadata"]["datasets_info"]["amazon_video_games"]
    fk_info = meta["fk_links"]

    fk_links = []
    for link_name, info in fk_info.items():
        parent_table = info["parent_table"]
        child_table = info["child_table"]
        fk_col = info["fk_col"]
        mean_card = info["cardinality_mean"]
        median_card = info["cardinality_median"]
        max_card = info["cardinality_max"]
        parent_feat_names = info["parent_feature_names"]
        child_feat_names = info["child_feature_names"]

        X_parent = np.array(info["aligned_parent_features_sample"], dtype=np.float64)
        X_child = np.array(info["aligned_child_features_sample"], dtype=np.float64)

        fk_links.append(make_fk_link(
            dataset="rel-amazon",
            parent_table=parent_table,
            child_table=child_table,
            fk_column=fk_col,
            X_parent=X_parent,
            X_child=X_child,
            mean_cardinality=mean_card,
            median_cardinality=median_card,
            max_cardinality=max_card,
            parent_feature_names=parent_feat_names,
            child_feature_names=child_feat_names,
        ))
        logger.info(f"  {link_name}: {X_parent.shape} parent, {X_child.shape} child, card_mean={mean_card:.2f}")

    del data
    gc.collect()
    logger.info(f"rel-amazon loaded {len(fk_links)} FK links in {time.time()-t0:.1f}s")
    return fk_links


# ---------------------------------------------------------------------------
# STEP 2: COMPUTE R² WITH RIDGE REGRESSION (5-FOLD CV)
# ---------------------------------------------------------------------------

def compute_r2_ridge(X_parent: np.ndarray, X_child: np.ndarray) -> dict[str, float]:
    """Compute R² using Ridge regression with 5-fold CV."""
    # Handle NaN/inf
    X_parent = np.nan_to_num(X_parent, nan=0.0, posinf=0.0, neginf=0.0)
    X_child = np.nan_to_num(X_child, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize parent features
    scaler = StandardScaler()
    X_parent_scaled = scaler.fit_transform(X_parent)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Per-dimension Ridge R²
    r2_per_dim = []
    r2_std_per_dim = []

    for dim_j in range(X_child.shape[1]):
        y = X_child[:, dim_j]
        # Check if target has variance
        if np.std(y) < 1e-10:
            r2_per_dim.append(0.0)
            r2_std_per_dim.append(0.0)
            continue

        fold_r2s = []
        for train_idx, test_idx in kf.split(X_parent_scaled):
            model = Ridge(alpha=1.0)
            model.fit(X_parent_scaled[train_idx], y[train_idx])
            y_pred = model.predict(X_parent_scaled[test_idx])
            r2 = r2_score(y[test_idx], y_pred)
            fold_r2s.append(max(0.0, r2))  # Cap at 0 for negative R²

        r2_per_dim.append(float(np.mean(fold_r2s)))
        r2_std_per_dim.append(float(np.std(fold_r2s)))

    R2_ridge = float(np.mean(r2_per_dim))
    R2_std = float(np.mean(r2_std_per_dim))

    # Multi-output Ridge R² (cross-check)
    try:
        model_multi = MultiOutputRegressor(Ridge(alpha=1.0))
        from sklearn.model_selection import cross_val_predict
        y_pred_all = cross_val_predict(model_multi, X_parent_scaled, X_child, cv=kf)
        R2_ridge_multi = max(0.0, float(r2_score(X_child, y_pred_all, multioutput='uniform_average')))
    except Exception as e:
        logger.warning(f"MultiOutput R² failed: {e}, using per-dim average")
        R2_ridge_multi = R2_ridge

    return {
        "R2_ridge": R2_ridge,
        "R2_ridge_multi": R2_ridge_multi,
        "R2_std": R2_std,
        "R2_per_dim": r2_per_dim,
    }


# ---------------------------------------------------------------------------
# STEP 2b: BASELINE - Random Forest R² for nonlinear comparison
# ---------------------------------------------------------------------------

def compute_r2_rf(X_parent: np.ndarray, X_child: np.ndarray) -> dict[str, float]:
    """Compute R² using Random Forest as a nonlinear baseline."""
    from sklearn.ensemble import RandomForestRegressor

    X_parent = np.nan_to_num(X_parent, nan=0.0, posinf=0.0, neginf=0.0)
    X_child = np.nan_to_num(X_child, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_parent_scaled = scaler.fit_transform(X_parent)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Subsample for RF to keep things fast
    MAX_RF_ROWS = 5000
    if X_parent_scaled.shape[0] > MAX_RF_ROWS:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_parent_scaled.shape[0], MAX_RF_ROWS, replace=False)
        X_parent_scaled = X_parent_scaled[idx]
        X_child = X_child[idx]

    r2_per_dim = []
    for dim_j in range(X_child.shape[1]):
        y = X_child[:, dim_j]
        if np.std(y) < 1e-10:
            r2_per_dim.append(0.0)
            continue

        fold_r2s = []
        for train_idx, test_idx in kf.split(X_parent_scaled):
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
            model.fit(X_parent_scaled[train_idx], y[train_idx])
            y_pred = model.predict(X_parent_scaled[test_idx])
            r2 = r2_score(y[test_idx], y_pred)
            fold_r2s.append(max(0.0, r2))
        r2_per_dim.append(float(np.mean(fold_r2s)))

    return {"R2_rf": float(np.mean(r2_per_dim))}


# ---------------------------------------------------------------------------
# STEP 3: COMPUTE MUTUAL INFORMATION
# ---------------------------------------------------------------------------

def compute_mutual_info(X_parent: np.ndarray, X_child: np.ndarray, max_mi_rows: int = 5000) -> float:
    """Compute average mutual information between parent and child features."""
    X_parent = np.nan_to_num(X_parent, nan=0.0, posinf=0.0, neginf=0.0)
    X_child = np.nan_to_num(X_child, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_parent_scaled = scaler.fit_transform(X_parent)

    # Subsample for MI to keep runtime manageable
    if X_parent_scaled.shape[0] > max_mi_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_parent_scaled.shape[0], max_mi_rows, replace=False)
        X_parent_scaled = X_parent_scaled[idx]
        X_child = X_child[idx]

    mi_values = []
    for dim_j in range(X_child.shape[1]):
        y = X_child[:, dim_j]
        if np.std(y) < 1e-10:
            mi_values.append(0.0)
            continue
        try:
            mi = mutual_info_regression(X_parent_scaled, y, random_state=42, n_neighbors=5)
            mi_values.append(float(np.mean(mi)))
        except Exception as e:
            logger.warning(f"MI computation failed for dim {dim_j}: {e}")
            mi_values.append(0.0)

    return float(np.mean(mi_values))


# ---------------------------------------------------------------------------
# Worker function for parallel processing
# ---------------------------------------------------------------------------

def process_single_fk_link(args: tuple) -> dict:
    """Process a single FK link: compute R², RF R², and MI."""
    idx, link_data = args
    name = f"{link_data['dataset']}/{link_data['parent_table']}->{link_data['child_table']}"

    t0 = time.time()
    result = {
        "dataset": link_data["dataset"],
        "parent_table": link_data["parent_table"],
        "child_table": link_data["child_table"],
        "fk_column": link_data["fk_column"],
        "mean_cardinality": link_data["mean_cardinality"],
        "median_cardinality": link_data["median_cardinality"],
        "max_cardinality": link_data["max_cardinality"],
        "parent_dim": link_data["X_parent"].shape[1],
        "child_dim": link_data["X_child"].shape[1],
        "n_samples": link_data["n_samples"],
        "parent_feature_names": link_data["parent_feature_names"],
        "child_feature_names": link_data["child_feature_names"],
    }

    X_parent = link_data["X_parent"]
    X_child = link_data["X_child"]

    # R² Ridge
    try:
        r2_results = compute_r2_ridge(X_parent, X_child)
        result.update(r2_results)
    except Exception as e:
        logger.exception(f"R² Ridge failed for {name}: {e}")
        result["R2_ridge"] = 0.0
        result["R2_ridge_multi"] = 0.0
        result["R2_std"] = 0.0
        result["R2_per_dim"] = []

    # R² Random Forest (baseline)
    try:
        rf_results = compute_r2_rf(X_parent, X_child)
        result.update(rf_results)
    except Exception as e:
        logger.exception(f"R² RF failed for {name}: {e}")
        result["R2_rf"] = 0.0

    # Mutual Information
    try:
        mi = compute_mutual_info(X_parent, X_child)
        result["mutual_info_mean"] = mi
    except Exception as e:
        logger.exception(f"MI failed for {name}: {e}")
        result["mutual_info_mean"] = 0.0

    result["log_mean_cardinality"] = float(np.log10(max(result["mean_cardinality"], 0.01)))
    result["runtime_seconds"] = time.time() - t0

    return result


# ---------------------------------------------------------------------------
# STEP 5: GENERATE SCATTER PLOTS
# ---------------------------------------------------------------------------

def generate_scatter_plots(results_table: list[dict], output_dir: Path) -> list[str]:
    """Generate 2D diagnostic scatter plots with adaptive axis scaling."""
    figure_files = []

    colors = {"rel-hm": "#e74c3c", "rel-stack": "#3498db", "rel-amazon": "#2ecc71"}
    markers = {"rel-hm": "o", "rel-stack": "s", "rel-amazon": "D"}

    def _adaptive_ylim(values: list[float], pad_frac: float = 0.15) -> tuple[float, float]:
        """Compute adaptive y-axis limits with padding."""
        vmin, vmax = min(values), max(values)
        span = vmax - vmin
        if span < 1e-6:
            span = max(abs(vmax), 0.01)
        return vmin - span * pad_frac, vmax + span * pad_frac

    from adjustText import adjust_text as _adjust_text  # type: ignore
    _has_adjust_text = True

    # --- Plot 1: Cardinality vs R² (Ridge) ---
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    plotted_labels: set[str] = set()
    texts = []

    for row in results_table:
        ds = row["dataset"]
        legend_label = ds if ds not in plotted_labels else None
        plotted_labels.add(ds)
        ax.scatter(
            row["mean_cardinality"], row["R2_ridge"],
            c=colors[ds], marker=markers[ds], s=120, edgecolors="black",
            linewidth=0.8, zorder=3, label=legend_label,
        )
        label = f"{row['parent_table']}\u2192{row['child_table']}"
        if _has_adjust_text:
            texts.append(ax.text(row["mean_cardinality"], row["R2_ridge"], label, fontsize=7))
        else:
            ax.annotate(label, (row["mean_cardinality"], row["R2_ridge"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=7)

    if _has_adjust_text and texts:
        try:
            _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except Exception:
            pass  # Fall back to default positioning

    ax.set_xscale("log")
    ax.set_xlabel("Mean Join Cardinality (log scale)", fontsize=12)
    ax.set_ylabel("Cross-Table Predictability (R², Ridge CV)", fontsize=12)
    ax.set_title(f"2D FK Diagnostic: Cardinality vs Predictability\nacross {len(results_table)} FK Links in 3 RelBench Datasets", fontsize=13)

    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper left")

    r2_vals = [r["R2_ridge"] for r in results_table]
    ylo, yhi = _adaptive_ylim(r2_vals)
    ax.set_ylim(max(ylo, -0.01), yhi)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname1 = "fk_diagnostic_2d_scatter.png"
    plt.savefig(output_dir / fname1, dpi=150, bbox_inches="tight")
    plt.close()
    figure_files.append(fname1)
    logger.info(f"Saved {fname1}")

    # --- Plot 2: Cardinality vs MI ---
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    plotted_labels = set()
    texts = []

    for row in results_table:
        ds = row["dataset"]
        legend_label = ds if ds not in plotted_labels else None
        plotted_labels.add(ds)
        ax.scatter(
            row["mean_cardinality"], row["mutual_info_mean"],
            c=colors[ds], marker=markers[ds], s=120, edgecolors="black",
            linewidth=0.8, zorder=3, label=legend_label,
        )
        label = f"{row['parent_table']}\u2192{row['child_table']}"
        if _has_adjust_text:
            texts.append(ax.text(row["mean_cardinality"], row["mutual_info_mean"], label, fontsize=7))
        else:
            ax.annotate(label, (row["mean_cardinality"], row["mutual_info_mean"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=7)

    if _has_adjust_text and texts:
        try:
            _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except Exception:
            pass

    ax.set_xscale("log")
    ax.set_xlabel("Mean Join Cardinality (log scale)", fontsize=12)
    ax.set_ylabel("Mutual Information (avg across dims)", fontsize=12)
    ax.set_title(f"2D FK Diagnostic: Cardinality vs Mutual Information\nacross {len(results_table)} FK Links in 3 RelBench Datasets", fontsize=13)

    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper left")

    mi_vals = [r["mutual_info_mean"] for r in results_table]
    ylo, yhi = _adaptive_ylim(mi_vals)
    ax.set_ylim(max(ylo, -0.005), yhi)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname2 = "fk_diagnostic_mi_scatter.png"
    plt.savefig(output_dir / fname2, dpi=150, bbox_inches="tight")
    plt.close()
    figure_files.append(fname2)
    logger.info(f"Saved {fname2}")

    # --- Plot 3: R² Ridge vs R² RF comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    plotted_labels = set()
    texts = []

    for row in results_table:
        ds = row["dataset"]
        legend_label = ds if ds not in plotted_labels else None
        plotted_labels.add(ds)
        ax.scatter(
            row["R2_ridge"], row.get("R2_rf", 0.0),
            c=colors[ds], marker=markers[ds], s=120, edgecolors="black",
            linewidth=0.8, zorder=3, label=legend_label,
        )
        label = f"{row['parent_table']}\u2192{row['child_table']}"
        if _has_adjust_text:
            texts.append(ax.text(row["R2_ridge"], row.get("R2_rf", 0.0), label, fontsize=7))
        else:
            ax.annotate(label, (row["R2_ridge"], row.get("R2_rf", 0.0)),
                        textcoords="offset points", xytext=(8, 4), fontsize=7)

    if _has_adjust_text and texts:
        try:
            _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except Exception:
            pass

    all_r2 = r2_vals + [r.get("R2_rf", 0.0) for r in results_table]
    lim_max = max(all_r2) * 1.2 + 0.01
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("R² (Ridge, linear)", fontsize=12)
    ax.set_ylabel("R² (Random Forest, nonlinear)", fontsize=12)
    ax.set_title("Linear vs Nonlinear Cross-Table Predictability", fontsize=13)

    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper left")

    ax.set_xlim(-lim_max * 0.05, lim_max)
    ax.set_ylim(-lim_max * 0.05, lim_max)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fname3 = "fk_diagnostic_ridge_vs_rf.png"
    plt.savefig(output_dir / fname3, dpi=150, bbox_inches="tight")
    plt.close()
    figure_files.append(fname3)
    logger.info(f"Saved {fname3}")

    return figure_files


# ---------------------------------------------------------------------------
# STEP 6: COMPUTE CORRELATIONS
# ---------------------------------------------------------------------------

def compute_correlations(results_table: list[dict]) -> dict:
    """Compute Pearson/Spearman correlations between metrics."""
    cardinalities = [r["mean_cardinality"] for r in results_table]
    r2_values = [r["R2_ridge"] for r in results_table]
    mi_values = [r["mutual_info_mean"] for r in results_table]
    log_cards = [r["log_mean_cardinality"] for r in results_table]
    rf_values = [r.get("R2_rf", 0.0) for r in results_table]

    correlations = {}

    # Pearson(log_cardinality, R²)
    if len(log_cards) >= 3:
        r, p = scipy.stats.pearsonr(log_cards, r2_values)
        correlations["pearson_logcard_r2"] = {"r": float(r), "p": float(p)}
    else:
        correlations["pearson_logcard_r2"] = {"r": 0.0, "p": 1.0}

    # Spearman(cardinality, R²)
    if len(cardinalities) >= 3:
        rho, p = scipy.stats.spearmanr(cardinalities, r2_values)
        correlations["spearman_card_r2"] = {"rho": float(rho), "p": float(p)}
    else:
        correlations["spearman_card_r2"] = {"rho": 0.0, "p": 1.0}

    # Pearson(R², MI)
    if len(r2_values) >= 3:
        r, p = scipy.stats.pearsonr(r2_values, mi_values)
        correlations["pearson_r2_mi"] = {"r": float(r), "p": float(p)}
    else:
        correlations["pearson_r2_mi"] = {"r": 0.0, "p": 1.0}

    # Pearson(R² Ridge, R² RF)
    if len(r2_values) >= 3:
        r, p = scipy.stats.pearsonr(r2_values, rf_values)
        correlations["pearson_ridge_rf"] = {"r": float(r), "p": float(p)}
    else:
        correlations["pearson_ridge_rf"] = {"r": 0.0, "p": 1.0}

    return correlations


# ---------------------------------------------------------------------------
# STEP 7: REGIME CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_regimes(results_table: list[dict]) -> dict:
    """Classify each FK link into quadrants using median thresholds."""
    cardinalities = [r["mean_cardinality"] for r in results_table]
    r2_values = [r["R2_ridge"] for r in results_table]

    card_threshold = float(np.median(cardinalities))
    r2_threshold = float(np.median(r2_values))

    regimes = {
        "card_threshold": card_threshold,
        "r2_threshold": r2_threshold,
        "high_card_high_pred": [],
        "high_card_low_pred": [],
        "low_card_high_pred": [],
        "low_card_low_pred": [],
    }

    for r in results_table:
        label = f"{r['dataset']}/{r['parent_table']}->{r['child_table']}"
        high_card = r["mean_cardinality"] >= card_threshold
        high_pred = r["R2_ridge"] >= r2_threshold

        if high_card and high_pred:
            regimes["high_card_high_pred"].append(label)
        elif high_card and not high_pred:
            regimes["high_card_low_pred"].append(label)
        elif not high_card and high_pred:
            regimes["low_card_high_pred"].append(label)
        else:
            regimes["low_card_low_pred"].append(label)

    return regimes


# ---------------------------------------------------------------------------
# STEP 8: BUILD OUTPUT IN exp_gen_sol_out FORMAT
# ---------------------------------------------------------------------------

def build_output(results_table: list[dict], correlations: dict, regimes: dict, figure_files: list[str]) -> dict:
    """Build output in exp_gen_sol_out.json schema format."""

    cardinalities = [r["mean_cardinality"] for r in results_table]
    r2_values = [r["R2_ridge"] for r in results_table]
    mi_values = [r["mutual_info_mean"] for r in results_table]
    rf_values = [r.get("R2_rf", 0.0) for r in results_table]

    r2_range = [float(min(r2_values)), float(max(r2_values))]
    card_range = [float(min(cardinalities)), float(max(cardinalities))]
    mi_range = [float(min(mi_values)), float(max(mi_values))]

    spans_meaningful = (r2_range[1] - r2_range[0] > 0.1) and (card_range[1] / max(card_range[0], 0.01) > 10)

    # Build per-FK-link table (serializable, no numpy arrays)
    fk_link_table = []
    for r in results_table:
        entry = {
            "dataset": r["dataset"],
            "parent_table": r["parent_table"],
            "child_table": r["child_table"],
            "fk_column": r["fk_column"],
            "mean_cardinality": r["mean_cardinality"],
            "median_cardinality": r["median_cardinality"],
            "max_cardinality": r["max_cardinality"],
            "parent_dim": r["parent_dim"],
            "child_dim": r["child_dim"],
            "n_samples": r["n_samples"],
            "R2_ridge": r["R2_ridge"],
            "R2_ridge_multi": r.get("R2_ridge_multi", 0.0),
            "R2_std": r.get("R2_std", 0.0),
            "R2_rf": r.get("R2_rf", 0.0),
            "mutual_info_mean": r["mutual_info_mean"],
            "log_mean_cardinality": r["log_mean_cardinality"],
            "runtime_seconds": r.get("runtime_seconds", 0.0),
        }
        fk_link_table.append(entry)

    summary = {
        "R2_range": r2_range,
        "R2_mean": float(np.mean(r2_values)),
        "R2_rf_mean": float(np.mean(rf_values)),
        "cardinality_range": card_range,
        "MI_range": mi_range,
        "spans_meaningful_range": spans_meaningful,
    }

    # Build method_out.json in exp_gen_sol_out schema
    # Each FK link becomes one example: input=FK link info, output=R² and MI results
    examples = []
    for r in fk_link_table:
        input_dict = {
            "dataset": r["dataset"],
            "parent_table": r["parent_table"],
            "child_table": r["child_table"],
            "fk_column": r["fk_column"],
            "mean_cardinality": r["mean_cardinality"],
            "median_cardinality": r["median_cardinality"],
            "max_cardinality": r["max_cardinality"],
            "parent_dim": r["parent_dim"],
            "child_dim": r["child_dim"],
            "n_samples": r["n_samples"],
        }
        output_dict = {
            "R2_ridge": r["R2_ridge"],
            "R2_ridge_multi": r["R2_ridge_multi"],
            "R2_std": r["R2_std"],
            "R2_rf": r["R2_rf"],
            "mutual_info_mean": r["mutual_info_mean"],
            "log_mean_cardinality": r["log_mean_cardinality"],
        }
        example = {
            "input": json.dumps(input_dict),
            "output": json.dumps(output_dict),
            "predict_ridge_r2": str(r["R2_ridge"]),
            "predict_rf_r2": str(r["R2_rf"]),
            "metadata_dataset": r["dataset"],
            "metadata_parent_table": r["parent_table"],
            "metadata_child_table": r["child_table"],
            "metadata_fk_column": r["fk_column"],
            "metadata_mean_cardinality": r["mean_cardinality"],
            "metadata_R2_ridge": r["R2_ridge"],
            "metadata_R2_rf": r["R2_rf"],
            "metadata_mutual_info": r["mutual_info_mean"],
        }
        examples.append(example)

    output = {
        "metadata": {
            "description": "2D FK diagnostic: cross-table predictability and cardinality for 13 FK links across 3 RelBench datasets",
            "datasets_analyzed": ["rel-hm", "rel-stack", "rel-amazon"],
            "num_fk_links": len(results_table),
            "methods": ["Ridge regression (5-fold CV)", "Random Forest (5-fold CV, baseline)", "Mutual information (sklearn)"],
            "figure_files": figure_files,
            "correlations": correlations,
            "summary_statistics": summary,
            "regime_classification": regimes,
            "fk_link_table": fk_link_table,
        },
        "datasets": [
            {
                "dataset": "fk_diagnostic_2d",
                "examples": examples,
            }
        ],
    }

    return output


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

@logger.catch
def main():
    t_start = time.time()
    os.chdir(WORKSPACE)
    logger.info(f"Working directory: {WORKSPACE}")

    # ---- Load all FK links ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading FK links from all 3 datasets")
    logger.info("=" * 60)

    all_fk_links = []

    hm_links = load_rel_hm()
    all_fk_links.extend(hm_links)
    gc.collect()

    stack_links = load_rel_stack()
    all_fk_links.extend(stack_links)
    gc.collect()

    amazon_links = load_rel_amazon()
    all_fk_links.extend(amazon_links)
    gc.collect()

    logger.info(f"Total FK links loaded: {len(all_fk_links)}")
    assert len(all_fk_links) >= 5, f"Need at least 5 FK links, got {len(all_fk_links)}"

    # ---- Compute metrics for each FK link ----
    logger.info("=" * 60)
    logger.info("STEP 2-3: Computing R², RF R², and MI for all FK links")
    logger.info("=" * 60)

    results_table = []
    for i, link in enumerate(all_fk_links):
        name = f"{link['dataset']}/{link['parent_table']}->{link['child_table']}"
        logger.info(f"Processing [{i+1}/{len(all_fk_links)}]: {name}")

        result = process_single_fk_link((i, link))
        results_table.append(result)

        logger.info(
            f"  R2_ridge={result['R2_ridge']:.4f}, "
            f"R2_rf={result.get('R2_rf', 0):.4f}, "
            f"MI={result['mutual_info_mean']:.4f}, "
            f"time={result['runtime_seconds']:.1f}s"
        )

        # Free the large arrays
        link["X_parent"] = None
        link["X_child"] = None
        gc.collect()

    # ---- Print summary table ----
    logger.info("=" * 60)
    logger.info("STEP 4: Results Summary Table")
    logger.info("=" * 60)

    df_results = pd.DataFrame([{
        "dataset": r["dataset"],
        "parent->child": f"{r['parent_table']}->{r['child_table']}",
        "fk_col": r["fk_column"],
        "card_mean": f"{r['mean_cardinality']:.2f}",
        "p_dim": r["parent_dim"],
        "c_dim": r["child_dim"],
        "n": r["n_samples"],
        "R2_ridge": f"{r['R2_ridge']:.4f}",
        "R2_rf": f"{r.get('R2_rf', 0):.4f}",
        "MI": f"{r['mutual_info_mean']:.4f}",
    } for r in results_table])
    logger.info(f"\n{df_results.to_string(index=False)}")

    # ---- Generate plots ----
    logger.info("=" * 60)
    logger.info("STEP 5: Generating scatter plots")
    logger.info("=" * 60)

    figure_files = generate_scatter_plots(results_table, WORKSPACE)

    # ---- Compute correlations ----
    logger.info("=" * 60)
    logger.info("STEP 6: Computing correlations")
    logger.info("=" * 60)

    correlations = compute_correlations(results_table)
    for name, vals in correlations.items():
        logger.info(f"  {name}: {vals}")

    # ---- Regime classification ----
    regimes = classify_regimes(results_table)
    logger.info(f"Regime thresholds: card={regimes['card_threshold']:.2f}, R2={regimes['r2_threshold']:.4f}")
    for regime_name in ["high_card_high_pred", "high_card_low_pred", "low_card_high_pred", "low_card_low_pred"]:
        logger.info(f"  {regime_name}: {regimes[regime_name]}")

    # ---- Summary statistics ----
    cardinalities = [r["mean_cardinality"] for r in results_table]
    r2_values = [r["R2_ridge"] for r in results_table]
    mi_values = [r["mutual_info_mean"] for r in results_table]

    logger.info(f"R² range: [{min(r2_values):.4f}, {max(r2_values):.4f}], mean={np.mean(r2_values):.4f}")
    logger.info(f"Cardinality range: [{min(cardinalities):.2f}, {max(cardinalities):.2f}]")
    logger.info(f"MI range: [{min(mi_values):.4f}, {max(mi_values):.4f}]")

    spans_meaningful = (max(r2_values) - min(r2_values) > 0.1) and (max(cardinalities) / max(min(cardinalities), 0.01) > 10)
    logger.info(f"Spans meaningful range in 2D: {spans_meaningful}")

    # ---- Build and save output ----
    logger.info("=" * 60)
    logger.info("STEP 7: Saving method_out.json")
    logger.info("=" * 60)

    output = build_output(results_table, correlations, regimes, figure_files)

    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {output_path}")

    total_time = time.time() - t_start
    logger.info(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
