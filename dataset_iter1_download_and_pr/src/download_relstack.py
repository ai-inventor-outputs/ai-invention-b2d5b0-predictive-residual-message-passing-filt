#!/usr/bin/env python3
"""Download and prepare RelBench rel-stack (Stack Exchange) dataset.

Extracts all tables with features, computes cardinality distributions per FK link,
builds aligned parent-child feature matrices for predictability analysis,
enumerates all RelBench tasks, and outputs standardized data_out.json.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger

# ── Logging setup ──────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection (cgroup-aware) ──────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM total, {AVAILABLE_RAM_GB:.1f}GB available")

# ── Memory limits ──────────────────────────────────────────────────────────
# Budget: ~18GB for this script (leaving headroom for OS + agent)
RAM_BUDGET_BYTES = int(18 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET_BYTES < _avail, f"Budget {RAM_BUDGET_BYTES/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time
logger.info(f"RAM budget: {RAM_BUDGET_BYTES/1e9:.1f}GB, CPU limit: 3600s")

# ── Constants ──────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ALIGNED_ROWS = 50_000  # Max rows per aligned feature matrix (keep JSON manageable)
MAX_SAMPLE_ROWS = 5  # Sample rows for table metadata

def timestamp_to_str(val):
    """Convert pandas Timestamp or datetime to ISO string safely."""
    if pd.isna(val):
        return None
    if hasattr(val, 'isoformat'):
        return val.isoformat()
    return str(val)


def safe_json_value(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return timestamp_to_str(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def df_sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Get sample rows as list of dicts with JSON-safe values."""
    sample = df.head(n)
    rows = []
    for _, row in sample.iterrows():
        rows.append({k: safe_json_value(v) for k, v in row.items()})
    return rows


@logger.catch
def main():
    t0 = time.time()

    # ── Step 1: Load the rel-stack dataset ─────────────────────────────────
    logger.info("Loading RelBench rel-stack dataset...")
    from relbench.datasets import get_dataset

    dataset = get_dataset(name="rel-stack", download=True)
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s")

    db = dataset.get_db()
    table_names = list(db.table_dict.keys())
    logger.info(f"Tables found: {table_names}")

    # ── Step 2: Inspect all tables ─────────────────────────────────────────
    logger.info("Inspecting database schema...")
    tables_info = {}
    for table_name in table_names:
        table = db.table_dict[table_name]
        df = table.df
        logger.info(f"  {table_name}: {len(df)} rows, {len(df.columns)} cols")
        logger.info(f"    Columns: {list(df.columns)}")
        logger.info(f"    Dtypes: {df.dtypes.to_dict()}")
        logger.info(f"    PKey: {table.pkey_col}")
        logger.info(f"    FKeys: {table.fkey_col_to_pkey_table}")
        logger.info(f"    Time col: {table.time_col}")

        # Collect table info
        tables_info[table_name] = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "pkey_col": table.pkey_col,
            "time_col": table.time_col,
            "fkey_col_to_pkey_table": dict(table.fkey_col_to_pkey_table) if table.fkey_col_to_pkey_table else {},
            "sample_rows": df_sample_rows(df, MAX_SAMPLE_ROWS),
        }
        # Free memory for large tables
        del df
        gc.collect()

    logger.info(f"Schema inspection done. {len(tables_info)} tables found.")

    # ── Step 3: Enumerate FK links and compute cardinality ─────────────────
    logger.info("Computing FK cardinality distributions...")
    fk_links = []

    for child_table_name in table_names:
        child_table = db.table_dict[child_table_name]
        fkey_map = child_table.fkey_col_to_pkey_table
        if not fkey_map:
            continue

        for fkey_col, parent_table_name in fkey_map.items():
            parent_table = db.table_dict[parent_table_name]
            child_df = child_table.df
            parent_df = parent_table.df

            logger.info(f"  FK: {child_table_name}.{fkey_col} -> {parent_table_name}")

            # Compute cardinality: children per parent
            # Only count non-null FK values
            valid_fk = child_df[fkey_col].dropna()
            cardinality = valid_fk.groupby(valid_fk).size()

            fk_info = {
                "child_table": child_table_name,
                "parent_table": parent_table_name,
                "fkey_col": fkey_col,
                "link_id": f"{child_table_name}__{fkey_col}__{parent_table_name}",
                "num_child_rows": len(child_df),
                "num_parent_rows": len(parent_df),
                "num_valid_fk_values": len(valid_fk),
                "num_null_fk_values": int(child_df[fkey_col].isna().sum()),
                "cardinality_mean": float(cardinality.mean()) if len(cardinality) > 0 else 0.0,
                "cardinality_median": float(cardinality.median()) if len(cardinality) > 0 else 0.0,
                "cardinality_std": float(cardinality.std()) if len(cardinality) > 1 else 0.0,
                "cardinality_max": int(cardinality.max()) if len(cardinality) > 0 else 0,
                "cardinality_min": int(cardinality.min()) if len(cardinality) > 0 else 0,
                "cardinality_p25": float(np.percentile(cardinality, 25)) if len(cardinality) > 0 else 0.0,
                "cardinality_p75": float(np.percentile(cardinality, 75)) if len(cardinality) > 0 else 0.0,
                "cardinality_p95": float(np.percentile(cardinality, 95)) if len(cardinality) > 0 else 0.0,
                "cardinality_p99": float(np.percentile(cardinality, 99)) if len(cardinality) > 0 else 0.0,
                "num_parents_with_children": int(len(cardinality)),
                "coverage": float(len(cardinality) / len(parent_df)) if len(parent_df) > 0 else 0.0,
            }
            fk_links.append(fk_info)
            logger.info(f"    Cardinality: mean={fk_info['cardinality_mean']:.2f}, "
                        f"median={fk_info['cardinality_median']:.1f}, "
                        f"max={fk_info['cardinality_max']}, "
                        f"coverage={fk_info['coverage']:.3f}")

            del valid_fk, cardinality
            gc.collect()

    logger.info(f"Found {len(fk_links)} FK links")

    # ── Step 4: Build aligned parent-child feature matrices ────────────────
    logger.info("Building aligned parent-child feature matrices...")

    for fk_info in fk_links:
        child_table = db.table_dict[fk_info["child_table"]]
        parent_table = db.table_dict[fk_info["parent_table"]]
        child_df = child_table.df
        parent_df = parent_table.df
        fkey_col = fk_info["fkey_col"]
        pkey_col = parent_table.pkey_col

        # Select numeric + bool columns — exclude keys, timestamps, text
        child_numeric_cols = child_df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        parent_numeric_cols = parent_df.select_dtypes(include=[np.number, 'bool']).columns.tolist()

        # Get all FK columns and PK columns to exclude (actual join keys only)
        child_fk_cols = set(child_table.fkey_col_to_pkey_table.keys()) if child_table.fkey_col_to_pkey_table else set()
        child_pk = child_table.pkey_col
        parent_fk_cols = set(parent_table.fkey_col_to_pkey_table.keys()) if parent_table.fkey_col_to_pkey_table else set()
        parent_pk = parent_table.pkey_col

        # Only exclude actual PK/FK columns, NOT TypeId columns (those are features)
        child_exclude = child_fk_cols | {child_pk}
        parent_exclude = parent_fk_cols | {parent_pk}
        # Also exclude AccountId (users) - it's an external ID, not a feature
        child_exclude.add("AccountId")
        parent_exclude.add("AccountId")

        child_numeric_cols = [c for c in child_numeric_cols if c not in child_exclude]
        parent_numeric_cols = [c for c in parent_numeric_cols if c not in parent_exclude]

        link_id = fk_info["link_id"]
        logger.info(f"  {link_id}: child_feats={child_numeric_cols}, parent_feats={parent_numeric_cols}")

        if not child_numeric_cols and not parent_numeric_cols:
            logger.info(f"    No numeric features for this FK link, skipping matrix")
            fk_info["child_feature_cols"] = []
            fk_info["parent_feature_cols"] = []
            fk_info["aligned_matrix_rows"] = 0
            continue

        # Build aligned matrix by joining child with parent on FK
        # Use suffixes to handle self-referential FKs (same table on both sides)
        child_subset = child_df[[fkey_col] + child_numeric_cols].copy()
        parent_subset = parent_df[[pkey_col] + parent_numeric_cols].copy()

        merged = child_subset.merge(
            parent_subset,
            left_on=fkey_col, right_on=pkey_col, how="inner",
            suffixes=("_child", "_parent")
        )
        del child_subset, parent_subset
        gc.collect()

        # Sample if too large
        if len(merged) > MAX_ALIGNED_ROWS:
            merged = merged.sample(n=MAX_ALIGNED_ROWS, random_state=42)
            logger.info(f"    Sampled down to {MAX_ALIGNED_ROWS} rows")

        fk_info["child_feature_cols"] = child_numeric_cols
        fk_info["parent_feature_cols"] = parent_numeric_cols
        fk_info["aligned_matrix_rows"] = len(merged)

        # Resolve suffixed column names after merge
        def resolve_cols(orig_cols: list[str], suffix: str) -> list[str]:
            """Find actual column names in merged df, handling suffixes."""
            resolved = []
            for c in orig_cols:
                if c in merged.columns:
                    resolved.append(c)
                elif f"{c}{suffix}" in merged.columns:
                    resolved.append(f"{c}{suffix}")
                else:
                    logger.warning(f"    Column {c} not found in merged (tried {c}{suffix})")
            return resolved

        child_merged_cols = resolve_cols(child_numeric_cols, "_child")
        parent_merged_cols = resolve_cols(parent_numeric_cols, "_parent")

        # Store feature values as lists (for JSON output)
        if child_merged_cols:
            child_vals = merged[child_merged_cols].astype(float).values
            # Replace NaN/inf with 0.0
            child_vals = np.where(np.isfinite(child_vals), child_vals, 0.0)
            fk_info["child_features"] = child_vals.tolist()
        else:
            fk_info["child_features"] = []

        if parent_merged_cols:
            parent_vals = merged[parent_merged_cols].astype(float).values
            parent_vals = np.where(np.isfinite(parent_vals), parent_vals, 0.0)
            fk_info["parent_features"] = parent_vals.tolist()
        else:
            fk_info["parent_features"] = []

        logger.info(f"    Aligned matrix: {len(merged)} rows, "
                    f"{len(child_numeric_cols)} child feats, {len(parent_numeric_cols)} parent feats")

        del merged
        gc.collect()

    # ── Step 5: Enumerate RelBench tasks ───────────────────────────────────
    logger.info("Enumerating RelBench tasks for rel-stack...")
    task_info = []

    from relbench.tasks import get_task, get_task_names

    task_names_list = get_task_names("rel-stack")
    logger.info(f"Registered task names: {task_names_list}")

    for task_name in task_names_list:
        try:
            task = get_task("rel-stack", task_name, download=True)
            info = {
                "task_name": task_name,
                "entity_table": getattr(task, 'entity_table', 'unknown'),
                "entity_col": getattr(task, 'entity_col', 'unknown'),
                "target_col": getattr(task, 'target_col', 'unknown'),
                "task_type": str(getattr(task, 'task_type', 'unknown')),
                "num_eval_timestamps": getattr(task, 'num_eval_timestamps', None),
                "timedelta": str(getattr(task, 'timedelta', None)),
                "metrics": [m.__name__ if hasattr(m, '__name__') else str(m)
                            for m in getattr(task, 'metrics', [])],
            }
            # Try to get train table info
            try:
                train_table = task.get_table("train")
                info["train_rows"] = len(train_table.df)
                info["train_cols"] = list(train_table.df.columns)
            except Exception as e:
                logger.debug(f"  Could not get train table for {task_name}: {e}")
                info["train_rows"] = 0
                info["train_cols"] = []

            task_info.append(info)
            logger.info(f"  Task: {task_name} -> entity={info['entity_table']}, "
                        f"type={info['task_type']}, train_rows={info['train_rows']}")
        except Exception:
            logger.exception(f"  Could not load task '{task_name}'")

    logger.info(f"Found {len(task_info)} tasks")

    # ── Step 6: Assemble output JSON ───────────────────────────────────────
    logger.info("Assembling data_out.json...")

    # Build schema topology (adjacency list)
    topology = {}
    for fk in fk_links:
        child = fk["child_table"]
        parent = fk["parent_table"]
        topology.setdefault(child, []).append(parent)
        topology.setdefault(parent, []).append(child)
    topology = {k: sorted(set(v)) for k, v in topology.items()}

    output = {
        "dataset_name": "rel-stack",
        "source": "RelBench (Stanford SNAP)",
        "domain": "social/Q&A platform",
        "description": (
            "Stack Exchange Stats site Q&A dataset from RelBench. Contains users, posts, "
            "votes, badges, comments, and postHistory tables with rich FK relationships. "
            "Derived from the stats.stackexchange.com data dump (2023-09-12). "
            "Part of the RelBench benchmark (NeurIPS 2024 Datasets and Benchmarks track)."
        ),
        "license": "CC BY-SA 4.0",
        "citation": (
            "@misc{robinson2024relbenchbenchmarkdeeplearning, "
            "title={RelBench: A Benchmark for Deep Learning on Relational Databases}, "
            "author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang "
            "and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan Eric Lenssen "
            "and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec}, "
            "year={2024}, eprint={2407.20060}, archivePrefix={arXiv}}"
        ),
        "tables": tables_info,
        "fk_links": fk_links,
        "tasks": task_info,
        "schema_topology": topology,
    }

    # ── Step 7: Write output ───────────────────────────────────────────────
    out_path = WORKSPACE / "data_out.json"
    logger.info(f"Writing output to {out_path}")
    out_text = json.dumps(output, indent=2, default=safe_json_value)
    out_path.write_text(out_text)
    logger.info(f"Output written: {len(out_text)} chars, {len(out_text)/1e6:.1f}MB")

    elapsed = time.time() - t0
    logger.info(f"Done! Total time: {elapsed:.1f}s")
    logger.info(f"Tables: {len(tables_info)}, FK links: {len(fk_links)}, Tasks: {len(task_info)}")

    return output


if __name__ == "__main__":
    main()
