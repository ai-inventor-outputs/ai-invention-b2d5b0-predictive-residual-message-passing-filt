#!/usr/bin/env python3
"""Process a RelBench dataset: enumerate tables, FK links, cardinality stats,
extract aligned parent-child feature matrices, collect task metadata, and
produce data_out.json matching the prior artifact schema."""

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
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────
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
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# Set RAM limit to 80% of available (leave headroom)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ── Configuration ────────────────────────────────────────────────────────
DATASET_NAME = os.environ.get("DATASET_NAME", "rel-avito")
MAX_SAMPLE_ROWS = 500_000       # Sample large tables to this many rows
MAX_ALIGNED_PAIRS = 5_000       # Max aligned parent-child pairs per FK link
MIN_FK_VALID = 100              # Skip FK links with fewer valid pairs
MAX_UNIQUE_CAT = 1000           # Drop categorical cols with more unique values
SAMPLE_ROWS_FOR_TABLE = 5       # Number of sample rows per table in output
OUTPUT_DIR = Path(".")
OUTPUT_FILE = OUTPUT_DIR / "data_out.json"

# ── Helpers ──────────────────────────────────────────────────────────────

def safe_convert(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def df_to_sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Convert first n rows to JSON-serializable list of dicts."""
    rows = []
    for _, row in df.head(n).iterrows():
        rows.append({col: safe_convert(row[col]) for col in df.columns})
    return rows


def get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get columns suitable for feature extraction (numeric, low-cardinality categorical)."""
    feature_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            feature_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            feature_cols.append(col)
        elif dtype == object:
            nunique = df[col].nunique()
            if nunique <= MAX_UNIQUE_CAT:
                feature_cols.append(col)
    return feature_cols


def preprocess_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Preprocess feature columns: encode categoricals, convert datetimes, fill NaN."""
    result = pd.DataFrame(index=df.index)
    for col in cols:
        series = df[col]
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            # Convert to epoch seconds
            result[col] = series.astype(np.int64) // 10**9
            result[col] = result[col].fillna(0)
        elif dtype == object:
            # Label encode
            codes, _ = pd.factorize(series)
            result[col] = codes.astype(np.float64)
            result[col] = result[col].replace(-1, np.nan)
            result[col] = result[col].fillna(-1)
        else:
            result[col] = series.astype(np.float64)
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)
    return result


@logger.catch
def main():
    t0 = time.time()

    # ── Step 1: Load dataset ─────────────────────────────────────────────
    from relbench.datasets import get_dataset

    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = get_dataset(DATASET_NAME, download=True)
    db = dataset.get_db()
    table_names = list(db.table_dict.keys())
    logger.info(f"Loaded {len(table_names)} tables: {table_names}")

    # ── Step 2: Enumerate tables and schema ──────────────────────────────
    tables_meta = {}
    sampled_dfs = {}  # Store sampled DataFrames
    total_rows = 0
    total_cols = 0

    for tname in table_names:
        table = db.table_dict[tname]
        df = table.df
        nrows = len(df)
        ncols = len(df.columns)
        total_rows += nrows
        total_cols += ncols

        pkey = table.pkey_col
        fks = dict(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else {}
        time_col = table.time_col if hasattr(table, 'time_col') else None

        # Identify column roles
        columns_info = []
        for col in df.columns:
            role = "feature"
            if col == pkey:
                role = "pk"
            elif col in fks:
                role = "fk"
            elif col == time_col:
                role = "timestamp"
            columns_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
            })

        # Strategic sampling
        sampled_nrows = None
        if nrows > MAX_SAMPLE_ROWS:
            sampled_df = df.sample(n=MAX_SAMPLE_ROWS, random_state=42)
            sampled_nrows = MAX_SAMPLE_ROWS
            logger.info(f"  Sampled {tname}: {nrows} -> {MAX_SAMPLE_ROWS} rows")
        else:
            sampled_df = df
            logger.info(f"  Kept full {tname}: {nrows} rows")

        sampled_dfs[tname] = sampled_df

        tables_meta[tname] = {
            "num_rows": nrows,
            "num_cols": ncols,
            "columns": columns_info,
            "pkey_col": pkey,
            "time_col": time_col,
            "fkey_col_to_pkey_table": fks,
            "sample_rows": df_to_sample_rows(df, SAMPLE_ROWS_FOR_TABLE),
        }
        if sampled_nrows is not None:
            tables_meta[tname]["num_rows_sampled"] = sampled_nrows

        # Free the original df reference early
        del df

    gc.collect()
    logger.info(f"Total: {total_rows} rows, {total_cols} columns across {len(table_names)} tables")

    # ── Step 3: Enumerate FK links and compute cardinality stats ─────────
    fk_links = []

    for tname in table_names:
        table = db.table_dict[tname]
        fks = dict(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else {}

        for fk_col, parent_table_name in fks.items():
            parent_table = db.table_dict[parent_table_name]
            parent_pk = parent_table.pkey_col

            # Use FULL table for cardinality stats (not sampled)
            child_df = table.df

            # Count valid/null FK values
            valid_mask = child_df[fk_col].notna()
            num_valid = int(valid_mask.sum())
            num_null = int((~valid_mask).sum())

            if num_valid < MIN_FK_VALID:
                logger.warning(f"  Skipping {tname}.{fk_col} -> {parent_table_name}: only {num_valid} valid FK values")
                continue

            # Cardinality: count children per parent
            cardinality = child_df.loc[valid_mask, fk_col].value_counts()

            # How many parents actually have children
            num_parents_with_children = len(cardinality)
            total_parents = len(parent_table.df)
            coverage = num_parents_with_children / total_parents if total_parents > 0 else 0

            link_id = f"{tname}__{fk_col}__{parent_table_name}"

            fk_link = {
                "child_table": tname,
                "parent_table": parent_table_name,
                "fkey_col": fk_col,
                "link_id": link_id,
                "num_child_rows": len(child_df),
                "num_parent_rows": total_parents,
                "num_valid_fk_values": num_valid,
                "num_null_fk_values": num_null,
                "cardinality_mean": float(cardinality.mean()),
                "cardinality_median": float(cardinality.median()),
                "cardinality_std": float(cardinality.std()) if len(cardinality) > 1 else 0.0,
                "cardinality_max": int(cardinality.max()),
                "cardinality_min": int(cardinality.min()),
                "cardinality_p25": float(cardinality.quantile(0.25)),
                "cardinality_p75": float(cardinality.quantile(0.75)),
                "cardinality_p95": float(cardinality.quantile(0.95)),
                "cardinality_p99": float(cardinality.quantile(0.99)),
                "num_parents_with_children": num_parents_with_children,
                "coverage": coverage,
            }

            logger.info(f"  FK: {tname}.{fk_col} -> {parent_table_name} | "
                        f"valid={num_valid}, mean_card={fk_link['cardinality_mean']:.1f}, "
                        f"max_card={fk_link['cardinality_max']}")

            # ── Step 4: Extract aligned parent-child feature matrices ────
            try:
                # Use sampled child DataFrame for feature extraction
                child_sampled = sampled_dfs[tname]
                parent_df_full = parent_table.df

                # Get feature columns for both tables (excluding pk, fk, timestamp, text)
                child_all_cols = list(child_sampled.columns)
                parent_all_cols = list(parent_df_full.columns)

                # Exclude pk, fk, time cols from features
                child_exclude = {table.pkey_col, fk_col}
                if table.time_col:
                    child_exclude.add(table.time_col)
                parent_exclude = {parent_pk}
                if parent_table.time_col:
                    parent_exclude.add(parent_table.time_col)
                # Also exclude other FK columns
                for other_fk in (table.fkey_col_to_pkey_table or {}):
                    if other_fk != fk_col:
                        child_exclude.add(other_fk)
                for other_fk in (parent_table.fkey_col_to_pkey_table or {}):
                    parent_exclude.add(other_fk)

                child_feat_candidates = [c for c in child_all_cols if c not in child_exclude]
                parent_feat_candidates = [c for c in parent_all_cols if c not in parent_exclude]

                child_feat_cols = get_numeric_feature_cols(child_sampled[child_feat_candidates]) if child_feat_candidates else []
                parent_feat_cols = get_numeric_feature_cols(parent_df_full[parent_feat_candidates]) if parent_feat_candidates else []

                # Only proceed if we have at least some features
                if child_feat_cols or parent_feat_cols:
                    # Join child to parent on FK
                    child_valid = child_sampled[child_sampled[fk_col].notna()].copy()

                    # Sample to MAX_ALIGNED_PAIRS
                    if len(child_valid) > MAX_ALIGNED_PAIRS:
                        child_valid = child_valid.sample(n=MAX_ALIGNED_PAIRS, random_state=42)

                    # Create parent lookup (index by PK for fast join)
                    parent_indexed = parent_df_full.set_index(parent_pk)

                    # Get parent rows for each child
                    child_fk_values = child_valid[fk_col].values
                    parent_matched = parent_indexed.loc[
                        parent_indexed.index.intersection(child_fk_values)
                    ]

                    # Merge via join
                    merged = child_valid.merge(
                        parent_df_full,
                        left_on=fk_col,
                        right_on=parent_pk,
                        how="inner",
                        suffixes=("_child", "_parent")
                    )

                    if len(merged) > MAX_ALIGNED_PAIRS:
                        merged = merged.sample(n=MAX_ALIGNED_PAIRS, random_state=42)

                    # Resolve suffix conflicts
                    child_cols_in_merged = []
                    for c in child_feat_cols:
                        if c + "_child" in merged.columns:
                            child_cols_in_merged.append(c + "_child")
                        elif c in merged.columns:
                            child_cols_in_merged.append(c)

                    parent_cols_in_merged = []
                    for c in parent_feat_cols:
                        if c + "_parent" in merged.columns:
                            parent_cols_in_merged.append(c + "_parent")
                        elif c in merged.columns:
                            parent_cols_in_merged.append(c)

                    # Preprocess features
                    if child_cols_in_merged:
                        child_features_df = preprocess_features(merged, child_cols_in_merged)
                        child_features = child_features_df.values.tolist()
                    else:
                        child_features = []

                    if parent_cols_in_merged:
                        parent_features_df = preprocess_features(merged, parent_cols_in_merged)
                        parent_features = parent_features_df.values.tolist()
                    else:
                        parent_features = []

                    fk_link["child_feature_cols"] = child_feat_cols
                    fk_link["parent_feature_cols"] = parent_feat_cols
                    fk_link["aligned_matrix_rows"] = len(merged)
                    fk_link["child_features"] = child_features
                    fk_link["parent_features"] = parent_features

                    logger.info(f"    Aligned pairs: {len(merged)}, "
                                f"child_feats={len(child_feat_cols)}, parent_feats={len(parent_feat_cols)}")

                    # Clean up
                    del merged, child_valid, parent_indexed
                else:
                    fk_link["child_feature_cols"] = []
                    fk_link["parent_feature_cols"] = []
                    fk_link["aligned_matrix_rows"] = 0
                    fk_link["child_features"] = []
                    fk_link["parent_features"] = []
                    logger.warning(f"    No suitable feature columns for {link_id}")

            except Exception:
                logger.exception(f"Failed to extract features for {link_id}")
                fk_link["child_feature_cols"] = []
                fk_link["parent_feature_cols"] = []
                fk_link["aligned_matrix_rows"] = 0
                fk_link["child_features"] = []
                fk_link["parent_features"] = []

            fk_links.append(fk_link)
            gc.collect()

    logger.info(f"Computed {len(fk_links)} FK links with cardinality stats")

    # ── Step 5: Collect benchmark task metadata ──────────────────────────
    from relbench.tasks import get_task_names, get_task

    tasks_meta = []
    candidate_tasks = get_task_names(DATASET_NAME)
    logger.info(f"Found {len(candidate_tasks)} tasks for {DATASET_NAME}: {candidate_tasks}")

    for task_name in candidate_tasks:
        try:
            task = get_task(DATASET_NAME, task_name, download=True)

            task_info = {
                "task_name": task_name,
                "task_type": str(task.task_type) if hasattr(task, 'task_type') else str(type(task).__name__),
            }

            # Entity info
            if hasattr(task, 'entity_table'):
                task_info["entity_table"] = task.entity_table
            elif hasattr(task, 'src_entity_table'):
                task_info["entity_table"] = f"{task.src_entity_table} -> {task.dst_entity_table}"
            else:
                task_info["entity_table"] = "unknown"

            if hasattr(task, 'entity_col'):
                task_info["entity_col"] = task.entity_col
            elif hasattr(task, 'src_entity_col'):
                task_info["entity_col"] = f"{task.src_entity_col} -> {task.dst_entity_col}"
            else:
                task_info["entity_col"] = "unknown"

            if hasattr(task, 'target_col'):
                task_info["target_col"] = task.target_col
            else:
                task_info["target_col"] = "unknown"

            # Timedelta and eval timestamps
            if hasattr(task, 'timedelta'):
                task_info["timedelta"] = str(task.timedelta)
            if hasattr(task, 'num_eval_timestamps'):
                task_info["num_eval_timestamps"] = task.num_eval_timestamps

            # Metrics - extract function names from metric functions
            if hasattr(task, 'metrics'):
                task_info["metrics"] = [
                    m.__name__ if hasattr(m, '__name__') else str(m)
                    for m in task.metrics
                ]

            # Train/val/test sizes
            for split_name in ['train_table', 'val_table', 'test_table']:
                try:
                    split_table = getattr(task, split_name, None)
                    if split_table is not None:
                        key = split_name.replace('_table', '') + '_rows'
                        if hasattr(split_table, 'df'):
                            task_info[key] = len(split_table.df)
                            task_info[split_name.replace('_table', '') + '_cols'] = list(split_table.df.columns)
                        else:
                            task_info[key] = len(split_table)
                except Exception:
                    pass

            tasks_meta.append(task_info)
            logger.info(f"  Task: {task_name} -> {task_info.get('task_type', 'unknown')}")

        except Exception:
            logger.exception(f"Failed to load task: {task_name}")
            tasks_meta.append({
                "task_name": task_name,
                "task_type": "unknown",
                "error": "Failed to load task",
            })

    # ── Step 6: Build schema topology ────────────────────────────────────
    schema_topology = {}
    for tname in table_names:
        connected = set()
        table = db.table_dict[tname]
        fks = dict(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else {}
        # Tables this table points to (as child)
        for parent in fks.values():
            connected.add(parent)
        # Tables that point to this table (as parent)
        for other_name in table_names:
            other = db.table_dict[other_name]
            other_fks = dict(other.fkey_col_to_pkey_table) if hasattr(other, 'fkey_col_to_pkey_table') else {}
            for parent in other_fks.values():
                if parent == tname:
                    connected.add(other_name)
        schema_topology[tname] = sorted(connected)

    # ── Step 7: Determine dataset metadata ───────────────────────────────
    dataset_info = {
        "dataset_name": DATASET_NAME,
        "source": "RelBench (Stanford SNAP)",
    }

    if DATASET_NAME == "rel-avito":
        dataset_info.update({
            "domain": "e-commerce/advertising",
            "description": "Avito online advertisement platform dataset from RelBench. Contains ads, users, visits, searches, phone requests, locations, and categories with rich FK relationships. Derived from the Avito Context Ad Clicks Kaggle competition (2015). Part of the RelBench benchmark (NeurIPS 2024 Datasets and Benchmarks track).",
            "license": "Custom (Kaggle competition data)",
            "time_range": {"start": "2015-04-25", "end": "2015-05-14"},
        })
    elif DATASET_NAME == "rel-event":
        dataset_info.update({
            "domain": "social/event recommendation",
            "description": "Event recommendation dataset from RelBench. Hangtime mobile app data with users, events, attendance, interests, and social relations. Part of the RelBench benchmark (NeurIPS 2024).",
            "license": "Custom (Kaggle competition data)",
        })
    else:
        dataset_info.update({
            "domain": "unknown",
            "description": f"RelBench dataset: {DATASET_NAME}",
            "license": "See RelBench documentation",
        })

    dataset_info["citation"] = "@misc{robinson2024relbenchbenchmarkdeeplearning, title={RelBench: A Benchmark for Deep Learning on Relational Databases}, author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan Eric Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec}, year={2024}, eprint={2407.20060}, archivePrefix={arXiv}}"

    # ── Step 8: Assemble data_out.json in exp_sel_data_out schema ────────
    # Build examples from aligned parent-child feature pairs
    examples = []
    for fk in fk_links:
        link_id = fk["link_id"]
        child_feats_list = fk.get("child_features", [])
        parent_feats_list = fk.get("parent_features", [])
        n_pairs = min(len(child_feats_list), len(parent_feats_list))

        for i in range(n_pairs):
            example = {
                "input": json.dumps({
                    "fk_link_id": f"{fk['parent_table']}->{fk['child_table']}",
                    "parent_features": parent_feats_list[i],
                    "child_features": child_feats_list[i],
                }),
                "output": json.dumps({
                    "cardinality_mean": fk["cardinality_mean"],
                    "cardinality_median": fk["cardinality_median"],
                    "cardinality_max": fk["cardinality_max"],
                    "cardinality_p95": fk["cardinality_p95"],
                }),
                "metadata_fold": "all",
                "metadata_fk_link": f"{fk['parent_table']}->{fk['child_table']}",
                "metadata_parent_table": fk["parent_table"],
                "metadata_child_table": fk["child_table"],
                "metadata_link_id": link_id,
            }
            examples.append(example)

    logger.info(f"Created {len(examples)} examples from {len(fk_links)} FK links")

    # Build FK links metadata (without large feature arrays, for metadata)
    fk_links_meta = []
    for fk in fk_links:
        fk_meta = {k: v for k, v in fk.items() if k not in ("child_features", "parent_features")}
        fk_links_meta.append(fk_meta)

    # Assemble in exp_sel_data_out schema
    output = {
        "metadata": {
            **dataset_info,
            "num_tables": len(tables_meta),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "tables": tables_meta,
            "fk_links": fk_links_meta,
            "tasks": tasks_meta,
            "schema_topology": schema_topology,
        },
        "datasets": [
            {
                "dataset": DATASET_NAME,
                "examples": examples,
            }
        ]
    }

    # Write output
    logger.info(f"Writing output to {OUTPUT_FILE}")
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=safe_convert))

    fsize = OUTPUT_FILE.stat().st_size
    logger.info(f"Output file size: {fsize / 1e6:.1f} MB")

    elapsed = time.time() - t0
    logger.info(f"Total processing time: {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    main()
