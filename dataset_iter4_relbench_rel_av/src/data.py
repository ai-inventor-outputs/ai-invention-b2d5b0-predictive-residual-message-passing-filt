#!/usr/bin/env python3
"""Load RelBench rel-avito dataset, extract aligned parent-child feature pairs
per FK link, and produce full_data_out.json in exp_sel_data_out schema."""

import gc
import json
import math
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

# ── Configuration ────────────────────────────────────────────────────────
DATASET_NAME = "rel-avito"
MAX_SAMPLE_ROWS = 500_000
MAX_ALIGNED_PAIRS = 5_000       # per FK link
MIN_FK_VALID = 100
MAX_UNIQUE_CAT = 1000
MAX_FEATURE_COLS = 15           # cap features per side to limit JSON size
OUTPUT_FILE = Path("full_data_out.json")


# ── Helpers ──────────────────────────────────────────────────────────────

def safe_float(val) -> float:
    """Convert value to a JSON-safe float."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0.0
    return float(val)


def get_feature_cols(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    """Get columns suitable for numeric feature extraction."""
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            feature_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            feature_cols.append(col)
        elif dtype == object:
            nunique = df[col].nunique()
            if 1 < nunique <= MAX_UNIQUE_CAT:
                feature_cols.append(col)
    return feature_cols


def preprocess_col(series: pd.Series) -> pd.Series:
    """Convert a single column to numeric, handling categoricals and datetimes."""
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        result = series.astype(np.int64) // 10**9
        return result.fillna(0).astype(np.float64)
    elif dtype == object:
        codes, _ = pd.factorize(series)
        result = pd.Series(codes, index=series.index, dtype=np.float64)
        result = result.replace(-1, np.nan)
        return result.fillna(-1.0)
    else:
        result = series.astype(np.float64)
        median_val = result.median()
        if pd.isna(median_val):
            median_val = 0.0
        return result.fillna(median_val)


def round_list(vals: list, decimals: int = 4) -> list:
    """Round floats in a list to reduce JSON size."""
    return [round(v, decimals) if isinstance(v, float) else v for v in vals]


@logger.catch
def main():
    t0 = time.time()
    logger.info(f"Starting data.py — processing {DATASET_NAME}")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task_names, get_task

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset = get_dataset(DATASET_NAME, download=True)
    db = dataset.get_db()
    table_names = list(db.table_dict.keys())
    logger.info(f"Tables: {table_names}")

    # ── Sample large tables ──────────────────────────────────────────────
    sampled_dfs: dict[str, pd.DataFrame] = {}
    for tname in table_names:
        table = db.table_dict[tname]
        df = table.df
        nrows = len(df)
        if nrows > MAX_SAMPLE_ROWS:
            sampled_dfs[tname] = df.sample(n=MAX_SAMPLE_ROWS, random_state=42)
            logger.info(f"  Sampled {tname}: {nrows} -> {MAX_SAMPLE_ROWS}")
        else:
            sampled_dfs[tname] = df
            logger.info(f"  Full {tname}: {nrows}")

    # ── Build examples from FK links ─────────────────────────────────────
    examples: list[dict] = []
    fk_link_details: dict[str, dict] = {}
    fk_link_count = 0

    for tname in table_names:
        table = db.table_dict[tname]
        fks = dict(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else {}

        for fk_col, parent_table_name in fks.items():
            parent_table = db.table_dict[parent_table_name]
            parent_pk = parent_table.pkey_col
            child_df_full = table.df

            valid_mask = child_df_full[fk_col].notna()
            num_valid = int(valid_mask.sum())
            if num_valid < MIN_FK_VALID:
                logger.warning(f"  Skip {tname}.{fk_col}->{parent_table_name}: only {num_valid} valid")
                continue

            # Cardinality stats on full data
            cardinality = child_df_full.loc[valid_mask, fk_col].value_counts()
            card_mean = safe_float(cardinality.mean())
            card_median = safe_float(cardinality.median())
            card_max = int(cardinality.max())
            card_p95 = safe_float(cardinality.quantile(0.95))
            card_std = safe_float(cardinality.std()) if len(cardinality) > 1 else 0.0

            fk_link_id = f"{parent_table_name}->{tname}"
            link_id = f"{tname}__{fk_col}__{parent_table_name}"

            # Determine feature columns (exclude pk, fk, time, other fk cols)
            child_exclude = {table.pkey_col or "", fk_col}
            if table.time_col:
                child_exclude.add(table.time_col)
            for other_fk in fks:
                if other_fk != fk_col:
                    child_exclude.add(other_fk)

            parent_exclude = {parent_pk}
            if parent_table.time_col:
                parent_exclude.add(parent_table.time_col)
            parent_fks = dict(parent_table.fkey_col_to_pkey_table) if hasattr(parent_table, 'fkey_col_to_pkey_table') else {}
            for pfk in parent_fks:
                parent_exclude.add(pfk)

            child_sampled = sampled_dfs[tname]
            parent_df = parent_table.df

            child_feat_cols = get_feature_cols(child_sampled, child_exclude)[:MAX_FEATURE_COLS]
            parent_feat_cols = get_feature_cols(parent_df, parent_exclude)[:MAX_FEATURE_COLS]

            if not child_feat_cols and not parent_feat_cols:
                logger.warning(f"  No feature cols for {link_id}")
                continue

            # Join child to parent
            child_valid = child_sampled[child_sampled[fk_col].notna()].copy()
            if len(child_valid) > MAX_ALIGNED_PAIRS:
                child_valid = child_valid.sample(n=MAX_ALIGNED_PAIRS, random_state=42)

            merged = child_valid.merge(
                parent_df,
                left_on=fk_col,
                right_on=parent_pk,
                how="inner",
                suffixes=("_child", "_parent"),
            )
            if len(merged) > MAX_ALIGNED_PAIRS:
                merged = merged.sample(n=MAX_ALIGNED_PAIRS, random_state=42)

            # Resolve column names after merge (suffixes may be added)
            def resolve_cols(orig_cols: list[str], suffix: str) -> list[str]:
                resolved = []
                for c in orig_cols:
                    if c + suffix in merged.columns:
                        resolved.append(c + suffix)
                    elif c in merged.columns:
                        resolved.append(c)
                return resolved

            child_cols_merged = resolve_cols(child_feat_cols, "_child")
            parent_cols_merged = resolve_cols(parent_feat_cols, "_parent")

            if not child_cols_merged and not parent_cols_merged:
                logger.warning(f"  No resolved cols for {link_id}")
                continue

            # Preprocess features to numeric
            child_matrix = pd.concat(
                [preprocess_col(merged[c]).rename(c) for c in child_cols_merged],
                axis=1,
            )
            parent_matrix = pd.concat(
                [preprocess_col(merged[c]).rename(c) for c in parent_cols_merged],
                axis=1,
            )

            logger.info(
                f"  FK {link_id}: {len(merged)} pairs, "
                f"child_feats={len(child_cols_merged)}, parent_feats={len(parent_cols_merged)}, "
                f"card_mean={card_mean:.1f}, card_max={card_max}"
            )

            # Build examples — each aligned pair is one example
            child_vals = child_matrix.values
            parent_vals = parent_matrix.values
            for i in range(len(merged)):
                parent_feats = round_list(parent_vals[i].tolist())
                child_feats = round_list(child_vals[i].tolist())
                examples.append({
                    "input": json.dumps(parent_feats),
                    "output": json.dumps(child_feats),
                    "metadata_fk_link": fk_link_id,
                    "metadata_link_id": link_id,
                    "metadata_row_index": i,
                })

            # Store per-link details in top-level metadata
            fk_link_details[link_id] = {
                "fk_link": fk_link_id,
                "parent_table": parent_table_name,
                "child_table": tname,
                "fk_column": fk_col,
                "parent_pk_column": parent_pk,
                "parent_feature_cols": parent_feat_cols,
                "child_feature_cols": child_feat_cols,
                "num_aligned_pairs": len(merged),
                "cardinality_mean": card_mean,
                "cardinality_median": card_median,
                "cardinality_max": card_max,
                "cardinality_p95": card_p95,
                "cardinality_std": card_std,
                "num_child_rows": len(child_df_full),
                "num_parent_rows": len(parent_df),
                "num_valid_fk": num_valid,
            }

            fk_link_count += 1
            del merged, child_valid, child_matrix, parent_matrix
            gc.collect()

    logger.info(f"{DATASET_NAME}: {len(examples)} examples from {fk_link_count} FK links")

    # ── Collect task metadata ────────────────────────────────────────────
    task_list = []
    try:
        for tn in get_task_names(DATASET_NAME):
            try:
                task = get_task(DATASET_NAME, tn, download=True)
                info: dict = {
                    "task_name": tn,
                    "task_type": str(task.task_type) if hasattr(task, 'task_type') else type(task).__name__,
                }
                for attr in ("entity_table", "entity_col", "target_col"):
                    if hasattr(task, attr):
                        info[attr] = getattr(task, attr)
                if hasattr(task, 'timedelta'):
                    info["timedelta"] = str(task.timedelta)
                if hasattr(task, 'metrics'):
                    info["metrics"] = [m.__name__ if hasattr(m, '__name__') else str(m) for m in task.metrics]
                if hasattr(task, 'num_eval_timestamps'):
                    info["num_eval_timestamps"] = task.num_eval_timestamps
                task_list.append(info)
                logger.info(f"  Task: {tn} -> {info.get('task_type')}")
            except Exception:
                logger.exception(f"Failed task {tn}")
                task_list.append({"task_name": tn, "error": "failed"})
    except Exception:
        logger.exception("Failed to get task names")

    # ── Build table metadata ─────────────────────────────────────────────
    table_meta = {}
    for tname, table in db.table_dict.items():
        df = table.df
        fks = dict(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else {}
        table_meta[tname] = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "pkey_col": table.pkey_col,
            "time_col": table.time_col if hasattr(table, 'time_col') else None,
            "fkey_col_to_pkey_table": fks,
        }

    # ── Assemble output ──────────────────────────────────────────────────
    output = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "source": "RelBench (Stanford SNAP)",
            "domain": "e-commerce/advertising",
            "description": "Avito online advertisement platform dataset from RelBench. 8 tables, ~20.7M rows. Each example is one aligned (parent, child) row pair joined on a foreign key, for downstream R-squared computation.",
            "citation": "@misc{robinson2024relbenchbenchmarkdeeplearning, title={RelBench: A Benchmark for Deep Learning on Relational Databases}, author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan Eric Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec}, year={2024}, eprint={2407.20060}, archivePrefix={arXiv}}",
            "tables": table_meta,
            "tasks": task_list,
            "fk_links": fk_link_details,
        },
        "datasets": [
            {
                "dataset": DATASET_NAME,
                "examples": examples,
            }
        ],
    }

    # ── Write output ─────────────────────────────────────────────────────
    logger.info(f"Writing {OUTPUT_FILE}")
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    fsize = OUTPUT_FILE.stat().st_size
    logger.info(f"Output: {fsize / 1e6:.1f} MB, {len(examples)} examples")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
