#!/usr/bin/env python3
"""Download and prepare the Ergast Formula 1 relational database for PRMP hypothesis analysis.

Downloads the Ergast F1 CSV database, extracts all tables, enumerates FK links,
computes cardinality statistics per FK link, generates aligned parent-child feature
matrices (parquet), and outputs everything in exp_sel_data_out.json schema format.

Each FK link = one example, with cardinality stats, table metadata, and aligned
feature matrix references.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import math
import gc
import resource
import zipfile
import io

import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)  # 60% of available
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
WS = Path(__file__).parent
TEMP_DIR = WS / "temp" / "datasets"
PARQUET_DIR = WS / "parquet_features"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

ERGAST_CSV_URL = "https://github.com/rubenv/ergast-mrd/raw/master/f1db_csv.zip"
MAX_ALIGNED_ROWS = 200_000

# ── Known Ergast F1 Schema (FK links) ───────────────────────────────────────
# Format: (child_table, child_fk_col, parent_table, parent_pk_col)
FK_LINKS = [
    ("results", "raceId", "races", "raceId"),
    ("results", "driverId", "drivers", "driverId"),
    ("results", "constructorId", "constructors", "constructorId"),
    ("results", "statusId", "status", "statusId"),
    ("qualifying", "raceId", "races", "raceId"),
    ("qualifying", "driverId", "drivers", "driverId"),
    ("qualifying", "constructorId", "constructors", "constructorId"),
    ("lap_times", "raceId", "races", "raceId"),
    ("lap_times", "driverId", "drivers", "driverId"),
    ("pit_stops", "raceId", "races", "raceId"),
    ("pit_stops", "driverId", "drivers", "driverId"),
    ("races", "circuitId", "circuits", "circuitId"),
    ("driver_standings", "raceId", "races", "raceId"),
    ("driver_standings", "driverId", "drivers", "driverId"),
    ("constructor_standings", "raceId", "races", "raceId"),
    ("constructor_standings", "constructorId", "constructors", "constructorId"),
    ("constructor_results", "raceId", "races", "raceId"),
    ("constructor_results", "constructorId", "constructors", "constructorId"),
    ("sprint_results", "raceId", "races", "raceId"),
    ("sprint_results", "driverId", "drivers", "driverId"),
    ("sprint_results", "constructorId", "constructors", "constructorId"),
    ("sprint_results", "statusId", "status", "statusId"),
]

# Known ID columns per table (primary keys and foreign keys)
ID_COLUMNS = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "constructor_results": ["constructorResultsId", "raceId", "constructorId"],
    "constructor_standings": ["constructorStandingsId", "raceId", "constructorId"],
    "drivers": ["driverId"],
    "driver_standings": ["driverStandingsId", "raceId", "driverId"],
    "lap_times": ["raceId", "driverId"],
    "pit_stops": ["raceId", "driverId"],
    "qualifying": ["qualifyId", "raceId", "driverId", "constructorId"],
    "races": ["raceId", "circuitId"],
    "results": ["resultId", "raceId", "driverId", "constructorId", "statusId"],
    "seasons": ["year"],
    "sprint_results": ["resultId", "raceId", "driverId", "constructorId", "statusId"],
    "status": ["statusId"],
}

# rel-f1 task metadata (from RelBench paper)
TASKS = [
    {
        "name": "driver-dnf",
        "task_type": "binary_classification",
        "target_table": "drivers",
        "time_horizon": "1 race",
        "metrics": ["average_precision"],
        "description": "Predict whether a driver will DNF (did not finish) in an upcoming race"
    },
    {
        "name": "driver-top3",
        "task_type": "binary_classification",
        "target_table": "drivers",
        "time_horizon": "1 race",
        "metrics": ["average_precision"],
        "description": "Predict whether a driver will finish in the top 3 in an upcoming race"
    },
    {
        "name": "driver-position",
        "task_type": "regression",
        "target_table": "drivers",
        "time_horizon": "1 race",
        "metrics": ["mean_absolute_error"],
        "description": "Predict a driver's finishing position in an upcoming race"
    },
]


def download_ergast_csv() -> dict[str, pd.DataFrame]:
    """Download and extract all CSV tables from Ergast F1 database."""
    zip_path = TEMP_DIR / "f1db_csv.zip"

    if zip_path.exists():
        logger.info(f"Using cached zip: {zip_path}")
    else:
        logger.info(f"Downloading Ergast F1 CSV from {ERGAST_CSV_URL}...")
        resp = requests.get(ERGAST_CSV_URL, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        logger.info(f"Downloaded {len(resp.content) / 1e6:.1f} MB")

    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        logger.info(f"Found {len(csv_files)} CSV files in zip")
        for csv_name in sorted(csv_files):
            table_name = csv_name.replace(".csv", "")
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, na_values=["\\N", "NULL", ""])
                tables[table_name] = df
                logger.info(f"  {table_name}: {df.shape[0]} rows x {df.shape[1]} cols")

    return tables


def encode_table(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, list[dict], int]:
    """Encode a table's features: label-encode categoricals, datetime to epoch, handle NaN.

    Returns: (encoded_df, column_info_list, feature_dim)
    """
    encoded = df.copy()
    id_cols = set(ID_COLUMNS.get(table_name, []))
    column_info = []

    for col in encoded.columns:
        is_id = col in id_cols
        original_dtype = str(df[col].dtype)

        # Try datetime conversion
        if encoded[col].dtype == "object":
            try:
                dt_parsed = pd.to_datetime(encoded[col], format="mixed", errors="coerce")
                if dt_parsed.notna().sum() > len(dt_parsed) * 0.5:
                    encoded[col] = dt_parsed.astype("int64") // 10**9
                    encoded[col] = encoded[col].where(dt_parsed.notna(), -1)
                    column_info.append({
                        "name": col, "dtype": "datetime_epoch", "original_dtype": original_dtype, "is_id": is_id
                    })
                    continue
            except Exception:
                pass

        # Label-encode strings/objects
        if encoded[col].dtype == "object":
            codes, _ = pd.factorize(encoded[col], sort=False)
            encoded[col] = codes  # NaN → -1 by pd.factorize
            column_info.append({
                "name": col, "dtype": "categorical_encoded", "original_dtype": original_dtype, "is_id": is_id
            })
            continue

        # Boolean
        if encoded[col].dtype == "bool":
            encoded[col] = encoded[col].astype(int)
            column_info.append({
                "name": col, "dtype": "boolean_int", "original_dtype": original_dtype, "is_id": is_id
            })
            continue

        # Numeric - fill NaN with -1
        if pd.api.types.is_numeric_dtype(encoded[col]):
            encoded[col] = encoded[col].fillna(-1)
            column_info.append({
                "name": col, "dtype": "numeric", "original_dtype": original_dtype, "is_id": is_id
            })
            continue

        # Fallback
        codes, _ = pd.factorize(encoded[col].astype(str), sort=False)
        encoded[col] = codes
        column_info.append({
            "name": col, "dtype": "other_encoded", "original_dtype": original_dtype, "is_id": is_id
        })

    # Feature dim = non-ID numeric columns
    feature_cols = [ci["name"] for ci in column_info if not ci["is_id"]]
    feature_dim = len(feature_cols)

    return encoded, column_info, feature_dim


def compute_cardinality_stats(child_df: pd.DataFrame, child_col: str) -> dict:
    """Compute cardinality statistics for a FK link."""
    counts = child_df[child_col].value_counts()
    return {
        "mean": round(float(counts.mean()), 2),
        "median": round(float(counts.median()), 2),
        "max": int(counts.max()),
        "p95": round(float(counts.quantile(0.95)), 2),
        "std": round(float(counts.std()), 2),
        "num_parents": int(counts.shape[0]),
        "num_children": int(child_df.shape[0]),
    }


def extract_aligned_features(
    child_encoded: pd.DataFrame,
    parent_encoded: pd.DataFrame,
    child_col: str,
    parent_col: str,
    child_table: str,
    parent_table: str,
    child_id_cols: set[str],
    parent_id_cols: set[str],
) -> tuple[str, int, int, int]:
    """Extract aligned parent-child feature pairs and save as parquet.

    Returns: (parquet_filename, num_rows, parent_feature_dim, child_feature_dim)
    """
    # Get feature columns (non-ID)
    child_feature_cols = [c for c in child_encoded.columns if c not in child_id_cols]
    parent_feature_cols = [c for c in parent_encoded.columns if c not in parent_id_cols]

    if not child_feature_cols or not parent_feature_cols:
        logger.warning(f"  No features for {child_table}->{parent_table}, skipping parquet")
        return "", 0, len(parent_feature_cols), len(child_feature_cols)

    # Merge on FK
    merged = child_encoded.merge(
        parent_encoded,
        left_on=child_col,
        right_on=parent_col,
        how="inner",
        suffixes=("_child", "_parent")
    )

    if len(merged) == 0:
        logger.warning(f"  Empty merge for {child_table}->{parent_table}")
        return "", 0, len(parent_feature_cols), len(child_feature_cols)

    # Sample if too large
    if len(merged) > MAX_ALIGNED_ROWS:
        merged = merged.sample(n=MAX_ALIGNED_ROWS, random_state=42)
        logger.info(f"  Sampled {MAX_ALIGNED_ROWS} rows from {child_table}->{parent_table}")

    # Build aligned feature matrices with proper prefixed columns
    # After merge with suffixes, figure out which columns belong to which table
    result_cols = {}
    for c in child_feature_cols:
        if f"{c}_child" in merged.columns:
            result_cols[f"child__{c}"] = merged[f"{c}_child"].values
        elif c in merged.columns:
            result_cols[f"child__{c}"] = merged[c].values

    for c in parent_feature_cols:
        if f"{c}_parent" in merged.columns:
            result_cols[f"parent__{c}"] = merged[f"{c}_parent"].values
        elif c in merged.columns:
            result_cols[f"parent__{c}"] = merged[c].values

    if not result_cols:
        return "", 0, len(parent_feature_cols), len(child_feature_cols)

    aligned_df = pd.DataFrame(result_cols)
    filename = f"{child_table}__{parent_table}__{child_col}_features.parquet"
    filepath = PARQUET_DIR / filename
    aligned_df.to_parquet(filepath, engine="pyarrow", index=False)

    parent_dim = sum(1 for c in aligned_df.columns if c.startswith("parent__"))
    child_dim = sum(1 for c in aligned_df.columns if c.startswith("child__"))

    logger.info(f"  Saved {filepath.name}: {len(aligned_df)} rows, parent_dim={parent_dim}, child_dim={child_dim}")

    del merged, aligned_df
    gc.collect()

    return filename, len(result_cols.get(list(result_cols.keys())[0], [])) if result_cols else 0, parent_dim, child_dim


@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("RelBench rel-f1 (Ergast Formula 1) Dataset Preparation")
    logger.info("=" * 60)

    # ── Step 1: Download ─────────────────────────────────────────────────
    tables = download_ergast_csv()
    logger.info(f"Loaded {len(tables)} tables, total rows: {sum(df.shape[0] for df in tables.values())}")

    # ── Step 2: Encode tables ────────────────────────────────────────────
    encoded_tables: dict[str, pd.DataFrame] = {}
    table_metadata: list[dict] = []

    for table_name, df in tables.items():
        logger.info(f"Encoding table: {table_name}")
        try:
            encoded_df, col_info, feature_dim = encode_table(df, table_name)
            encoded_tables[table_name] = encoded_df

            # Sample rows (first 3) from ORIGINAL table for display
            sample_rows = []
            for _, row in df.head(3).iterrows():
                sample_row = {}
                for k, v in row.items():
                    if pd.isna(v):
                        sample_row[k] = None
                    elif isinstance(v, (np.integer,)):
                        sample_row[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        sample_row[k] = float(v)
                    else:
                        sample_row[k] = str(v)
                sample_rows.append(sample_row)

            table_metadata.append({
                "name": table_name,
                "row_count": int(df.shape[0]),
                "col_count": int(df.shape[1]),
                "feature_dim": feature_dim,
                "columns": col_info,
                "sample_rows": sample_rows,
            })
        except Exception:
            logger.exception(f"Failed to encode table {table_name}")

    logger.info(f"Encoded {len(encoded_tables)} tables")

    # ── Step 3: Process FK links ─────────────────────────────────────────
    fk_link_data: list[dict] = []
    valid_fk_count = 0

    for child_table, child_col, parent_table, parent_col in FK_LINKS:
        logger.info(f"Processing FK: {child_table}.{child_col} -> {parent_table}.{parent_col}")

        if child_table not in encoded_tables:
            logger.warning(f"  Child table '{child_table}' not found, skipping")
            continue
        if parent_table not in encoded_tables:
            logger.warning(f"  Parent table '{parent_table}' not found, skipping")
            continue

        child_enc = encoded_tables[child_table]
        parent_enc = encoded_tables[parent_table]

        if child_col not in child_enc.columns:
            logger.warning(f"  FK column '{child_col}' not in {child_table}, skipping")
            continue
        if parent_col not in parent_enc.columns:
            logger.warning(f"  PK column '{parent_col}' not in {parent_table}, skipping")
            continue

        # Cardinality stats
        card_stats = compute_cardinality_stats(child_enc, child_col)

        # Aligned features
        child_id_cols_set = set(ID_COLUMNS.get(child_table, []))
        parent_id_cols_set = set(ID_COLUMNS.get(parent_table, []))

        parquet_file, num_aligned, parent_fdim, child_fdim = extract_aligned_features(
            child_enc, parent_enc, child_col, parent_col,
            child_table, parent_table,
            child_id_cols_set, parent_id_cols_set,
        )

        fk_entry = {
            "child_table": child_table,
            "child_fk_col": child_col,
            "parent_table": parent_table,
            "parent_pk_col": parent_col,
            "cardinality_stats": card_stats,
            "num_parents": card_stats["num_parents"],
            "num_children": card_stats["num_children"],
            "aligned_features_file": parquet_file,
            "aligned_features_num_rows": num_aligned,
            "parent_feature_dim": parent_fdim,
            "child_feature_dim": child_fdim,
        }
        fk_link_data.append(fk_entry)
        valid_fk_count += 1
        logger.info(f"  Cardinality: mean={card_stats['mean']}, max={card_stats['max']}, p95={card_stats['p95']}")

    logger.info(f"Processed {valid_fk_count} valid FK links")

    # ── Step 4: Build output in exp_sel_data_out.json schema ─────────────
    # Each FK link = one example
    examples = []
    for i, fk in enumerate(fk_link_data):
        # Input: FK link description with table metadata
        child_meta = next((t for t in table_metadata if t["name"] == fk["child_table"]), {})
        parent_meta = next((t for t in table_metadata if t["name"] == fk["parent_table"]), {})

        input_data = {
            "fk_link": {
                "child_table": fk["child_table"],
                "child_fk_col": fk["child_fk_col"],
                "parent_table": fk["parent_table"],
                "parent_pk_col": fk["parent_pk_col"],
            },
            "child_table_info": {
                "row_count": child_meta.get("row_count", 0),
                "col_count": child_meta.get("col_count", 0),
                "feature_dim": child_meta.get("feature_dim", 0),
            },
            "parent_table_info": {
                "row_count": parent_meta.get("row_count", 0),
                "col_count": parent_meta.get("col_count", 0),
                "feature_dim": parent_meta.get("feature_dim", 0),
            },
        }

        output_data = {
            "cardinality_stats": fk["cardinality_stats"],
            "aligned_features_file": fk["aligned_features_file"],
            "aligned_features_num_rows": fk["aligned_features_num_rows"],
            "parent_feature_dim": fk["parent_feature_dim"],
            "child_feature_dim": fk["child_feature_dim"],
        }

        example = {
            "input": json.dumps(input_data, separators=(",", ":")),
            "output": json.dumps(output_data, separators=(",", ":")),
            "metadata_fk_index": i,
            "metadata_child_table": fk["child_table"],
            "metadata_parent_table": fk["parent_table"],
            "metadata_child_fk_col": fk["child_fk_col"],
            "metadata_cardinality_mean": fk["cardinality_stats"]["mean"],
            "metadata_cardinality_max": fk["cardinality_stats"]["max"],
            "metadata_cardinality_p95": fk["cardinality_stats"]["p95"],
            "metadata_cardinality_std": fk["cardinality_stats"]["std"],
            "metadata_num_parents": fk["num_parents"],
            "metadata_num_children": fk["num_children"],
            "metadata_parent_feature_dim": fk["parent_feature_dim"],
            "metadata_child_feature_dim": fk["child_feature_dim"],
            "metadata_aligned_rows": fk["aligned_features_num_rows"],
            "metadata_task_type": "relational_predictability",
        }
        examples.append(example)

    # Final output
    data_out = {
        "datasets": [
            {
                "dataset": "rel-f1",
                "examples": examples,
            }
        ],
        "metadata": {
            "dataset_name": "rel-f1",
            "source": "ergast_f1_database",
            "description": "Formula 1 relational database from Ergast Motor Racing Data (as used in RelBench). Contains 14 tables with ~74K total rows, 22 FK links with cardinality stats, and aligned parent-child feature matrices for cross-table predictability (R²) computation. High-priority for PRMP hypothesis: RelGNN reported largest improvements on rel-f1.",
            "tables": table_metadata,
            "fk_links": fk_link_data,
            "tasks": TASKS,
            "total_tables": len(table_metadata),
            "total_fk_links": len(fk_link_data),
            "total_rows_across_tables": sum(t["row_count"] for t in table_metadata),
            "license": "CC-BY-4.0",
            "data_source": "Ergast API (final 2024 release)",
        },
    }

    # Save
    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(data_out, indent=2, ensure_ascii=False))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Tables: {len(table_metadata)}")
    logger.info(f"  FK links: {len(fk_link_data)}")
    logger.info(f"  Total examples: {len(examples)}")
    logger.info(f"  Total rows across tables: {sum(t['row_count'] for t in table_metadata)}")
    logger.info(f"  Parquet files: {len(list(PARQUET_DIR.glob('*.parquet')))}")
    logger.info("=" * 60)

    # Print cardinality summary for PRMP analysis
    logger.info("Cardinality summary (sorted by mean):")
    for fk in sorted(fk_link_data, key=lambda x: x["cardinality_stats"]["mean"], reverse=True):
        cs = fk["cardinality_stats"]
        logger.info(
            f"  {fk['child_table']}->{fk['parent_table']} via {fk['child_fk_col']}: "
            f"mean={cs['mean']}, max={cs['max']}, p95={cs['p95']}, std={cs['std']}"
        )


if __name__ == "__main__":
    main()
