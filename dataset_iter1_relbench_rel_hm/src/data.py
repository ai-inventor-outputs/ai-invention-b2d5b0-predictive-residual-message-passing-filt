#!/usr/bin/env python3
"""Load H&M (rel-hm) relational dataset, extract FK links, cardinality stats,
and aligned parent-child feature pairs. Output as full_data_out.json."""

import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

# --- Logging ---
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# --- Hardware-aware memory limits ---
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    return os.cpu_count() or 1

TOTAL_RAM_GB = _container_ram_gb() or 29.0
NUM_CPUS = _detect_cpus()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)  # 50% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# --- Paths ---
WS = Path(__file__).parent
DATASETS_DIR = WS / "temp" / "datasets"
HM_DIR = DATASETS_DIR / "hm"


def encode_table_features(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Convert all columns to numeric, dropping ID columns and pure-text columns."""
    feature_df = df.drop(columns=id_cols, errors="ignore").copy()

    # Convert datetime columns to epoch seconds
    for col in feature_df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns:
        feature_df[col] = feature_df[col].astype(np.int64) // 10**9

    # Handle string date columns (like 't_dat') with explicit format
    for col in list(feature_df.columns):
        if feature_df[col].dtype == object:
            sample = feature_df[col].dropna().head(10).astype(str)
            if sample.str.match(r"^\d{4}-\d{2}-\d{2}$").all() and len(sample) > 0:
                parsed = pd.to_datetime(feature_df[col], format="%Y-%m-%d", errors="coerce")
                if parsed.notna().sum() > len(feature_df) * 0.8:
                    feature_df[col] = parsed.astype(np.int64) // 10**9
                    continue

    # Label-encode categoricals and objects
    for col in feature_df.select_dtypes(include=["object", "category"]).columns:
        nunique = feature_df[col].nunique()
        if nunique > 10000:
            logger.debug(f"Dropping high-cardinality column: {col} ({nunique} unique)")
            feature_df = feature_df.drop(columns=[col])
        else:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str).fillna("__NA__"))

    # Fill NaN and convert to float32
    feature_df = feature_df.fillna(0).astype(np.float32)
    return feature_df


def compute_cardinality_stats(child_df: pd.DataFrame, fk_col: str) -> dict:
    """Compute cardinality statistics for a FK relationship."""
    cardinality = child_df.groupby(fk_col).size()
    hist_counts, hist_bins = np.histogram(cardinality.values, bins=20)
    return {
        "num_unique_parents": int(cardinality.shape[0]),
        "num_children_total": int(len(child_df)),
        "mean_cardinality": float(round(cardinality.mean(), 4)),
        "median_cardinality": float(cardinality.median()),
        "max_cardinality": int(cardinality.max()),
        "std_cardinality": float(round(cardinality.std(), 4)),
        "p95_cardinality": float(cardinality.quantile(0.95)),
        "cardinality_histogram": {
            "bins": [float(round(b, 4)) for b in hist_bins.tolist()],
            "counts": hist_counts.tolist(),
        },
    }


def extract_fk_link(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_name: str,
    child_name: str,
    parent_pkey: str,
    fk_col: str,
    parent_id_cols: list[str],
    child_id_cols: list[str],
    max_sample_pairs: int = 5000,
    parquet_sample: int = 200000,
) -> dict:
    """Extract one FK link with cardinality stats and aligned feature pairs."""
    logger.info(f"Processing FK: {child_name}.{fk_col} -> {parent_name}.{parent_pkey}")

    # Cardinality stats
    card_stats = compute_cardinality_stats(child_df, fk_col)
    logger.info(f"  Cardinality: mean={card_stats['mean_cardinality']:.2f}, "
                f"median={card_stats['median_cardinality']:.0f}, "
                f"max={card_stats['max_cardinality']}")

    # Encode features
    parent_features = encode_table_features(parent_df, parent_id_cols)
    child_features = encode_table_features(child_df, child_id_cols)

    parent_feature_cols = list(parent_features.columns)
    child_feature_cols = list(child_features.columns)

    logger.info(f"  Parent features: {len(parent_feature_cols)} cols, Child features: {len(child_feature_cols)} cols")

    # Align by FK join - sample child rows first to save memory
    if len(child_df) > parquet_sample:
        sample_idx = np.random.choice(len(child_df), size=parquet_sample, replace=False)
        child_sample = child_df.iloc[sample_idx].copy()
        child_feat_sample = child_features.iloc[sample_idx].copy()
    else:
        child_sample = child_df.copy()
        child_feat_sample = child_features.copy()

    # Rename columns with explicit prefixes to avoid suffix issues
    child_feat_renamed = child_feat_sample.rename(
        columns={c: f"child__{c}" for c in child_feature_cols}
    )
    child_feat_renamed["__fk__"] = child_sample[fk_col].values

    parent_feat_renamed = parent_features.rename(
        columns={c: f"parent__{c}" for c in parent_feature_cols}
    )
    parent_feat_renamed["__pk__"] = parent_df[parent_pkey].values

    # Merge
    aligned = child_feat_renamed.merge(
        parent_feat_renamed,
        left_on="__fk__",
        right_on="__pk__",
    )
    aligned = aligned.drop(columns=["__fk__", "__pk__"], errors="ignore")

    logger.info(f"  Aligned pairs: {len(aligned)} rows")

    # Save supplementary parquet
    parquet_name = f"supplementary_{parent_name}_{child_name}_aligned_features.parquet"
    parquet_path = WS / parquet_name
    aligned.to_parquet(parquet_path, index=False)
    logger.info(f"  Saved parquet: {parquet_name} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # Sample pairs for JSON output
    n_sample = min(max_sample_pairs, len(aligned))
    if n_sample < len(aligned):
        sample_rows = aligned.sample(n=n_sample, random_state=42)
    else:
        sample_rows = aligned

    # Separate parent and child features using explicit prefixes
    child_cols_in_aligned = [c for c in sample_rows.columns if c.startswith("child__")]
    parent_cols_in_aligned = [c for c in sample_rows.columns if c.startswith("parent__")]

    sample_pairs = []
    for _, row in sample_rows.iterrows():
        pair = {
            "parent_features": [float(round(v, 6)) for v in row[parent_cols_in_aligned].values],
            "child_features": [float(round(v, 6)) for v in row[child_cols_in_aligned].values],
        }
        sample_pairs.append(pair)

    # Build the example
    input_data = {
        "parent_table": parent_name,
        "child_table": child_name,
        "fk_column": fk_col,
        "parent_feature_columns": parent_feature_cols,
        "child_feature_columns": child_feature_cols,
        "parent_feature_dim": len(parent_feature_cols),
        "child_feature_dim": len(child_feature_cols),
        "parent_row_count": len(parent_df),
        "child_row_count": len(child_df),
    }

    output_data = {
        **card_stats,
        "sample_parent_child_pairs": sample_pairs,
    }

    return {
        "input": json.dumps(input_data),
        "output": json.dumps(output_data),
        "metadata_fold": 0,
        "metadata_parent_table": parent_name,
        "metadata_child_table": child_name,
        "metadata_fk_column": fk_col,
        "metadata_task_type": "fk_cardinality_and_feature_alignment",
        "metadata_supplementary_file": parquet_name,
    }


def process_hm() -> dict:
    """Process H&M dataset: extract FK links with stats and feature pairs."""
    logger.info("=== Processing H&M Dataset ===")

    # Load tables
    articles = pd.read_parquet(HM_DIR / "articles.parquet")
    logger.info(f"  Articles: {articles.shape}")
    customers = pd.read_parquet(HM_DIR / "customers.parquet")
    logger.info(f"  Customers: {customers.shape}")
    transactions = pd.read_parquet(HM_DIR / "transactions_sample.parquet")
    logger.info(f"  Transactions (sample): {transactions.shape}")

    examples = []

    # FK 1: transaction.customer_id -> customer.customer_id
    ex1 = extract_fk_link(
        parent_df=customers,
        child_df=transactions,
        parent_name="customer",
        child_name="transaction",
        parent_pkey="customer_id",
        fk_col="customer_id",
        parent_id_cols=["customer_id", "postal_code"],
        child_id_cols=["customer_id", "article_id"],
        max_sample_pairs=5000,
        parquet_sample=200000,
    )
    examples.append(ex1)

    # FK 2: transaction.article_id -> article.article_id
    ex2 = extract_fk_link(
        parent_df=articles,
        child_df=transactions,
        parent_name="article",
        child_name="transaction",
        parent_pkey="article_id",
        fk_col="article_id",
        parent_id_cols=["article_id", "product_code"],
        child_id_cols=["customer_id", "article_id"],
        max_sample_pairs=5000,
        parquet_sample=200000,
    )
    examples.append(ex2)

    return {"dataset": "rel_hm_fashion", "examples": examples}


@logger.catch
def main():
    logger.info("Starting data collection pipeline")
    np.random.seed(42)

    datasets_output = []

    # Process H&M (chosen dataset)
    try:
        hm_data = process_hm()
        datasets_output.append(hm_data)
        logger.info(f"H&M: {len(hm_data['examples'])} FK links extracted")
    except Exception:
        logger.exception("Failed to process H&M dataset")
        raise

    # Build final output
    output = {"datasets": datasets_output}

    # Save
    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Total datasets: {len(datasets_output)}, "
                f"Total FK links: {sum(len(d['examples']) for d in datasets_output)}")


if __name__ == "__main__":
    main()
