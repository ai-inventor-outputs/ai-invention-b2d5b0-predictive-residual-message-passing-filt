#!/usr/bin/env python3
"""Load Amazon Video Games and MovieLens relational datasets, build relational
tables with FK links, compute cardinality statistics, encode features with
meaningful parent aggregations, and output in exp_sel_data_out.json schema for
PRMP cross-table predictability diagnostic.

Two datasets prepared for comparison:
1. Amazon Video Games (50K reviews, 3.4K products, 10.9K customers)
2. MovieLens (891K ratings, 15K movies, 43K users)

Both have the same relational triangle structure (parent1→child←parent2)
with meaningful aggregated parent features for R² computation.
Memory-safe with cgroup-aware limits, output kept under 30MB total.
"""

import gc
import json
import math
import os
import resource
import sys
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# Set memory limit — budget 8 GB (conservative, well under 29 GB container)
RAM_BUDGET = int(8 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
DATASETS_DIR = WS / "temp" / "datasets"
OUTPUT_FILE = WS / "full_data_out.json"

# Keep output manageable: 10K examples, 2K alignment samples
MAX_EXAMPLES = 10_000
ALIGN_SAMPLE = 2_000
TEXT_HASH_DIM = 16


# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------

def encode_numeric(series: pd.Series) -> np.ndarray:
    """Encode a numeric column with StandardScaler."""
    vals = series.fillna(0).values.astype(np.float32).reshape(-1, 1)
    return StandardScaler().fit_transform(vals).astype(np.float32)


def encode_categorical(series: pd.Series) -> np.ndarray:
    """Label-encode a categorical column."""
    le = LabelEncoder()
    encoded = le.fit_transform(series.fillna("MISSING").astype(str))
    return encoded.reshape(-1, 1).astype(np.float32)


def encode_timestamp(series: pd.Series) -> np.ndarray:
    """Extract numeric features from unix timestamps."""
    ts = pd.to_datetime(series, unit="s", errors="coerce")
    feats = np.column_stack([
        ts.dt.year.fillna(2000).values,
        ts.dt.month.fillna(1).values,
        ts.dt.dayofweek.fillna(0).values,
    ]).astype(np.float32)
    return StandardScaler().fit_transform(feats)


def encode_text_hash(series: pd.Series, n_features: int = TEXT_HASH_DIM) -> np.ndarray:
    """Simple hash-based text encoding (fast, memory-efficient)."""
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


def parse_helpful(helpful_str) -> tuple[int, int]:
    """Parse helpful string like '[8, 12]' into (up, total)."""
    try:
        if isinstance(helpful_str, str):
            vals = eval(helpful_str)
            return int(vals[0]), int(vals[1])
        elif isinstance(helpful_str, list):
            return int(helpful_str[0]), int(helpful_str[1])
    except Exception:
        pass
    return 0, 0


def compute_cardinality_stats(child_df: pd.DataFrame, fk_col: str) -> dict:
    """Compute FK cardinality statistics."""
    card = child_df.groupby(fk_col).size()
    return {
        "cardinality_mean": round(float(card.mean()), 4),
        "cardinality_median": round(float(card.median()), 4),
        "cardinality_std": round(float(card.std()), 4),
        "cardinality_min": int(card.min()),
        "cardinality_max": int(card.max()),
        "cardinality_p25": round(float(card.quantile(0.25)), 4),
        "cardinality_p75": round(float(card.quantile(0.75)), 4),
    }


# ---------------------------------------------------------------------------
# Amazon Video Games dataset processing
# ---------------------------------------------------------------------------

def build_parent_features(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meaningful parent features by aggregating child (review) data."""
    logger.info("Building aggregated parent features from reviews...")

    helpful_parsed = reviews["helpful"].apply(parse_helpful)
    helpful_up = helpful_parsed.apply(lambda x: x[0]).astype(np.float32)
    helpful_total = helpful_parsed.apply(lambda x: x[1]).astype(np.float32)

    agg_df = reviews[["asin", "reviewerID", "overall"]].copy()
    agg_df["helpful_up"] = helpful_up
    agg_df["helpful_total"] = helpful_total

    # Product features: aggregate by asin
    product_agg = agg_df.groupby("asin").agg(
        prod_mean_rating=("overall", "mean"),
        prod_std_rating=("overall", "std"),
        prod_review_count=("overall", "count"),
        prod_mean_helpful_up=("helpful_up", "mean"),
        prod_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    product_agg["prod_std_rating"] = product_agg["prod_std_rating"].fillna(0)
    logger.info(f"Product features: {product_agg.shape}")

    # Customer features: aggregate by reviewerID
    customer_agg = agg_df.groupby("reviewerID").agg(
        cust_mean_rating=("overall", "mean"),
        cust_std_rating=("overall", "std"),
        cust_review_count=("overall", "count"),
        cust_mean_helpful_up=("helpful_up", "mean"),
        cust_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    customer_agg["cust_std_rating"] = customer_agg["cust_std_rating"].fillna(0)
    logger.info(f"Customer features: {customer_agg.shape}")

    del agg_df
    gc.collect()
    return product_agg, customer_agg


def process_amazon(file_path: Path) -> tuple[list[dict], dict]:
    """Process Amazon Video Games reviews into relational examples with
    meaningful parent features for R-squared cross-table predictability."""
    logger.info(f"Loading Amazon dataset from {file_path}")
    raw = pd.read_json(file_path)
    logger.info(f"Loaded {len(raw)} rows, columns: {list(raw.columns)}")

    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])

    # === Build relational tables ===
    products = raw[["asin"]].drop_duplicates().reset_index(drop=True)
    logger.info(f"Products table: {len(products)} unique products")

    customer_cols = ["reviewerID"]
    if "reviewerName" in raw.columns:
        customer_cols.append("reviewerName")
    customers = raw[customer_cols].drop_duplicates("reviewerID").reset_index(drop=True)
    logger.info(f"Customers table: {len(customers)} unique customers")

    reviews = raw.copy()
    del raw
    gc.collect()
    logger.info(f"Reviews table: {len(reviews)} reviews")

    # === FK cardinality statistics ===
    fk_product_stats = compute_cardinality_stats(reviews, "asin")
    fk_customer_stats = compute_cardinality_stats(reviews, "reviewerID")
    logger.info(f"FK product->review: mean_card={fk_product_stats['cardinality_mean']:.2f}")
    logger.info(f"FK customer->review: mean_card={fk_customer_stats['cardinality_mean']:.2f}")

    # === Build meaningful parent features via aggregation ===
    product_feats_df, customer_feats_df = build_parent_features(reviews)
    product_feat_names = list(product_feats_df.columns)
    customer_feat_names = list(customer_feats_df.columns)

    # Standardize parent features
    prod_scaler = StandardScaler()
    prod_features_scaled = prod_scaler.fit_transform(product_feats_df.values).astype(np.float32)
    product_feats_scaled = pd.DataFrame(
        prod_features_scaled, index=product_feats_df.index, columns=product_feat_names
    )
    del product_feats_df, prod_features_scaled
    gc.collect()

    cust_scaler = StandardScaler()
    cust_features_scaled = cust_scaler.fit_transform(customer_feats_df.values).astype(np.float32)
    customer_feats_scaled = pd.DataFrame(
        cust_features_scaled, index=customer_feats_df.index, columns=customer_feat_names
    )
    del customer_feats_df, cust_features_scaled
    gc.collect()

    # === Encode review (child) features ===
    feature_names = []
    feature_arrays = []

    if "unixReviewTime" in reviews.columns:
        ts_feats = encode_timestamp(reviews["unixReviewTime"])
        feature_names.extend(["time_year", "time_month", "time_dayofweek"])
        feature_arrays.append(ts_feats)

    if "summary" in reviews.columns:
        summary_feats = encode_text_hash(reviews["summary"], n_features=TEXT_HASH_DIM)
        feature_names.extend([f"summary_h{i}" for i in range(TEXT_HASH_DIM)])
        feature_arrays.append(summary_feats)

    # Helpful votes (numeric)
    helpful_parsed = reviews["helpful"].apply(parse_helpful)
    helpful_up = helpful_parsed.apply(lambda x: x[0]).values.astype(np.float32).reshape(-1, 1)
    helpful_total = helpful_parsed.apply(lambda x: x[1]).values.astype(np.float32).reshape(-1, 1)
    feature_arrays.extend([
        StandardScaler().fit_transform(helpful_up),
        StandardScaler().fit_transform(helpful_total),
    ])
    feature_names.extend(["helpful_up", "helpful_total"])
    del helpful_parsed, helpful_up, helpful_total
    gc.collect()

    X_child = np.hstack(feature_arrays).astype(np.float32)
    del feature_arrays
    gc.collect()
    logger.info(f"Child (review) feature matrix: {X_child.shape} -- {len(feature_names)} features")

    # === Sample examples ===
    n_total = len(reviews)
    n_sample = min(n_total, MAX_EXAMPLES)
    rng = np.random.RandomState(42)
    if n_sample < n_total:
        sample_indices = np.sort(rng.choice(n_total, n_sample, replace=False))
    else:
        sample_indices = np.arange(n_total)
    logger.info(f"Sampled {len(sample_indices)} examples from {n_total} total")

    # === Build aligned parent-child feature matrices for R-squared diagnostic ===
    align_n = min(ALIGN_SAMPLE, n_total)
    rng2 = np.random.RandomState(123)
    align_idx = np.sort(rng2.choice(n_total, align_n, replace=False))

    # Product->review: align product features with each review
    align_asins = reviews.iloc[align_idx]["asin"].values
    align_product_feats = product_feats_scaled.loc[align_asins].values.astype(np.float32)
    align_child_feats_prod = X_child[align_idx]

    # Customer->review: align customer features with each review
    align_reviewer_ids = reviews.iloc[align_idx]["reviewerID"].values
    align_customer_feats = customer_feats_scaled.loc[align_reviewer_ids].values.astype(np.float32)
    align_child_feats_cust = X_child[align_idx]

    def to_rounded_list(arr: np.ndarray) -> list:
        return [[round(float(v), 3) for v in row] for row in arr]

    fk_links = {
        "product_to_review": {
            "parent_table": "product",
            "child_table": "review",
            "fk_col": "asin",
            "num_parents": len(products),
            "num_children": n_total,
            **fk_product_stats,
            "parent_feature_names": product_feat_names,
            "child_feature_names": feature_names,
            "aligned_parent_features_sample": to_rounded_list(align_product_feats),
            "aligned_child_features_sample": to_rounded_list(align_child_feats_prod),
        },
        "customer_to_review": {
            "parent_table": "customer",
            "child_table": "review",
            "fk_col": "reviewerID",
            "num_parents": len(customers),
            "num_children": n_total,
            **fk_customer_stats,
            "parent_feature_names": customer_feat_names,
            "child_feature_names": feature_names,
            "aligned_parent_features_sample": to_rounded_list(align_customer_feats),
            "aligned_child_features_sample": to_rounded_list(align_child_feats_cust),
        },
    }

    del align_product_feats, align_customer_feats
    del align_child_feats_prod, align_child_feats_cust
    del product_feats_scaled, customer_feats_scaled
    gc.collect()

    # === Build examples ===
    logger.info("Building example dicts...")
    examples = []
    ratings = reviews["overall"].values
    asins = reviews["asin"].values
    reviewer_ids = reviews["reviewerID"].values

    for idx in sample_indices:
        row_features = {fn: round(float(X_child[idx, j]), 3) for j, fn in enumerate(feature_names)}
        example = {
            "input": json.dumps(row_features),
            "output": str(float(ratings[idx])),
            "metadata_fold": int(idx % 5),
            "metadata_row_index": int(idx),
            "metadata_task_type": "regression",
            "metadata_feature_names": feature_names,
            "metadata_product_id": str(asins[idx]),
            "metadata_customer_id": str(reviewer_ids[idx]),
        }
        examples.append(example)

    logger.info(f"Built {len(examples)} examples")

    # Table summary metadata
    time_start = str(pd.to_datetime(reviews["unixReviewTime"].min(), unit="s").date()) \
        if "unixReviewTime" in reviews.columns else None
    time_end = str(pd.to_datetime(reviews["unixReviewTime"].max(), unit="s").date()) \
        if "unixReviewTime" in reviews.columns else None

    dataset_metadata = {
        "source": "LoganKells/amazon_product_reviews_video_games (McAuley UCSD)",
        "description": "Amazon Video Games product reviews relational database with "
                       "product, customer, and review tables. Part of the Amazon product "
                       "reviews family used by RelBench.",
        "total_reviews": int(n_total),
        "num_examples_sampled": int(n_sample),
        "align_sample_size": int(align_n),
        "tables": {
            "product": {
                "num_rows": len(products),
                "columns": ["asin"],
                "pkey_col": "asin",
                "aggregated_feature_names": product_feat_names,
            },
            "customer": {
                "num_rows": len(customers),
                "columns": list(customers.columns),
                "pkey_col": "reviewerID",
                "aggregated_feature_names": customer_feat_names,
            },
            "review": {
                "num_rows": int(n_total),
                "columns": list(reviews.columns),
                "pkey_col": None,
                "fkey_col_to_pkey_table": {
                    "asin": "product",
                    "reviewerID": "customer",
                },
                "time_col": "unixReviewTime",
                "feature_names": feature_names,
            },
        },
        "fk_links": fk_links,
        "time_range": {"start": time_start, "end": time_end},
    }

    del reviews, X_child
    gc.collect()

    return examples, dataset_metadata


# ---------------------------------------------------------------------------
# MovieLens dataset processing
# ---------------------------------------------------------------------------

def build_movielens_parent_features(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meaningful parent features for movies and users by aggregation."""
    logger.info("Building aggregated parent features for MovieLens...")

    # Movie features: aggregate by movie_id
    movie_agg = ratings.groupby("movie_id").agg(
        movie_mean_rating=("rating", "mean"),
        movie_std_rating=("rating", "std"),
        movie_rating_count=("rating", "count"),
        movie_min_rating=("rating", "min"),
        movie_max_rating=("rating", "max"),
    ).astype(np.float32)
    movie_agg["movie_std_rating"] = movie_agg["movie_std_rating"].fillna(0)
    logger.info(f"Movie features: {movie_agg.shape}")

    # User features: aggregate by user_id
    user_agg = ratings.groupby("user_id").agg(
        user_mean_rating=("rating", "mean"),
        user_std_rating=("rating", "std"),
        user_rating_count=("rating", "count"),
        user_min_rating=("rating", "min"),
        user_max_rating=("rating", "max"),
    ).astype(np.float32)
    user_agg["user_std_rating"] = user_agg["user_std_rating"].fillna(0)
    logger.info(f"User features: {user_agg.shape}")

    return movie_agg, user_agg


def process_movielens(file_path: Path) -> tuple[list[dict], dict]:
    """Process MovieLens ratings into relational examples with meaningful
    parent features for R-squared cross-table predictability."""
    logger.info(f"Loading MovieLens dataset from {file_path}")
    raw = pd.read_json(file_path)
    logger.info(f"Loaded {len(raw)} rows, columns: {list(raw.columns)}")

    # === Build relational tables ===
    movie_cols = [c for c in ["movie_id", "title", "genres", "imdbId", "tmdbId"]
                  if c in raw.columns]
    movies = raw[movie_cols].drop_duplicates("movie_id").reset_index(drop=True)
    logger.info(f"Movies table: {len(movies)} unique movies")

    users = raw[["user_id"]].drop_duplicates().reset_index(drop=True)
    logger.info(f"Users table: {len(users)} unique users")

    ratings = raw[["movie_id", "user_id", "rating"]].copy()
    # Add genres for child features
    if "genres" in raw.columns:
        ratings["genres"] = raw["genres"].values
    if "title" in raw.columns:
        ratings["title"] = raw["title"].values
    del raw
    gc.collect()
    logger.info(f"Ratings table: {len(ratings)} ratings")

    # === FK cardinality statistics ===
    fk_movie_stats = compute_cardinality_stats(ratings, "movie_id")
    fk_user_stats = compute_cardinality_stats(ratings, "user_id")
    logger.info(f"FK movie->rating: mean_card={fk_movie_stats['cardinality_mean']:.2f}")
    logger.info(f"FK user->rating: mean_card={fk_user_stats['cardinality_mean']:.2f}")

    # === Build meaningful parent features via aggregation ===
    movie_feats_df, user_feats_df = build_movielens_parent_features(ratings)
    movie_feat_names = list(movie_feats_df.columns)
    user_feat_names = list(user_feats_df.columns)

    # Standardize parent features
    movie_scaler = StandardScaler()
    movie_feats_scaled = pd.DataFrame(
        movie_scaler.fit_transform(movie_feats_df.values).astype(np.float32),
        index=movie_feats_df.index, columns=movie_feat_names,
    )
    del movie_feats_df
    gc.collect()

    user_scaler = StandardScaler()
    user_feats_scaled = pd.DataFrame(
        user_scaler.fit_transform(user_feats_df.values).astype(np.float32),
        index=user_feats_df.index, columns=user_feat_names,
    )
    del user_feats_df
    gc.collect()

    # === Encode child (rating) features ===
    feature_names = []
    feature_arrays = []

    # Genre multi-hot encoding
    if "genres" in ratings.columns:
        all_genres = set()
        for g in ratings["genres"].dropna().unique():
            all_genres.update(g.split("|"))
        all_genres = sorted(all_genres - {"(no genres listed)", ""})
        genre_matrix = np.zeros((len(ratings), len(all_genres)), dtype=np.float32)
        genre_vals = ratings["genres"].fillna("").values
        for i, g in enumerate(genre_vals):
            for genre in g.split("|"):
                if genre in all_genres:
                    genre_matrix[i, all_genres.index(genre)] = 1.0
        feature_names.extend([f"genre_{g}" for g in all_genres])
        feature_arrays.append(genre_matrix)
        del genre_matrix, genre_vals
        gc.collect()

    # Title hash features
    if "title" in ratings.columns:
        title_feats = encode_text_hash(ratings["title"], n_features=TEXT_HASH_DIM)
        feature_names.extend([f"title_h{i}" for i in range(TEXT_HASH_DIM)])
        feature_arrays.append(title_feats)
        del title_feats
        gc.collect()

    X_child = np.hstack(feature_arrays).astype(np.float32)
    del feature_arrays
    gc.collect()
    logger.info(f"Child (rating) feature matrix: {X_child.shape} -- {len(feature_names)} features")

    # === Sample examples ===
    n_total = len(ratings)
    n_sample = min(n_total, MAX_EXAMPLES)
    rng = np.random.RandomState(42)
    if n_sample < n_total:
        sample_indices = np.sort(rng.choice(n_total, n_sample, replace=False))
    else:
        sample_indices = np.arange(n_total)
    logger.info(f"Sampled {len(sample_indices)} examples from {n_total} total")

    # === Build aligned parent-child feature matrices ===
    align_n = min(ALIGN_SAMPLE, n_total)
    rng2 = np.random.RandomState(123)
    align_idx = np.sort(rng2.choice(n_total, align_n, replace=False))

    align_movie_ids = ratings.iloc[align_idx]["movie_id"].values
    align_movie_feats = movie_feats_scaled.loc[align_movie_ids].values.astype(np.float32)
    align_child_feats_movie = X_child[align_idx]

    align_user_ids = ratings.iloc[align_idx]["user_id"].values
    align_user_feats = user_feats_scaled.loc[align_user_ids].values.astype(np.float32)
    align_child_feats_user = X_child[align_idx]

    def to_rounded_list(arr: np.ndarray) -> list:
        return [[round(float(v), 3) for v in row] for row in arr]

    fk_links = {
        "movie_to_rating": {
            "parent_table": "movie",
            "child_table": "rating",
            "fk_col": "movie_id",
            "num_parents": len(movies),
            "num_children": n_total,
            **fk_movie_stats,
            "parent_feature_names": movie_feat_names,
            "child_feature_names": feature_names,
            "aligned_parent_features_sample": to_rounded_list(align_movie_feats),
            "aligned_child_features_sample": to_rounded_list(align_child_feats_movie),
        },
        "user_to_rating": {
            "parent_table": "user",
            "child_table": "rating",
            "fk_col": "user_id",
            "num_parents": len(users),
            "num_children": n_total,
            **fk_user_stats,
            "parent_feature_names": user_feat_names,
            "child_feature_names": feature_names,
            "aligned_parent_features_sample": to_rounded_list(align_user_feats),
            "aligned_child_features_sample": to_rounded_list(align_child_feats_user),
        },
    }

    del align_movie_feats, align_user_feats
    del align_child_feats_movie, align_child_feats_user
    del movie_feats_scaled, user_feats_scaled
    gc.collect()

    # === Build examples ===
    logger.info("Building MovieLens example dicts...")
    examples = []
    rating_vals = ratings["rating"].values
    movie_ids = ratings["movie_id"].values
    user_ids_arr = ratings["user_id"].values

    for idx in sample_indices:
        row_features = {fn: round(float(X_child[idx, j]), 3) for j, fn in enumerate(feature_names)}
        example = {
            "input": json.dumps(row_features),
            "output": str(float(rating_vals[idx])),
            "metadata_fold": int(idx % 5),
            "metadata_row_index": int(idx),
            "metadata_task_type": "regression",
            "metadata_feature_names": feature_names,
            "metadata_movie_id": int(movie_ids[idx]),
            "metadata_user_id": int(user_ids_arr[idx]),
        }
        examples.append(example)

    logger.info(f"Built {len(examples)} MovieLens examples")

    dataset_metadata = {
        "source": "ashraq/movielens_ratings (GroupLens Research)",
        "description": "MovieLens movie ratings relational database with movie, user, "
                       "and rating tables. Standard collaborative filtering benchmark.",
        "total_ratings": int(n_total),
        "num_examples_sampled": int(n_sample),
        "align_sample_size": int(align_n),
        "tables": {
            "movie": {
                "num_rows": len(movies),
                "columns": movie_cols,
                "pkey_col": "movie_id",
                "aggregated_feature_names": movie_feat_names,
            },
            "user": {
                "num_rows": len(users),
                "columns": ["user_id"],
                "pkey_col": "user_id",
                "aggregated_feature_names": user_feat_names,
            },
            "rating": {
                "num_rows": int(n_total),
                "columns": list(ratings.columns),
                "pkey_col": None,
                "fkey_col_to_pkey_table": {
                    "movie_id": "movie",
                    "user_id": "user",
                },
                "feature_names": feature_names,
            },
        },
        "fk_links": fk_links,
    }

    del ratings, X_child
    gc.collect()

    return examples, dataset_metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main() -> None:
    logger.info("Starting PRMP relational data preparation (Amazon Video Games)")

    amazon_file = DATASETS_DIR / "full_LoganKells_amazon_product_reviews_video_games_train.json"

    if not amazon_file.exists():
        logger.error(f"Amazon dataset not found at {amazon_file}")
        sys.exit(1)

    # Process Amazon Video Games (chosen as best dataset for PRMP diagnostic)
    amazon_examples, amazon_meta = process_amazon(amazon_file)

    data_out = {
        "metadata": {
            "description": "Amazon Video Games relational dataset for PRMP cross-table "
                           "predictability diagnostic. Product/customer/review triangle "
                           "structure with FK links, cardinality statistics, and aligned "
                           "parent-child feature matrices for R-squared computation.",
            "source": "HuggingFace Hub - LoganKells/amazon_product_reviews_video_games",
            "compute_profile": "cpu_heavy",
            "datasets_info": {
                "amazon_video_games": amazon_meta,
            },
        },
        "datasets": [
            {
                "dataset": "amazon_video_games_reviews",
                "examples": amazon_examples,
            },
        ],
    }

    logger.info(f"Writing output to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data_out, f)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {size_mb:.1f} MB")
    logger.info(f"Total examples: {len(amazon_examples)}")

    del data_out, amazon_examples
    gc.collect()

    logger.info("Done!")


if __name__ == "__main__":
    main()
