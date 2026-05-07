#!/usr/bin/env python3
"""Full-Scale Amazon PRMP Validation.

Train 4 GNN variants (Standard, PRMP, Wide control, Auxiliary MLP control)
x 5 seeds on Amazon Video Games review-rating regression (50K reviews).
Key deliverable: Cohen's d effect size at full scale vs the d~1.0 at 10K.
"""

import gc
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
from scipy import stats
from sklearn.linear_model import Ridge
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
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# Memory limits
RAM_BUDGET = int(20 * 1024**3)  # 20 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(16 * 1024**3)  # 16 GB of 20 GB
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), device={DEVICE}")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
DEP_WS = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
    "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)

TEXT_HASH_DIM = 16
SEEDS = [42, 123, 456, 789, 1024]
HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 20
MAX_EPOCHS = 200
GRAD_CLIP = 1.0

# ---------------------------------------------------------------------------
# Feature encoding (replicating data.py exactly)
# ---------------------------------------------------------------------------

def encode_timestamp(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, unit="s", errors="coerce")
    feats = np.column_stack([
        ts.dt.year.fillna(2000).values,
        ts.dt.month.fillna(1).values,
        ts.dt.dayofweek.fillna(0).values,
    ]).astype(np.float32)
    return StandardScaler().fit_transform(feats)


def encode_text_hash(series: pd.Series, n_features: int = TEXT_HASH_DIM) -> np.ndarray:
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
    try:
        if isinstance(helpful_str, str):
            vals = eval(helpful_str)  # noqa: S307
            return int(vals[0]), int(vals[1])
        if isinstance(helpful_str, list):
            return int(helpful_str[0]), int(helpful_str[1])
    except Exception:
        pass
    return 0, 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_encode_data(max_examples: int | None = None):
    """Load Amazon Video Games data and encode features.

    Returns a tuple:
        X_child            (N, 21)
        product_features   (P, 5)
        customer_features  (C, 5)
        ratings            (N,)
        product_ids        (N,)  int index per review
        customer_ids       (N,)  int index per review
        product_id_map     dict asin  -> int
        customer_id_map    dict revID -> int
        timestamps         (N,)
        feature_names      list[str]
    """
    logger.info("Loading Amazon Video Games dataset ...")

    # --- try HuggingFace first ---
    try:
        from datasets import load_dataset
        logger.info("Downloading from HuggingFace: "
                    "LoganKells/amazon_product_reviews_video_games")
        ds = load_dataset(
            "LoganKells/amazon_product_reviews_video_games", split="train"
        )
        raw_df = ds.to_pandas()
        logger.info(f"Downloaded {len(raw_df)} rows, cols={list(raw_df.columns)}")
        del ds; gc.collect()
    except Exception as exc:
        logger.warning(f"HuggingFace download failed: {exc}")
        logger.info("Falling back to dependency data (10K examples)")
        return _load_from_dependency(DEP_WS / "full_data_out.json", max_examples)

    if max_examples and max_examples < len(raw_df):
        raw_df = raw_df.head(max_examples)
        logger.info(f"Limited to {max_examples} examples for testing")

    n_total = len(raw_df)

    # --- unique-ID maps ---
    unique_asins = sorted(raw_df["asin"].unique())
    unique_rids  = sorted(raw_df["reviewerID"].unique())
    product_id_map  = {a: i for i, a in enumerate(unique_asins)}
    customer_id_map = {r: i for i, r in enumerate(unique_rids)}
    logger.info(f"Unique products={len(unique_asins)}, customers={len(unique_rids)}")

    product_ids  = np.array([product_id_map[a]  for a in raw_df["asin"]],       dtype=np.int64)
    customer_ids = np.array([customer_id_map[r] for r in raw_df["reviewerID"]], dtype=np.int64)

    # --- child (review) features ---
    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []

    # 3 time features
    ts_feats = encode_timestamp(raw_df["unixReviewTime"])
    feature_names.extend(["time_year", "time_month", "time_dayofweek"])
    feature_arrays.append(ts_feats)

    # 16 text-hash features
    summary_feats = encode_text_hash(raw_df["summary"], n_features=TEXT_HASH_DIM)
    feature_names.extend([f"summary_h{i}" for i in range(TEXT_HASH_DIM)])
    feature_arrays.append(summary_feats)

    # 2 helpful-vote features
    h_parsed = raw_df["helpful"].apply(parse_helpful)
    h_up    = h_parsed.apply(lambda x: x[0]).values.astype(np.float32).reshape(-1, 1)
    h_total = h_parsed.apply(lambda x: x[1]).values.astype(np.float32).reshape(-1, 1)
    feature_arrays.extend([
        StandardScaler().fit_transform(h_up),
        StandardScaler().fit_transform(h_total),
    ])
    feature_names.extend(["helpful_up", "helpful_total"])
    del h_parsed, h_up, h_total; gc.collect()

    X_child = np.hstack(feature_arrays).astype(np.float32)  # (N, 21)
    del feature_arrays; gc.collect()
    logger.info(f"Child features: {X_child.shape}")

    # --- parent features via aggregation ---
    agg_df = pd.DataFrame({
        "asin": raw_df["asin"].values,
        "reviewerID": raw_df["reviewerID"].values,
        "overall": raw_df["overall"].values.astype(np.float32),
    })
    hp = raw_df["helpful"].apply(parse_helpful)
    agg_df["helpful_up"]    = hp.apply(lambda x: x[0]).astype(np.float32)
    agg_df["helpful_total"] = hp.apply(lambda x: x[1]).astype(np.float32)
    del hp

    product_agg = agg_df.groupby("asin").agg(
        prod_mean_rating=("overall", "mean"),
        prod_std_rating=("overall", "std"),
        prod_review_count=("overall", "count"),
        prod_mean_helpful_up=("helpful_up", "mean"),
        prod_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    product_agg["prod_std_rating"] = product_agg["prod_std_rating"].fillna(0)

    customer_agg = agg_df.groupby("reviewerID").agg(
        cust_mean_rating=("overall", "mean"),
        cust_std_rating=("overall", "std"),
        cust_review_count=("overall", "count"),
        cust_mean_helpful_up=("helpful_up", "mean"),
        cust_mean_helpful_total=("helpful_total", "mean"),
    ).astype(np.float32)
    customer_agg["cust_std_rating"] = customer_agg["cust_std_rating"].fillna(0)
    del agg_df; gc.collect()

    product_features = StandardScaler().fit_transform(
        product_agg.loc[unique_asins].values
    ).astype(np.float32)   # (P, 5)

    customer_features = StandardScaler().fit_transform(
        customer_agg.loc[unique_rids].values
    ).astype(np.float32)   # (C, 5)

    logger.info(f"Product feats: {product_features.shape}, "
                f"Customer feats: {customer_features.shape}")

    ratings    = raw_df["overall"].values.astype(np.float32)
    timestamps = raw_df["unixReviewTime"].values.astype(np.int64)

    del raw_df, product_agg, customer_agg; gc.collect()

    return (X_child, product_features, customer_features, ratings,
            product_ids, customer_ids, product_id_map, customer_id_map,
            timestamps, feature_names)


def _load_from_dependency(data_path: Path, max_examples: int | None = None):
    """Fallback: reconstruct from dependency JSON (10K examples)."""
    logger.info(f"Loading from {data_path}")
    data = json.loads(data_path.read_text())
    examples = data["datasets"][0]["examples"]
    if max_examples:
        examples = examples[:max_examples]

    n = len(examples)
    feature_names = examples[0]["metadata_feature_names"]
    n_features = len(feature_names)

    X_child = np.zeros((n, n_features), dtype=np.float32)
    ratings = np.zeros(n, dtype=np.float32)
    prod_strs: list[str] = []
    cust_strs: list[str] = []

    for i, ex in enumerate(examples):
        feats = json.loads(ex["input"])
        for j, fn in enumerate(feature_names):
            X_child[i, j] = feats.get(fn, 0.0)
        ratings[i] = float(ex["output"])
        prod_strs.append(ex["metadata_product_id"])
        cust_strs.append(ex["metadata_customer_id"])

    unique_prods = sorted(set(prod_strs))
    unique_custs = sorted(set(cust_strs))
    product_id_map  = {p: i for i, p in enumerate(unique_prods)}
    customer_id_map = {c: i for i, c in enumerate(unique_custs)}

    product_ids  = np.array([product_id_map[p]  for p in prod_strs], dtype=np.int64)
    customer_ids = np.array([customer_id_map[c] for c in cust_strs], dtype=np.int64)

    n_products  = len(unique_prods)
    n_customers = len(unique_custs)

    # aggregate parent features
    prod_r: dict[int, list[float]] = {i: [] for i in range(n_products)}
    cust_r: dict[int, list[float]] = {i: [] for i in range(n_customers)}
    for i in range(n):
        prod_r[product_ids[i]].append(float(ratings[i]))
        cust_r[customer_ids[i]].append(float(ratings[i]))

    product_features = np.zeros((n_products, 5), dtype=np.float32)
    for pid in range(n_products):
        r = prod_r[pid]
        if r:
            product_features[pid] = [np.mean(r), np.std(r), len(r), 0, 0]

    customer_features = np.zeros((n_customers, 5), dtype=np.float32)
    for cid in range(n_customers):
        r = cust_r[cid]
        if r:
            customer_features[cid] = [np.mean(r), np.std(r), len(r), 0, 0]

    product_features  = StandardScaler().fit_transform(product_features).astype(np.float32)
    customer_features = StandardScaler().fit_transform(customer_features).astype(np.float32)

    timestamps = np.arange(n, dtype=np.int64)

    del data, examples; gc.collect()
    return (X_child, product_features, customer_features, ratings,
            product_ids, customer_ids, product_id_map, customer_id_map,
            timestamps, feature_names)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(X_child, product_features, customer_features, ratings,
                product_ids, customer_ids, timestamps):
    n_reviews   = len(X_child)
    n_products  = len(product_features)
    n_customers = len(customer_features)
    logger.info(f"Building graph: {n_reviews} reviews, "
                f"{n_products} products, {n_customers} customers")

    review_x   = torch.tensor(X_child,            dtype=torch.float32)
    product_x  = torch.tensor(product_features,   dtype=torch.float32)
    customer_x = torch.tensor(customer_features,  dtype=torch.float32)
    y          = torch.tensor(ratings,             dtype=torch.float32)

    prod_ids_t = torch.tensor(product_ids,  dtype=torch.long)
    cust_ids_t = torch.tensor(customer_ids, dtype=torch.long)
    rev_ids_t  = torch.arange(n_reviews,    dtype=torch.long)

    # temporal split 70/15/15
    sorted_idx = np.argsort(timestamps)
    n_train = int(0.70 * n_reviews)
    n_val   = int(0.15 * n_reviews)

    train_mask = torch.zeros(n_reviews, dtype=torch.bool)
    val_mask   = torch.zeros(n_reviews, dtype=torch.bool)
    test_mask  = torch.zeros(n_reviews, dtype=torch.bool)
    train_mask[sorted_idx[:n_train]]                   = True
    val_mask[sorted_idx[n_train:n_train + n_val]]      = True
    test_mask[sorted_idx[n_train + n_val:]]            = True

    logger.info(f"Split: train={train_mask.sum().item()}, "
                f"val={val_mask.sum().item()}, test={test_mask.sum().item()}")

    return {
        "review_x":   review_x,
        "product_x":  product_x,
        "customer_x": customer_x,
        "y":          y,
        "prod_to_rev_edge": torch.stack([prod_ids_t, rev_ids_t]),
        "cust_to_rev_edge": torch.stack([cust_ids_t, rev_ids_t]),
        "rev_to_prod_edge": torch.stack([rev_ids_t,  prod_ids_t]),
        "rev_to_cust_edge": torch.stack([rev_ids_t,  cust_ids_t]),
        "train_mask": train_mask,
        "val_mask":   val_mask,
        "test_mask":  test_mask,
        "n_reviews":   n_reviews,
        "n_products":  n_products,
        "n_customers": n_customers,
    }


# ---------------------------------------------------------------------------
# Scatter mean (manual, avoids PyG / torch-scatter dependency)
# ---------------------------------------------------------------------------

def scatter_mean(src: torch.Tensor, index: torch.Tensor,
                 dim_size: int) -> torch.Tensor:
    """Mean-pool *src* rows by *index* into *dim_size* buckets."""
    out   = torch.zeros(dim_size, src.size(1), device=src.device,
                        dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx   = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(0, index.unsqueeze(1),
                       torch.ones(index.size(0), 1,
                                  device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1)


# ===================================================================
# MODEL A – Standard heterogeneous GNN
# ===================================================================

class StandardHeteroLayer(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.msg_p2r = nn.Linear(h, h)
        self.msg_c2r = nn.Linear(h, h)
        self.msg_r2p = nn.Linear(h, h)
        self.msg_r2c = nn.Linear(h, h)
        self.upd_r = nn.Linear(h * 3, h)
        self.upd_p = nn.Linear(h * 2, h)
        self.upd_c = nn.Linear(h * 2, h)

    def forward(self, hr, hp, hc, g):
        nr, nprod, ncust = g["n_reviews"], g["n_products"], g["n_customers"]
        p2r = g["prod_to_rev_edge"]; c2r = g["cust_to_rev_edge"]
        r2p = g["rev_to_prod_edge"]; r2c = g["rev_to_cust_edge"]

        agg_p2r = scatter_mean(self.msg_p2r(hp[p2r[0]]), p2r[1], nr)
        agg_c2r = scatter_mean(self.msg_c2r(hc[c2r[0]]), c2r[1], nr)
        agg_r2p = scatter_mean(self.msg_r2p(hr[r2p[0]]), r2p[1], nprod)
        agg_r2c = scatter_mean(self.msg_r2c(hr[r2c[0]]), r2c[1], ncust)

        new_r = F.relu(self.upd_r(torch.cat([hr, agg_p2r, agg_c2r], 1)))
        new_p = F.relu(self.upd_p(torch.cat([hp, agg_r2p], 1)))
        new_c = F.relu(self.upd_c(torch.cat([hc, agg_r2c], 1)))
        return new_r, new_p, new_c


class StandardHeteroGNN(nn.Module):
    def __init__(self, cd=21, pd_=5, cud=5, h=128, nl=2):
        super().__init__()
        self.proj_r = nn.Linear(cd, h)
        self.proj_p = nn.Linear(pd_, h)
        self.proj_c = nn.Linear(cud, h)
        self.layers = nn.ModuleList([StandardHeteroLayer(h) for _ in range(nl)])
        self.head = nn.Linear(h, 1)
        self.variant = "A_standard"

    def forward(self, g, return_emb=False):
        hr = F.relu(self.proj_r(g["review_x"]))
        hp = F.relu(self.proj_p(g["product_x"]))
        hc = F.relu(self.proj_c(g["customer_x"]))
        for layer in self.layers:
            hr, hp, hc = layer(hr, hp, hc, g)
        out = self.head(hr).squeeze(-1)
        return (out, hr, hp, hc) if return_emb else out


# ===================================================================
# MODEL B – PRMP (Predictive Residual Message Passing)
# ===================================================================

class PRMPHeteroLayer(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        # prediction MLPs  parent → child
        self.pred_p2r = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.pred_c2r = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        # parent update (self + agg residuals)
        self.upd_p = nn.Linear(h * 2, h)
        self.upd_c = nn.Linear(h * 2, h)
        # child update (self + parent messages)
        self.msg_p2r = nn.Linear(h, h)
        self.msg_c2r = nn.Linear(h, h)
        self.upd_r   = nn.Linear(h * 3, h)

    def forward(self, hr, hp, hc, g):
        nr   = g["n_reviews"]
        nprod = g["n_products"]
        ncust = g["n_customers"]
        p2r  = g["prod_to_rev_edge"]
        c2r  = g["cust_to_rev_edge"]

        # --- PRMP product → review ---
        pred_rv_p  = self.pred_p2r(hp[p2r[0]])          # predicted child
        resid_p    = hr[p2r[1]] - pred_rv_p             # residual
        agg_res_p  = scatter_mean(resid_p, p2r[0], nprod)
        new_p      = F.relu(self.upd_p(torch.cat([hp, agg_res_p], 1)))

        # --- PRMP customer → review ---
        pred_rv_c  = self.pred_c2r(hc[c2r[0]])
        resid_c    = hr[c2r[1]] - pred_rv_c
        agg_res_c  = scatter_mean(resid_c, c2r[0], ncust)
        new_c      = F.relu(self.upd_c(torch.cat([hc, agg_res_c], 1)))

        # --- standard child update ---
        agg_p = scatter_mean(self.msg_p2r(hp[p2r[0]]), p2r[1], nr)
        agg_c = scatter_mean(self.msg_c2r(hc[c2r[0]]), c2r[1], nr)
        new_r = F.relu(self.upd_r(torch.cat([hr, agg_p, agg_c], 1)))

        return new_r, new_p, new_c


class PRMPHeteroGNN(nn.Module):
    def __init__(self, cd=21, pd_=5, cud=5, h=128, nl=2):
        super().__init__()
        self.proj_r = nn.Linear(cd, h)
        self.proj_p = nn.Linear(pd_, h)
        self.proj_c = nn.Linear(cud, h)
        self.layers = nn.ModuleList([PRMPHeteroLayer(h) for _ in range(nl)])
        self.head = nn.Linear(h, 1)
        self.variant = "B_prmp"

    def forward(self, g, return_emb=False):
        hr = F.relu(self.proj_r(g["review_x"]))
        hp = F.relu(self.proj_p(g["product_x"]))
        hc = F.relu(self.proj_c(g["customer_x"]))
        for layer in self.layers:
            hr, hp, hc = layer(hr, hp, hc, g)
        out = self.head(hr).squeeze(-1)
        return (out, hr, hp, hc) if return_emb else out


# ===================================================================
# MODEL D – AuxMLP (Standard + auxiliary prediction loss, NO residual)
# ===================================================================

class AuxMLPHeteroGNN(nn.Module):
    def __init__(self, cd=21, pd_=5, cud=5, h=128, nl=2):
        super().__init__()
        self.proj_r = nn.Linear(cd, h)
        self.proj_p = nn.Linear(pd_, h)
        self.proj_c = nn.Linear(cud, h)
        self.layers  = nn.ModuleList([StandardHeteroLayer(h) for _ in range(nl)])
        self.aux_p   = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                                     nn.Linear(h, h))
                                       for _ in range(nl)])
        self.aux_c   = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                                     nn.Linear(h, h))
                                       for _ in range(nl)])
        self.head = nn.Linear(h, 1)
        self.variant = "D_aux_mlp"
        self._aux_loss = torch.tensor(0.0)

    def forward(self, g, return_emb=False):
        hr = F.relu(self.proj_r(g["review_x"]))
        hp = F.relu(self.proj_p(g["product_x"]))
        hc = F.relu(self.proj_c(g["customer_x"]))

        p2r = g["prod_to_rev_edge"]
        c2r = g["cust_to_rev_edge"]
        aux = torch.tensor(0.0, device=hr.device)

        for i, layer in enumerate(self.layers):
            # compute aux prediction loss (gradient flows, but NO subtraction)
            pred_r_p = self.aux_p[i](hp[p2r[0]])
            aux = aux + F.mse_loss(pred_r_p, hr[p2r[1]].detach())
            pred_r_c = self.aux_c[i](hc[c2r[0]])
            aux = aux + F.mse_loss(pred_r_c, hr[c2r[1]].detach())
            # standard message passing
            hr, hp, hc = layer(hr, hp, hc, g)

        self._aux_loss = aux
        out = self.head(hr).squeeze(-1)
        return (out, hr, hp, hc) if return_emb else out


# ===================================================================
# MODEL C – Wide parameter-matched control
# ===================================================================

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def make_wide_model(target_params: int, cd=21, pd_=5, cud=5, nl=2):
    lo, hi = 64, 512
    while lo < hi:
        mid = (lo + hi) // 2
        p = _count_params(StandardHeteroGNN(cd, pd_, cud, mid, nl))
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid
    m = StandardHeteroGNN(cd, pd_, cud, lo, nl)
    m.variant = "C_wide"
    return m, lo


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _to_device(graph_data: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in graph_data.items()}


def train_one_run(model: nn.Module, graph_data: dict,
                  seed: int, device: torch.device):
    set_seeds(seed)
    gd = _to_device(graph_data, device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    epoch_times: list[float] = []

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        pred = model(gd)
        loss = F.l1_loss(pred[gd["train_mask"]], gd["y"][gd["train_mask"]])
        if hasattr(model, "_aux_loss"):
            loss = loss + 0.1 * model._aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        epoch_times.append(time.time() - t0)

        # validate
        model.eval()
        with torch.no_grad():
            vp = model(gd)
            vm = F.l1_loss(vp[gd["val_mask"]], gd["y"][gd["val_mask"]]).item()
        if vm < best_val:
            best_val = vm
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if (epoch + 1) % 50 == 0:
            logger.debug(f"    ep={epoch+1} loss={loss.item():.4f} val={vm:.4f}")

    # test with best checkpoint
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out_tuple = model(gd, return_emb=True)
        pred_all, hr, hp, hc = out_tuple
        pred_np = pred_all.cpu().numpy()
        y_np    = gd["y"].cpu().numpy()
        tm      = gd["test_mask"].cpu().numpy()

    t_mae  = float(np.mean(np.abs(pred_np[tm] - y_np[tm])))
    t_rmse = float(np.sqrt(np.mean((pred_np[tm] - y_np[tm]) ** 2)))
    ss_res = float(np.sum((y_np[tm] - pred_np[tm]) ** 2))
    ss_tot = float(np.sum((y_np[tm] - y_np[tm].mean()) ** 2))
    t_r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    emb = {"review": hr.cpu().numpy(),
           "product": hp.cpu().numpy(),
           "customer": hc.cpu().numpy()}

    n_epochs = epoch + 1
    avg_et   = float(np.mean(epoch_times))

    del gd; model.cpu(); torch.cuda.empty_cache(); gc.collect()

    return {
        "test_mae": t_mae, "test_rmse": t_rmse, "test_r2": t_r2,
        "best_val_mae": best_val, "n_epochs": n_epochs,
        "avg_epoch_time": avg_et, "predictions": pred_np,
    }, emb


# ---------------------------------------------------------------------------
# Ridge R² (embedding-space cross-table predictability)
# ---------------------------------------------------------------------------

def compute_ridge_r2(emb: dict, graph_data: dict) -> dict:
    p2r = graph_data["prod_to_rev_edge"].numpy()
    c2r = graph_data["cust_to_rev_edge"].numpy()
    rng = np.random.RandomState(0)
    results = {}

    for name, edge, pkey, ckey in [
        ("product_to_review",  p2r, "product",  "review"),
        ("customer_to_review", c2r, "customer", "review"),
    ]:
        parent = emb[pkey][edge[0]]
        child  = emb[ckey][edge[1]]
        n = min(2000, len(parent))
        idx = rng.choice(len(parent), n, replace=False)
        ridge = Ridge(alpha=1.0)
        ridge.fit(parent[idx], child[idx])
        results[name] = float(ridge.score(parent[idx], child[idx]))
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    ps = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return 0.0 if ps < 1e-12 else float((a.mean() - b.mean()) / ps)


def compare_variants(ra, rb):
    ma = [r["test_mae"] for r in ra]
    mb = [r["test_mae"] for r in rb]
    if len(ma) < 2:
        md = float(np.mean(ma) - np.mean(mb))
        return {"mean_diff": round(md, 6), "cohens_d": 0.0,
                "p_value": 1.0, "ci_95": [round(md, 6), round(md, 6)],
                "mae_a_mean": round(float(np.mean(ma)), 6),
                "mae_b_mean": round(float(np.mean(mb)), 6)}
    t, p = stats.ttest_rel(ma, mb)
    d = cohens_d(ma, mb)
    diff = np.array(ma) - np.array(mb)
    md = float(diff.mean())
    se = float(stats.sem(diff)) if len(diff) > 1 else 0.0
    ci = [round(md - 1.96 * se, 6), round(md + 1.96 * se, 6)]
    return {"mean_diff": round(md, 6), "cohens_d": round(d, 4),
            "p_value": round(float(p), 6), "ci_95": ci,
            "mae_a_mean": round(float(np.mean(ma)), 6),
            "mae_b_mean": round(float(np.mean(mb)), 6)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main(max_examples: int | None = None):
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Full-Scale Amazon PRMP Validation")
    logger.info("=" * 60)

    # ---- load & encode ----
    result = load_and_encode_data(max_examples=max_examples)
    (X_child, prod_feats, cust_feats, ratings,
     prod_ids, cust_ids, prod_map, cust_map,
     timestamps, feature_names) = result

    n_reviews   = len(X_child)
    n_products  = len(prod_feats)
    n_customers = len(cust_feats)
    cd, pd_, cud = X_child.shape[1], prod_feats.shape[1], cust_feats.shape[1]
    logger.info(f"Data: {n_reviews} reviews, {n_products} products, "
                f"{n_customers} customers | dims child={cd} prod={pd_} cust={cud}")

    # ---- build graph ----
    graph_data = build_graph(X_child, prod_feats, cust_feats, ratings,
                             prod_ids, cust_ids, timestamps)

    # ---- param counts ----
    prmp_params = _count_params(PRMPHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS))
    std_params  = _count_params(StandardHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS))
    wide_tmp, wide_h = make_wide_model(prmp_params, cd, pd_, cud, NUM_LAYERS)
    wide_params = _count_params(wide_tmp); del wide_tmp
    aux_params  = _count_params(AuxMLPHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS))
    logger.info(f"Params: Std={std_params}, PRMP={prmp_params}, "
                f"Wide(h={wide_h})={wide_params}, Aux={aux_params}")

    # ---- factories ----
    factories = [
        ("A_standard", lambda: StandardHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS)),
        ("B_prmp",     lambda: PRMPHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS)),
        ("C_wide",     lambda: make_wide_model(prmp_params, cd, pd_, cud, NUM_LAYERS)[0]),
        ("D_aux_mlp",  lambda: AuxMLPHeteroGNN(cd, pd_, cud, HIDDEN_DIM, NUM_LAYERS)),
    ]

    all_results:  dict[str, list[dict]] = {}
    all_ridge:    dict[str, list[dict]] = {}
    all_preds:    dict[str, np.ndarray] = {}

    for vname, factory in factories:
        logger.info(f"\n{'=' * 40}  {vname}  {'=' * 40}")
        vr, vri, vp = [], [], []
        for seed in SEEDS:
            logger.info(f"  seed={seed} ...")
            t0 = time.time()
            model = factory()
            model.variant = vname
            metrics, emb = train_one_run(model, graph_data, seed, DEVICE)
            elapsed = time.time() - t0
            logger.info(f"  seed={seed}: MAE={metrics['test_mae']:.4f} "
                        f"RMSE={metrics['test_rmse']:.4f} R²={metrics['test_r2']:.4f} "
                        f"ep={metrics['n_epochs']} t={elapsed:.1f}s")
            ridge = compute_ridge_r2(emb, graph_data)
            logger.info(f"  Ridge R²: p→r={ridge['product_to_review']:.4f} "
                        f"c→r={ridge['customer_to_review']:.4f}")
            vr.append(metrics)
            vri.append(ridge)
            vp.append(metrics["predictions"])
            del model, emb; torch.cuda.empty_cache(); gc.collect()

        all_results[vname] = vr
        all_ridge[vname]   = vri
        all_preds[vname]   = np.mean(vp, axis=0)

        elapsed_total = time.time() - t_start
        logger.info(f"Total elapsed so far: {elapsed_total / 60:.1f} min")

    # ---- comparisons ----
    logger.info(f"\n{'=' * 40}  STATS  {'=' * 40}")
    comparisons = {}
    for tag, a, b in [("B_vs_A", "A_standard", "B_prmp"),
                       ("B_vs_C", "C_wide",     "B_prmp"),
                       ("B_vs_D", "D_aux_mlp",  "B_prmp")]:
        c = compare_variants(all_results[a], all_results[b])
        comparisons[tag] = c
        logger.info(f"  {tag}: diff={c['mean_diff']:.4f} d={c['cohens_d']:.4f} "
                    f"p={c['p_value']:.4f} CI={c['ci_95']}")

    d_ba = comparisons["B_vs_A"]["cohens_d"]
    if abs(d_ba) > 0.5:
        kf = f"PRMP d={d_ba:.3f} persists at full {n_reviews}-review scale"
    elif abs(d_ba) > 0.2:
        kf = f"PRMP d={d_ba:.3f} moderate at full {n_reviews}-review scale"
    else:
        kf = f"PRMP d={d_ba:.3f} vanishes at full {n_reviews}-review scale"
    logger.info(f"KEY FINDING: {kf}")

    # ---- build exp_gen_sol_out.json ----
    logger.info("Building output JSON ...")
    rev_prod = {v: k for k, v in prod_map.items()}
    rev_cust = {v: k for k, v in cust_map.items()}

    test_mask_np = graph_data["test_mask"].numpy()
    test_idx = np.where(test_mask_np)[0]

    examples = []
    for idx in test_idx:
        feat_dict = {fn: round(float(X_child[idx, j]), 3)
                     for j, fn in enumerate(feature_names)}
        ex = {
            "input":  json.dumps(feat_dict),
            "output": str(float(ratings[idx])),
            "predict_A_standard": str(round(float(all_preds["A_standard"][idx]), 3)),
            "predict_B_prmp":     str(round(float(all_preds["B_prmp"][idx]),     3)),
            "predict_C_wide":     str(round(float(all_preds["C_wide"][idx]),     3)),
            "predict_D_aux_mlp":  str(round(float(all_preds["D_aux_mlp"][idx]),  3)),
            "metadata_product_id":  rev_prod.get(int(prod_ids[idx]),
                                                  str(prod_ids[idx])),
            "metadata_customer_id": rev_cust.get(int(cust_ids[idx]),
                                                  str(cust_ids[idx])),
            "metadata_task_type": "regression",
            "metadata_split":     "test",
        }
        examples.append(ex)

    # variant summaries
    param_map = {"A_standard": std_params, "B_prmp": prmp_params,
                 "C_wide": wide_params, "D_aux_mlp": aux_params}
    variant_summary = {}
    for vn in ["A_standard", "B_prmp", "C_wide", "D_aux_mlp"]:
        vr = all_results[vn]
        variant_summary[vn] = {
            "mae":  [round(r["test_mae"],  4) for r in vr],
            "rmse": [round(r["test_rmse"], 4) for r in vr],
            "r2":   [round(r["test_r2"],   4) for r in vr],
            "mae_mean": round(float(np.mean([r["test_mae"]  for r in vr])), 4),
            "mae_std":  round(float(np.std([r["test_mae"]   for r in vr])), 4),
            "params": param_map[vn],
            "avg_epoch_time": round(float(np.mean([r["avg_epoch_time"] for r in vr])), 4),
            "avg_epochs":     round(float(np.mean([r["n_epochs"]       for r in vr])), 1),
        }

    ridge_summary: dict = {}
    for fk in ["product_to_review", "customer_to_review"]:
        ridge_summary[fk] = {}
        for vn in ["A_standard", "B_prmp", "C_wide", "D_aux_mlp"]:
            vals = [r[fk] for r in all_ridge[vn]]
            ridge_summary[fk][vn] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "values": [round(v, 4) for v in vals],
            }

    output = {
        "metadata": {
            "experiment": "full_scale_amazon_prmp_validation",
            "description": (
                f"Full-scale PRMP validation on Amazon Video Games "
                f"({n_reviews} reviews, 4 GNN variants x 5 seeds x "
                f"{MAX_EPOCHS} epochs max). Cohen's d effect size."
            ),
            "dataset": {
                "name": "amazon_video_games_full",
                "num_reviews":   n_reviews,
                "num_products":  n_products,
                "num_customers": n_customers,
                "fk_cardinality_product":  round(n_reviews / max(n_products, 1),  1),
                "fk_cardinality_customer": round(n_reviews / max(n_customers, 1), 1),
                "subsampled": n_reviews < 50000,
            },
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE, "max_epochs": MAX_EPOCHS,
                "grad_clip": GRAD_CLIP, "seeds": SEEDS,
            },
            "variants":          variant_summary,
            "comparisons":       comparisons,
            "embedding_ridge_r2": ridge_summary,
            "key_finding":       kf,
            "total_time_min":    round((time.time() - t_start) / 60, 1),
        },
        "datasets": [{
            "dataset":  "amazon_video_games_reviews",
            "examples": examples,
        }],
    }

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_path} ({size_mb:.1f} MB, {len(examples)} examples)")
    logger.info(f"Total time: {(time.time() - t_start)/60:.1f} min")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples (for testing)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (e.g. '42,123')")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override MAX_EPOCHS")
    args = parser.parse_args()

    if args.max_epochs:
        MAX_EPOCHS = args.max_epochs
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",")]

    main(max_examples=args.max_examples)
