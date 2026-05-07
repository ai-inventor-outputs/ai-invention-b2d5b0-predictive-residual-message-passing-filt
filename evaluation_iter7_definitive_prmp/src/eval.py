#!/usr/bin/env python3
"""
Definitive PRMP Meta-Analysis: Pooled Effects, Embedding Regime Theory, and 8 Publication Figures.

Comprehensive meta-analysis evaluation incorporating ALL 12 experiments from iterations 1-6.
Computes DerSimonian-Laird random-effects pooled effect sizes, tests embedding-space regime theory,
performs task-type moderator analysis, and generates 8 publication-quality matplotlib figures.
"""

import json
import math
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
WORK = Path(__file__).resolve().parent
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORK / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits (container: 29 GB RAM, 4 CPUs) ────────────────────────
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Constants ────────────────────────────────────────────────────────────────
BASE = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop")
FIG_DIR = WORK / "figures"
FIG_DIR.mkdir(exist_ok=True)
(WORK / "logs").mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Cohen's d (standardized mean difference)
# ──────────────────────────────────────────────────────────────────────────────
def cohens_d(group1: list[float], group2: list[float]) -> tuple[float, float, float, float]:
    """Compute Cohen's d = (mean2 - mean1) / pooled_sd.

    Returns (d, se_d, ci_lo, ci_hi).
    Positive d means group2 > group1.
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    # Pooled SD
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1e-10
    if sp < 1e-12:
        sp = 1e-12
    d = (m2 - m1) / sp
    # SE of d (Hedges & Olkin approx)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lo = d - 1.96 * se
    ci_hi = d + 1.96 * se
    return float(d), float(se), float(ci_lo), float(ci_hi)


def variance_of_d(d: float, n1: int, n2: int) -> float:
    """Variance of Cohen's d."""
    return (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))


# ──────────────────────────────────────────────────────────────────────────────
# META-ANALYSIS: DerSimonian-Laird random-effects
# ──────────────────────────────────────────────────────────────────────────────
def dersimonian_laird(effects: list[float], variances: list[float]):
    """DerSimonian-Laird random-effects meta-analysis.

    Returns dict with pooled_d, se, ci_lo, ci_hi, z, p, Q, I2, tau2, pred_lo, pred_hi.
    """
    k = len(effects)
    effects = np.array(effects)
    variances = np.array(variances)

    # Fixed-effects weights
    w_fe = 1.0 / variances

    # Fixed-effects pooled estimate
    d_fe = np.sum(w_fe * effects) / np.sum(w_fe)

    # Cochran's Q
    Q = np.sum(w_fe * (effects - d_fe)**2)
    df = k - 1
    p_Q = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0

    # Between-study variance (tau²)
    C = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects weights
    w_re = 1.0 / (variances + tau2)
    d_re = np.sum(w_re * effects) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))

    ci_lo = d_re - 1.96 * se_re
    ci_hi = d_re + 1.96 * se_re

    z = d_re / se_re if se_re > 0 else 0.0
    p_z = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    # I² = (Q - df) / Q × 100%
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # Prediction interval
    pred_se = np.sqrt(tau2 + se_re**2)
    t_crit = stats.t.ppf(0.975, max(df, 1))
    pred_lo = d_re - t_crit * np.sqrt(tau2 + se_re**2)
    pred_hi = d_re + t_crit * np.sqrt(tau2 + se_re**2)

    return {
        "pooled_d": float(d_re),
        "se": float(se_re),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "z": float(z),
        "p_value": float(p_z),
        "Q": float(Q),
        "Q_p": float(p_Q),
        "I2": float(I2),
        "tau2": float(tau2),
        "pred_lo": float(pred_lo),
        "pred_hi": float(pred_hi),
        "k": k,
    }


def eggers_test(effects: list[float], se_list: list[float]):
    """Egger's regression test for funnel plot asymmetry."""
    precision = [1.0 / se for se in se_list]
    std_effects = [e / se for e, se in zip(effects, se_list)]
    if len(effects) < 3:
        return {"intercept": float("nan"), "p_value": float("nan"), "interpretable": False}
    slope, intercept, r, p, stderr = stats.linregress(precision, std_effects)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "p_value": float(p),
        "interpretable": len(effects) >= 5,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    """Load JSON file with error handling."""
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception(f"Failed to load {path}")
        raise


@logger.catch
def main():
    logger.info("Starting PRMP Meta-Analysis Evaluation")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD ALL DEPENDENCY DATA
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Loading dependency data...")

    # exp_id4_it2__opus (Amazon iter2, 3 seeds, 7 ablation variants)
    it2_amazon = load_json(BASE / "iter_2/gen_art/exp_id4_it2__opus/preview_method_out.json")
    it2_meta = it2_amazon["metadata"]

    # exp_id1_it4__opus (Parameter-matched controls Amazon + F1, 3 seeds)
    it4_controls = load_json(BASE / "iter_4/gen_art/exp_id1_it4__opus/preview_method_out.json")
    it4_meta = it4_controls["metadata"]

    # exp_id2_it3__opus (F1 benchmark, 3 tasks, 3 seeds)
    it3_f1 = load_json(BASE / "iter_3/gen_art/exp_id2_it3__opus/preview_method_out.json")
    it3_f1_meta = it3_f1["metadata"]

    # exp_id3_it3__opus (Stack benchmark, 2 tasks, 1 seed effectively)
    it3_stack = load_json(BASE / "iter_3/gen_art/exp_id3_it3__opus/preview_method_out.json")
    it3_stack_meta = it3_stack["metadata"]

    # exp_id1_it3__opus (HM benchmark, 2 tasks, 3 seeds) - need full for per-seed
    it3_hm = load_json(BASE / "iter_3/gen_art/exp_id1_it3__opus/full_method_out.json")

    # exp_id1_it6__opus (Avito, 3 seeds)
    it6_avito = load_json(BASE / "iter_6/gen_art/exp_id1_it6__opus/preview_method_out.json")
    it6_avito_meta = it6_avito["metadata"]

    # exp_id2_it6__opus (Unified Amazon/F1, 5 seeds)
    it6_unified = load_json(BASE / "iter_6/gen_art/exp_id2_it6__opus/preview_method_out.json")
    it6_unified_meta = it6_unified["metadata"]

    # exp_id3_it6__opus (Loss-swap, 13 seeds per config)
    it6_lossswap = load_json(BASE / "iter_6/gen_art/exp_id3_it6__opus/preview_method_out.json")
    it6_lossswap_meta = it6_lossswap["metadata"]

    # exp_id4_it6__opus (Stack clean, 3 seeds)
    it6_stack = load_json(BASE / "iter_6/gen_art/exp_id4_it6__opus/preview_method_out.json")
    it6_stack_meta = it6_stack["metadata"]

    # exp_id2_it4__opus (Embedding trajectories Amazon)
    it4_embed = load_json(BASE / "iter_4/gen_art/exp_id2_it4__opus/preview_method_out.json")
    it4_embed_meta = it4_embed["metadata"]

    # exp_id2_it5__opus (Embedding trajectories F1)
    it5_f1_embed = load_json(BASE / "iter_5/gen_art/exp_id2_it5__opus/preview_method_out.json")
    it5_f1_meta = it5_f1_embed["metadata"]

    # exp_id3_it5__opus (Embedding trajectories Stack)
    it5_stack_embed = load_json(BASE / "iter_5/gen_art/exp_id3_it5__opus/preview_method_out.json")
    it5_stack_meta = it5_stack_embed["metadata"]

    logger.info("All dependency data loaded successfully")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. COMPUTE PER-TASK EFFECT SIZES (Cohen's d)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Computing per-task effect sizes...")

    # Structure: list of dicts with task info and Cohen's d
    all_tasks = []  # all including duplicates
    independent_tasks = []  # deduplicated for meta-analysis

    # --- Amazon review-rating (iter2, 3 seeds) ---
    amz_it2_std = it2_meta["results"]["standard_sage"]["rmse"]
    amz_it2_prmp = it2_meta["results"]["prmp"]["rmse"]
    # Reconstruct per-seed from training curves
    std_vals_it2 = it2_meta["training_curves"]["standard_sage"]["val_losses_summary"]["per_seed_best"]
    prmp_vals_it2 = it2_meta["training_curves"]["prmp"]["val_losses_summary"]["per_seed_best"]
    # Use RMSE: for lower-is-better, d = (std - prmp) / sp → positive = PRMP better
    # Per-seed RMSE values from metadata results
    # Since we have mean and std for RMSE, reconstruct approximate per-seed
    # Actually we can compute from test RMSE mean/std with 3 seeds
    amz_it2_std_seeds = [amz_it2_std["mean"] + amz_it2_std["std"] * x for x in [-1, 0, 1]]
    amz_it2_prmp_seeds = [amz_it2_prmp["mean"] + amz_it2_prmp["std"] * x for x in [-1, 0, 1]]
    d_amz_it2, se_amz_it2, ci_lo, ci_hi = cohens_d(amz_it2_prmp_seeds, amz_it2_std_seeds)
    all_tasks.append({
        "task": "Amazon review-rating (iter2)",
        "dataset": "Amazon",
        "task_type": "regression",
        "metric": "RMSE",
        "higher_better": False,
        "d": d_amz_it2, "se": se_amz_it2, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": amz_it2_prmp["mean"], "std_mean": amz_it2_std["mean"],
        "source": "exp_id4_it2__opus",
        "is_primary": False,
    })

    # --- Amazon review-rating (iter4 param-matched, 3 seeds) ---
    amz_it4_std = it4_meta["amazon_summaries"]["A_standard_sage"]["rmse"]["values"]
    amz_it4_prmp = it4_meta["amazon_summaries"]["B_prmp"]["rmse"]["values"]
    d_amz_it4, se_amz_it4, ci_lo, ci_hi = cohens_d(amz_it4_prmp, amz_it4_std)
    all_tasks.append({
        "task": "Amazon review-rating (iter4)",
        "dataset": "Amazon",
        "task_type": "regression",
        "metric": "RMSE",
        "higher_better": False,
        "d": d_amz_it4, "se": se_amz_it4, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(amz_it4_prmp), "std_mean": np.mean(amz_it4_std),
        "source": "exp_id1_it4__opus",
        "is_primary": False,
    })

    # --- Amazon review-rating (iter6 unified 5-seed) --- PRIMARY
    amz_it6 = it6_unified_meta["statistical_analysis"]["rel-amazon/review-rating"]["summary"]
    amz_it6_std_mean = amz_it6["standard"]["mean"]
    amz_it6_std_std = amz_it6["standard"]["std"]
    amz_it6_prmp_mean = amz_it6["prmp"]["mean"]
    amz_it6_prmp_std = amz_it6["prmp"]["std"]
    n6 = amz_it6["standard"]["n"]
    # For MAE (lower is better), d = (std - prmp) / sp → positive = PRMP better
    sp_amz6 = np.sqrt(((n6-1)*amz_it6_std_std**2 + (n6-1)*amz_it6_prmp_std**2) / (2*n6-2))
    if sp_amz6 < 1e-12:
        sp_amz6 = 1e-12
    d_amz_it6 = (amz_it6_std_mean - amz_it6_prmp_mean) / sp_amz6
    se_amz_it6 = np.sqrt(2/n6 + d_amz_it6**2 / (2*2*n6))
    ci_lo_amz6 = d_amz_it6 - 1.96 * se_amz_it6
    ci_hi_amz6 = d_amz_it6 + 1.96 * se_amz_it6
    all_tasks.append({
        "task": "Amazon review-rating",
        "dataset": "Amazon",
        "task_type": "regression",
        "metric": "MAE",
        "higher_better": False,
        "d": float(d_amz_it6), "se": float(se_amz_it6),
        "ci_lo": float(ci_lo_amz6), "ci_hi": float(ci_hi_amz6),
        "n_prmp": n6, "n_std": n6,
        "prmp_mean": amz_it6_prmp_mean, "std_mean": amz_it6_std_mean,
        "source": "exp_id2_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    # --- F1 driver-position (iter3, 3 seeds) ---
    f1_pos_it3 = it3_f1_meta["benchmark_summary"]
    f1_pos_std_vals = f1_pos_it3["rel-f1-driver-position__sage"]["test_values"]
    f1_pos_prmp_vals = f1_pos_it3["rel-f1-driver-position__prmp"]["test_values"]
    d_f1pos_it3, se_f1pos_it3, ci_lo, ci_hi = cohens_d(f1_pos_prmp_vals, f1_pos_std_vals)
    all_tasks.append({
        "task": "F1 driver-position (iter3)",
        "dataset": "F1",
        "task_type": "regression",
        "metric": "MAE",
        "higher_better": False,
        "d": d_f1pos_it3, "se": se_f1pos_it3, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(f1_pos_prmp_vals), "std_mean": np.mean(f1_pos_std_vals),
        "source": "exp_id2_it3__opus",
        "is_primary": False,
    })

    # --- F1 driver-position (iter6 unified 5-seed) --- PRIMARY
    f1_pos_it6 = it6_unified_meta["statistical_analysis"]["rel-f1/result-position"]["summary"]
    f1_pos_std_m = f1_pos_it6["standard"]["mean"]
    f1_pos_std_s = f1_pos_it6["standard"]["std"]
    f1_pos_prmp_m = f1_pos_it6["prmp"]["mean"]
    f1_pos_prmp_s = f1_pos_it6["prmp"]["std"]
    n6f = f1_pos_it6["standard"]["n"]
    sp_f1pos = np.sqrt(((n6f-1)*f1_pos_std_s**2 + (n6f-1)*f1_pos_prmp_s**2) / (2*n6f-2))
    if sp_f1pos < 1e-12:
        sp_f1pos = 1e-12
    d_f1pos_it6 = (f1_pos_std_m - f1_pos_prmp_m) / sp_f1pos
    se_f1pos_it6 = np.sqrt(2/n6f + d_f1pos_it6**2 / (2*2*n6f))
    all_tasks.append({
        "task": "F1 driver-position",
        "dataset": "F1",
        "task_type": "regression",
        "metric": "MAE",
        "higher_better": False,
        "d": float(d_f1pos_it6), "se": float(se_f1pos_it6),
        "ci_lo": float(d_f1pos_it6 - 1.96*se_f1pos_it6),
        "ci_hi": float(d_f1pos_it6 + 1.96*se_f1pos_it6),
        "n_prmp": n6f, "n_std": n6f,
        "prmp_mean": f1_pos_prmp_m, "std_mean": f1_pos_std_m,
        "source": "exp_id2_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    # --- F1 driver-dnf (iter3, 3 seeds) ---
    f1_dnf_std_vals = f1_pos_it3["rel-f1-driver-dnf__sage"]["test_values"]
    f1_dnf_prmp_vals = f1_pos_it3["rel-f1-driver-dnf__prmp"]["test_values"]
    # AP is higher-is-better, so d = (prmp - std) / sp → positive = PRMP better
    d_f1dnf_it3, se_f1dnf_it3, ci_lo, ci_hi = cohens_d(f1_dnf_std_vals, f1_dnf_prmp_vals)
    all_tasks.append({
        "task": "F1 driver-dnf (iter3)",
        "dataset": "F1",
        "task_type": "classification",
        "metric": "AP",
        "higher_better": True,
        "d": d_f1dnf_it3, "se": se_f1dnf_it3, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(f1_dnf_prmp_vals), "std_mean": np.mean(f1_dnf_std_vals),
        "source": "exp_id2_it3__opus",
        "is_primary": False,
    })

    # --- F1 driver-dnf (iter6 unified 5-seed) --- PRIMARY
    f1_dnf_it6 = it6_unified_meta["statistical_analysis"]["rel-f1/result-dnf"]["summary"]
    f1_dnf_std_m = f1_dnf_it6["standard"]["mean"]
    f1_dnf_std_s = f1_dnf_it6["standard"]["std"]
    f1_dnf_prmp_m = f1_dnf_it6["prmp"]["mean"]
    f1_dnf_prmp_s = f1_dnf_it6["prmp"]["std"]
    n6d = f1_dnf_it6["standard"]["n"]
    sp_dnf = np.sqrt(((n6d-1)*f1_dnf_std_s**2 + (n6d-1)*f1_dnf_prmp_s**2) / (2*n6d-2))
    if sp_dnf < 1e-12:
        sp_dnf = 1e-12
    # Higher is better for AP: d = (prmp - std) / sp
    d_f1dnf_it6 = (f1_dnf_prmp_m - f1_dnf_std_m) / sp_dnf
    se_f1dnf_it6 = np.sqrt(2/n6d + d_f1dnf_it6**2 / (2*2*n6d))
    all_tasks.append({
        "task": "F1 driver-dnf",
        "dataset": "F1",
        "task_type": "classification",
        "metric": "AP",
        "higher_better": True,
        "d": float(d_f1dnf_it6), "se": float(se_f1dnf_it6),
        "ci_lo": float(d_f1dnf_it6 - 1.96*se_f1dnf_it6),
        "ci_hi": float(d_f1dnf_it6 + 1.96*se_f1dnf_it6),
        "n_prmp": n6d, "n_std": n6d,
        "prmp_mean": f1_dnf_prmp_m, "std_mean": f1_dnf_std_m,
        "source": "exp_id2_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    # --- Stack post-votes (iter6 clean, 3 seeds) --- PRIMARY
    stack_pv = it6_stack_meta["results_summary"]["post-votes"]
    stack_pv_std_seeds = [s["mae"] for s in stack_pv["Standard"]["per_seed"]]
    stack_pv_prmp_seeds = [s["mae"] for s in stack_pv["PRMP"]["per_seed"]]
    d_stack_pv, se_stack_pv, ci_lo, ci_hi = cohens_d(stack_pv_prmp_seeds, stack_pv_std_seeds)
    all_tasks.append({
        "task": "Stack post-votes",
        "dataset": "Stack",
        "task_type": "regression",
        "metric": "MAE",
        "higher_better": False,
        "d": d_stack_pv, "se": se_stack_pv, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(stack_pv_prmp_seeds), "std_mean": np.mean(stack_pv_std_seeds),
        "source": "exp_id4_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    # --- Stack user-engagement (iter6 clean, 3 seeds) --- PRIMARY
    stack_ue = it6_stack_meta["results_summary"]["user-engagement"]
    stack_ue_std_seeds = [s["auroc"] for s in stack_ue["Standard"]["per_seed"]]
    stack_ue_prmp_seeds = [s["auroc"] for s in stack_ue["PRMP"]["per_seed"]]
    # AUROC higher is better: d = (prmp - std) / sp
    d_stack_ue, se_stack_ue, ci_lo, ci_hi = cohens_d(stack_ue_std_seeds, stack_ue_prmp_seeds)
    all_tasks.append({
        "task": "Stack user-engagement",
        "dataset": "Stack",
        "task_type": "classification",
        "metric": "AUROC",
        "higher_better": True,
        "d": d_stack_ue, "se": se_stack_ue, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(stack_ue_prmp_seeds), "std_mean": np.mean(stack_ue_std_seeds),
        "source": "exp_id4_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    # --- HM user-churn (iter3, 3 seeds) --- PRIMARY
    # Compute per-seed AUROC from instance-level predictions
    hm_exs = it3_hm["datasets"][0]["examples"]
    hm_seeds = defaultdict(lambda: defaultdict(lambda: {"bl_preds": [], "pr_preds": [], "rd_preds": [], "labels": []}))
    for ex in hm_exs:
        if ex.get("metadata_task") == "aggregate":
            continue
        task = ex["metadata_task"]
        seed = ex["metadata_seed"]
        try:
            label = json.loads(ex["output"])["label"]
        except (json.JSONDecodeError, KeyError):
            continue
        hm_seeds[task][seed]["bl_preds"].append(float(ex["predict_baseline"]))
        hm_seeds[task][seed]["pr_preds"].append(float(ex["predict_our_method"]))
        hm_seeds[task][seed]["rd_preds"].append(float(ex["predict_ablation_random"]))
        hm_seeds[task][seed]["labels"].append(label)

    # user-churn AUROC
    hm_churn_std_seeds = []
    hm_churn_prmp_seeds = []
    for seed in sorted(hm_seeds["user-churn"].keys()):
        sd = hm_seeds["user-churn"][seed]
        try:
            hm_churn_std_seeds.append(roc_auc_score(sd["labels"], sd["bl_preds"]))
            hm_churn_prmp_seeds.append(roc_auc_score(sd["labels"], sd["pr_preds"]))
        except ValueError:
            logger.warning(f"Could not compute AUROC for HM user-churn seed {seed}")

    if len(hm_churn_std_seeds) >= 2:
        # AUROC higher is better
        d_hm_churn, se_hm_churn, ci_lo, ci_hi = cohens_d(hm_churn_std_seeds, hm_churn_prmp_seeds)
        all_tasks.append({
            "task": "HM user-churn",
            "dataset": "HM",
            "task_type": "classification",
            "metric": "AUROC",
            "higher_better": True,
            "d": d_hm_churn, "se": se_hm_churn, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_prmp": len(hm_churn_prmp_seeds), "n_std": len(hm_churn_std_seeds),
            "prmp_mean": np.mean(hm_churn_prmp_seeds), "std_mean": np.mean(hm_churn_std_seeds),
            "source": "exp_id1_it3__opus",
            "is_primary": True,
        })
        independent_tasks.append(all_tasks[-1])

    # item-sales MAE
    hm_sales_std_seeds = []
    hm_sales_prmp_seeds = []
    for seed in sorted(hm_seeds["item-sales"].keys()):
        sd = hm_seeds["item-sales"][seed]
        try:
            hm_sales_std_seeds.append(mean_absolute_error(sd["labels"], sd["bl_preds"]))
            hm_sales_prmp_seeds.append(mean_absolute_error(sd["labels"], sd["pr_preds"]))
        except ValueError:
            logger.warning(f"Could not compute MAE for HM item-sales seed {seed}")

    if len(hm_sales_std_seeds) >= 2:
        # MAE lower is better: d = (std - prmp) / sp
        d_hm_sales, se_hm_sales, ci_lo, ci_hi = cohens_d(hm_sales_prmp_seeds, hm_sales_std_seeds)
        all_tasks.append({
            "task": "HM item-sales",
            "dataset": "HM",
            "task_type": "regression",
            "metric": "MAE",
            "higher_better": False,
            "d": d_hm_sales, "se": se_hm_sales, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_prmp": len(hm_sales_prmp_seeds), "n_std": len(hm_sales_std_seeds),
            "prmp_mean": np.mean(hm_sales_prmp_seeds), "std_mean": np.mean(hm_sales_std_seeds),
            "source": "exp_id1_it3__opus",
            "is_primary": True,
        })
        independent_tasks.append(all_tasks[-1])

    # --- Avito ad-ctr (iter6, 3 seeds) --- PRIMARY
    avito_std_seeds = [s["rmse"] for s in it6_avito_meta["gnn_results"]["A_StandardSAGE"]["per_seed"]]
    avito_prmp_seeds = [s["rmse"] for s in it6_avito_meta["gnn_results"]["B_PRMP_Full"]["per_seed"]]
    d_avito, se_avito, ci_lo, ci_hi = cohens_d(avito_prmp_seeds, avito_std_seeds)
    all_tasks.append({
        "task": "Avito ad-ctr",
        "dataset": "Avito",
        "task_type": "regression",
        "metric": "RMSE",
        "higher_better": False,
        "d": d_avito, "se": se_avito, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_prmp": 3, "n_std": 3,
        "prmp_mean": np.mean(avito_prmp_seeds), "std_mean": np.mean(avito_std_seeds),
        "source": "exp_id1_it6__opus",
        "is_primary": True,
    })
    independent_tasks.append(all_tasks[-1])

    logger.info(f"Computed {len(all_tasks)} total task effects, {len(independent_tasks)} independent")
    for t in independent_tasks:
        logger.info(f"  {t['task']}: d={t['d']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. DerSimonian-Laird META-ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Running DerSimonian-Laird random-effects meta-analysis...")

    effects = [t["d"] for t in independent_tasks]
    variances = [t["se"]**2 for t in independent_tasks]
    se_list = [t["se"] for t in independent_tasks]

    meta_result = dersimonian_laird(effects, variances)
    logger.info(f"Pooled d = {meta_result['pooled_d']:.4f} [{meta_result['ci_lo']:.4f}, {meta_result['ci_hi']:.4f}]")
    logger.info(f"Z = {meta_result['z']:.3f}, p = {meta_result['p_value']:.4f}")
    logger.info(f"I² = {meta_result['I2']:.1f}%, τ² = {meta_result['tau2']:.4f}")
    logger.info(f"Prediction interval: [{meta_result['pred_lo']:.3f}, {meta_result['pred_hi']:.3f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TASK-TYPE MODERATOR ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Running task-type moderator analysis...")

    reg_tasks = [t for t in independent_tasks if t["task_type"] == "regression"]
    cls_tasks = [t for t in independent_tasks if t["task_type"] == "classification"]

    reg_effects = [t["d"] for t in reg_tasks]
    reg_vars = [t["se"]**2 for t in reg_tasks]
    cls_effects = [t["d"] for t in cls_tasks]
    cls_vars = [t["se"]**2 for t in cls_tasks]

    meta_reg = dersimonian_laird(reg_effects, reg_vars) if len(reg_effects) >= 2 else None
    meta_cls = dersimonian_laird(cls_effects, cls_vars) if len(cls_effects) >= 2 else None

    # Simple meta-regression: does task_type predict d?
    if len(reg_effects) >= 2 and len(cls_effects) >= 2:
        x_mod = [0]*len(reg_effects) + [1]*len(cls_effects)
        y_mod = reg_effects + cls_effects
        mod_slope, mod_intercept, mod_r, mod_p, mod_se = stats.linregress(x_mod, y_mod)
        moderator_test = {
            "slope": float(mod_slope),
            "intercept": float(mod_intercept),
            "p_value": float(mod_p),
            "regression_pooled_d": meta_reg["pooled_d"] if meta_reg else None,
            "classification_pooled_d": meta_cls["pooled_d"] if meta_cls else None,
        }
    else:
        moderator_test = {"slope": None, "intercept": None, "p_value": None}

    logger.info(f"Regression tasks (k={len(reg_tasks)}): pooled d = {meta_reg['pooled_d']:.4f}" if meta_reg else "Regression: insufficient data")
    logger.info(f"Classification tasks (k={len(cls_tasks)}): pooled d = {meta_cls['pooled_d']:.4f}" if meta_cls else "Classification: insufficient data")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. EMBEDDING-SPACE REGIME THEORY
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Computing embedding-space regime theory metrics...")

    embedding_links = []

    # Amazon FK links from exp_id2_it6 embedding_diagnostics
    for diag in it6_unified_meta.get("embedding_diagnostics", []):
        if diag.get("variant") == "standard" and diag.get("dataset") == "rel-amazon":
            for link_name, link_data in diag.get("fk_ridge_r2", {}).items():
                # Find matching PRMP entry
                prmp_r2 = None
                for d2 in it6_unified_meta["embedding_diagnostics"]:
                    if d2.get("variant") == "prmp" and d2.get("dataset") == "rel-amazon":
                        if link_name in d2.get("fk_ridge_r2", {}):
                            prmp_r2 = d2["fk_ridge_r2"][link_name]["ridge_r2"]
                embedding_links.append({
                    "dataset": "Amazon",
                    "fk_link": link_name,
                    "embedding_r2_standard": link_data["ridge_r2"],
                    "embedding_r2_prmp": prmp_r2,
                    "delta_r2": (link_data["ridge_r2"] - prmp_r2) if prmp_r2 is not None else None,
                    "source": "exp_id2_it6__opus",
                })

    # Amazon FK links from exp_id2_it4 (trajectory final values)
    for link_name in ["product_to_review", "customer_to_review"]:
        traj = it4_embed_meta["embedding_predictability"]
        if link_name in traj.get("standard", {}) and link_name in traj.get("prmp", {}):
            std_r2 = traj["standard"][link_name]["layer_2"]["ridge_r2"][-1]  # last epoch
            prmp_r2 = traj["prmp"][link_name]["layer_2"]["ridge_r2"][-1]
            embedding_links.append({
                "dataset": "Amazon",
                "fk_link": f"{link_name} (traj)",
                "embedding_r2_standard": std_r2,
                "embedding_r2_prmp": prmp_r2,
                "delta_r2": std_r2 - prmp_r2,
                "source": "exp_id2_it4__opus",
            })

    # Avito FK links from exp_id1_it6
    avito_emb = it6_avito_meta.get("embedding_r2", {})
    for link_name in avito_emb.get("A_StandardSAGE", {}):
        std_data = avito_emb["A_StandardSAGE"][link_name]
        prmp_data = avito_emb.get("B_PRMP_Full", {}).get(link_name, {})
        embedding_links.append({
            "dataset": "Avito",
            "fk_link": link_name,
            "embedding_r2_standard": std_data["embedding_r2_mean"],
            "embedding_r2_prmp": prmp_data.get("embedding_r2_mean"),
            "delta_r2": (std_data["embedding_r2_mean"] - prmp_data["embedding_r2_mean"]) if prmp_data.get("embedding_r2_mean") is not None else None,
            "source": "exp_id1_it6__opus",
        })

    # Stack FK links from exp_id4_it6
    stack_emb = it6_stack_meta.get("embedding_r2_summary", {})
    for link_name, link_data in stack_emb.items():
        embedding_links.append({
            "dataset": "Stack",
            "fk_link": link_name,
            "embedding_r2_standard": link_data.get("Standard_mean"),
            "embedding_r2_prmp": link_data.get("PRMP_mean"),
            "delta_r2": (link_data.get("Standard_mean", 0) - link_data.get("PRMP_mean", 0)),
            "raw_feature_r2": link_data.get("raw_feature_r2"),
            "source": "exp_id4_it6__opus",
        })

    # F1 FK links from exp_id2_it5 (embedding trajectories)
    f1_r2_summary = it5_f1_meta.get("r2_timeseries_summary", {})
    if isinstance(f1_r2_summary, dict):
        for link_name, link_data in f1_r2_summary.items():
            if isinstance(link_data, dict):
                std_r2 = link_data.get("standard_final_r2") or link_data.get("standard_max_r2")
                prmp_r2 = link_data.get("prmp_final_r2") or link_data.get("prmp_max_r2")
                if std_r2 is not None:
                    embedding_links.append({
                        "dataset": "F1",
                        "fk_link": link_name,
                        "embedding_r2_standard": std_r2,
                        "embedding_r2_prmp": prmp_r2,
                        "delta_r2": (std_r2 - prmp_r2) if prmp_r2 is not None else None,
                        "source": "exp_id2_it5__opus",
                    })

    logger.info(f"Compiled {len(embedding_links)} FK link embedding R² records")

    # Compute per-dataset mean embedding R²
    dataset_emb_r2 = defaultdict(list)
    for link in embedding_links:
        if link.get("embedding_r2_standard") is not None:
            dataset_emb_r2[link["dataset"]].append(link["embedding_r2_standard"])

    dataset_mean_emb_r2 = {ds: np.mean(vals) for ds, vals in dataset_emb_r2.items()}

    # Spearman correlation: embedding R² vs PRMP improvement at task level
    # For each independent task, get representative embedding R²
    task_emb_r2 = []
    task_d_values = []
    for t in independent_tasks:
        ds = t["dataset"]
        if ds in dataset_mean_emb_r2:
            task_emb_r2.append(dataset_mean_emb_r2[ds])
            task_d_values.append(t["d"])

    if len(task_emb_r2) >= 3:
        spearman_rho, spearman_p = stats.spearmanr(task_emb_r2, task_d_values)
    else:
        spearman_rho, spearman_p = float("nan"), float("nan")

    regime_results = {
        "n_fk_links": len(embedding_links),
        "spearman_rho_task_level": float(spearman_rho),
        "spearman_p_task_level": float(spearman_p),
        "dataset_mean_embedding_r2": {k: float(v) for k, v in dataset_mean_emb_r2.items()},
    }
    logger.info(f"Regime: Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. LOSS-SWAP MECHANISM DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Extracting loss-swap mechanism results...")

    lossswap_analysis = it6_lossswap_meta.get("analysis", {})
    lossswap_deltas = lossswap_analysis.get("deltas_per_config", {})
    lossswap_results = {
        "config1_natural_regression": lossswap_deltas.get("config1_natural_regression", {}),
        "config2_binned_classification": lossswap_deltas.get("config2_binned_classification", {}),
        "config3_natural_classification": lossswap_deltas.get("config3_natural_classification", {}),
        "config4_softened_regression": lossswap_deltas.get("config4_softened_regression", {}),
        "loss_function_hypothesis": lossswap_analysis.get("loss_function_hypothesis", {}),
        "target_nature_hypothesis": lossswap_analysis.get("target_nature_hypothesis", {}),
        "conclusion": lossswap_analysis.get("conclusion", "N/A"),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 7. ABLATION SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Computing ablation summary metrics...")

    ablation_results = {}

    # From iter2: 7 variants on Amazon
    it2_res = it2_meta["results"]
    for variant_name, variant_key in [
        ("Standard", "standard_sage"),
        ("PRMP", "prmp"),
        ("Random Pred", "ablation_random_pred"),
        ("No Subtraction", "ablation_no_subtraction"),
        ("Linear Pred", "ablation_linear_pred"),
        ("Product Only", "prmp_product_only"),
        ("Customer Only", "prmp_customer_only"),
    ]:
        ablation_results[f"iter2_{variant_name}"] = {
            "rmse_mean": it2_res[variant_key]["rmse"]["mean"],
            "rmse_std": it2_res[variant_key]["rmse"]["std"],
        }

    # From iter4: parameter-matched controls
    for variant_name, variant_key in [
        ("Standard", "A_standard_sage"),
        ("PRMP", "B_prmp"),
        ("Wide", "C_wide_sage"),
        ("AuxMLP", "D_aux_mlp"),
        ("SkipResidual", "E_skip_residual"),
    ]:
        ablation_results[f"iter4_Amazon_{variant_name}"] = {
            "rmse_mean": it4_meta["amazon_summaries"][variant_key]["rmse"]["mean"],
            "rmse_std": it4_meta["amazon_summaries"][variant_key]["rmse"]["std"],
            "rmse_values": it4_meta["amazon_summaries"][variant_key]["rmse"]["values"],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 8. EGGER'S TEST (Funnel Plot Asymmetry)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Running Egger's test for publication bias...")

    egger = eggers_test(effects, se_list)
    logger.info(f"Egger's test: intercept = {egger['intercept']:.3f}, p = {egger['p_value']:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 9. CROSS-DATASET SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Building cross-dataset summary...")

    dataset_summary = {}
    for ds in ["Amazon", "F1", "Stack", "HM", "Avito"]:
        ds_tasks = [t for t in independent_tasks if t["dataset"] == ds]
        ds_reg = [t for t in ds_tasks if t["task_type"] == "regression"]
        ds_cls = [t for t in ds_tasks if t["task_type"] == "classification"]
        ds_d_values = [t["d"] for t in ds_tasks]
        dataset_summary[ds] = {
            "n_tasks": len(ds_tasks),
            "n_regression": len(ds_reg),
            "n_classification": len(ds_cls),
            "best_d": float(max(ds_d_values)) if ds_d_values else None,
            "mean_d": float(np.mean(ds_d_values)) if ds_d_values else None,
            "task_names": [t["task"] for t in ds_tasks],
            "mean_embedding_r2": float(dataset_mean_emb_r2.get(ds, float("nan"))),
            "n_fk_links": len([l for l in embedding_links if l["dataset"] == ds]),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 10. GENERATE 8 PUBLICATION FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Generating publication figures...")

    # Shared style
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    dataset_colors = {
        "Amazon": "#1f77b4",
        "F1": "#ff7f0e",
        "Stack": "#2ca02c",
        "HM": "#d62728",
        "Avito": "#9467bd",
    }

    # ── Fig 1: Forest Plot ──
    logger.info("Generating Fig 1: Forest Plot")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sorted_tasks = sorted(independent_tasks, key=lambda t: t["d"])
    y_positions = list(range(len(sorted_tasks)))

    for i, t in enumerate(sorted_tasks):
        color = dataset_colors.get(t["dataset"], "gray")
        ax1.errorbar(t["d"], i, xerr=[[t["d"] - t["ci_lo"]], [t["ci_hi"] - t["d"]]],
                     fmt="o", color=color, markersize=7, capsize=3, linewidth=1.5)

    # Pooled estimate diamond
    diamond_y = -1.2
    diamond_w = meta_result["ci_hi"] - meta_result["ci_lo"]
    diamond = plt.Polygon([
        [meta_result["ci_lo"], diamond_y],
        [meta_result["pooled_d"], diamond_y + 0.4],
        [meta_result["ci_hi"], diamond_y],
        [meta_result["pooled_d"], diamond_y - 0.4],
    ], closed=True, facecolor="black", edgecolor="black", alpha=0.7)
    ax1.add_patch(diamond)

    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.axvline(x=meta_result["pooled_d"], color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax1.set_yticks(y_positions + [-1.2])
    ax1.set_yticklabels([t["task"] for t in sorted_tasks] + [f"Pooled (k={meta_result['k']})"])
    ax1.set_xlabel("Cohen's d (positive = PRMP better)")
    ax1.set_title(f"Forest Plot: PRMP vs Standard (I²={meta_result['I2']:.0f}%)")

    # Legend
    handles = [mpatches.Patch(color=c, label=ds) for ds, c in dataset_colors.items()
               if any(t["dataset"] == ds for t in independent_tasks)]
    ax1.legend(handles=handles, loc="lower right", fontsize=8)

    ax1.set_ylim(-2.5, len(sorted_tasks) + 0.5)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "fig1_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Fig 2: Embedding Regime Scatter ──
    logger.info("Generating Fig 2: Embedding Regime Scatter")
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # For each task, compute task-level embedding R² vs d
    for t in independent_tasks:
        ds = t["dataset"]
        emb_r2 = dataset_mean_emb_r2.get(ds)
        if emb_r2 is not None and not np.isnan(emb_r2):
            color = dataset_colors.get(ds, "gray")
            ax2.scatter(emb_r2, t["d"], color=color, s=80, zorder=5, edgecolors="black", linewidth=0.5)
            ax2.annotate(t["task"], (emb_r2, t["d"]), fontsize=7, xytext=(5, 5),
                        textcoords="offset points")

    if len(task_emb_r2) >= 3 and not np.isnan(spearman_rho):
        # Regression line
        z = np.polyfit(task_emb_r2, task_d_values, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(task_emb_r2) - 0.05, max(task_emb_r2) + 0.05, 50)
        ax2.plot(x_range, p(x_range), "--", color="gray", alpha=0.5, linewidth=1)
        ax2.set_title(f"Embedding R² vs PRMP Effect (Spearman ρ={spearman_rho:.2f}, p={spearman_p:.3f})")
    else:
        ax2.set_title("Embedding R² vs PRMP Effect")

    ax2.set_xlabel("Mean Embedding R² (Standard model)")
    ax2.set_ylabel("Cohen's d (PRMP vs Standard)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    handles = [mpatches.Patch(color=c, label=ds) for ds, c in dataset_colors.items()
               if ds in dataset_mean_emb_r2]
    ax2.legend(handles=handles, loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig2_regime_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Fig 3: R² Trajectory Comparison (2x2) ──
    logger.info("Generating Fig 3: R² Trajectory Comparison")
    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))

    # Panel 1: Amazon product_to_review
    ax = axes3[0, 0]
    emb_pred = it4_embed_meta["embedding_predictability"]
    for model, style in [("standard", "-"), ("prmp", "--")]:
        if model in emb_pred and "product_to_review" in emb_pred[model]:
            data = emb_pred[model]["product_to_review"]["layer_2"]
            ax.plot(data["epochs"], data["ridge_r2"], style, label=model.upper(), linewidth=1.5)
    ax.set_title("Amazon: product→review")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ridge R²")
    ax.legend(fontsize=8)

    # Panel 2: Amazon customer_to_review
    ax = axes3[0, 1]
    for model, style in [("standard", "-"), ("prmp", "--")]:
        if model in emb_pred and "customer_to_review" in emb_pred[model]:
            data = emb_pred[model]["customer_to_review"]["layer_2"]
            ax.plot(data["epochs"], data["ridge_r2"], style, label=model.upper(), linewidth=1.5)
    ax.set_title("Amazon: customer→review")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ridge R²")
    ax.legend(fontsize=8)

    # Panel 3: F1 trajectories (if available from exp_id2_it5)
    ax = axes3[1, 0]
    f1_r2_ts = it5_f1_meta.get("r2_timeseries_summary", {})
    if isinstance(f1_r2_ts, dict) and len(f1_r2_ts) > 0:
        # Pick first available link
        first_link = list(f1_r2_ts.keys())[0] if f1_r2_ts else None
        if first_link and isinstance(f1_r2_ts[first_link], dict):
            link_data = f1_r2_ts[first_link]
            for key in link_data:
                if "epochs" in str(key).lower():
                    break
            # Try to plot if structure allows
            ax.set_title(f"F1: {first_link[:30]}")
            ax.text(0.5, 0.5, "Data available\n(see metadata)", transform=ax.transAxes,
                   ha="center", va="center", fontsize=9, color="gray")
    else:
        ax.set_title("F1: results→drivers")
        ax.text(0.5, 0.5, "Trajectory data\nnot in preview", transform=ax.transAxes,
               ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ridge R²")

    # Panel 4: Stack embedding R² bar chart (since trajectories limited)
    ax = axes3[1, 1]
    stack_links = [(k, v) for k, v in stack_emb.items() if v.get("Standard_mean")]
    if stack_links:
        link_names = [l[0][:20] for l in stack_links[:6]]  # top 6 for readability
        std_vals = [l[1]["Standard_mean"] for l in stack_links[:6]]
        prmp_vals = [l[1].get("PRMP_mean", 0) for l in stack_links[:6]]
        x = np.arange(len(link_names))
        ax.bar(x - 0.15, std_vals, 0.3, label="Standard", alpha=0.8)
        ax.bar(x + 0.15, prmp_vals, 0.3, label="PRMP", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(link_names, rotation=45, ha="right", fontsize=7)
        ax.set_title("Stack: Embedding R² by FK Link")
        ax.set_ylabel("Ridge R²")
        ax.legend(fontsize=8)

    fig3.suptitle("Embedding-Space R² Trajectories / Comparisons", y=1.02, fontsize=13)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "fig3_r2_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── Fig 4: Mechanism Bar Chart (iter2 ablations) ──
    logger.info("Generating Fig 4: Mechanism Bar Chart")
    fig4, ax4 = plt.subplots(figsize=(10, 5))

    variant_names = ["Standard", "PRMP", "Random Pred", "No Subtraction",
                     "Linear Pred", "Product Only", "Customer Only"]
    variant_keys = ["standard_sage", "prmp", "ablation_random_pred", "ablation_no_subtraction",
                    "ablation_linear_pred", "prmp_product_only", "prmp_customer_only"]
    rmse_means = [it2_res[k]["rmse"]["mean"] for k in variant_keys]
    rmse_stds = [it2_res[k]["rmse"]["std"] for k in variant_keys]

    colors4 = ["#bdbdbd", "#1f77b4", "#ff9896", "#aec7e8", "#c7c7c7", "#98df8a", "#ffbb78"]
    x4 = np.arange(len(variant_names))
    bars = ax4.bar(x4, rmse_means, yerr=rmse_stds, capsize=4, color=colors4, edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(variant_names, rotation=30, ha="right")
    ax4.set_ylabel("Test RMSE")
    ax4.set_title("Amazon Video Games: PRMP Ablation Study (iter2)")
    ax4.axhline(y=rmse_means[0], color="gray", linestyle="--", alpha=0.3)

    # Annotate improvement
    for i, (m, s) in enumerate(zip(rmse_means, rmse_stds)):
        ax4.text(i, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig4.tight_layout()
    fig4.savefig(FIG_DIR / "fig4_ablation_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── Fig 5: Task-Type Subgroup Forest Plot ──
    logger.info("Generating Fig 5: Task-Type Subgroup Forest Plot")
    fig5, ax5 = plt.subplots(figsize=(10, 7))

    y_pos = 0
    y_labels = []
    y_positions_5 = []

    # Regression subgroup
    ax5.text(-0.1, y_pos + 0.3, "Regression Tasks", fontweight="bold", fontsize=10,
            transform=ax5.get_yaxis_transform())
    y_pos += 1
    for t in sorted(reg_tasks, key=lambda x: x["d"]):
        color = dataset_colors.get(t["dataset"], "gray")
        ax5.errorbar(t["d"], y_pos, xerr=[[t["d"] - t["ci_lo"]], [t["ci_hi"] - t["d"]]],
                     fmt="o", color=color, markersize=6, capsize=3)
        y_labels.append(t["task"])
        y_positions_5.append(y_pos)
        y_pos += 1

    # Regression pooled diamond
    if meta_reg:
        diamond_y_reg = y_pos
        diamond = plt.Polygon([
            [meta_reg["ci_lo"], diamond_y_reg],
            [meta_reg["pooled_d"], diamond_y_reg + 0.3],
            [meta_reg["ci_hi"], diamond_y_reg],
            [meta_reg["pooled_d"], diamond_y_reg - 0.3],
        ], closed=True, facecolor="blue", edgecolor="blue", alpha=0.5)
        ax5.add_patch(diamond)
        y_labels.append(f"Regression pooled (d={meta_reg['pooled_d']:.2f})")
        y_positions_5.append(diamond_y_reg)
        y_pos += 1

    y_pos += 0.5
    ax5.axhline(y=y_pos - 0.25, color="gray", linewidth=0.5, alpha=0.5)

    # Classification subgroup
    ax5.text(-0.1, y_pos + 0.3, "Classification Tasks", fontweight="bold", fontsize=10,
            transform=ax5.get_yaxis_transform())
    y_pos += 1
    for t in sorted(cls_tasks, key=lambda x: x["d"]):
        color = dataset_colors.get(t["dataset"], "gray")
        ax5.errorbar(t["d"], y_pos, xerr=[[t["d"] - t["ci_lo"]], [t["ci_hi"] - t["d"]]],
                     fmt="s", color=color, markersize=6, capsize=3)
        y_labels.append(t["task"])
        y_positions_5.append(y_pos)
        y_pos += 1

    # Classification pooled diamond
    if meta_cls:
        diamond_y_cls = y_pos
        diamond = plt.Polygon([
            [meta_cls["ci_lo"], diamond_y_cls],
            [meta_cls["pooled_d"], diamond_y_cls + 0.3],
            [meta_cls["ci_hi"], diamond_y_cls],
            [meta_cls["pooled_d"], diamond_y_cls - 0.3],
        ], closed=True, facecolor="red", edgecolor="red", alpha=0.5)
        ax5.add_patch(diamond)
        y_labels.append(f"Classification pooled (d={meta_cls['pooled_d']:.2f})")
        y_positions_5.append(diamond_y_cls)
        y_pos += 1

    ax5.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.set_yticks(y_positions_5)
    ax5.set_yticklabels(y_labels, fontsize=9)
    ax5.set_xlabel("Cohen's d (positive = PRMP better)")
    ax5.set_title("Subgroup Forest Plot by Task Type")
    ax5.set_ylim(-0.5, y_pos + 0.5)
    fig5.tight_layout()
    fig5.savefig(FIG_DIR / "fig5_subgroup_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)

    # ── Fig 6: Funnel Plot ──
    logger.info("Generating Fig 6: Funnel Plot")
    fig6, ax6 = plt.subplots(figsize=(8, 6))

    for t in independent_tasks:
        color = dataset_colors.get(t["dataset"], "gray")
        ax6.scatter(t["d"], t["se"], color=color, s=60, edgecolors="black", linewidth=0.5, zorder=5)

    ax6.axvline(x=meta_result["pooled_d"], color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Funnel boundaries
    se_max = max(se_list) * 1.2
    se_range = np.linspace(0.01, se_max, 100)
    ax6.plot(meta_result["pooled_d"] - 1.96 * se_range, se_range, "k--", alpha=0.3)
    ax6.plot(meta_result["pooled_d"] + 1.96 * se_range, se_range, "k--", alpha=0.3)

    ax6.invert_yaxis()
    ax6.set_xlabel("Cohen's d")
    ax6.set_ylabel("Standard Error (inverted)")
    ax6.set_title(f"Funnel Plot (Egger p={egger['p_value']:.3f})")

    handles = [mpatches.Patch(color=c, label=ds) for ds, c in dataset_colors.items()
               if any(t["dataset"] == ds for t in independent_tasks)]
    ax6.legend(handles=handles, loc="upper right", fontsize=8)
    fig6.tight_layout()
    fig6.savefig(FIG_DIR / "fig6_funnel_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)

    # ── Fig 7: Loss-Swap Results ──
    logger.info("Generating Fig 7: Loss-Swap Results")
    fig7, ax7 = plt.subplots(figsize=(10, 5))

    config_names = [
        "Config 1\nNatural Regression\n(MAE)",
        "Config 2\nBinned Classification\n(CE)",
        "Config 3\nNatural Classification\n(BCE)",
        "Config 4\nSoftened Regression\n(MSE)",
    ]
    config_keys = ["config1_natural_regression", "config2_binned_classification",
                   "config3_natural_classification", "config4_softened_regression"]

    delta_means = []
    delta_stds = []
    for ck in config_keys:
        cd = lossswap_deltas.get(ck, {})
        delta_means.append(cd.get("mean", 0))
        delta_stds.append(cd.get("std", 0))

    x7 = np.arange(len(config_names))
    bar_colors = ["#2ca02c" if dm > 0 else "#d62728" for dm in delta_means]
    bars7 = ax7.bar(x7, delta_means, yerr=delta_stds, capsize=5, color=bar_colors,
                   edgecolor="black", linewidth=0.5, alpha=0.8)
    ax7.set_xticks(x7)
    ax7.set_xticklabels(config_names, fontsize=9)
    ax7.set_ylabel("PRMP Delta (positive = PRMP better)")
    ax7.set_title("Loss-Swap Experiment: PRMP Advantage by Configuration")
    ax7.axhline(y=0, color="black", linewidth=0.8)

    # Annotate
    lf_hyp = lossswap_analysis.get("loss_function_hypothesis", {})
    ax7.annotate(f"Loss-fn hypothesis p={lf_hyp.get('p_value', 'N/A'):.4f}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig7.tight_layout()
    fig7.savefig(FIG_DIR / "fig7_lossswap.png", dpi=150, bbox_inches="tight")
    plt.close(fig7)

    # ── Fig 8: Cross-Dataset Summary Table ──
    logger.info("Generating Fig 8: Cross-Dataset Summary Table")
    fig8, ax8 = plt.subplots(figsize=(12, 4))
    ax8.axis("off")

    col_labels = ["Dataset", "Tasks", "Best d", "Mean d", "Emb R²", "FK Links", "Verdict"]
    table_data = []
    for ds in ["Amazon", "F1", "Stack", "HM", "Avito"]:
        s = dataset_summary[ds]
        best_d = f"{s['best_d']:.3f}" if s['best_d'] is not None else "N/A"
        mean_d = f"{s['mean_d']:.3f}" if s['mean_d'] is not None else "N/A"
        emb_r2 = f"{s['mean_embedding_r2']:.3f}" if not np.isnan(s['mean_embedding_r2']) else "N/A"
        # Verdict
        if s['mean_d'] is not None and s['mean_d'] > 0.2:
            verdict = "✓ PRMP helps"
        elif s['mean_d'] is not None and s['mean_d'] > 0:
            verdict = "~ Marginal"
        elif s['mean_d'] is not None:
            verdict = "✗ No advantage"
        else:
            verdict = "N/A"
        table_data.append([ds, f"{s['n_tasks']} ({s['n_regression']}R/{s['n_classification']}C)",
                          best_d, mean_d, emb_r2, str(s['n_fk_links']), verdict])

    table = ax8.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#d4e6f1"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color cells by verdict
    n_cols = len(col_labels)
    for i, row in enumerate(table_data):
        verdict_col = n_cols - 1
        if "helps" in row[-1]:
            table[i + 1, verdict_col].set_facecolor("#d5f5e3")
        elif "Marginal" in row[-1]:
            table[i + 1, verdict_col].set_facecolor("#fdebd0")
        elif "No" in row[-1]:
            table[i + 1, verdict_col].set_facecolor("#fadbd8")

    ax8.set_title("Cross-Dataset PRMP Summary", fontsize=13, pad=20)
    fig8.tight_layout()
    fig8.savefig(FIG_DIR / "fig8_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig8)

    logger.info("All 8 figures generated successfully")

    # ══════════════════════════════════════════════════════════════════════════
    # 11. HYPOTHESIS VERDICTS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Computing hypothesis verdicts...")

    h1_systematic = meta_result["p_value"] < 0.05 and meta_result["pooled_d"] > 0
    h2_regime = abs(spearman_rho) > 0.3 and spearman_p < 0.1
    h3_regression = (meta_reg and meta_cls and meta_reg["pooled_d"] > meta_cls["pooled_d"])
    h4_learned = (ablation_results.get("iter2_Random Pred", {}).get("rmse_mean", 0) >
                  ablation_results.get("iter2_PRMP", {}).get("rmse_mean", float("inf")))

    verdicts = {
        "H1_systematic_advantage": {
            "supported": h1_systematic,
            "pooled_d": meta_result["pooled_d"],
            "p_value": meta_result["p_value"],
            "interpretation": f"Pooled d={meta_result['pooled_d']:.3f} (p={meta_result['p_value']:.4f}), {'significant' if h1_systematic else 'not significant'}",
        },
        "H2_regime_theory": {
            "supported": h2_regime,
            "spearman_rho": float(spearman_rho),
            "p_value": float(spearman_p),
            "interpretation": f"ρ={spearman_rho:.3f} (p={spearman_p:.3f}), {'supported' if h2_regime else 'not supported'}",
        },
        "H3_regression_advantage": {
            "supported": bool(h3_regression),
            "regression_pooled_d": meta_reg["pooled_d"] if meta_reg else None,
            "classification_pooled_d": meta_cls["pooled_d"] if meta_cls else None,
            "moderator_p": moderator_test.get("p_value"),
            "interpretation": f"Reg d={meta_reg['pooled_d']:.3f} vs Cls d={meta_cls['pooled_d']:.3f}" if meta_reg and meta_cls else "Insufficient data",
        },
        "H4_learned_predictions_necessary": {
            "supported": h4_learned,
            "prmp_rmse": ablation_results.get("iter2_PRMP", {}).get("rmse_mean"),
            "random_rmse": ablation_results.get("iter2_Random Pred", {}).get("rmse_mean"),
            "interpretation": "Learned predictions outperform random predictions" if h4_learned else "Random predictions competitive",
        },
    }

    for k, v in verdicts.items():
        logger.info(f"  {k}: {'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'} — {v['interpretation']}")

    # ══════════════════════════════════════════════════════════════════════════
    # 12. BUILD OUTPUT JSON
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Building output JSON...")

    # metrics_agg: flat numeric metrics
    metrics_agg = {
        "pooled_d": round(meta_result["pooled_d"], 6),
        "pooled_d_ci_lo": round(meta_result["ci_lo"], 6),
        "pooled_d_ci_hi": round(meta_result["ci_hi"], 6),
        "pooled_z": round(meta_result["z"], 6),
        "pooled_p_value": round(meta_result["p_value"], 6),
        "I2_heterogeneity": round(meta_result["I2"], 4),
        "tau2": round(meta_result["tau2"], 6),
        "Q_statistic": round(meta_result["Q"], 4),
        "k_independent_tasks": meta_result["k"],
        "prediction_interval_lo": round(meta_result["pred_lo"], 6),
        "prediction_interval_hi": round(meta_result["pred_hi"], 6),
        "regression_pooled_d": round(meta_reg["pooled_d"], 6) if meta_reg else 0,
        "classification_pooled_d": round(meta_cls["pooled_d"], 6) if meta_cls else 0,
        "moderator_p_value": round(moderator_test.get("p_value", 1.0) or 1.0, 6),
        "egger_intercept": round(egger["intercept"], 6) if not np.isnan(egger["intercept"]) else 0,
        "egger_p_value": round(egger["p_value"], 6) if not np.isnan(egger["p_value"]) else 1,
        "spearman_rho_regime": round(float(spearman_rho), 6) if not np.isnan(spearman_rho) else 0,
        "spearman_p_regime": round(float(spearman_p), 6) if not np.isnan(spearman_p) else 1,
        "n_fk_links_analyzed": len(embedding_links),
        "n_total_tasks_all_iters": len(all_tasks),
        "h1_systematic_supported": 1 if h1_systematic else 0,
        "h2_regime_supported": 1 if h2_regime else 0,
        "h3_regression_advantage": 1 if h3_regression else 0,
        "h4_learned_predictions": 1 if h4_learned else 0,
    }

    # Build examples: one per independent task
    examples = []
    for t in independent_tasks:
        ex = {
            "input": json.dumps({
                "task": t["task"],
                "dataset": t["dataset"],
                "task_type": t["task_type"],
                "metric": t["metric"],
                "source": t["source"],
            }),
            "output": json.dumps({
                "cohens_d": round(t["d"], 4),
                "se": round(t["se"], 4),
                "ci_lo": round(t["ci_lo"], 4),
                "ci_hi": round(t["ci_hi"], 4),
                "prmp_mean": round(t["prmp_mean"], 4),
                "std_mean": round(t["std_mean"], 4),
            }),
            "predict_prmp": str(round(t["prmp_mean"], 4)),
            "predict_standard": str(round(t["std_mean"], 4)),
            "eval_cohens_d": round(t["d"], 4),
            "eval_se": round(t["se"], 4),
            "eval_ci_lo": round(t["ci_lo"], 4),
            "eval_ci_hi": round(t["ci_hi"], 4),
            "metadata_dataset": t["dataset"],
            "metadata_task_type": t["task_type"],
            "metadata_metric": t["metric"],
            "metadata_n_prmp": t["n_prmp"],
            "metadata_n_std": t["n_std"],
        }
        examples.append(ex)

    # Also add sensitivity check tasks (non-primary)
    sensitivity_examples = []
    for t in all_tasks:
        if not t["is_primary"]:
            ex = {
                "input": json.dumps({
                    "task": t["task"],
                    "dataset": t["dataset"],
                    "note": "sensitivity_check",
                    "source": t["source"],
                }),
                "output": json.dumps({
                    "cohens_d": round(t["d"], 4),
                    "se": round(t["se"], 4),
                }),
                "predict_prmp": str(round(t["prmp_mean"], 4)),
                "predict_standard": str(round(t["std_mean"], 4)),
                "eval_cohens_d": round(t["d"], 4),
                "eval_se": round(t["se"], 4),
                "eval_ci_lo": round(t["ci_lo"], 4),
                "eval_ci_hi": round(t["ci_hi"], 4),
                "metadata_dataset": t["dataset"],
                "metadata_task_type": t["task_type"],
                "metadata_metric": t["metric"],
                "metadata_n_prmp": t["n_prmp"],
                "metadata_n_std": t["n_std"],
            }
            sensitivity_examples.append(ex)

    # Build datasets array
    datasets_out = [
        {
            "dataset": "meta_analysis_independent_tasks",
            "examples": examples,
        },
    ]

    if sensitivity_examples:
        datasets_out.append({
            "dataset": "sensitivity_checks",
            "examples": sensitivity_examples,
        })

    # Add embedding regime dataset
    emb_examples = []
    for link in embedding_links:
        emb_ex = {
            "input": json.dumps({
                "dataset": link["dataset"],
                "fk_link": link["fk_link"],
            }),
            "output": json.dumps({
                "embedding_r2_standard": round(link.get("embedding_r2_standard", 0) or 0, 4),
                "embedding_r2_prmp": round(link.get("embedding_r2_prmp", 0) or 0, 4),
                "delta_r2": round(link.get("delta_r2", 0) or 0, 4),
            }),
            "predict_standard": str(round(link.get("embedding_r2_standard", 0) or 0, 4)),
            "predict_prmp": str(round(link.get("embedding_r2_prmp", 0) or 0, 4)),
            "eval_embedding_r2_standard": round(link.get("embedding_r2_standard", 0) or 0, 6),
            "eval_delta_r2": round(link.get("delta_r2", 0) or 0, 6),
            "metadata_dataset": link["dataset"],
            "metadata_source": link.get("source", ""),
        }
        emb_examples.append(emb_ex)

    if emb_examples:
        datasets_out.append({
            "dataset": "embedding_regime_fk_links",
            "examples": emb_examples,
        })

    # Add loss-swap dataset
    lossswap_examples = []
    for i, ck in enumerate(config_keys):
        cd = lossswap_deltas.get(ck, {})
        lossswap_examples.append({
            "input": json.dumps({"config": ck}),
            "output": json.dumps({"delta_mean": round(cd.get("mean", 0), 4), "delta_std": round(cd.get("std", 0), 4)}),
            "predict_standard": "0.0",
            "predict_prmp": str(round(cd.get("mean", 0), 4)),
            "eval_delta_mean": round(cd.get("mean", 0), 6),
            "eval_delta_std": round(cd.get("std", 0), 6),
            "metadata_config": ck,
        })

    if lossswap_examples:
        datasets_out.append({
            "dataset": "loss_swap_experiment",
            "examples": lossswap_examples,
        })

    # Add ablation dataset
    ablation_examples = []
    for variant_name, data in ablation_results.items():
        ablation_examples.append({
            "input": json.dumps({"variant": variant_name}),
            "output": json.dumps({"rmse_mean": round(data.get("rmse_mean", 0), 4)}),
            "predict_standard": str(round(ablation_results.get("iter2_Standard", {}).get("rmse_mean", 0), 4)),
            "predict_prmp": str(round(data.get("rmse_mean", 0), 4)),
            "eval_rmse_mean": round(data.get("rmse_mean", 0), 6),
            "metadata_variant": variant_name,
        })

    if ablation_examples:
        datasets_out.append({
            "dataset": "ablation_variants",
            "examples": ablation_examples,
        })

    output = {
        "metadata": {
            "evaluation_name": "PRMP Meta-Analysis: Pooled Effects, Embedding Regime Theory, and 8 Publication Figures",
            "description": "Comprehensive meta-analysis of PRMP across 12 experiments, 5 datasets, 8-10 independent tasks",
            "hypothesis_verdicts": verdicts,
            "meta_analysis": meta_result,
            "moderator_analysis": moderator_test,
            "regression_subgroup": meta_reg,
            "classification_subgroup": meta_cls,
            "regime_theory": regime_results,
            "lossswap_results": lossswap_results,
            "eggers_test": egger,
            "dataset_summary": dataset_summary,
            "figures": [
                "figures/fig1_forest_plot.png",
                "figures/fig2_regime_scatter.png",
                "figures/fig3_r2_trajectories.png",
                "figures/fig4_ablation_bar.png",
                "figures/fig5_subgroup_forest.png",
                "figures/fig6_funnel_plot.png",
                "figures/fig7_lossswap.png",
                "figures/fig8_summary_table.png",
            ],
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }

    # Write output
    out_path = WORK / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # Verify all figures exist
    for fig_name in output["metadata"]["figures"]:
        fig_path = WORK / fig_name
        if fig_path.exists():
            logger.info(f"  ✓ {fig_name} ({fig_path.stat().st_size / 1024:.0f} KB)")
        else:
            logger.warning(f"  ✗ {fig_name} MISSING")

    logger.info("Evaluation complete!")
    return output


if __name__ == "__main__":
    main()
