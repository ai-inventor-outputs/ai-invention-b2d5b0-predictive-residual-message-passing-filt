#!/usr/bin/env python3
"""Mechanistic Synthesis: PRMP Improvement Decomposition, Gradient/Embedding Analysis, and Publication Figures.

Synthesizes evidence from four prior experiments into a unified mechanistic narrative
explaining what PRMP does and when it helps.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware / Memory ────────────────────────────────────────────────────────
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)  # 50% of container
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f} GB, container total: {TOTAL_RAM_GB:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEP1_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus")
DEP2_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus")
DEP3_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_4/gen_art/exp_id3_it4__opus")
DEP4_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop/iter_2/gen_art/exp_id4_it2__opus")

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return json.loads(path.read_text())


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of values."""
    arr = np.array(values)
    if len(arr) < 2:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d (positive = group1 > group2)."""
    m1, m2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1e-9
    if pooled < 1e-12:
        pooled = 1e-12
    return float((m1 - m2) / pooled)


# =============================================================================
# METRIC 1: Mechanism Decomposition Table (Amazon) from exp_id1
# =============================================================================
@logger.catch
def compute_mechanism_decomposition(dep1_data: dict) -> dict:
    logger.info("Computing mechanism decomposition table (Amazon)")
    meta = dep1_data["metadata"]
    summaries = meta["amazon_summaries"]

    results = {}
    for metric_name in ["rmse", "mae", "r2"]:
        std_vals = summaries["A_standard_sage"][metric_name]["values"]
        prmp_vals = summaries["B_prmp"][metric_name]["values"]
        wide_vals = summaries["C_wide_sage"][metric_name]["values"]
        aux_vals = summaries["D_aux_mlp"][metric_name]["values"]
        skip_vals = summaries["E_skip_residual"][metric_name]["values"]

        std_mean = np.mean(std_vals)
        prmp_mean = np.mean(prmp_vals)
        total_improvement = std_mean - prmp_mean  # positive for RMSE/MAE, negative for R²

        # For R², improvement is prmp - standard (positive is better)
        if metric_name == "r2":
            total_improvement = prmp_mean - std_mean

        if abs(total_improvement) < 1e-12:
            logger.warning(f"No improvement for {metric_name}, skipping")
            continue

        # Compute fractions
        wide_mean = np.mean(wide_vals)
        aux_mean = np.mean(aux_vals)
        skip_mean = np.mean(skip_vals)

        if metric_name == "r2":
            extra_params_frac = (wide_mean - std_mean) / total_improvement
            aux_mlp_frac = (aux_mean - std_mean) / total_improvement
            skip_frac = (skip_mean - std_mean) / total_improvement
        else:
            extra_params_frac = (std_mean - wide_mean) / total_improvement
            aux_mlp_frac = (std_mean - aux_mean) / total_improvement
            skip_frac = (std_mean - skip_mean) / total_improvement

        residual_frac = 1.0 - max(extra_params_frac, aux_mlp_frac, skip_frac)

        # Bootstrap CIs for each fraction
        n_boot = 10000
        rng = np.random.default_rng(42)
        frac_boots = {"extra_params": [], "aux_mlp": [], "skip": [], "residual": []}
        for _ in range(n_boot):
            idx = rng.integers(0, len(std_vals), size=len(std_vals))
            s = np.mean([std_vals[i] for i in idx])
            p = np.mean([prmp_vals[i] for i in idx])
            w = np.mean([wide_vals[i] for i in idx])
            a = np.mean([aux_vals[i] for i in idx])
            sk = np.mean([skip_vals[i] for i in idx])
            ti = (p - s) if metric_name == "r2" else (s - p)
            if abs(ti) < 1e-12:
                continue
            if metric_name == "r2":
                ep = (w - s) / ti
                am = (a - s) / ti
                sf = (sk - s) / ti
            else:
                ep = (s - w) / ti
                am = (s - a) / ti
                sf = (s - sk) / ti
            rf = 1.0 - max(ep, am, sf)
            frac_boots["extra_params"].append(ep)
            frac_boots["aux_mlp"].append(am)
            frac_boots["skip"].append(sf)
            frac_boots["residual"].append(rf)

        ci = {}
        for k, v in frac_boots.items():
            if len(v) > 0:
                ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            else:
                ci[k] = (0.0, 0.0)

        results[metric_name] = {
            "total_improvement": float(total_improvement),
            "standard_mean": float(std_mean),
            "prmp_mean": float(prmp_mean),
            "extra_params_frac": float(extra_params_frac),
            "extra_params_ci": ci["extra_params"],
            "aux_mlp_frac": float(aux_mlp_frac),
            "aux_mlp_ci": ci["aux_mlp"],
            "skip_frac": float(skip_frac),
            "skip_ci": ci["skip"],
            "predict_subtract_residual_frac": float(residual_frac),
            "predict_subtract_ci": ci["residual"],
            "variant_means": {
                "standard": float(std_mean),
                "wide": float(wide_mean),
                "aux_mlp": float(aux_mean),
                "skip_residual": float(skip_mean),
                "prmp": float(prmp_mean),
            },
        }
        logger.info(
            f"  {metric_name}: extra_params={extra_params_frac:.3f}, aux_mlp={aux_mlp_frac:.3f}, "
            f"skip={skip_frac:.3f}, residual={residual_frac:.3f}"
        )

    return results


# =============================================================================
# METRIC 2: Gradient Norm Ratio Analysis from exp_id3
# =============================================================================
@logger.catch
def compute_gradient_analysis(dep3_data: dict) -> dict:
    logger.info("Computing gradient norm ratio analysis")
    examples = dep3_data["datasets"][0]["examples"]

    # Organize data: task -> epoch -> list of (update_grad, pred_grad) per edge
    task_epoch_ratios: dict[str, dict[int, list[float]]] = {}
    task_types: dict[str, str] = {}

    for ex in examples:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        task = inp.get("task", "")
        variant = inp.get("variant", "")
        task_type = inp.get("task_type", "")
        epoch = inp.get("epoch", -1)

        if variant != "prmp" or not task:
            continue
        task_types[task] = task_type
        grad_norms = out.get("grad_norms", {})
        if not grad_norms:
            continue

        update_keys = [k for k in grad_norms if "update_mlp" in k]
        if task not in task_epoch_ratios:
            task_epoch_ratios[task] = {}

        ratios_this_epoch = []
        for uk in update_keys:
            pk = uk.replace("update_mlp", "pred_mlp")
            u_val = grad_norms.get(uk, 0.0)
            p_val = grad_norms.get(pk, 0.0)
            if p_val > 1e-12:
                ratios_this_epoch.append(u_val / p_val)

        if ratios_this_epoch:
            task_epoch_ratios[task][epoch] = ratios_this_epoch

    # Per-task mean gradient ratio across all epochs and edges
    task_mean_ratios: dict[str, float] = {}
    task_all_ratios: dict[str, list[float]] = {}
    for task, epoch_data in task_epoch_ratios.items():
        all_ratios = []
        for epoch, ratios in epoch_data.items():
            all_ratios.extend(ratios)
        if all_ratios:
            task_mean_ratios[task] = float(np.mean(all_ratios))
            task_all_ratios[task] = all_ratios

    # Group by task type
    reg_ratios = []
    cls_ratios = []
    for task, ratios in task_all_ratios.items():
        if task_types.get(task) == "regression":
            reg_ratios.extend(ratios)
        else:
            cls_ratios.extend(ratios)

    # Mann-Whitney U test
    mw_stat, mw_pval = (float("nan"), float("nan"))
    if reg_ratios and cls_ratios:
        try:
            mw_result = stats.mannwhitneyu(reg_ratios, cls_ratios, alternative="two-sided")
            mw_stat = float(mw_result.statistic)
            mw_pval = float(mw_result.pvalue)
        except Exception:
            logger.exception("Mann-Whitney U test failed")

    # Build epoch trajectories for plotting
    epoch_trajectories: dict[str, dict[str, list]] = {}
    for task, epoch_data in task_epoch_ratios.items():
        epochs_sorted = sorted(epoch_data.keys())
        means = [float(np.mean(epoch_data[e])) for e in epochs_sorted]
        stds = [float(np.std(epoch_data[e])) for e in epochs_sorted]
        epoch_trajectories[task] = {
            "epochs": epochs_sorted,
            "mean_ratio": means,
            "std_ratio": stds,
            "task_type": task_types.get(task, "unknown"),
        }

    result = {
        "task_mean_gradient_ratios": task_mean_ratios,
        "regression_mean_ratio": float(np.mean(reg_ratios)) if reg_ratios else None,
        "classification_mean_ratio": float(np.mean(cls_ratios)) if cls_ratios else None,
        "regression_n_edges": len(reg_ratios),
        "classification_n_edges": len(cls_ratios),
        "mann_whitney_U": mw_stat,
        "mann_whitney_p": mw_pval,
        "epoch_trajectories": epoch_trajectories,
    }
    logger.info(
        f"  Regression mean ratio: {result['regression_mean_ratio']:.3f} (n={len(reg_ratios)}), "
        f"Classification: {result['classification_mean_ratio']:.3f} (n={len(cls_ratios)}), "
        f"MW p={mw_pval:.4f}"
    )
    return result


# =============================================================================
# METRIC 3: Embedding Effective Rank Analysis from exp_id3
# =============================================================================
@logger.catch
def compute_embedding_rank_analysis(dep3_data: dict) -> dict:
    logger.info("Computing embedding effective rank analysis")
    examples = dep3_data["datasets"][0]["examples"]

    # task -> variant -> epoch -> effective_rank
    rank_data: dict[str, dict[str, dict[int, float]]] = {}
    task_types: dict[str, str] = {}

    for ex in examples:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        task = inp.get("task", "")
        variant = inp.get("variant", "")
        task_type = inp.get("task_type", "")
        epoch = inp.get("epoch", -1)

        if not task:
            continue
        task_types[task] = task_type
        embed_stats = out.get("embed_stats", {})
        if not embed_stats or not isinstance(embed_stats, dict):
            continue
        eff_rank = embed_stats.get("effective_rank")
        if eff_rank is None:
            continue

        if task not in rank_data:
            rank_data[task] = {}
        if variant not in rank_data[task]:
            rank_data[task][variant] = {}
        rank_data[task][variant][epoch] = float(eff_rank)

    # Compute delta_rank at final epoch
    delta_ranks: dict[str, float] = {}
    final_ranks: dict[str, dict[str, float]] = {}
    for task in rank_data:
        std_epochs = rank_data[task].get("standard", {})
        prmp_epochs = rank_data[task].get("prmp", {})
        if not std_epochs or not prmp_epochs:
            continue
        # Get max epoch available for both
        std_final = max(std_epochs.keys())
        prmp_final = max(prmp_epochs.keys())
        final_ranks[task] = {
            "standard": std_epochs[std_final],
            "standard_epoch": std_final,
            "prmp": prmp_epochs[prmp_final],
            "prmp_epoch": prmp_final,
        }
        delta_ranks[task] = prmp_epochs[prmp_final] - std_epochs[std_final]

    # Get val_metric deltas for Spearman correlation
    val_metrics: dict[str, dict[str, float]] = {}
    for ex in examples:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        task = inp.get("task", "")
        variant = inp.get("variant", "")
        epoch = inp.get("epoch", -1)
        val_metric = out.get("val_metric")
        if not task or val_metric is None:
            continue
        if task not in val_metrics:
            val_metrics[task] = {}
        key = f"{variant}_{epoch}"
        val_metrics[task][key] = float(val_metric)

    # Best val_metric for each task/variant
    task_deltas: dict[str, float] = {}
    for task in delta_ranks:
        std_best = None
        prmp_best = None
        tt = task_types.get(task, "")
        for k, v in val_metrics.get(task, {}).items():
            if k.startswith("standard_"):
                if std_best is None or (tt == "regression" and v < std_best) or (tt == "classification" and v > std_best):
                    std_best = v
            elif k.startswith("prmp_"):
                if prmp_best is None or (tt == "regression" and v < prmp_best) or (tt == "classification" and v > prmp_best):
                    prmp_best = v
        if std_best is not None and prmp_best is not None:
            if tt == "regression":
                task_deltas[task] = std_best - prmp_best  # positive = PRMP better
            else:
                task_deltas[task] = prmp_best - std_best  # positive = PRMP better

    # Spearman correlation between delta_rank and delta improvement
    common_tasks = sorted(set(delta_ranks.keys()) & set(task_deltas.keys()))
    spearman_rho, spearman_p = (float("nan"), float("nan"))
    if len(common_tasks) >= 3:
        dr = [delta_ranks[t] for t in common_tasks]
        dd = [task_deltas[t] for t in common_tasks]
        try:
            res = stats.spearmanr(dr, dd)
            spearman_rho = float(res.correlation)
            spearman_p = float(res.pvalue)
        except Exception:
            logger.exception("Spearman failed")

    # Regression vs classification delta_rank comparison
    reg_delta = [delta_ranks[t] for t in delta_ranks if task_types.get(t) == "regression"]
    cls_delta = [delta_ranks[t] for t in delta_ranks if task_types.get(t) == "classification"]

    result = {
        "delta_ranks": delta_ranks,
        "final_ranks": final_ranks,
        "task_deltas": task_deltas,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "regression_delta_ranks": reg_delta,
        "classification_delta_ranks": cls_delta,
        "regression_mean_delta_rank": float(np.mean(reg_delta)) if reg_delta else None,
        "classification_mean_delta_rank": float(np.mean(cls_delta)) if cls_delta else None,
        "rank_trajectories": {
            task: {
                variant: {"epochs": sorted(epochs.keys()), "ranks": [epochs[e] for e in sorted(epochs.keys())]}
                for variant, epochs in variants.items()
            }
            for task, variants in rank_data.items()
        },
        "task_types": task_types,
    }
    logger.info(f"  Delta ranks: {delta_ranks}")
    logger.info(f"  Spearman rho={spearman_rho:.3f}, p={spearman_p:.4f}")
    return result


# =============================================================================
# METRIC 4: Embedding R² Correlation with Per-Link Improvement (exp_id2 × exp_id4)
# =============================================================================
@logger.catch
def compute_embedding_r2_correlation(dep2_data: dict, dep4_data: dict) -> dict:
    logger.info("Computing embedding R² correlation with per-link improvement")
    meta2 = dep2_data["metadata"]
    meta4 = dep4_data["metadata"]

    # From exp_id2: final-epoch layer-2 ridge_r2
    ep = meta2["embedding_predictability"]
    std_product_r2 = ep["standard"]["product_to_review"]["layer_2"]["ridge_r2"][-1]
    std_customer_r2 = ep["standard"]["customer_to_review"]["layer_2"]["ridge_r2"][-1]
    prmp_product_r2 = ep["prmp"]["product_to_review"]["layer_2"]["ridge_r2"][-1]
    prmp_customer_r2 = ep["prmp"]["customer_to_review"]["layer_2"]["ridge_r2"][-1]

    # From exp_id4: per-link PRMP improvement
    results4 = meta4["results"]
    std_rmse = results4["standard_sage"]["rmse"]["mean"]
    product_only_rmse = results4["prmp_product_only"]["rmse"]["mean"]
    customer_only_rmse = results4["prmp_customer_only"]["rmse"]["mean"]

    product_improvement = std_rmse - product_only_rmse
    customer_improvement = std_rmse - customer_only_rmse

    # R² reduction (filtering effect)
    product_r2_reduction = std_product_r2 - prmp_product_r2
    customer_r2_reduction = std_customer_r2 - prmp_customer_r2

    # Relationship: higher embedding R² → larger per-link improvement?
    higher_r2_link = "customer" if std_customer_r2 > std_product_r2 else "product"
    larger_improvement_link = "customer" if customer_improvement > product_improvement else "product"
    r2_predicts_improvement = higher_r2_link == larger_improvement_link

    # Full R² trajectories for figure
    r2_trajectories = {}
    for model in ["standard", "prmp"]:
        for link in ["product_to_review", "customer_to_review"]:
            key = f"{model}_{link}"
            layer2 = ep[model][link]["layer_2"]
            r2_trajectories[key] = {
                "epochs": layer2["epochs"],
                "ridge_r2": layer2["ridge_r2"],
            }

    result = {
        "standard_product_r2": float(std_product_r2),
        "standard_customer_r2": float(std_customer_r2),
        "prmp_product_r2": float(prmp_product_r2),
        "prmp_customer_r2": float(prmp_customer_r2),
        "product_link_improvement": float(product_improvement),
        "customer_link_improvement": float(customer_improvement),
        "product_r2_reduction": float(product_r2_reduction),
        "customer_r2_reduction": float(customer_r2_reduction),
        "higher_r2_link": higher_r2_link,
        "larger_improvement_link": larger_improvement_link,
        "r2_predicts_improvement": r2_predicts_improvement,
        "r2_trajectories": r2_trajectories,
    }
    logger.info(
        f"  Product R²={std_product_r2:.3f}, improvement={product_improvement:.4f}; "
        f"Customer R²={std_customer_r2:.3f}, improvement={customer_improvement:.4f}; "
        f"R² predicts improvement: {r2_predicts_improvement}"
    )
    return result


# =============================================================================
# METRIC 5: Cross-Experiment Summary Statistics
# =============================================================================
@logger.catch
def compute_cross_experiment_summary(
    dep1_data: dict, dep3_data: dict, dep4_data: dict
) -> dict:
    logger.info("Computing cross-experiment summary statistics")

    # Regression tasks: Amazon (exp_id1), driver-position (exp_id3), post-votes (exp_id3)
    # Classification tasks: driver-dnf, driver-top3, user-engagement (exp_id3)

    # Amazon PRMP delta (RMSE improvement) from exp_id1
    meta1 = dep1_data["metadata"]["amazon_summaries"]
    amazon_std_vals = meta1["A_standard_sage"]["rmse"]["values"]
    amazon_prmp_vals = meta1["B_prmp"]["rmse"]["values"]
    amazon_deltas = [s - p for s, p in zip(amazon_std_vals, amazon_prmp_vals)]

    # From exp_id3: get best val_metric per task/variant
    examples = dep3_data["datasets"][0]["examples"]
    task_variant_best: dict[str, dict[str, float]] = {}
    task_types_map: dict[str, str] = {}
    for ex in examples:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        task = inp.get("task", "")
        variant = inp.get("variant", "")
        task_type = inp.get("task_type", "")
        val_metric = out.get("val_metric")
        if not task or val_metric is None:
            continue
        task_types_map[task] = task_type
        if task not in task_variant_best:
            task_variant_best[task] = {}
        if variant not in task_variant_best[task]:
            task_variant_best[task][variant] = val_metric
        else:
            if task_type == "regression":
                task_variant_best[task][variant] = min(task_variant_best[task][variant], val_metric)
            else:
                task_variant_best[task][variant] = max(task_variant_best[task][variant], val_metric)

    # Compute deltas (positive = PRMP better)
    task_deltas: dict[str, float] = {}
    for task in task_variant_best:
        s = task_variant_best[task].get("standard")
        p = task_variant_best[task].get("prmp")
        if s is not None and p is not None:
            tt = task_types_map.get(task, "")
            if tt == "regression":
                task_deltas[task] = s - p  # lower is better
            else:
                task_deltas[task] = p - s  # higher is better

    # Also add Amazon from exp_id4 for extra reference
    reg_deltas = amazon_deltas.copy()  # Amazon RMSE deltas per seed
    for task, delta in task_deltas.items():
        if task_types_map.get(task) == "regression":
            reg_deltas.append(delta)

    cls_deltas = [task_deltas[t] for t in task_deltas if task_types_map.get(t) == "classification"]

    reg_mean = float(np.mean(reg_deltas))
    cls_mean = float(np.mean(cls_deltas))
    reg_ci = bootstrap_ci(reg_deltas)
    cls_ci = bootstrap_ci(cls_deltas)

    # Cohen's d for regression vs classification
    effect_size = cohens_d(reg_deltas, cls_deltas) if (reg_deltas and cls_deltas) else float("nan")

    result = {
        "regression_deltas": {
            "amazon_per_seed": amazon_deltas,
            "driver_position": task_deltas.get("driver-position"),
            "post_votes": task_deltas.get("post-votes"),
            "all_values": reg_deltas,
            "mean": reg_mean,
            "ci_95": reg_ci,
        },
        "classification_deltas": {
            "driver_dnf": task_deltas.get("driver-dnf"),
            "driver_top3": task_deltas.get("driver-top3"),
            "user_engagement": task_deltas.get("user-engagement"),
            "all_values": cls_deltas,
            "mean": cls_mean,
            "ci_95": cls_ci,
        },
        "regression_vs_classification_cohens_d": effect_size,
        "task_deltas": task_deltas,
        "task_types": task_types_map,
    }
    logger.info(
        f"  Regression mean delta: {reg_mean:.4f} CI={reg_ci}, "
        f"Classification mean delta: {cls_mean:.4f} CI={cls_ci}, "
        f"Cohen's d: {effect_size:.3f}"
    )
    return result


# =============================================================================
# FIGURE 1: Mechanism Decomposition Bar Chart
# =============================================================================
def plot_figure1(decomp: dict, dep1_data: dict) -> str:
    logger.info("Plotting Figure 1: Mechanism Decomposition Bar Chart")
    meta = dep1_data["metadata"]["amazon_summaries"]
    variants = ["A_standard_sage", "C_wide_sage", "D_aux_mlp", "E_skip_residual", "B_prmp"]
    labels = ["Standard", "Wide\n(+params)", "AuxMLP\n(+MLP)", "Skip-Res\n(+shortcut)", "PRMP\n(predict-subtract)"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    means = [np.mean(meta[v]["rmse"]["values"]) for v in variants]
    stds = [np.std(meta[v]["rmse"]["values"]) for v in variants]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5, width=0.65)

    # Dashed lines
    std_mean = means[0]
    prmp_mean = means[-1]
    ax.axhline(std_mean, color="#7f7f7f", linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(prmp_mean, color="#d62728", linestyle="--", alpha=0.6, linewidth=1)

    # Annotate fractions
    rmse_decomp = decomp.get("rmse", {})
    fracs = {
        "Wide": rmse_decomp.get("extra_params_frac", 0),
        "AuxMLP": rmse_decomp.get("aux_mlp_frac", 0),
        "Skip-Res": rmse_decomp.get("skip_frac", 0),
    }
    for i, (lbl, frac) in enumerate(fracs.items(), start=1):
        ax.annotate(
            f"{frac:.0%}",
            xy=(x[i], means[i]),
            xytext=(x[i], means[i] - 0.012),
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color=colors[i],
        )

    # Residual annotation
    res_frac = rmse_decomp.get("predict_subtract_residual_frac", 0)
    ax.annotate(
        f"Residual: {res_frac:.0%}",
        xy=(x[-1], prmp_mean),
        xytext=(x[-1] + 0.35, prmp_mean + 0.015),
        ha="left", va="bottom",
        fontsize=9, fontweight="bold",
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE (↓ better)", fontsize=11)
    ax.set_title("Amazon Video Games: Mechanism Decomposition", fontsize=13, fontweight="bold")
    ax.set_ylim(0.48, 0.60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(FIGURES_DIR / "fig1_mechanism_decomposition.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")
    return path


# =============================================================================
# FIGURE 2: Gradient Norm Ratio Trajectories
# =============================================================================
def plot_figure2(grad_result: dict) -> str:
    logger.info("Plotting Figure 2: Gradient Norm Ratio Trajectories")
    trajectories = grad_result["epoch_trajectories"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    reg_tasks = [t for t, d in trajectories.items() if d["task_type"] == "regression"]
    cls_tasks = [t for t, d in trajectories.items() if d["task_type"] == "classification"]

    line_styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v"]
    blue_palette = ["#1f77b4", "#4a90d9"]
    red_palette = ["#d62728", "#e8775a", "#c44e52"]

    for i, task in enumerate(sorted(reg_tasks)):
        d = trajectories[task]
        axes[0].plot(d["epochs"], d["mean_ratio"], linestyle=line_styles[i % len(line_styles)],
                     marker=markers[i % len(markers)], color=blue_palette[i % len(blue_palette)],
                     label=task, linewidth=2, markersize=5)
        if d["std_ratio"]:
            means = np.array(d["mean_ratio"])
            stds = np.array(d["std_ratio"])
            axes[0].fill_between(d["epochs"], means - stds, means + stds,
                                 alpha=0.15, color=blue_palette[i % len(blue_palette)])

    for i, task in enumerate(sorted(cls_tasks)):
        d = trajectories[task]
        axes[1].plot(d["epochs"], d["mean_ratio"], linestyle=line_styles[i % len(line_styles)],
                     marker=markers[i % len(markers)], color=red_palette[i % len(red_palette)],
                     label=task, linewidth=2, markersize=5)
        if d["std_ratio"]:
            means = np.array(d["mean_ratio"])
            stds = np.array(d["std_ratio"])
            axes[1].fill_between(d["epochs"], means - stds, means + stds,
                                 alpha=0.15, color=red_palette[i % len(red_palette)])

    axes[0].set_title("Regression Tasks", fontsize=12, fontweight="bold", color="#1f77b4")
    axes[1].set_title("Classification Tasks", fontsize=12, fontweight="bold", color="#d62728")
    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Gradient Norm Ratio (update_mlp / pred_mlp)", fontsize=10)

    fig.suptitle("Gradient Routing: update_mlp vs pred_mlp by Task Type", fontsize=13, fontweight="bold", y=1.02)
    path = str(FIGURES_DIR / "fig2_gradient_ratio_trajectories.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")
    return path


# =============================================================================
# FIGURE 3: Embedding Effective Rank Comparison
# =============================================================================
def plot_figure3(rank_result: dict) -> str:
    logger.info("Plotting Figure 3: Embedding Effective Rank Comparison")
    final_ranks = rank_result["final_ranks"]
    task_types = rank_result["task_types"]

    tasks = sorted(final_ranks.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))

    for i, task in enumerate(tasks):
        tt = task_types.get(task, "unknown")
        color = "#1f77b4" if tt == "regression" else "#d62728"
        std_rank = final_ranks[task]["standard"]
        prmp_rank = final_ranks[task]["prmp"]

        # Connect with line
        ax.plot([x[i], x[i]], [std_rank, prmp_rank], color=color, alpha=0.4, linewidth=2)
        # Standard = circle
        ax.scatter(x[i], std_rank, marker="o", s=120, color=color, edgecolors="black",
                   linewidths=0.8, zorder=5, label="Standard" if i == 0 else "")
        # PRMP = star
        ax.scatter(x[i], prmp_rank, marker="*", s=200, color=color, edgecolors="black",
                   linewidths=0.5, zorder=5, label="PRMP" if i == 0 else "")

        # Delta annotation
        delta = prmp_rank - std_rank
        sign = "+" if delta > 0 else ""
        ax.annotate(f"{sign}{delta:.1f}", xy=(x[i] + 0.15, (std_rank + prmp_rank) / 2),
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Embedding Effective Rank: Standard (○) vs PRMP (★)", fontsize=13, fontweight="bold")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Standard"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=14, label="PRMP"),
        Line2D([0], [0], color="#1f77b4", linewidth=3, label="Regression"),
        Line2D([0], [0], color="#d62728", linewidth=3, label="Classification"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(FIGURES_DIR / "fig3_embedding_rank_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")
    return path


# =============================================================================
# FIGURE 4: Embedding-Space Predictability vs PRMP Improvement
# =============================================================================
def plot_figure4(r2_result: dict) -> str:
    logger.info("Plotting Figure 4: Embedding-Space Predictability vs PRMP Improvement")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.2, 1]})

    # Top panel: R² trajectories
    ax = axes[0]
    traj = r2_result["r2_trajectories"]
    styles = {
        "standard_product_to_review": ("-", "#1f77b4", "Std - Product"),
        "standard_customer_to_review": ("-", "#d62728", "Std - Customer"),
        "prmp_product_to_review": ("--", "#1f77b4", "PRMP - Product"),
        "prmp_customer_to_review": ("--", "#d62728", "PRMP - Customer"),
    }
    for key, (ls, color, label) in styles.items():
        d = traj[key]
        ax.plot(d["epochs"], d["ridge_r2"], linestyle=ls, color=color, label=label,
                linewidth=2, marker="o" if ls == "-" else "s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Embedding Ridge R²", fontsize=10)
    ax.set_title("Embedding R² Trajectories: Standard vs PRMP", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom panel: scatter
    ax2 = axes[1]
    links = ["product_to_review", "customer_to_review"]
    r2_vals = [r2_result["standard_product_r2"], r2_result["standard_customer_r2"]]
    improvements = [r2_result["product_link_improvement"], r2_result["customer_link_improvement"]]
    colors = ["#1f77b4", "#d62728"]
    labels = ["Product→Review", "Customer→Review"]

    for i in range(2):
        ax2.scatter(r2_vals[i], improvements[i], s=200, color=colors[i],
                    edgecolors="black", linewidths=1, zorder=5)
        ax2.annotate(labels[i], xy=(r2_vals[i], improvements[i]),
                     xytext=(r2_vals[i] + 0.02, improvements[i] + 0.001),
                     fontsize=10, fontweight="bold", color=colors[i])

    # Draw line connecting the two points
    ax2.plot(r2_vals, improvements, "--", color="gray", alpha=0.5, linewidth=1.5)
    ax2.set_xlabel("Standard Model Embedding R²", fontsize=10)
    ax2.set_ylabel("PRMP Per-Link RMSE Improvement", fontsize=10)
    ax2.set_title("Higher Embedding R² → Larger PRMP Benefit", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Annotation about relationship
    predicts = r2_result["r2_predicts_improvement"]
    ax2.text(0.5, 0.05, f"R² predicts per-link improvement: {predicts}",
             transform=ax2.transAxes, fontsize=10, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    path = str(FIGURES_DIR / "fig4_embedding_r2_vs_improvement.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")
    return path


# =============================================================================
# BUILD OUTPUT JSON
# =============================================================================
def build_output(
    decomp: dict,
    grad: dict,
    rank: dict,
    r2_corr: dict,
    summary: dict,
    figure_paths: list[str],
    dep1_data: dict,
    dep3_data: dict,
) -> dict:
    """Build the output JSON conforming to exp_eval_sol_out schema."""
    logger.info("Building output JSON")

    # ── metrics_agg ──
    rmse_decomp = decomp.get("rmse", {})
    metrics_agg = {
        # Mechanism decomposition
        "decomp_extra_params_frac": round(rmse_decomp.get("extra_params_frac", 0), 4),
        "decomp_aux_mlp_frac": round(rmse_decomp.get("aux_mlp_frac", 0), 4),
        "decomp_skip_frac": round(rmse_decomp.get("skip_frac", 0), 4),
        "decomp_predict_subtract_frac": round(rmse_decomp.get("predict_subtract_residual_frac", 0), 4),
        # Gradient analysis
        "grad_regression_mean_ratio": round(grad.get("regression_mean_ratio") or 0, 4),
        "grad_classification_mean_ratio": round(grad.get("classification_mean_ratio") or 0, 4),
        "grad_mann_whitney_p": round(grad.get("mann_whitney_p", float("nan")), 6),
        # Embedding rank
        "rank_spearman_rho": round(rank.get("spearman_rho", float("nan")), 4),
        "rank_regression_mean_delta": round(rank.get("regression_mean_delta_rank") or 0, 4),
        "rank_classification_mean_delta": round(rank.get("classification_mean_delta_rank") or 0, 4),
        # Embedding R² correlation
        "r2_predicts_improvement": 1 if r2_corr.get("r2_predicts_improvement") else 0,
        "r2_product_link_improvement": round(r2_corr.get("product_link_improvement", 0), 4),
        "r2_customer_link_improvement": round(r2_corr.get("customer_link_improvement", 0), 4),
        # Cross-experiment summary
        "summary_regression_mean_delta": round(summary["regression_deltas"]["mean"], 4),
        "summary_classification_mean_delta": round(summary["classification_deltas"]["mean"], 4),
        "summary_reg_vs_cls_cohens_d": round(summary.get("regression_vs_classification_cohens_d", 0), 4),
    }

    # Replace NaN with 0 for JSON compliance
    for k, v in metrics_agg.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics_agg[k] = 0.0

    # ── Build examples from all 4 experiments ──
    all_examples = []

    # Examples from exp_id1: mechanism decomposition per-seed
    meta1 = dep1_data["metadata"]["amazon_summaries"]
    for seed_idx in range(3):
        seed_vals = {}
        for var_key in ["A_standard_sage", "B_prmp", "C_wide_sage", "D_aux_mlp", "E_skip_residual"]:
            seed_vals[var_key] = {
                m: meta1[var_key][m]["values"][seed_idx] for m in ["rmse", "mae", "r2"]
            }
        ex = {
            "input": json.dumps({
                "experiment": "exp_id1_mechanism_decomposition",
                "dataset": "amazon_video_games",
                "seed_index": seed_idx,
            }),
            "output": json.dumps(seed_vals),
            "eval_seed_prmp_rmse": seed_vals["B_prmp"]["rmse"],
            "eval_seed_standard_rmse": seed_vals["A_standard_sage"]["rmse"],
            "eval_seed_rmse_delta": seed_vals["A_standard_sage"]["rmse"] - seed_vals["B_prmp"]["rmse"],
            "metadata_experiment": "exp_id1_mechanism_decomposition",
            "metadata_seed_index": seed_idx,
        }
        all_examples.append(ex)

    # Examples from exp_id3: per-task results
    examples3 = dep3_data["datasets"][0]["examples"]
    seen_tasks = set()
    for ex in examples3:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        task = inp.get("task", "")
        variant = inp.get("variant", "")
        task_type = inp.get("task_type", "")
        epoch = inp.get("epoch", -1)

        if not task:
            continue
        task_key = f"{task}_{variant}_{epoch}"
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)

        # Only include checkpointed epochs (every 5) for conciseness
        if epoch % 5 != 0 and epoch != max(e for e2 in examples3 if (json.loads(e2["input"]) if isinstance(e2["input"], str) else e2["input"]).get("task") == task and (json.loads(e2["input"]) if isinstance(e2["input"], str) else e2["input"]).get("variant") == variant for e in [(json.loads(e2["input"]) if isinstance(e2["input"], str) else e2["input"]).get("epoch", -1)]):
            continue

        eval_fields = {}
        val_metric = out.get("val_metric")
        if val_metric is not None:
            eval_fields["eval_val_metric"] = float(val_metric)

        embed_stats = out.get("embed_stats", {})
        if isinstance(embed_stats, dict) and embed_stats.get("effective_rank"):
            eval_fields["eval_effective_rank"] = float(embed_stats["effective_rank"])

        # Gradient ratio for PRMP
        grad_norms = out.get("grad_norms", {})
        if grad_norms and variant == "prmp":
            update_vals = [v for k, v in grad_norms.items() if "update_mlp" in k]
            pred_vals = [v for k, v in grad_norms.items() if "pred_mlp" in k and v > 1e-12]
            if update_vals and pred_vals:
                eval_fields["eval_mean_grad_ratio"] = round(float(np.mean(update_vals) / np.mean(pred_vals)), 4)

        new_ex = {
            "input": json.dumps({
                "experiment": "exp_id3_task_type_effect",
                "dataset": inp.get("dataset", ""),
                "task": task,
                "variant": variant,
                "task_type": task_type,
                "epoch": epoch,
            }),
            "output": json.dumps({
                "val_metric": val_metric,
                "train_loss": out.get("train_loss"),
            }),
            "metadata_experiment": "exp_id3_task_type_effect",
            "metadata_task": task,
            "metadata_variant": variant,
            "metadata_task_type": task_type,
            "metadata_epoch": epoch,
            **eval_fields,
        }
        all_examples.append(new_ex)

    # Examples from exp_id2/exp_id4: R² correlation
    for link in ["product_to_review", "customer_to_review"]:
        improvement = r2_corr[f"{link.split('_')[0]}_link_improvement"]
        std_r2 = r2_corr[f"standard_{link.split('_')[0]}_r2"]
        prmp_r2 = r2_corr[f"prmp_{link.split('_')[0]}_r2"]
        ex = {
            "input": json.dumps({
                "experiment": "exp_id2_x_exp_id4_r2_correlation",
                "link": link,
            }),
            "output": json.dumps({
                "standard_embedding_r2": std_r2,
                "prmp_embedding_r2": prmp_r2,
                "per_link_improvement": improvement,
            }),
            "eval_standard_r2": round(std_r2, 4),
            "eval_prmp_r2": round(prmp_r2, 4),
            "eval_per_link_improvement": round(improvement, 4),
            "eval_r2_reduction": round(std_r2 - prmp_r2, 4),
            "metadata_experiment": "exp_id2_x_exp_id4_r2_correlation",
            "metadata_link": link,
        }
        all_examples.append(ex)

    output = {
        "metadata": {
            "evaluation_name": "PRMP Mechanistic Synthesis",
            "description": (
                "Synthesizes evidence from 4 prior experiments into unified mechanistic narrative. "
                "Produces mechanism decomposition, gradient analysis, embedding rank analysis, "
                "R² correlation, and 4 publication figures."
            ),
            "experiments_used": [
                "exp_id1_it4__opus (parameter-matched controls)",
                "exp_id2_it4__opus (embedding R² trajectories)",
                "exp_id3_it4__opus (task-type instrumented comparison)",
                "exp_id4_it2__opus (per-link PRMP ablation)",
            ],
            "mechanism_decomposition": decomp,
            "gradient_analysis": {k: v for k, v in grad.items() if k != "epoch_trajectories"},
            "embedding_rank_analysis": {k: v for k, v in rank.items() if k != "rank_trajectories"},
            "embedding_r2_correlation": {k: v for k, v in r2_corr.items() if k != "r2_trajectories"},
            "cross_experiment_summary": summary,
            "figures": figure_paths,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "prmp_mechanistic_synthesis",
                "examples": all_examples,
            }
        ],
    }
    return output


# =============================================================================
# MAIN
# =============================================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("PRMP Mechanistic Synthesis Evaluation")
    logger.info("=" * 60)

    # Load data
    dep1_data = load_json(DEP1_DIR / "full_method_out.json")
    dep2_data = load_json(DEP2_DIR / "full_method_out.json")
    dep3_data = load_json(DEP3_DIR / "full_method_out.json")
    dep4_data = load_json(DEP4_DIR / "full_method_out.json")

    # Compute metrics
    decomp = compute_mechanism_decomposition(dep1_data)
    grad = compute_gradient_analysis(dep3_data)
    rank = compute_embedding_rank_analysis(dep3_data)
    r2_corr = compute_embedding_r2_correlation(dep2_data, dep4_data)
    summary = compute_cross_experiment_summary(dep1_data, dep3_data, dep4_data)

    # Generate figures
    fig1 = plot_figure1(decomp, dep1_data)
    fig2 = plot_figure2(grad)
    fig3 = plot_figure3(rank)
    fig4 = plot_figure4(r2_corr)
    figure_paths = [fig1, fig2, fig3, fig4]

    # Build output
    output = build_output(decomp, grad, rank, r2_corr, summary, figure_paths, dep1_data, dep3_data)

    # Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved eval_out.json ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Log key results
    ma = output["metrics_agg"]
    logger.info("=" * 60)
    logger.info("KEY RESULTS:")
    logger.info(f"  Predict-subtract residual fraction: {ma['decomp_predict_subtract_frac']:.1%}")
    logger.info(f"  Gradient MW p-value (reg vs cls): {ma['grad_mann_whitney_p']:.4f}")
    logger.info(f"  Rank Spearman rho: {ma['rank_spearman_rho']:.3f}")
    logger.info(f"  R² predicts improvement: {bool(ma['r2_predicts_improvement'])}")
    logger.info(f"  Regression mean delta: {ma['summary_regression_mean_delta']:.4f}")
    logger.info(f"  Classification mean delta: {ma['summary_classification_mean_delta']:.4f}")
    logger.info(f"  Reg vs Cls Cohen's d: {ma['summary_reg_vs_cls_cohens_d']:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
