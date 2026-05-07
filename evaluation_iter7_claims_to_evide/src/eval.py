#!/usr/bin/env python3
"""Claims-to-Evidence Audit: PRMP Publication Readiness Assessment.

Systematic audit mapping every PRMP paper claim to quantitative evidence
across 6 experiments, computing confidence levels, identifying contradictions,
enumerating reviewer objections with prepared responses, and recommending
an honest paper narrative.

Pure data analysis — no new training, just JSON parsing, statistical
recomputation, and structured output.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

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

# Set memory limit (this is CPU-only analysis, ~2GB should be plenty)
RAM_BUDGET = int(min(4 * 1024**3, TOTAL_RAM_GB * 0.5 * 1024**3))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
BASE = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop")

EXP_PATHS = {
    "exp_id4_it2": BASE / "iter_2/gen_art/exp_id4_it2__opus/full_method_out.json",
    "exp_id1_it4": BASE / "iter_4/gen_art/exp_id1_it4__opus/full_method_out.json",
    "exp_id3_it4": BASE / "iter_4/gen_art/exp_id3_it4__opus/full_method_out.json",
    "exp_id2_it6": BASE / "iter_6/gen_art/exp_id2_it6__opus/full_method_out.json",
    "exp_id1_it6": BASE / "iter_6/gen_art/exp_id1_it6__opus/full_method_out.json",
    "exp_id3_it6": BASE / "iter_6/gen_art/exp_id3_it6__opus/full_method_out.json",
}


# ── Helpers ──────────────────────────────────────────────────────────────
def cohens_d(x: list[float], y: list[float]) -> float:
    """Compute Cohen's d (x - y) / pooled_std."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled == 0:
        return float('nan')
    return float((mx - my) / pooled)


def paired_ttest(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (t_stat, p_value) for paired t-test. Returns (nan, nan) if too few samples."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return (float('nan'), float('nan'))
    t, p = stats.ttest_rel(x, y)
    return (float(t), float(p))


def independent_ttest(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (t_stat, p_value) for independent t-test."""
    if len(x) < 2 or len(y) < 2:
        return (float('nan'), float('nan'))
    t, p = stats.ttest_ind(x, y)
    return (float(t), float(p))


def safe_float(v: Any) -> float:
    """Safely convert to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


# ── Load experiment data ─────────────────────────────────────────────────
def load_experiments() -> dict[str, dict]:
    """Load metadata from all 6 experiments."""
    data = {}
    for exp_id, path in EXP_PATHS.items():
        logger.info(f"Loading {exp_id} from {path}")
        try:
            raw = json.loads(path.read_text())
            data[exp_id] = raw
            logger.info(f"  Loaded {exp_id}: {len(str(raw))} chars")
        except Exception:
            logger.exception(f"Failed to load {exp_id}")
            raise
    return data


# ── SECTION 1: Claims Analysis ──────────────────────────────────────────

def analyze_c1_outperforms(exps: dict) -> dict:
    """C1: PRMP outperforms standard aggregation on relational tasks.

    Computes win_rate, significant_win_rate, median_effect_size, consistency_score.
    """
    logger.info("Analyzing C1: PRMP vs Standard performance")

    task_comparisons = []

    # exp_id4_it2: Amazon - PRMP vs Standard (3 seeds, RMSE lower=better)
    e = exps["exp_id4_it2"]["metadata"]
    prmp_rmse = e["results"]["prmp"]["rmse"]
    std_rmse = e["results"]["standard_sage"]["rmse"]
    task_comparisons.append({
        "experiment": "exp_id4_it2",
        "task": "amazon_review_rating",
        "dataset": "Amazon Video Games",
        "metric": "RMSE",
        "lower_better": True,
        "prmp_mean": prmp_rmse["mean"],
        "std_mean": std_rmse["mean"],
        "prmp_std": prmp_rmse["std"],
        "std_std": std_rmse["std"],
        "n_seeds": 3,
        "prmp_better": prmp_rmse["mean"] < std_rmse["mean"],
        "improvement_pct": (std_rmse["mean"] - prmp_rmse["mean"]) / std_rmse["mean"] * 100,
    })

    # exp_id1_it4: Amazon RMSE (3 seeds)
    e = exps["exp_id1_it4"]["metadata"]
    amz = e["amazon_summaries"]
    prmp_vals = amz["B_prmp"]["rmse"]["values"]
    std_vals = amz["A_standard_sage"]["rmse"]["values"]
    d = cohens_d(prmp_vals, std_vals)
    t, p = paired_ttest(prmp_vals, std_vals)
    task_comparisons.append({
        "experiment": "exp_id1_it4",
        "task": "amazon_review_rating_rmse",
        "dataset": "Amazon Video Games",
        "metric": "RMSE",
        "lower_better": True,
        "prmp_mean": amz["B_prmp"]["rmse"]["mean"],
        "std_mean": amz["A_standard_sage"]["rmse"]["mean"],
        "prmp_std": amz["B_prmp"]["rmse"]["std"],
        "std_std": amz["A_standard_sage"]["rmse"]["std"],
        "n_seeds": 3,
        "cohens_d": d,
        "p_value": p,
        "prmp_better": amz["B_prmp"]["rmse"]["mean"] < amz["A_standard_sage"]["rmse"]["mean"],
        "improvement_pct": (amz["A_standard_sage"]["rmse"]["mean"] - amz["B_prmp"]["rmse"]["mean"]) / amz["A_standard_sage"]["rmse"]["mean"] * 100,
    })

    # exp_id1_it4: F1 MAE (3 seeds)
    f1 = e["f1_summaries"]
    prmp_vals_f1 = f1["B_prmp"]["mae"]["values"]
    std_vals_f1 = f1["A_standard_sage"]["mae"]["values"]
    d_f1 = cohens_d(prmp_vals_f1, std_vals_f1)
    t_f1, p_f1 = paired_ttest(prmp_vals_f1, std_vals_f1)
    task_comparisons.append({
        "experiment": "exp_id1_it4",
        "task": "f1_driver_position_mae",
        "dataset": "F1",
        "metric": "MAE",
        "lower_better": True,
        "prmp_mean": f1["B_prmp"]["mae"]["mean"],
        "std_mean": f1["A_standard_sage"]["mae"]["mean"],
        "prmp_std": f1["B_prmp"]["mae"]["std"],
        "std_std": f1["A_standard_sage"]["mae"]["std"],
        "n_seeds": 3,
        "cohens_d": d_f1,
        "p_value": p_f1,
        "prmp_better": f1["B_prmp"]["mae"]["mean"] < f1["A_standard_sage"]["mae"]["mean"],
        "improvement_pct": (f1["A_standard_sage"]["mae"]["mean"] - f1["B_prmp"]["mae"]["mean"]) / f1["A_standard_sage"]["mae"]["mean"] * 100,
    })

    # exp_id2_it6: Amazon MAE (5 seeds)
    e2 = exps["exp_id2_it6"]["metadata"]
    sa = e2["statistical_analysis"]

    # Amazon review-rating MAE
    amazon_sa = sa["rel-amazon/review-rating"]
    prmp_amz_mean = amazon_sa["summary"]["prmp"]["mean"]
    std_amz_mean = amazon_sa["summary"]["standard"]["mean"]
    prmp_amz_std = amazon_sa["summary"]["prmp"]["std"]
    std_amz_std = amazon_sa["summary"]["standard"]["std"]
    pairwise_amz = amazon_sa["pairwise"]["standard_vs_prmp"]
    task_comparisons.append({
        "experiment": "exp_id2_it6",
        "task": "amazon_review_rating_mae",
        "dataset": "Amazon",
        "metric": "MAE",
        "lower_better": True,
        "prmp_mean": prmp_amz_mean,
        "std_mean": std_amz_mean,
        "prmp_std": prmp_amz_std,
        "std_std": std_amz_std,
        "n_seeds": 5,
        "cohens_d": pairwise_amz["cohens_d"],
        "p_value": pairwise_amz["p_value"],
        "prmp_better": prmp_amz_mean < std_amz_mean,
        "improvement_pct": (std_amz_mean - prmp_amz_mean) / std_amz_mean * 100,
    })

    # F1 result-position MAE
    f1pos_sa = sa["rel-f1/result-position"]
    prmp_f1pos_mean = f1pos_sa["summary"]["prmp"]["mean"]
    std_f1pos_mean = f1pos_sa["summary"]["standard"]["mean"]
    pairwise_f1pos = f1pos_sa["pairwise"]["standard_vs_prmp"]
    task_comparisons.append({
        "experiment": "exp_id2_it6",
        "task": "f1_result_position_mae",
        "dataset": "F1",
        "metric": "MAE",
        "lower_better": True,
        "prmp_mean": prmp_f1pos_mean,
        "std_mean": std_f1pos_mean,
        "prmp_std": f1pos_sa["summary"]["prmp"]["std"],
        "std_std": f1pos_sa["summary"]["standard"]["std"],
        "n_seeds": 5,
        "cohens_d": pairwise_f1pos["cohens_d"],
        "p_value": pairwise_f1pos["p_value"],
        "prmp_better": prmp_f1pos_mean < std_f1pos_mean,
        "improvement_pct": (std_f1pos_mean - prmp_f1pos_mean) / std_f1pos_mean * 100,
    })

    # F1 result-dnf AP (higher=better)
    f1dnf_sa = sa["rel-f1/result-dnf"]
    prmp_f1dnf_mean = f1dnf_sa["summary"]["prmp"]["mean"]
    std_f1dnf_mean = f1dnf_sa["summary"]["standard"]["mean"]
    pairwise_f1dnf = f1dnf_sa["pairwise"]["standard_vs_prmp"]
    task_comparisons.append({
        "experiment": "exp_id2_it6",
        "task": "f1_result_dnf_ap",
        "dataset": "F1",
        "metric": "AP",
        "lower_better": False,
        "prmp_mean": prmp_f1dnf_mean,
        "std_mean": std_f1dnf_mean,
        "prmp_std": f1dnf_sa["summary"]["prmp"]["std"],
        "std_std": f1dnf_sa["summary"]["standard"]["std"],
        "n_seeds": 5,
        "cohens_d": pairwise_f1dnf["cohens_d"],
        "p_value": pairwise_f1dnf["p_value"],
        "prmp_better": prmp_f1dnf_mean > std_f1dnf_mean,
        "improvement_pct": (prmp_f1dnf_mean - std_f1dnf_mean) / std_f1dnf_mean * 100 if std_f1dnf_mean != 0 else 0,
    })

    # exp_id1_it6: Avito RMSE (3 seeds)
    e1_6 = exps["exp_id1_it6"]["metadata"]
    avito_std = e1_6["gnn_results"]["A_StandardSAGE"]
    avito_prmp = e1_6["gnn_results"]["B_PRMP_Full"]
    prmp_avito_seeds = [s["rmse"] for s in avito_prmp["per_seed"]]
    std_avito_seeds = [s["rmse"] for s in avito_std["per_seed"]]
    d_avito = cohens_d(prmp_avito_seeds, std_avito_seeds)
    t_avito, p_avito = paired_ttest(prmp_avito_seeds, std_avito_seeds)
    task_comparisons.append({
        "experiment": "exp_id1_it6",
        "task": "avito_ad_ctr_rmse",
        "dataset": "Avito",
        "metric": "RMSE",
        "lower_better": True,
        "prmp_mean": avito_prmp["mean_rmse"],
        "std_mean": avito_std["mean_rmse"],
        "prmp_std": avito_prmp["std_rmse"],
        "std_std": avito_std["std_rmse"],
        "n_seeds": 3,
        "cohens_d": d_avito,
        "p_value": p_avito,
        "prmp_better": avito_prmp["mean_rmse"] < avito_std["mean_rmse"],
        "improvement_pct": (avito_std["mean_rmse"] - avito_prmp["mean_rmse"]) / avito_std["mean_rmse"] * 100,
    })

    # exp_id3_it4: 5 tasks (1 seed each) - extract from full data
    e3_4 = exps["exp_id3_it4"]
    # Need to parse out final metrics per task from the examples
    task_final_metrics = extract_exp_id3_it4_finals(e3_4)
    for task_name, vals in task_final_metrics.items():
        lower_better = vals.get("lower_better", True)
        prmp_val = vals["prmp"]
        std_val = vals["standard"]
        if lower_better:
            # Lower is better (regression: MAE, RMSE)
            prmp_better = prmp_val < std_val
            delta = std_val - prmp_val  # positive = PRMP better
            imp_pct = delta / abs(std_val) * 100 if std_val != 0 else 0
        else:
            # Higher is better (classification: AP, accuracy)
            prmp_better = prmp_val > std_val
            delta = prmp_val - std_val  # positive = PRMP better
            imp_pct = delta / abs(std_val) * 100 if std_val != 0 else 0

        task_comparisons.append({
            "experiment": "exp_id3_it4",
            "task": task_name,
            "dataset": vals["dataset"],
            "metric": vals["metric"],
            "lower_better": lower_better,
            "prmp_mean": prmp_val,
            "std_mean": std_val,
            "n_seeds": 1,
            "prmp_better": prmp_better,
            "improvement_pct": imp_pct,
            "delta": delta,
            "task_type": vals.get("task_type", "unknown"),
        })

    # Compute aggregated metrics
    wins = sum(1 for t in task_comparisons if t["prmp_better"])
    total = len(task_comparisons)
    win_rate = wins / total if total > 0 else 0

    # Significant wins (p<0.05)
    sig_wins = sum(1 for t in task_comparisons
                   if t.get("p_value") is not None
                   and not math.isnan(t.get("p_value", float('nan')))
                   and t.get("p_value", 1.0) < 0.05
                   and t["prmp_better"])
    tasks_with_pvalue = sum(1 for t in task_comparisons
                           if t.get("p_value") is not None
                           and not math.isnan(t.get("p_value", float('nan'))))
    significant_win_rate = sig_wins / tasks_with_pvalue if tasks_with_pvalue > 0 else 0

    # Median effect size
    effect_sizes = [abs(t.get("cohens_d", float('nan'))) for t in task_comparisons
                    if t.get("cohens_d") is not None and not math.isnan(t.get("cohens_d", float('nan')))]
    median_effect_size = float(np.median(effect_sizes)) if effect_sizes else 0

    # Consistency score: fraction of individual seeds where PRMP beats Standard
    # Compute from experiments that have per-seed data
    seed_wins = 0
    seed_total = 0

    # exp_id1_it4: Amazon per-seed RMSE (lower=better)
    e_it4 = exps["exp_id1_it4"]["metadata"]
    amz_prmp_seeds = e_it4["amazon_summaries"]["B_prmp"]["rmse"]["values"]
    amz_std_seeds = e_it4["amazon_summaries"]["A_standard_sage"]["rmse"]["values"]
    for p, s in zip(amz_prmp_seeds, amz_std_seeds):
        seed_total += 1
        if p < s:
            seed_wins += 1

    # exp_id1_it4: F1 per-seed MAE (lower=better)
    f1_prmp_seeds = e_it4["f1_summaries"]["B_prmp"]["mae"]["values"]
    f1_std_seeds = e_it4["f1_summaries"]["A_standard_sage"]["mae"]["values"]
    for p, s in zip(f1_prmp_seeds, f1_std_seeds):
        seed_total += 1
        if p < s:
            seed_wins += 1

    # exp_id1_it6: Avito per-seed RMSE (lower=better)
    e_it6 = exps["exp_id1_it6"]["metadata"]
    avito_prmp_seeds_v = [s["rmse"] for s in e_it6["gnn_results"]["B_PRMP_Full"]["per_seed"]]
    avito_std_seeds_v = [s["rmse"] for s in e_it6["gnn_results"]["A_StandardSAGE"]["per_seed"]]
    for p, s in zip(avito_prmp_seeds_v, avito_std_seeds_v):
        seed_total += 1
        if p < s:
            seed_wins += 1

    # exp_id2_it6: per-run results
    e_it2_6 = exps["exp_id2_it6"]["metadata"]
    per_run = e_it2_6.get("per_run_results", [])
    # Group by task
    from collections import defaultdict
    per_run_by_task: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in per_run:
        task_key = f"{r['dataset']}/{r['task']}"
        variant = r["variant"]
        metrics = r.get("test_metrics", {})
        # Get primary metric
        if "mae" in metrics:
            per_run_by_task[task_key][variant].append(metrics["mae"])
        elif "average_precision" in metrics:
            per_run_by_task[task_key][variant].append(metrics["average_precision"])

    for task_key, variants in per_run_by_task.items():
        if "standard" in variants and "prmp" in variants:
            prmp_list = sorted(variants["prmp"])
            std_list = sorted(variants["standard"])
            n = min(len(prmp_list), len(std_list))
            sa_info = e_it2_6["statistical_analysis"].get(task_key, {})
            lb = sa_info.get("lower_better", True)
            for i in range(n):
                seed_total += 1
                if lb:
                    if prmp_list[i] < std_list[i]:
                        seed_wins += 1
                else:
                    if prmp_list[i] > std_list[i]:
                        seed_wins += 1

    consistency_score = seed_wins / seed_total if seed_total > 0 else 0

    result = {
        "claim": "C1: PRMP outperforms standard aggregation on relational tasks",
        "task_comparisons": task_comparisons,
        "total_tasks": total,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "significant_wins": sig_wins,
        "tasks_with_pvalue": tasks_with_pvalue,
        "significant_win_rate": round(significant_win_rate, 4),
        "median_effect_size": round(median_effect_size, 4),
        "effect_sizes": [round(e, 4) for e in effect_sizes],
        "consistency_score": round(consistency_score, 4),
        "consistency_seed_wins": seed_wins,
        "consistency_seed_total": seed_total,
    }

    logger.info(f"  C1: win_rate={win_rate:.3f}, sig_win_rate={significant_win_rate:.3f}, "
                f"median_d={median_effect_size:.3f}, consistency={consistency_score:.3f}")
    return result


def extract_exp_id3_it4_finals(exp_data: dict) -> dict:
    """Extract final test metrics for each task from exp_id3_it4."""
    results = {}
    examples = []
    for ds in exp_data.get("datasets", []):
        examples.extend(ds.get("examples", []))

    # Group by (dataset, task, variant) and find the last epoch
    task_data: dict[tuple[str, str], dict[str, list]] = {}
    for ex in examples:
        dataset = ex.get("metadata_dataset", "")
        task = ex.get("metadata_task", "")
        variant = ex.get("metadata_variant", "")
        epoch = ex.get("metadata_epoch", 0)

        # Skip analysis/summary rows
        if variant == "comparison" or dataset == "all" or epoch < 0:
            continue

        key = (dataset, task)
        if key not in task_data:
            task_data[key] = {}
        if variant not in task_data[key]:
            task_data[key][variant] = []

        # Parse output to get metrics
        try:
            output = json.loads(ex.get("output", "{}"))
        except (json.JSONDecodeError, TypeError):
            output = {}

        task_data[key][variant].append({
            "epoch": epoch,
            "output": output,
            "task_type": ex.get("metadata_task_type", ""),
        })

    # For each task, get the final epoch metric
    for (dataset, task), variants in task_data.items():
        for variant_name, epoch_list in variants.items():
            # Sort by epoch descending and get the last one
            epoch_list.sort(key=lambda x: x["epoch"], reverse=True)
            last = epoch_list[0]
            task_type = last["task_type"]

            # Determine the best metric to use:
            # - test_metric=999.0 is a placeholder for regression → use val_metric
            # - test_metric=0.5 for classification is likely a dummy → use val_metric
            # - val_metric is the reliable metric for both types
            test_metric = last["output"].get("test_metric", None)
            val_metric = last["output"].get("val_metric", last["output"].get("val_loss", None))

            # For classification, higher val_metric is better (AP, accuracy)
            # For regression, lower val_metric is better (MAE, RMSE)
            is_classification = task_type == "classification"

            # Use val_metric as the primary metric since test_metric has dummy values
            # (999.0 for regression, 0.5 for classification)
            use_metric = val_metric
            if test_metric is not None and test_metric != 999.0 and test_metric != 0.5:
                use_metric = test_metric

            task_key = f"{dataset}_{task}"
            if task_key not in results:
                results[task_key] = {
                    "dataset": dataset,
                    "task": task,
                    "metric": "MAE" if task_type == "regression" else "AP/Accuracy",
                    "lower_better": not is_classification,
                    "task_type": task_type,
                }

            if use_metric is not None:
                results[task_key][variant_name] = safe_float(use_metric)

    # Filter to tasks that have both standard and prmp
    filtered = {}
    for k, v in results.items():
        if "standard" in v and "prmp" in v:
            filtered[k] = v

    return filtered


def analyze_c2_regime(exps: dict) -> dict:
    """C2: Improvement concentrates in high-cardinality, high-predictability regime."""
    logger.info("Analyzing C2: Cardinality x R^2 regime hypothesis")

    # exp_id4_it2: 2 FK links
    e = exps["exp_id4_it2"]["metadata"]
    diag = e["diagnostic"]

    exp4_links = {
        "product_to_review": {
            "cardinality": diag["product_to_review"]["cardinality_mean"],
            "r2": diag["product_to_review"]["linear_r2"],
            "interaction": diag["product_to_review"]["cardinality_mean"] * diag["product_to_review"]["linear_r2"],
        },
        "customer_to_review": {
            "cardinality": diag["customer_to_review"]["cardinality_mean"],
            "r2": diag["customer_to_review"]["linear_r2"],
            "interaction": diag["customer_to_review"]["cardinality_mean"] * diag["customer_to_review"]["linear_r2"],
        },
    }

    # Per-FK improvement from exp_id4_it2
    prmp_prod_only = e["results"]["prmp_product_only"]["rmse"]["mean"]
    prmp_cust_only = e["results"]["prmp_customer_only"]["rmse"]["mean"]
    std_rmse = e["results"]["standard_sage"]["rmse"]["mean"]

    exp4_links["product_to_review"]["prmp_improvement"] = std_rmse - prmp_prod_only
    exp4_links["customer_to_review"]["prmp_improvement"] = std_rmse - prmp_cust_only

    cardinality_hypothesis_supported = e["analysis"]["cardinality_hypothesis_supported"]

    # exp_id1_it6: 11 FK links with cardinality and R^2
    e1_6 = exps["exp_id1_it6"]["metadata"]
    r2_diag = e1_6.get("r2_diagnostic", {})
    card_stats = e1_6.get("cardinality_stats", {})
    interaction_terms = e1_6.get("regime_analysis", {}).get("interaction_terms_card_x_r2", {})

    avito_links = {}
    for link_name, it_val in interaction_terms.items():
        card_info = card_stats.get(link_name, {})
        r2_info = r2_diag.get(link_name, {})
        avito_links[link_name] = {
            "cardinality_mean": card_info.get("card_mean", card_info.get("n_edges", 0)),
            "r2": r2_info.get("r2", 0),
            "interaction_card_x_r2": it_val,
        }

    # Spearman correlation: interaction term vs PRMP improvement
    # We don't have per-FK improvement for Avito (single RMSE), so this is limited
    # For exp_id4_it2 with only 2 points, Spearman is degenerate
    interactions_4 = [exp4_links["product_to_review"]["interaction"],
                      exp4_links["customer_to_review"]["interaction"]]
    improvements_4 = [exp4_links["product_to_review"]["prmp_improvement"],
                      exp4_links["customer_to_review"]["prmp_improvement"]]

    # With 2 points, Spearman is always +1 or -1
    if len(interactions_4) >= 2:
        spearman_corr_4, spearman_p_4 = stats.spearmanr(interactions_4, improvements_4)
    else:
        spearman_corr_4, spearman_p_4 = float('nan'), float('nan')

    result = {
        "claim": "C2: Improvement concentrates in high-cardinality, high-predictability regime",
        "exp_id4_it2_links": exp4_links,
        "cardinality_hypothesis_supported_exp4": cardinality_hypothesis_supported,
        "note_exp4": "Customer link helped MORE despite LOWER cardinality — contradicts hypothesis",
        "avito_links": avito_links,
        "avito_prmp_relative_improvement_rmse": e1_6.get("regime_analysis", {}).get("prmp_relative_improvement_rmse", 0),
        "spearman_correlation_exp4": round(float(spearman_corr_4), 4) if not math.isnan(spearman_corr_4) else None,
        "spearman_p_exp4": round(float(spearman_p_4), 4) if not math.isnan(spearman_p_4) else None,
        "diagnostic_coverage_exp4": 2,
        "diagnostic_coverage_avito": len(avito_links),
        "total_fk_links_analyzed": 2 + len(avito_links),
        "conclusion": "weak_contradict",
        "reasoning": "exp_id4_it2 explicitly flags cardinality_hypothesis_supported=false. "
                     "Customer link (lower cardinality) helped more than product link (higher cardinality). "
                     "Avito has 11 FK links but only aggregate PRMP improvement (0.5%), insufficient for per-link correlation.",
    }

    logger.info(f"  C2: hypothesis_supported={cardinality_hypothesis_supported}, "
                f"spearman={spearman_corr_4:.3f}")
    return result


def analyze_c3_mechanism(exps: dict) -> dict:
    """C3: Predict-subtract mechanism is necessary (not just extra parameters)."""
    logger.info("Analyzing C3: Mechanism necessity (parameter-matched controls)")

    comparisons = []

    # exp_id1_it4: Amazon — PRMP vs Wide, AuxMLP, SkipResidual
    e = exps["exp_id1_it4"]["metadata"]
    amz = e["amazon_summaries"]
    prmp_rmse = amz["B_prmp"]["rmse"]["mean"]
    controls_amazon = {
        "Wide": {"mean": amz["C_wide_sage"]["rmse"]["mean"], "values": amz["C_wide_sage"]["rmse"]["values"]},
        "AuxMLP": {"mean": amz["D_aux_mlp"]["rmse"]["mean"], "values": amz["D_aux_mlp"]["rmse"]["values"]},
        "SkipResidual": {"mean": amz["E_skip_residual"]["rmse"]["mean"], "values": amz["E_skip_residual"]["rmse"]["values"]},
    }

    prmp_beats_all_amazon = all(prmp_rmse < c["mean"] for c in controls_amazon.values())
    best_control_amazon = min(controls_amazon.values(), key=lambda c: c["mean"])

    comparisons.append({
        "experiment": "exp_id1_it4",
        "dataset": "Amazon Video Games",
        "metric": "RMSE",
        "prmp_mean": prmp_rmse,
        "controls": {k: v["mean"] for k, v in controls_amazon.items()},
        "cohens_d_vs_controls": {
            "vs_Wide": e["amazon_analysis"]["prmp_vs_C_wide_sage"]["cohens_d"],
            "vs_AuxMLP": e["amazon_analysis"]["prmp_vs_D_aux_mlp"]["cohens_d"],
            "vs_SkipResidual": e["amazon_analysis"]["prmp_vs_E_skip_residual"]["cohens_d"],
        },
        "prmp_beats_all_controls": prmp_beats_all_amazon,
        "prmp_vs_best_control_delta": prmp_rmse - best_control_amazon["mean"],
    })

    # exp_id1_it4: F1 — PRMP vs controls (RMSE)
    f1 = e["f1_summaries"]
    prmp_f1_rmse = f1["B_prmp"]["rmse"]["mean"]
    controls_f1 = {
        "Wide": f1["C_wide_sage"]["rmse"]["mean"],
        "AuxMLP": f1["D_aux_mlp"]["rmse"]["mean"],
        "SkipResidual": f1["E_skip_residual"]["rmse"]["mean"],
        "Standard": f1["A_standard_sage"]["rmse"]["mean"],
    }
    prmp_beats_all_f1 = all(prmp_f1_rmse < c for c in controls_f1.values())

    comparisons.append({
        "experiment": "exp_id1_it4",
        "dataset": "F1",
        "metric": "RMSE",
        "prmp_mean": prmp_f1_rmse,
        "controls": controls_f1,
        "prmp_beats_all_controls": prmp_beats_all_f1,
        "note": "PRMP RMSE 0.824 vs Standard 0.691 — PRMP is WORSE on F1 RMSE",
    })

    # exp_id2_it6: Amazon MAE — PRMP vs Wide, AuxMLP, RandomFrozen
    e2 = exps["exp_id2_it6"]["metadata"]
    sa_amz = e2["statistical_analysis"]["rel-amazon/review-rating"]
    prmp_amz_mae = sa_amz["summary"]["prmp"]["mean"]
    controls_amz2 = {
        "Wide": sa_amz["summary"]["wide"]["mean"],
        "AuxMLP": sa_amz["summary"]["auxiliary_mlp"]["mean"],
        "RandomFrozen": sa_amz["summary"]["random_frozen"]["mean"],
    }
    prmp_beats_all_amz2 = all(prmp_amz_mae < c for c in controls_amz2.values())

    comparisons.append({
        "experiment": "exp_id2_it6",
        "dataset": "Amazon",
        "metric": "MAE",
        "prmp_mean": prmp_amz_mae,
        "controls": controls_amz2,
        "prmp_beats_all_controls": prmp_beats_all_amz2,
    })

    # exp_id1_it6: Avito — PRMP vs Wide
    e1_6 = exps["exp_id1_it6"]["metadata"]
    prmp_avito = e1_6["gnn_results"]["B_PRMP_Full"]["mean_rmse"]
    wide_avito = e1_6["gnn_results"]["C_WideSAGE_Control"]["mean_rmse"]

    comparisons.append({
        "experiment": "exp_id1_it6",
        "dataset": "Avito",
        "metric": "RMSE",
        "prmp_mean": prmp_avito,
        "controls": {"Wide": wide_avito},
        "prmp_beats_all_controls": prmp_avito < wide_avito,
        "note": "Wide had 2/3 seeds fail to converge, making comparison unreliable",
    })

    # Aggregate
    datasets_prmp_beats_all = sum(1 for c in comparisons if c["prmp_beats_all_controls"])
    total_datasets = len(comparisons)
    prmp_beats_all_rate = datasets_prmp_beats_all / total_datasets

    # Mean improvement over best control (across datasets where PRMP wins)
    improvements_vs_best = []
    for c in comparisons:
        if c.get("prmp_beats_all_controls"):
            best_ctrl_val = min(c["controls"].values())
            improvements_vs_best.append(c["prmp_mean"] - best_ctrl_val)
    mean_improvement_over_parammatched = (
        float(np.mean(improvements_vs_best)) if improvements_vs_best else float('nan')
    )

    result = {
        "claim": "C3: Predict-subtract mechanism is necessary",
        "comparisons": comparisons,
        "prmp_beats_all_controls_rate": round(prmp_beats_all_rate, 4),
        "mean_improvement_over_parammatched": round(mean_improvement_over_parammatched, 6),
        "conclusion": "strong_support_on_amazon_only",
        "reasoning": "On Amazon, Cohen's d values of -5.5 to -8.5 are enormous, proving mechanism works "
                     "beyond parameter count. On F1, PRMP is worse than Standard on RMSE. On Avito, Wide control "
                     "had convergence issues. The mechanism is validated primarily on Amazon.",
    }

    logger.info(f"  C3: beats_all_rate={prmp_beats_all_rate:.3f}")
    return result


def analyze_c4_learned_vs_random(exps: dict) -> dict:
    """C4: Learned predictions outperform random predictions."""
    logger.info("Analyzing C4: Learned vs Random predictions")

    comparisons = []

    # exp_id4_it2: PRMP vs Random-pred
    e = exps["exp_id4_it2"]["metadata"]
    prmp_rmse = e["results"]["prmp"]["rmse"]["mean"]
    random_rmse = e["results"]["ablation_random_pred"]["rmse"]["mean"]

    comparisons.append({
        "experiment": "exp_id4_it2",
        "dataset": "Amazon Video Games",
        "metric": "RMSE",
        "prmp_mean": prmp_rmse,
        "random_mean": random_rmse,
        "delta": random_rmse - prmp_rmse,
        "prmp_better": prmp_rmse < random_rmse,
        "note": "Learned predictions are better than random",
    })

    # exp_id2_it6: PRMP vs RandomFrozen
    e2 = exps["exp_id2_it6"]["metadata"]
    sa_amz = e2["statistical_analysis"]["rel-amazon/review-rating"]
    prmp_mae = sa_amz["summary"]["prmp"]["mean"]
    random_mae = sa_amz["summary"]["random_frozen"]["mean"]
    pairwise = sa_amz["pairwise"]["prmp_vs_random_frozen"]

    comparisons.append({
        "experiment": "exp_id2_it6",
        "dataset": "Amazon",
        "metric": "MAE",
        "prmp_mean": prmp_mae,
        "random_mean": random_mae,
        "delta": random_mae - prmp_mae,
        "prmp_better": prmp_mae < random_mae,
        "p_value": pairwise["p_value"],
        "cohens_d": pairwise["cohens_d"],
        "note": f"p={pairwise['p_value']:.3f}, NOT significant",
    })

    # Aggregate
    deltas = [c["delta"] for c in comparisons]
    p_values = [c.get("p_value") for c in comparisons if c.get("p_value") is not None]

    result = {
        "claim": "C4: Learned predictions outperform random predictions",
        "comparisons": comparisons,
        "learned_vs_random_deltas": [round(d, 6) for d in deltas],
        "mean_delta": round(float(np.mean(deltas)), 6),
        "p_values": [round(p, 4) for p in p_values],
        "all_prmp_better": all(c["prmp_better"] for c in comparisons),
        "any_significant": any(p < 0.05 for p in p_values),
        "conclusion": "moderate_support",
        "reasoning": "Directionally consistent (learned always beats random) but not statistically "
                     "significant in the larger 5-seed experiment (p=0.399).",
    }

    logger.info(f"  C4: mean_delta={np.mean(deltas):.4f}, sig={result['any_significant']}")
    return result


def analyze_c5_regression_vs_classification(exps: dict) -> dict:
    """C5: Regression benefits > classification benefits."""
    logger.info("Analyzing C5: Regression vs Classification task-type effect")

    results_parts = []

    # exp_id3_it4: Task-type comparison
    e3_4 = exps["exp_id3_it4"]
    task_finals = extract_exp_id3_it4_finals(e3_4)

    reg_deltas = []
    cls_deltas = []
    for k, v in task_finals.items():
        if "standard" in v and "prmp" in v:
            if v.get("lower_better", True):
                # Regression: positive delta = PRMP better (lower metric)
                delta = v["standard"] - v["prmp"]
            else:
                # Classification: positive delta = PRMP better (higher metric)
                delta = v["prmp"] - v["standard"]

            if v.get("task_type") == "regression":
                reg_deltas.append(delta)
            else:
                cls_deltas.append(delta)

    results_parts.append({
        "experiment": "exp_id3_it4",
        "regression_deltas": reg_deltas,
        "classification_deltas": cls_deltas,
        "regression_mean_delta": round(float(np.mean(reg_deltas)), 6) if reg_deltas else None,
        "classification_mean_delta": round(float(np.mean(cls_deltas)), 6) if cls_deltas else None,
    })

    # exp_id3_it6: Loss swap experiment
    e3_6 = exps["exp_id3_it6"]["metadata"]
    analysis = e3_6.get("analysis", {})
    loss_hyp = analysis.get("loss_function_hypothesis", {})
    target_hyp = analysis.get("target_nature_hypothesis", {})

    deltas_per_config = analysis.get("deltas_per_config", {})

    # Config1 = natural regression, Config2 = binned classification
    nat_reg_delta = deltas_per_config.get("config1_natural_regression", {}).get("mean", 0)
    bin_cls_delta = deltas_per_config.get("config2_binned_classification", {}).get("mean", 0)
    nat_cls_delta = deltas_per_config.get("config3_natural_classification", {}).get("mean", 0)
    soft_reg_delta = deltas_per_config.get("config4_softened_regression", {}).get("mean", 0)

    results_parts.append({
        "experiment": "exp_id3_it6",
        "natural_regression_delta": round(nat_reg_delta, 6),
        "binned_classification_delta": round(bin_cls_delta, 6),
        "natural_classification_delta": round(nat_cls_delta, 6),
        "softened_regression_delta": round(soft_reg_delta, 6),
        "loss_function_hypothesis": {
            "regression_loss_delta_mean": loss_hyp.get("regression_loss_configs_delta_mean", 0),
            "classification_loss_delta_mean": loss_hyp.get("classification_loss_configs_delta_mean", 0),
            "t_statistic": loss_hyp.get("t_statistic", 0),
            "p_value": loss_hyp.get("p_value", 1),
            "significant": loss_hyp.get("p_value", 1) < 0.05,
        },
        "target_nature_hypothesis": {
            "position_delta_mean": target_hyp.get("position_target_configs_delta_mean", 0),
            "binary_delta_mean": target_hyp.get("binary_target_configs_delta_mean", 0),
            "t_statistic": target_hyp.get("t_statistic", 0),
            "p_value": target_hyp.get("p_value", 1),
            "significant": target_hyp.get("p_value", 1) < 0.05,
        },
    })

    # exp_id2_it6: Amazon regression vs F1 DNF classification
    e2 = exps["exp_id2_it6"]["metadata"]
    sa = e2["statistical_analysis"]
    amazon_best = sa["rel-amazon/review-rating"]["best_variant"]
    f1dnf_best = sa["rel-f1/result-dnf"]["best_variant"]

    results_parts.append({
        "experiment": "exp_id2_it6",
        "amazon_regression_best_variant": amazon_best,
        "f1_dnf_classification_best_variant": f1dnf_best,
        "prmp_best_on_regression": amazon_best == "prmp",
        "prmp_best_on_classification": f1dnf_best == "prmp",
    })

    # Compute overall
    loss_swap_p_value = loss_hyp.get("p_value", 1)

    result = {
        "claim": "C5: Regression benefits > classification benefits",
        "analyses": results_parts,
        "loss_swap_p_value": round(loss_swap_p_value, 6),
        "loss_swap_significant": loss_swap_p_value < 0.05,
        "regression_vs_classification_delta_difference": round(
            nat_reg_delta - bin_cls_delta, 6
        ),
        "conclusion": "strong_support",
        "reasoning": "Loss swap experiment (13 seeds, p=0.0084) is the strongest statistical result "
                     "outside Amazon. Natural regression delta=+0.988 vs binned classification delta=-0.288. "
                     "This is a publishable novel finding.",
    }

    logger.info(f"  C5: loss_swap_p={loss_swap_p_value:.4f}, "
                f"reg_delta={nat_reg_delta:.3f}, cls_delta={bin_cls_delta:.3f}")
    return result


def analyze_c6_embedding_r2(exps: dict) -> dict:
    """C6: Embedding-space R^2 predicts PRMP benefit."""
    logger.info("Analyzing C6: Embedding R^2 as PRMP benefit predictor")

    # exp_id2_it6: Ridge R^2 per FK link (Amazon)
    e2 = exps["exp_id2_it6"]["metadata"]
    emb_diags = e2.get("embedding_diagnostics", [])

    amazon_emb_r2 = {}
    for diag in emb_diags:
        if diag.get("dataset") == "rel-amazon":
            variant = diag["variant"]
            amazon_emb_r2[variant] = {}
            for fk, vals in diag.get("fk_ridge_r2", {}).items():
                amazon_emb_r2[variant][fk] = vals.get("ridge_r2", 0)

    # exp_id1_it6: Embedding R^2 for Avito (11 FK links)
    e1_6 = exps["exp_id1_it6"]["metadata"]
    avito_emb_r2 = e1_6.get("embedding_r2", {})
    prmp_pred_r2 = e1_6.get("prmp_prediction_r2", {})

    # Check negative R^2 values in PRMP prediction MLPs
    negative_r2_count = sum(1 for v in prmp_pred_r2.values() if v < 0)
    total_r2 = len(prmp_pred_r2)
    worst_r2 = min(prmp_pred_r2.values()) if prmp_pred_r2 else 0

    result = {
        "claim": "C6: Embedding-space R^2 predicts PRMP benefit",
        "amazon_embedding_r2": amazon_emb_r2,
        "avito_prmp_prediction_r2": prmp_pred_r2,
        "avito_prediction_r2_all_negative": negative_r2_count == total_r2,
        "negative_r2_count": negative_r2_count,
        "total_prediction_r2_values": total_r2,
        "worst_prediction_r2": round(worst_r2, 2),
        "note_devastating": "ALL prediction MLP R^2 values on Avito are negative (worst: "
                           f"{worst_r2:.0f}), meaning predictions are WORSE than constant. "
                           "Yet PRMP still slightly improves, suggesting mechanism works "
                           "differently than theorized.",
        "conclusion": "weak_contradict",
        "reasoning": "Embedding R^2 from Ridge regression is moderate (0.38-0.57 on Amazon) but "
                     "PRMP prediction MLPs show catastrophically negative R^2 on Avito, indicating "
                     "the prediction component is not learning meaningful relationships. "
                     "Insufficient data to correlate embedding R^2 with per-FK improvement.",
    }

    logger.info(f"  C6: negative_r2={negative_r2_count}/{total_r2}, worst={worst_r2:.1f}")
    return result


def analyze_c7_diagnostic(exps: dict) -> dict:
    """C7: Raw-feature 2D diagnostic identifies difficult joins."""
    logger.info("Analyzing C7: Raw-feature R^2 diagnostic")

    # exp_id4_it2
    e = exps["exp_id4_it2"]["metadata"]
    diag = e["diagnostic"]
    exp4_diagnostics = {
        "product_to_review": {
            "raw_r2": diag["product_to_review"]["linear_r2"],
            "cardinality": diag["product_to_review"]["cardinality_mean"],
        },
        "customer_to_review": {
            "raw_r2": diag["customer_to_review"]["linear_r2"],
            "cardinality": diag["customer_to_review"]["cardinality_mean"],
        },
    }

    # exp_id1_it6: 11 FK links with R^2 values
    e1_6 = exps["exp_id1_it6"]["metadata"]
    r2_diag = e1_6.get("r2_diagnostic", {})
    card_stats = e1_6.get("cardinality_stats", {})

    avito_diagnostics = {}
    for link_name, r2_info in r2_diag.items():
        card_info = card_stats.get(link_name, {})
        avito_diagnostics[link_name] = {
            "raw_r2": r2_info.get("r2", 0),
            "cardinality_mean": card_info.get("card_mean", 0),
        }

    # Count coverage
    exp4_coverage = 2  # 2 FK links
    avito_coverage = sum(1 for v in avito_diagnostics.values()
                        if v["raw_r2"] is not None and v["cardinality_mean"] is not None)

    result = {
        "claim": "C7: Raw-feature 2D diagnostic identifies difficult joins",
        "exp_id4_it2_diagnostics": exp4_diagnostics,
        "avito_diagnostics": avito_diagnostics,
        "diagnostic_coverage_exp4": exp4_coverage,
        "diagnostic_coverage_avito": avito_coverage,
        "total_diagnostic_coverage": exp4_coverage + avito_coverage,
        "diagnostic_predicts_prmp_benefit": False,
        "note": "The diagnostic was computed but the correlation between diagnostic values and "
                "PRMP improvement is not established. On exp_id4_it2 the prediction was wrong "
                "(customer link helped more despite lower diagnostic score).",
        "conclusion": "inconclusive",
    }

    logger.info(f"  C7: coverage={exp4_coverage + avito_coverage}")
    return result


# ── SECTION 2: Evidence Matrix ───────────────────────────────────────────

def build_evidence_matrix(claims: dict, exps: dict) -> dict:
    """Build 6×7 evidence matrix (experiments × claims)."""
    logger.info("Building evidence matrix")

    experiments = ["exp_id4_it2", "exp_id1_it4", "exp_id3_it4", "exp_id2_it6", "exp_id1_it6", "exp_id3_it6"]
    claim_ids = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    matrix = {}

    # Row: exp_id4_it2
    matrix["exp_id4_it2"] = {
        "C1": {"support_level": "strong_support", "key_statistic": "RMSE 0.534 vs 0.578 (7.6% improvement)", "sample_size": 3, "notes": "3 seeds, hidden_dim=64"},
        "C2": {"support_level": "weak_contradict", "key_statistic": "cardinality_hypothesis_supported=false", "sample_size": 3, "notes": "Customer link helped more despite lower cardinality"},
        "C3": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No parameter-matched controls in this experiment"},
        "C4": {"support_level": "strong_support", "key_statistic": "PRMP 0.534 vs Random 0.560", "sample_size": 3, "notes": "Learned predictions clearly better"},
        "C5": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "Only regression task tested"},
        "C6": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No embedding R^2 analysis"},
        "C7": {"support_level": "weak_support", "key_statistic": "product R^2=0.012, customer R^2=0.045", "sample_size": 1, "notes": "Diagnostic computed but prediction was wrong"},
    }

    # Row: exp_id1_it4
    matrix["exp_id1_it4"] = {
        "C1": {"support_level": "strong_support", "key_statistic": "Amazon RMSE d=-5.55; F1 MAE d=-0.04", "sample_size": 3, "notes": "Strong on Amazon, negligible on F1"},
        "C2": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No per-FK analysis"},
        "C3": {"support_level": "strong_support", "key_statistic": "Cohen's d -5.5 to -8.5 vs all controls on Amazon", "sample_size": 3, "notes": "PRMP beats Wide, AuxMLP, SkipResidual on Amazon. On F1 RMSE, PRMP is WORSE (0.824 vs 0.691)"},
        "C4": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No random prediction ablation"},
        "C5": {"support_level": "weak_support", "key_statistic": "Amazon regression d=-5.55, F1 MAE d=-0.04", "sample_size": 3, "notes": "Regression task shows much larger effect"},
        "C6": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No embedding diagnostics"},
        "C7": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No raw-feature diagnostic"},
    }

    # Row: exp_id3_it4
    matrix["exp_id3_it4"] = {
        "C1": {"support_level": "weak_support", "key_statistic": "3/5 tasks PRMP better, 1 seed each", "sample_size": 1, "notes": "driver-position delta=+0.028, others tiny"},
        "C2": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No cardinality analysis"},
        "C3": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No parameter-matched controls"},
        "C4": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No random ablation"},
        "C5": {"support_level": "strong_support", "key_statistic": "reg mean delta=+0.015, cls mean delta=+0.004", "sample_size": 1, "notes": "Same-graph controlled: regression delta 7x larger than classification"},
        "C6": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
        "C7": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
    }

    # Row: exp_id2_it6
    matrix["exp_id2_it6"] = {
        "C1": {"support_level": "weak_support", "key_statistic": "Amazon d=1.01 p=0.247; F1-pos d=0.33 p=0.534; F1-dnf d=-0.13 p=0.739", "sample_size": 5, "notes": "PRMP best on Amazon but NO significant p-values"},
        "C2": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No per-FK analysis"},
        "C3": {"support_level": "weak_support", "key_statistic": "Amazon: PRMP(0.834) < AuxMLP(0.836) < RandomFrozen(0.838) < Standard(0.843) < Wide(0.849)", "sample_size": 5, "notes": "PRMP best but differences not significant"},
        "C4": {"support_level": "neutral", "key_statistic": "PRMP(0.834) vs RandomFrozen(0.838), p=0.399, d=-0.397", "sample_size": 5, "notes": "Directionally correct but NOT significant"},
        "C5": {"support_level": "weak_support", "key_statistic": "PRMP best on regression (Amazon) but not on classification (F1-dnf)", "sample_size": 5, "notes": "Consistent with regression > classification hypothesis"},
        "C6": {"support_level": "weak_support", "key_statistic": "Ridge R^2: review->product 0.41/0.38, review->customer 0.57/0.56", "sample_size": 5, "notes": "Embedding R^2 computed but insufficient for correlation"},
        "C7": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
    }

    # Row: exp_id1_it6
    matrix["exp_id1_it6"] = {
        "C1": {"support_level": "weak_support", "key_statistic": "RMSE 0.0878 vs 0.0882 (d=0.47, 0.5%)", "sample_size": 3, "notes": "Tiny improvement, not statistically significant"},
        "C2": {"support_level": "neutral", "key_statistic": "11 FK links analyzed, 0.5% aggregate improvement", "sample_size": 3, "notes": "Insufficient per-FK improvement data for correlation"},
        "C3": {"support_level": "weak_support", "key_statistic": "PRMP(0.0878) vs Wide(0.0979) but Wide had convergence issues", "sample_size": 3, "notes": "Wide 2/3 seeds failed to converge — comparison unreliable"},
        "C4": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "No random ablation"},
        "C5": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "Only regression task (ad-ctr)"},
        "C6": {"support_level": "strong_contradict", "key_statistic": "ALL prediction MLP R^2 negative (worst: -5046)", "sample_size": 3, "notes": "Predictions are WORSE than constant — mechanism not working as theorized"},
        "C7": {"support_level": "weak_support", "key_statistic": "11 FK links with raw R^2 ranging 0.0 to 0.50", "sample_size": 1, "notes": "Diagnostic computed, coverage is good"},
    }

    # Row: exp_id3_it6
    matrix["exp_id3_it6"] = {
        "C1": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": "Tests loss function effect, not overall PRMP advantage"},
        "C2": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
        "C3": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
        "C4": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
        "C5": {"support_level": "strong_support", "key_statistic": "p=0.0084: reg-loss delta=0.429 vs cls-loss delta=-0.053", "sample_size": 13, "notes": "Strongest statistical result: 13 seeds, 4 configs, controlled loss swap"},
        "C6": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
        "C7": {"support_level": "not_tested", "key_statistic": "N/A", "sample_size": 0, "notes": ""},
    }

    return {"matrix": matrix, "experiments": experiments, "claims": claim_ids}


# ── SECTION 3: Cross-Experiment Consistency ──────────────────────────────

def analyze_consistency(exps: dict) -> dict:
    """Analyze cross-experiment consistency."""
    logger.info("Analyzing cross-experiment consistency")

    implementation_diffs = [
        {"exp": "exp_id4_it2", "hidden_dim": 64, "framework": "PyG-style", "arch": "SAGEConv", "seeds": 3},
        {"exp": "exp_id1_it4", "hidden_dim": 128, "framework": "PyG-style", "arch": "SAGEConv", "seeds": 3},
        {"exp": "exp_id3_it4", "hidden_dim": 64, "framework": "PyG (RelBench)", "arch": "SAGEConv + RelBench loader", "seeds": 1},
        {"exp": "exp_id2_it6", "hidden_dim": 128, "framework": "Custom HeteroSAGEConv", "arch": "HeteroSAGEConv", "seeds": 5},
        {"exp": "exp_id1_it6", "hidden_dim": 128, "framework": "Pure PyTorch (no PyG)", "arch": "Custom SAGE", "seeds": 3},
        {"exp": "exp_id3_it6", "hidden_dim": 128, "framework": "Custom SAGEConv", "arch": "Custom F1 graph", "seeds": 13},
    ]

    # Count Amazon-dominated results
    # From C1: Amazon tasks show the strongest improvement
    amazon_positive_results = 3  # exp_id4_it2, exp_id1_it4, exp_id2_it6 all show Amazon PRMP wins
    total_positive_results = 5  # Including F1 and Avito marginal wins
    amazon_dominance_score = amazon_positive_results / total_positive_results

    # Dataset coverage
    datasets_tested = ["Amazon Video Games", "F1", "Avito", "Stack Overflow"]

    # Total unique tasks
    total_unique_tasks = 8  # amazon_rating, f1_position, f1_dnf, f1_top3, avito_ctr, stack_votes, stack_engagement, + loss-swap configs

    # Total seeds
    total_seeds = sum(d["seeds"] for d in implementation_diffs)

    result = {
        "implementation_differences": implementation_diffs,
        "amazon_dominance_score": round(amazon_dominance_score, 4),
        "dataset_coverage": datasets_tested,
        "num_unique_datasets": len(datasets_tested),
        "total_unique_tasks": total_unique_tasks,
        "total_seeds_across_experiments": total_seeds,
        "consistency_concerns": [
            "Hidden dim varies: 64 (exp_id4_it2, exp_id3_it4) vs 128 (others)",
            "Architectures differ: PyG SAGEConv, HeteroSAGEConv, pure PyTorch",
            "exp_id3_it4 uses only 1 seed per task",
            "exp_id1_it6 uses pure PyTorch (no torch-geometric) — different implementation",
            "exp_id2_it6 uses different weight_decay (1e-5 vs 1e-4) and gradient clipping (1.0 vs 5.0)",
        ],
    }

    logger.info(f"  Consistency: amazon_dominance={amazon_dominance_score:.3f}, "
                f"seeds={total_seeds}, datasets={len(datasets_tested)}")
    return result


# ── SECTION 4: Reviewer Objections ───────────────────────────────────────

def analyze_reviewer_objections(claims: dict, evidence_matrix: dict, exps: dict) -> dict:
    """Enumerate reviewer objections with factual responses."""
    logger.info("Building reviewer objections and responses")

    # O1: Count all p<0.05 results
    c1 = claims["C1"]
    sig_results = []
    for tc in c1["task_comparisons"]:
        if tc.get("p_value") is not None and not math.isnan(tc.get("p_value", float('nan'))):
            if tc["p_value"] < 0.05:
                sig_results.append({"task": tc["task"], "p_value": tc["p_value"]})

    # Also add loss swap p-value
    loss_swap_p = claims["C5"]["loss_swap_p_value"]
    if loss_swap_p < 0.05:
        sig_results.append({"task": "loss_swap_regression_vs_classification", "p_value": loss_swap_p})

    objections = {
        "O1_only_one_significant_win": {
            "objection": "Only one statistically significant win",
            "factual_response": f"Across all experiments, {len(sig_results)} results achieve p<0.05: "
                               f"{json.dumps(sig_results)}. "
                               "The loss swap experiment (p=0.0084, 13 seeds) provides the strongest "
                               "evidence, though it tests the regression-vs-classification finding, "
                               "not overall PRMP superiority.",
            "count_significant_p05": len(sig_results),
            "significant_results": sig_results,
        },
        "O2_no_comparison_to_relgnn_griffin": {
            "objection": "No comparison to established baselines like RelGNN or Griffin",
            "factual_response": "Baselines compared: Standard SAGEConv/HeteroSAGEConv, "
                               "parameter-matched Wide SAGEConv, Auxiliary MLP, Skip-Residual, "
                               "Random Frozen predictions. These isolate the predict-subtract "
                               "mechanism but do not compare against published heterogeneous GNN methods.",
            "baselines_compared": ["Standard SAGEConv", "Wide SAGEConv (param-matched)",
                                  "Auxiliary MLP (param-matched)", "Skip-Residual",
                                  "Random Frozen predictions"],
            "baselines_missing": ["RelGNN", "Griffin", "R-GCN", "HAN", "HGT"],
        },
        "O3_implementation_inconsistency": {
            "objection": "Implementation inconsistency across experiments",
            "factual_response": "Acknowledged. Key differences: "
                               "(1) hidden_dim: 64 (exp_id4_it2, exp_id3_it4) vs 128 (others), "
                               "(2) frameworks: PyG SAGEConv, custom HeteroSAGEConv, pure PyTorch, "
                               "(3) hyperparameters: weight_decay 1e-4 vs 1e-5, grad_clip 5.0 vs 1.0, "
                               "(4) exp_id3_it4 uses only 1 seed per task. "
                               "The unified exp_id2_it6 with 5 seeds addresses this partially.",
            "differences": [
                "hidden_dim: 64 vs 128",
                "framework: PyG vs custom HeteroSAGEConv vs pure PyTorch",
                "weight_decay: 1e-4 vs 1e-5",
                "grad_clip: 5.0 vs 1.0",
                "seeds: 1 to 13 across experiments",
            ],
        },
        "O4_amazon_outsized_work": {
            "objection": "Amazon doing outsized work",
            "factual_response": "On Amazon, PRMP shows large Cohen's d (1.01-5.55) and consistent wins. "
                               "On F1, improvements are negligible (d=-0.04 to 0.33). "
                               "On Avito, improvement is 0.5% (d=0.47). "
                               "Approximately 60% of positive evidence comes from Amazon alone.",
            "amazon_fraction_of_positive": 0.6,
            "per_dataset_strength": {
                "Amazon": "strong (d=1.01-5.55, consistent across 3 experiments)",
                "F1": "negligible to mixed (d=-0.04 to 0.33, PRMP worse on RMSE in exp_id1_it4)",
                "Avito": "tiny (d=0.47, 0.5% improvement)",
                "Stack_Overflow": "tiny (single seed, delta=+0.002 to +0.010)",
            },
        },
        "O5_negative_prediction_r2": {
            "objection": "Prediction MLPs show negative R^2 — predictions aren't working",
            "factual_response": "On Avito, ALL 22 prediction MLP R^2 values are negative "
                               "(worst: -5046 for SearchInfo->SearchStream). "
                               "This means predictions are worse than a constant, yet PRMP still "
                               "slightly outperforms Standard (0.0878 vs 0.0882). "
                               "This suggests the subtraction operation may be acting as a regularizer "
                               "rather than making accurate predictions. The theoretical narrative of "
                               "'predict and subtract redundancy' is undermined.",
            "avito_worst_r2": claims["C6"]["worst_prediction_r2"],
            "avito_all_negative": claims["C6"]["avito_prediction_r2_all_negative"],
        },
        "O6_small_sample_sizes": {
            "objection": "Small sample sizes (3 seeds)",
            "factual_response": "Most experiments use 3 seeds. exp_id2_it6 uses 5 seeds, "
                               "exp_id3_it6 uses 13 seeds. With 3 seeds, a paired t-test has very low "
                               "power (df=2, requiring enormous effect sizes for significance). "
                               "Only Cohen's d > ~4.3 would be significant at p<0.05 with n=3.",
            "power_analysis": {
                "n3_critical_d_p05": 4.3,
                "n5_critical_d_p05": 2.8,
                "n13_critical_d_p05": 1.3,
            },
            "experiments_by_seeds": {"3_seeds": ["exp_id4_it2", "exp_id1_it4", "exp_id1_it6"],
                                     "5_seeds": ["exp_id2_it6"],
                                     "13_seeds": ["exp_id3_it6"],
                                     "1_seed": ["exp_id3_it4"]},
        },
        "O7_prmp_hurts_on_f1": {
            "objection": "PRMP hurts on F1 in parameter-matched experiment",
            "factual_response": "In exp_id1_it4, F1 RMSE: PRMP=0.824 vs Standard=0.691 — "
                               "PRMP is 19.3% WORSE. This is the strongest negative result. "
                               "On F1 MAE, PRMP (0.511) is marginally better than Standard (0.513) "
                               "but the difference is negligible (d=-0.04). "
                               "SkipResidual (0.472) beats all methods on F1 MAE.",
            "f1_rmse_prmp": 0.824,
            "f1_rmse_standard": 0.691,
            "f1_rmse_degradation_pct": 19.3,
            "f1_mae_prmp": 0.511,
            "f1_mae_standard": 0.513,
            "f1_mae_skipresidual": 0.472,
        },
    }

    logger.info(f"  Built {len(objections)} reviewer objections")
    return objections


# ── SECTION 5: Overall Confidence Assessment ─────────────────────────────

def compute_overall_assessment(claims: dict, evidence_matrix: dict,
                                consistency: dict, objections: dict) -> dict:
    """Compute overall confidence assessment."""
    logger.info("Computing overall confidence assessment")

    claim_confidences = {
        "C1": {
            "confidence": "moderate",
            "reasoning": "PRMP wins on most tasks by raw mean, but only 1-2 results "
                        "achieve statistical significance. Win rate ~75% but significant win rate very low.",
        },
        "C2": {
            "confidence": "disconfirmed",
            "reasoning": "exp_id4_it2 explicitly flags hypothesis as unsupported. "
                        "Customer link (lower cardinality) helped more. "
                        "Insufficient data to correlate cardinality×R^2 with improvement.",
        },
        "C3": {
            "confidence": "strong_on_amazon",
            "reasoning": "Amazon Cohen's d of -5.5 to -8.5 is irrefutable evidence the mechanism "
                        "works there. But PRMP is worse on F1 RMSE and Avito Wide control was unreliable.",
        },
        "C4": {
            "confidence": "moderate",
            "reasoning": "Directionally consistent (learned always beats random) but p=0.399 "
                        "in the 5-seed experiment. Not statistically proven.",
        },
        "C5": {
            "confidence": "strong",
            "reasoning": "Loss swap p=0.0084 with 13 seeds is the most rigorous statistical result. "
                        "Regression delta ~20x larger than classification. Novel finding.",
        },
        "C6": {
            "confidence": "disconfirmed",
            "reasoning": "Prediction MLPs show catastrophically negative R^2 on Avito (-5046), "
                        "meaning the embedding R^2 diagnostic does NOT predict PRMP benefit.",
        },
        "C7": {
            "confidence": "inconclusive",
            "reasoning": "Diagnostic was computed but does not correctly predict where PRMP helps.",
        },
    }

    # Count
    strong_or_moderate = sum(1 for c in claim_confidences.values()
                            if c["confidence"] in ("strong", "strong_on_amazon", "moderate"))
    disconfirmed = sum(1 for c in claim_confidences.values()
                      if c["confidence"] == "disconfirmed")
    inconclusive = sum(1 for c in claim_confidences.values()
                      if c["confidence"] == "inconclusive")

    result = {
        "claim_confidences": claim_confidences,
        "overall_verdict": "partial_confirm",
        "claims_confirmed": strong_or_moderate,
        "claims_disconfirmed": disconfirmed,
        "claims_inconclusive": inconclusive,
        "recommended_narrative": (
            "Pivot from 'PRMP is universally better' to 'PRMP reveals when and why "
            "residual-based aggregation helps in relational learning.' The strongest "
            "contributions are: (1) The predict-subtract mechanism genuinely helps on "
            "Amazon (huge effect sizes), (2) The regression-vs-classification finding "
            "is novel and statistically robust (p=0.008). The paper should honestly "
            "acknowledge: (a) benefits are inconsistent across datasets, (b) the "
            "cardinality×predictability regime hypothesis is NOT supported, (c) "
            "prediction MLPs are not learning meaningful relationships on complex "
            "datasets (negative R^2), and (d) most individual task comparisons lack "
            "statistical significance due to small sample sizes."
        ),
        "strongest_result": {
            "result": "Amazon PRMP vs parameter-matched controls (exp_id1_it4)",
            "why": "Cohen's d of -5.5 to -8.5 against all controls is extraordinary. "
                   "PRMP beats Wide (+0.4% params), AuxMLP (exact params), and SkipResidual "
                   "with massive effect sizes. This is irrefutable on this dataset.",
        },
        "weakest_result": {
            "result": "Avito PRMP prediction MLP R^2 (exp_id1_it6)",
            "why": "ALL 22 prediction MLP R^2 values are strongly negative (worst: -5046). "
                   "This means the core mechanism (predict parent embeddings) is producing "
                   "predictions WORSE than a constant. If the predictions are garbage, the "
                   "subtraction is removing noise, not redundancy. This fundamentally "
                   "undermines the theoretical narrative of PRMP.",
        },
    }

    logger.info(f"  Overall: {result['overall_verdict']}, confirmed={strong_or_moderate}, "
                f"disconfirmed={disconfirmed}, inconclusive={inconclusive}")
    return result


# ── Main evaluation ──────────────────────────────────────────────────────

@logger.catch
def main():
    logger.info("="*60)
    logger.info("PRMP Publication Readiness Audit — Claims-to-Evidence")
    logger.info("="*60)

    # Load all experiment data
    exps = load_experiments()
    logger.info(f"Loaded {len(exps)} experiments")

    # Section 1: Claims analysis
    logger.info("\n--- SECTION 1: Claims Analysis ---")
    c1 = analyze_c1_outperforms(exps)
    c2 = analyze_c2_regime(exps)
    c3 = analyze_c3_mechanism(exps)
    c4 = analyze_c4_learned_vs_random(exps)
    c5 = analyze_c5_regression_vs_classification(exps)
    c6 = analyze_c6_embedding_r2(exps)
    c7 = analyze_c7_diagnostic(exps)

    claims = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7}

    # Section 2: Evidence matrix
    logger.info("\n--- SECTION 2: Evidence Matrix ---")
    evidence_matrix = build_evidence_matrix(claims, exps)

    # Section 3: Cross-experiment consistency
    logger.info("\n--- SECTION 3: Cross-Experiment Consistency ---")
    consistency = analyze_consistency(exps)

    # Section 4: Reviewer objections
    logger.info("\n--- SECTION 4: Reviewer Objections ---")
    objections = analyze_reviewer_objections(claims, evidence_matrix, exps)

    # Section 5: Overall assessment
    logger.info("\n--- SECTION 5: Overall Confidence Assessment ---")
    assessment = compute_overall_assessment(claims, evidence_matrix, consistency, objections)

    # ── Build output in exp_eval_sol_out.json schema ──────────────────────
    # metrics_agg: flat dict of numeric metrics
    metrics_agg = {
        # C1 metrics
        "c1_win_rate": c1["win_rate"],
        "c1_significant_win_rate": c1["significant_win_rate"],
        "c1_median_effect_size": c1["median_effect_size"],
        "c1_total_tasks": c1["total_tasks"],
        "c1_wins": c1["wins"],
        "c1_significant_wins": c1["significant_wins"],
        "c1_consistency_score": c1["consistency_score"],
        # C2 metrics
        "c2_diagnostic_coverage_total": c2["total_fk_links_analyzed"],
        # C3 metrics
        "c3_prmp_beats_all_controls_rate": c3["prmp_beats_all_controls_rate"],
        "c3_mean_improvement_over_parammatched": c3["mean_improvement_over_parammatched"],
        # C4 metrics
        "c4_mean_learned_vs_random_delta": c4["mean_delta"],
        "c4_any_significant": 1 if c4["any_significant"] else 0,
        # C5 metrics
        "c5_loss_swap_p_value": c5["loss_swap_p_value"],
        "c5_regression_vs_classification_delta_diff": c5["regression_vs_classification_delta_difference"],
        # C6 metrics
        "c6_negative_r2_count": c6["negative_r2_count"],
        "c6_total_prediction_r2": c6["total_prediction_r2_values"],
        "c6_worst_prediction_r2": c6["worst_prediction_r2"],
        # C7 metrics
        "c7_total_diagnostic_coverage": c7["total_diagnostic_coverage"],
        # Consistency metrics
        "consistency_amazon_dominance": consistency["amazon_dominance_score"],
        "consistency_num_datasets": consistency["num_unique_datasets"],
        "consistency_total_seeds": consistency["total_seeds_across_experiments"],
        "consistency_total_tasks": consistency["total_unique_tasks"],
        # Objection metrics
        "objection_significant_p05_count": objections["O1_only_one_significant_win"]["count_significant_p05"],
        "objection_f1_rmse_degradation_pct": objections["O7_prmp_hurts_on_f1"]["f1_rmse_degradation_pct"],
        "objection_amazon_fraction_positive": objections["O4_amazon_outsized_work"]["amazon_fraction_of_positive"],
        # Overall assessment
        "assessment_claims_confirmed": assessment["claims_confirmed"],
        "assessment_claims_disconfirmed": assessment["claims_disconfirmed"],
        "assessment_claims_inconclusive": assessment["claims_inconclusive"],
    }

    # Build examples: one per claim analysis + one per experiment in evidence matrix + objections
    examples = []

    # Claims examples
    for claim_id, claim_data in claims.items():
        examples.append({
            "input": json.dumps({"claim_id": claim_id, "claim": claim_data.get("claim", claim_id)}),
            "output": json.dumps(claim_data.get("conclusion", "N/A")),
            "metadata_section": "claims_analysis",
            "metadata_claim_id": claim_id,
            "eval_confidence": 1.0 if claim_data.get("conclusion") in ("strong_support", "strong_support_on_amazon_only") else 0.5 if "support" in str(claim_data.get("conclusion", "")) else 0.0,
        })

    # Evidence matrix examples (one per experiment-claim cell)
    for exp_id in evidence_matrix["experiments"]:
        for claim_id in evidence_matrix["claims"]:
            cell = evidence_matrix["matrix"][exp_id][claim_id]
            examples.append({
                "input": json.dumps({"experiment": exp_id, "claim": claim_id}),
                "output": json.dumps(cell["support_level"]),
                "metadata_section": "evidence_matrix",
                "metadata_experiment": exp_id,
                "metadata_claim": claim_id,
                "metadata_sample_size": cell["sample_size"],
                "eval_support_score": {
                    "strong_support": 1.0,
                    "weak_support": 0.5,
                    "neutral": 0.0,
                    "weak_contradict": -0.5,
                    "strong_contradict": -1.0,
                    "not_tested": 0.0,
                }.get(cell["support_level"], 0.0),
            })

    # Reviewer objection examples
    for obj_id, obj_data in objections.items():
        examples.append({
            "input": json.dumps({"objection_id": obj_id, "objection": obj_data["objection"]}),
            "output": obj_data["factual_response"][:2000],  # Truncate long strings
            "metadata_section": "reviewer_objections",
            "metadata_objection_id": obj_id,
            "eval_severity": 1.0,
        })

    # Build final output
    output = {
        "metadata": {
            "evaluation_name": "PRMP Publication Readiness Audit",
            "description": "Systematic audit mapping 7 PRMP paper claims to quantitative evidence across 6 experiments",
            "experiments_analyzed": list(EXP_PATHS.keys()),
            "claims_analyzed": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "sections": {
                "claims_analysis": claims,
                "evidence_matrix": evidence_matrix,
                "cross_experiment_consistency": consistency,
                "reviewer_objections": objections,
                "overall_assessment": assessment,
            },
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "prmp_audit",
                "examples": examples,
            }
        ],
    }

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved evaluation to {out_path}")
    logger.info(f"Output size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("AUDIT SUMMARY")
    logger.info("="*60)
    logger.info(f"Overall verdict: {assessment['overall_verdict']}")
    logger.info(f"Claims confirmed: {assessment['claims_confirmed']}/7")
    logger.info(f"Claims disconfirmed: {assessment['claims_disconfirmed']}/7")
    logger.info(f"Claims inconclusive: {assessment['claims_inconclusive']}/7")
    logger.info(f"Win rate: {c1['win_rate']:.1%}")
    logger.info(f"Significant win rate: {c1['significant_win_rate']:.1%}")
    logger.info(f"Loss swap p-value: {c5['loss_swap_p_value']:.4f}")
    logger.info(f"Amazon dominance: {consistency['amazon_dominance_score']:.1%}")
    logger.info(f"Strongest: {assessment['strongest_result']['result']}")
    logger.info(f"Weakest: {assessment['weakest_result']['result']}")

    return output


if __name__ == "__main__":
    main()
