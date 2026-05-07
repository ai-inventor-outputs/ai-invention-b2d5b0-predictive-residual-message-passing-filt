#!/usr/bin/env python3
"""Cross-Experiment PRMP Meta-Analysis: Effect Sizes, Forest Plots & Moderator Analysis.

Pools PRMP vs baseline results across 5 experiments (4 datasets, 8+ tasks).
Computes standardized effect sizes (Hedges' g), random-effects meta-analytic summary,
subgroup/moderator analyses, and produces publication-ready forest plots, funnel plots,
and summary tables.
"""

import json
import math
import sys
import os
import gc
import resource
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ============================================================
# Setup
# ============================================================
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# --- Hardware detection & memory limits ---
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

def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    import psutil
    return psutil.virtual_memory().total / 1e9

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET_BYTES = int(min(TOTAL_RAM_GB * 0.4, 20) * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except Exception:
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_BYTES/1e9:.1f} GB")

# --- Dependency paths ---
DEP_BASE = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/3_invention_loop")
EXP_AMAZON = DEP_BASE / "iter_2/gen_art/exp_id4_it2__opus"
EXP_RELHM_IT3 = DEP_BASE / "iter_3/gen_art/exp_id1_it3__opus"
EXP_RELF1 = DEP_BASE / "iter_3/gen_art/exp_id2_it3__opus"
EXP_RELSTACK = DEP_BASE / "iter_3/gen_art/exp_id3_it3__opus"
EXP_RELHM_IT2 = DEP_BASE / "iter_2/gen_art/exp_id2_it2__opus"

# ============================================================
# Data structure
# ============================================================
@dataclass
class TaskResult:
    """Per-task comparison results for meta-analysis."""
    dataset: str
    task: str
    metric_name: str
    task_type: str          # "regression" or "classification"
    higher_is_better: bool
    n_seeds: int
    baseline_mean: float
    baseline_std: float
    prmp_mean: float
    prmp_std: float
    baseline_values: Optional[List[float]] = None
    prmp_values: Optional[List[float]] = None
    random_mean: Optional[float] = None
    random_std: Optional[float] = None
    random_values: Optional[List[float]] = None
    mean_fk_cardinality: float = 0.0
    mean_cross_table_r2: float = 0.0
    is_saturated: bool = False

# ============================================================
# Statistical functions
# ============================================================
def compute_hedges_g(
    m_prmp: float, s_prmp: float, n_prmp: int,
    m_base: float, s_base: float, n_base: int,
    higher_is_better: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Compute Hedges' g (positive = PRMP better), SE, 95% CI, p-value."""
    df = n_prmp + n_base - 2
    if df <= 0:
        return 0.0, float('inf'), float('-inf'), float('inf'), 1.0
    if s_prmp == 0 and s_base == 0:
        # No variance — cannot compute standardized effect
        raw_diff = (m_prmp - m_base) if higher_is_better else (m_base - m_prmp)
        direction = 1.0 if raw_diff > 0 else (-1.0 if raw_diff < 0 else 0.0)
        return direction * 0.0, float('inf'), float('-inf'), float('inf'), 1.0

    s_pooled_sq = ((n_prmp - 1) * s_prmp**2 + (n_base - 1) * s_base**2) / df
    s_pooled = math.sqrt(max(s_pooled_sq, 1e-30))

    if higher_is_better:
        d = (m_prmp - m_base) / s_pooled
    else:
        d = (m_base - m_prmp) / s_pooled

    # Hedges' correction
    J = 1 - 3 / (4 * df - 1)
    g = d * J

    se = math.sqrt((n_prmp + n_base) / (n_prmp * n_base) + g**2 / (2 * df))
    ci_lower = g - 1.96 * se
    ci_upper = g + 1.96 * se

    if se > 0 and math.isfinite(se):
        z = g / se
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    else:
        p_value = 1.0
    return g, se, ci_lower, ci_upper, p_value


def dersimonian_laird(effects: List[float], variances: List[float]) -> Dict:
    """DerSimonian-Laird random-effects meta-analysis."""
    k = len(effects)
    if k == 0:
        return {"pooled_g": 0, "se": 0, "ci_lower": 0, "ci_upper": 0,
                "p_value": 1, "z": 0, "Q": 0, "Q_p_value": 1, "I2": 0, "tau2": 0, "k": 0}

    eff = np.array(effects, dtype=float)
    var = np.array(variances, dtype=float)
    var = np.clip(var, 1e-10, None)  # avoid div-by-zero

    w = 1.0 / var
    w_sum = np.sum(w)
    g_fixed = np.sum(w * eff) / w_sum
    Q = float(np.sum(w * (eff - g_fixed)**2))

    C = w_sum - np.sum(w**2) / w_sum
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

    w_star = 1.0 / (var + tau2)
    w_star_sum = np.sum(w_star)
    g_pooled = float(np.sum(w_star * eff) / w_star_sum)
    se_pooled = float(1.0 / math.sqrt(w_star_sum))

    ci_lo = g_pooled - 1.96 * se_pooled
    ci_hi = g_pooled + 1.96 * se_pooled
    z = g_pooled / se_pooled if se_pooled > 0 else 0.0
    p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    I2 = float(max(0, (Q - (k - 1)) / Q * 100)) if Q > 0 else 0.0
    Q_p = float(1 - sp_stats.chi2.cdf(Q, k - 1)) if k > 1 else 1.0

    return {"pooled_g": g_pooled, "se": se_pooled, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "p_value": p_val, "z": z, "Q": Q, "Q_p_value": Q_p, "I2": I2, "tau2": tau2, "k": k}


def wls_regression(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> Dict:
    """Weighted least squares meta-regression."""
    W = np.diag(w)
    k, p = X.shape
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX, XtWy)
        residuals = y - X @ beta
        sigma2 = float(residuals.T @ W @ residuals / max(k - p, 1))
        cov_beta = np.linalg.inv(XtWX) * max(sigma2, 1e-10)
        se_beta = np.sqrt(np.abs(np.diag(cov_beta)))
        z_vals = beta / np.where(se_beta > 0, se_beta, 1e-10)
        p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z_vals)))
        return {"coefficients": beta.tolist(), "standard_errors": se_beta.tolist(),
                "z_values": z_vals.tolist(), "p_values": p_vals.tolist()}
    except np.linalg.LinAlgError:
        logger.warning("WLS regression failed (singular matrix)")
        return {"coefficients": [0.0]*p, "standard_errors": [1e6]*p,
                "z_values": [0.0]*p, "p_values": [1.0]*p}

# ============================================================
# Data extraction
# ============================================================
def extract_amazon_results() -> TaskResult:
    """Amazon Video Games (exp_id4_it2__opus) — RMSE, 3 seeds."""
    logger.info("Extracting Amazon Video Games results...")
    data = json.loads((EXP_AMAZON / "preview_method_out.json").read_text())
    meta = data["metadata"]
    res = meta["results"]
    diag = meta["diagnostic"]

    bl = res["standard_sage"]["rmse"]
    pm = res["prmp"]["rmse"]
    rn = res["ablation_random_pred"]["rmse"]

    card_p = diag["product_to_review"]["cardinality_mean"]
    card_c = diag["customer_to_review"]["cardinality_mean"]
    r2_p = max(0.0, diag["product_to_review"]["linear_r2"])
    r2_c = max(0.0, diag["customer_to_review"]["linear_r2"])

    return TaskResult(
        dataset="Amazon Video Games", task="amazon-rating", metric_name="RMSE",
        task_type="regression", higher_is_better=False, n_seeds=3,
        baseline_mean=bl["mean"], baseline_std=bl["std"],
        prmp_mean=pm["mean"], prmp_std=pm["std"],
        random_mean=rn["mean"], random_std=rn["std"],
        mean_fk_cardinality=(card_p + card_c) / 2,
        mean_cross_table_r2=(r2_p + r2_c) / 2,
        is_saturated=False,
    )


def extract_relf1_results() -> List[TaskResult]:
    """rel-f1 (exp_id2_it3__opus) — 3 tasks, 3 seeds each."""
    logger.info("Extracting rel-f1 results...")
    data = json.loads((EXP_RELF1 / "preview_method_out.json").read_text())
    meta = data["metadata"]
    bench = meta["benchmark_summary"]
    diag_list = meta.get("diagnostic_results", [])

    cards = [d["cardinality_mean"] for d in diag_list]
    r2s = [max(0.0, d.get("nonlinear_r2_mean", 0.0)) for d in diag_list]
    mean_card = float(np.mean(cards)) if cards else 0.0
    mean_r2 = float(np.mean(r2s)) if r2s else 0.0

    cfgs = [
        ("rel-f1-driver-dnf", "AP", "classification", True),
        ("rel-f1-driver-top3", "AP", "classification", True),
        ("rel-f1-driver-position", "MAE", "regression", False),
    ]
    results = []
    for task_name, metric_name, task_type, higher in cfgs:
        sage = bench[f"{task_name}__sage"]
        prmp = bench[f"{task_name}__prmp"]
        rand = bench[f"{task_name}__random_pred"]
        is_sat = higher and sage["test_mean"] > 0.99

        results.append(TaskResult(
            dataset="rel-f1", task=task_name, metric_name=metric_name,
            task_type=task_type, higher_is_better=higher, n_seeds=3,
            baseline_mean=sage["test_mean"], baseline_std=sage["test_std"],
            prmp_mean=prmp["test_mean"], prmp_std=prmp["test_std"],
            baseline_values=sage["test_values"], prmp_values=prmp["test_values"],
            random_mean=rand["test_mean"], random_std=rand["test_std"],
            random_values=rand["test_values"],
            mean_fk_cardinality=mean_card, mean_cross_table_r2=mean_r2,
            is_saturated=is_sat,
        ))
    return results


def extract_relhm_it3_results() -> List[TaskResult]:
    """rel-hm iter3 (exp_id1_it3__opus) — user-churn + item-sales, 3 seeds.

    Computes per-seed AUROC / MAE from per-instance predictions.
    """
    logger.info("Extracting rel-hm (iter3) results from full data...")
    full_path = EXP_RELHM_IT3 / "full_method_out.json"
    data = json.loads(full_path.read_text())
    examples = data["datasets"][0]["examples"]
    logger.info(f"  Loaded {len(examples)} per-instance examples")

    # Group by (task, seed) — skip aggregate/summary rows
    groups: Dict[Tuple, Dict] = defaultdict(lambda: {"labels": [], "bl": [], "pm": [], "rn": []})
    skipped = 0
    for ex in examples:
        inp = json.loads(ex["input"])
        # Skip non-instance rows (summaries, aggregates)
        if inp.get("type") in ("run_summary", "aggregate_summary"):
            skipped += 1
            continue
        if "test_instance_idx" not in inp:
            skipped += 1
            continue

        task = inp.get("task", ex.get("metadata_task"))
        seed = inp.get("seed", ex.get("metadata_seed"))
        metric = inp.get("metric", ex.get("metadata_metric"))
        task_type = inp.get("task_type", ex.get("metadata_task_type", "unknown"))

        out = json.loads(ex["output"])
        label = float(out.get("label", out.get("value", 0)))

        bl = float(ex["predict_baseline"])
        pm = float(ex["predict_our_method"])
        rn_str = ex.get("predict_ablation_random", "0")
        rn = float(rn_str) if rn_str else 0.0

        key = (task, seed, metric, task_type)
        groups[key]["labels"].append(label)
        groups[key]["bl"].append(bl)
        groups[key]["pm"].append(pm)
        groups[key]["rn"].append(rn)
    logger.info(f"  Grouped {len(examples) - skipped} instances, skipped {skipped} summary rows")

    # Compute per-seed metrics
    task_seed_metrics: Dict[Tuple, Dict] = defaultdict(lambda: {"bl": [], "pm": [], "rn": []})
    for (task, seed, metric, task_type), vals in groups.items():
        labels = np.array(vals["labels"])
        bl_pred = np.array(vals["bl"])
        pm_pred = np.array(vals["pm"])
        rn_pred = np.array(vals["rn"])

        if metric == "auroc":
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logger.warning(f"  Skipping {task} seed={seed}: only one class in labels")
                continue
            try:
                bl_s = roc_auc_score(labels, bl_pred)
                pm_s = roc_auc_score(labels, pm_pred)
                rn_s = roc_auc_score(labels, rn_pred)
            except ValueError as e:
                logger.warning(f"  AUROC error for {task} seed={seed}: {e}")
                continue
        elif metric == "mae":
            bl_s = float(np.mean(np.abs(labels - bl_pred)))
            pm_s = float(np.mean(np.abs(labels - pm_pred)))
            rn_s = float(np.mean(np.abs(labels - rn_pred)))
        else:
            logger.warning(f"  Unknown metric '{metric}' for {task}")
            continue

        key = (task, metric, task_type)
        task_seed_metrics[key]["bl"].append(bl_s)
        task_seed_metrics[key]["pm"].append(pm_s)
        task_seed_metrics[key]["rn"].append(rn_s)
        logger.debug(f"  {task} seed={seed}: bl={bl_s:.6f}, pm={pm_s:.6f}, rn={rn_s:.6f}")

    # FK diagnostics (from exp_id2_it2 metadata)
    mean_card = (317.965 + 17.4414) / 2
    mean_r2 = (0.009082 + 0.072123) / 2

    results = []
    for (task, metric, task_type), vals in task_seed_metrics.items():
        bl_arr = np.array(vals["bl"])
        pm_arr = np.array(vals["pm"])
        rn_arr = np.array(vals["rn"])
        n = len(bl_arr)
        higher = (metric == "auroc")

        logger.info(f"  rel-hm-{task} ({metric}): n_seeds={n}, "
                     f"bl={np.mean(bl_arr):.4f}±{np.std(bl_arr, ddof=1) if n>1 else 0:.4f}, "
                     f"pm={np.mean(pm_arr):.4f}±{np.std(pm_arr, ddof=1) if n>1 else 0:.4f}")

        results.append(TaskResult(
            dataset="rel-hm", task=f"rel-hm-{task}", metric_name="AUROC" if metric == "auroc" else "MAE",
            task_type=task_type if task_type != "unknown" else ("classification" if metric == "auroc" else "regression"),
            higher_is_better=higher, n_seeds=n,
            baseline_mean=float(np.mean(bl_arr)),
            baseline_std=float(np.std(bl_arr, ddof=1)) if n > 1 else 0.0,
            prmp_mean=float(np.mean(pm_arr)),
            prmp_std=float(np.std(pm_arr, ddof=1)) if n > 1 else 0.0,
            baseline_values=bl_arr.tolist(), prmp_values=pm_arr.tolist(),
            random_mean=float(np.mean(rn_arr)),
            random_std=float(np.std(rn_arr, ddof=1)) if n > 1 else 0.0,
            random_values=rn_arr.tolist(),
            mean_fk_cardinality=mean_card, mean_cross_table_r2=mean_r2,
            is_saturated=False,
        ))

    del data, examples
    gc.collect()
    return results


def extract_relstack_results() -> List[TaskResult]:
    """rel-stack (exp_id3_it3__opus) — 2 tasks, 1 seed only."""
    logger.info("Extracting rel-stack results...")
    data = json.loads((EXP_RELSTACK / "mini_method_out.json").read_text())
    meta = data["metadata"]
    gnn = meta["gnn_results"]
    per_run = gnn["per_run_results"]
    fk_diag = gnn["fk_diagnostics"]

    cards = [v["card_mean"] for v in fk_diag.values()]
    r2s = [max(0.0, v.get("r_squared", 0.0)) for v in fk_diag.values()]
    mean_card = float(np.mean(cards))
    mean_r2 = float(np.mean(r2s))

    results = []

    # User-engagement (AUROC)
    sage_ue = per_run["user-engagement__sage__seed42"]["test_metrics"]
    prmp_ue = per_run["user-engagement__prmp__seed42"]["test_metrics"]
    results.append(TaskResult(
        dataset="rel-stack", task="rel-stack-user-engagement", metric_name="AUROC",
        task_type="classification", higher_is_better=True, n_seeds=1,
        baseline_mean=sage_ue["roc_auc"], baseline_std=0.0,
        prmp_mean=prmp_ue["roc_auc"], prmp_std=0.0,
        baseline_values=[sage_ue["roc_auc"]], prmp_values=[prmp_ue["roc_auc"]],
        mean_fk_cardinality=mean_card, mean_cross_table_r2=mean_r2,
        is_saturated=False,
    ))

    # Post-votes (MAE)
    sage_pv = per_run["post-votes__sage__seed42"]["test_metrics"]
    prmp_pv = per_run["post-votes__prmp__seed42"]["test_metrics"]
    results.append(TaskResult(
        dataset="rel-stack", task="rel-stack-post-votes", metric_name="MAE",
        task_type="regression", higher_is_better=False, n_seeds=1,
        baseline_mean=sage_pv["mae"], baseline_std=0.0,
        prmp_mean=prmp_pv["mae"], prmp_std=0.0,
        baseline_values=[sage_pv["mae"]], prmp_values=[prmp_pv["mae"]],
        mean_fk_cardinality=mean_card, mean_cross_table_r2=mean_r2,
        is_saturated=False,
    ))

    del data
    gc.collect()
    return results


def extract_relhm_it2_results() -> TaskResult:
    """rel-hm iter2 FK-link benchmark (exp_id2_it2__opus) — saturated, 1 seed."""
    logger.info("Extracting rel-hm (iter2 FK-link) results...")
    data = json.loads((EXP_RELHM_IT2 / "mini_method_out.json").read_text())
    examples = data["datasets"][0]["examples"]

    bl_aurocs = []
    pm_aurocs = []
    for ex in examples:
        std_pred = json.loads(ex["predict_standard"])
        prmp_pred = json.loads(ex["predict_prmp"])
        bl_aurocs.append(std_pred["test_metrics"]["auroc"])
        pm_aurocs.append(prmp_pred["test_metrics"]["auroc"])

    n = len(bl_aurocs)
    return TaskResult(
        dataset="rel-hm (FK-link)", task="rel-hm-fk-link-classification", metric_name="AUROC",
        task_type="classification", higher_is_better=True, n_seeds=1,
        baseline_mean=float(np.mean(bl_aurocs)),
        baseline_std=float(np.std(bl_aurocs, ddof=1)) if n > 1 else 0.0,
        prmp_mean=float(np.mean(pm_aurocs)),
        prmp_std=float(np.std(pm_aurocs, ddof=1)) if n > 1 else 0.0,
        baseline_values=bl_aurocs, prmp_values=pm_aurocs,
        mean_fk_cardinality=(317.965 + 17.4414) / 2,
        mean_cross_table_r2=(0.009082 + 0.072123) / 2,
        is_saturated=True,
    )

# ============================================================
# Plotting
# ============================================================
def create_forest_plot(all_tasks: List[TaskResult], meta_result: Dict, output_path: Path):
    """Forest plot with per-task Hedges' g and RE summary diamond."""
    logger.info("Creating forest plot...")

    plot_data = []
    for tr in all_tasks:
        g, se, ci_lo, ci_hi, pval = compute_hedges_g(
            tr.prmp_mean, tr.prmp_std, tr.n_seeds,
            tr.baseline_mean, tr.baseline_std, tr.n_seeds,
            tr.higher_is_better,
        )
        suffix = ""
        if tr.is_saturated:
            suffix = " [SAT]"
        if tr.n_seeds == 1:
            suffix += " [1 seed]"
        label = f"{tr.dataset}: {tr.task} ({tr.metric_name}){suffix}"
        plot_data.append((label, g, ci_lo, ci_hi, se, tr.dataset))

    plot_data.sort(key=lambda x: (x[5], x[0]))
    n_items = len(plot_data)

    fig, ax = plt.subplots(figsize=(14, max(5, n_items * 0.7 + 2.5)))

    dataset_colors = {
        "Amazon Video Games": "#1f77b4",
        "rel-hm": "#ff7f0e",
        "rel-f1": "#2ca02c",
        "rel-stack": "#d62728",
        "rel-hm (FK-link)": "#9467bd",
    }

    for i, (label, g, ci_lo, ci_hi, se, dataset) in enumerate(plot_data):
        y = n_items - i
        color = dataset_colors.get(dataset, "gray")

        # Clip CIs for display
        ci_lo_d = max(ci_lo, -8)
        ci_hi_d = min(ci_hi, 8)
        ax.plot([ci_lo_d, ci_hi_d], [y, y], color=color, linewidth=1.5, zorder=2)

        w = max(3, min(10, 20 / max(se, 0.1)))
        ax.plot(g, y, 's', color=color, markersize=w, zorder=3)

        ax.text(-8.8, y, label, ha='left', va='center', fontsize=7.5)

        if math.isfinite(ci_lo) and math.isfinite(ci_hi):
            ax.text(8.3, y, f"{g:+.2f} [{ci_lo:.2f}, {ci_hi:.2f}]",
                    ha='left', va='center', fontsize=7, family='monospace')

    # Summary diamond
    y0 = 0
    gp = meta_result["pooled_g"]
    cl = meta_result["ci_lower"]
    ch = meta_result["ci_upper"]
    diamond_x = [cl, gp, ch, gp]
    diamond_y = [y0, y0 + 0.3, y0, y0 - 0.3]
    ax.fill(diamond_x, diamond_y, color='black', alpha=0.8, zorder=3)
    ax.text(-8.8, y0, "RE Model (pooled)", ha='left', va='center', fontsize=8, fontweight='bold')
    ax.text(8.3, y0, f"{gp:+.2f} [{cl:.2f}, {ch:.2f}]",
            ha='left', va='center', fontsize=7, fontweight='bold', family='monospace')

    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    # Regions
    ax.axvspan(-8.8, 0, alpha=0.03, color='red')
    ax.axvspan(0, 8.8, alpha=0.03, color='green')
    ax.text(-0.3, n_items + 0.8, '← Baseline better', fontsize=8, color='red', ha='right')
    ax.text(0.3, n_items + 0.8, 'PRMP better →', fontsize=8, color='green', ha='left')

    ax.set_xlim(-9.5, 13)
    ax.set_ylim(-1, n_items + 1.5)
    ax.set_xlabel("Hedges' g (positive = PRMP better)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Forest Plot: PRMP vs Baseline Effect Sizes Across All Tasks",
                 fontsize=12, fontweight='bold', pad=12)

    info = (f"Random-effects: I² = {meta_result['I2']:.1f}%, "
            f"τ² = {meta_result['tau2']:.4f}, Q = {meta_result['Q']:.2f} "
            f"(p = {meta_result['Q_p_value']:.3f}), k = {meta_result['k']}")
    ax.text(0.5, -0.06, info, transform=ax.transAxes, ha='center', fontsize=8, style='italic')

    # Legend
    patches = [mpatches.Patch(color=c, label=d) for d, c in dataset_colors.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Forest plot saved: {output_path}")


def create_funnel_plot(primary_tasks: List[TaskResult], output_path: Path):
    """Funnel plot (effect size vs SE) for publication bias assessment."""
    logger.info("Creating funnel plot...")
    effects, ses, labels = [], [], []
    for tr in primary_tasks:
        g, se, _, _, _ = compute_hedges_g(
            tr.prmp_mean, tr.prmp_std, tr.n_seeds,
            tr.baseline_mean, tr.baseline_std, tr.n_seeds,
            tr.higher_is_better,
        )
        if math.isfinite(se) and se < 50:
            effects.append(g)
            ses.append(se)
            labels.append(tr.task)

    if not effects:
        logger.warning("  No valid data for funnel plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(effects, ses, s=60, c='steelblue', edgecolors='black', linewidth=0.5, zorder=3)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (effects[i], ses[i]), fontsize=7, xytext=(5, 5), textcoords='offset points')

    mean_g = np.mean(effects)
    ax.axvline(x=mean_g, color='red', linestyle='--', linewidth=0.8, label=f'Mean g = {mean_g:.2f}')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8)

    se_max = max(ses) * 1.3
    se_range = np.linspace(0.001, se_max, 100)
    ax.fill_betweenx(se_range, mean_g - 1.96 * se_range, mean_g + 1.96 * se_range,
                     alpha=0.1, color='gray', label='Pseudo 95% CI')

    ax.set_xlabel("Hedges' g", fontsize=11)
    ax.set_ylabel("Standard Error", fontsize=11)
    ax.set_title("Funnel Plot: Publication Bias Assessment", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Funnel plot saved: {output_path}")


def create_summary_table_plot(task_effects: List[Dict], output_path: Path):
    """Publication-ready summary table as an image."""
    logger.info("Creating summary table plot...")

    headers = ["Dataset", "Task", "Metric", "Baseline\n(mean±std)", "PRMP\n(mean±std)",
               "Δ", "Hedges' g\n[95% CI]", "p-value", "Sig?"]
    rows = []
    for te in task_effects:
        bl_str = f"{te['baseline_mean']:.4f}±{te['baseline_std']:.4f}"
        pm_str = f"{te['prmp_mean']:.4f}±{te['prmp_std']:.4f}"
        delta = te['delta']
        if te['higher_is_better']:
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = f"{delta:+.4f}"
        g = te['hedges_g']
        ci = f"{g:.2f} [{te['g_ci_lower']:.2f}, {te['g_ci_upper']:.2f}]"
        p = te['p_value']
        sig = "✓" if te['significant'] else ""
        if te['is_saturated']:
            sig += " [SAT]"
        rows.append([te['dataset'], te['task'], te['metric'], bl_str, pm_str,
                     delta_str, ci, f"{p:.4f}", sig])

    fig, ax = plt.subplots(figsize=(18, max(3, len(rows) * 0.4 + 1.5)))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        bg = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg)

    ax.set_title("Summary: PRMP vs Baseline Across All Tasks", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Summary table saved: {output_path}")

# ============================================================
# Main
# ============================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Cross-Experiment PRMP Meta-Analysis")
    logger.info("=" * 60)

    # ---- Step 1: Extract results ----
    logger.info("STEP 1: Extract results from all 5 experiments")
    all_tasks: List[TaskResult] = []

    amazon = extract_amazon_results()
    all_tasks.append(amazon)
    logger.info(f"  Amazon: bl RMSE={amazon.baseline_mean:.4f}, PRMP={amazon.prmp_mean:.4f}")

    relf1 = extract_relf1_results()
    all_tasks.extend(relf1)
    for t in relf1:
        logger.info(f"  {t.task}: bl {t.metric_name}={t.baseline_mean:.4f}, PRMP={t.prmp_mean:.4f}"
                     + (" [SATURATED]" if t.is_saturated else ""))

    relhm = extract_relhm_it3_results()
    all_tasks.extend(relhm)
    for t in relhm:
        logger.info(f"  {t.task}: bl {t.metric_name}={t.baseline_mean:.4f}, PRMP={t.prmp_mean:.4f} (n={t.n_seeds})")

    relstack = extract_relstack_results()
    all_tasks.extend(relstack)
    for t in relstack:
        logger.info(f"  {t.task}: bl {t.metric_name}={t.baseline_mean:.4f}, PRMP={t.prmp_mean:.4f} [1 seed]")

    relhm_it2 = extract_relhm_it2_results()
    all_tasks.append(relhm_it2)
    logger.info(f"  {relhm_it2.task}: bl AUROC={relhm_it2.baseline_mean:.4f}, PRMP={relhm_it2.prmp_mean:.4f} [SAT]")

    logger.info(f"Total tasks: {len(all_tasks)}")

    # ---- Step 2: Hedges' g for all tasks ----
    logger.info("STEP 2: Computing Hedges' g for all tasks")
    task_effects: List[Dict] = []
    for tr in all_tasks:
        g, se, ci_lo, ci_hi, pval = compute_hedges_g(
            tr.prmp_mean, tr.prmp_std, tr.n_seeds,
            tr.baseline_mean, tr.baseline_std, tr.n_seeds,
            tr.higher_is_better,
        )
        task_effects.append({
            "task": tr.task, "dataset": tr.dataset, "metric": tr.metric_name,
            "task_type": tr.task_type, "n_seeds": tr.n_seeds,
            "baseline_mean": tr.baseline_mean, "baseline_std": tr.baseline_std,
            "prmp_mean": tr.prmp_mean, "prmp_std": tr.prmp_std,
            "delta": tr.prmp_mean - tr.baseline_mean,
            "hedges_g": g, "g_se": se, "g_ci_lower": ci_lo, "g_ci_upper": ci_hi,
            "p_value": pval, "significant": pval < 0.05,
            "is_saturated": tr.is_saturated, "higher_is_better": tr.higher_is_better,
        })
        logger.info(f"  {tr.task}: g={g:+.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.4f}"
                     + (" *" if pval < 0.05 else ""))

    # ---- Step 3: Random-effects meta-analysis (≥3 seeds) ----
    logger.info("STEP 3: Random-effects meta-analysis (tasks with ≥3 seeds)")
    primary = [te for te in task_effects if te["n_seeds"] >= 3]
    logger.info(f"  {len(primary)} primary tasks for meta-analysis")

    eff_list = [te["hedges_g"] for te in primary]
    var_list = [te["g_se"]**2 for te in primary]
    meta = dersimonian_laird(eff_list, var_list)

    logger.info(f"  Pooled g = {meta['pooled_g']:+.4f} [{meta['ci_lower']:.4f}, {meta['ci_upper']:.4f}]")
    logger.info(f"  z = {meta['z']:.3f}, p = {meta['p_value']:.6f}")
    logger.info(f"  Q = {meta['Q']:.3f}, I² = {meta['I2']:.1f}%, τ² = {meta['tau2']:.4f}")

    # ---- Step 4: Subgroup analysis ----
    logger.info("STEP 4: Subgroup analysis (regression vs classification)")
    reg = [te for te in primary if te["task_type"] == "regression"]
    cls = [te for te in primary if te["task_type"] == "classification"]
    logger.info(f"  Regression tasks ({len(reg)}): {[t['task'] for t in reg]}")
    logger.info(f"  Classification tasks ({len(cls)}): {[t['task'] for t in cls]}")

    reg_meta = dersimonian_laird([t["hedges_g"] for t in reg], [t["g_se"]**2 for t in reg]) if len(reg) >= 2 else None
    cls_meta = dersimonian_laird([t["hedges_g"] for t in cls], [t["g_se"]**2 for t in cls]) if len(cls) >= 2 else None

    subgroup = {}
    if reg_meta and cls_meta:
        Q_within = reg_meta["Q"] + cls_meta["Q"]
        Q_between = max(0, meta["Q"] - Q_within)
        Q_between_p = float(1 - sp_stats.chi2.cdf(Q_between, 1))

        subgroup = {
            "regression_pooled_g": reg_meta["pooled_g"],
            "regression_ci": [reg_meta["ci_lower"], reg_meta["ci_upper"]],
            "regression_p": reg_meta["p_value"],
            "regression_tasks": [t["task"] for t in reg],
            "classification_pooled_g": cls_meta["pooled_g"],
            "classification_ci": [cls_meta["ci_lower"], cls_meta["ci_upper"]],
            "classification_p": cls_meta["p_value"],
            "classification_tasks": [t["task"] for t in cls],
            "Q_between": Q_between, "Q_between_p": Q_between_p,
            "subgroup_difference_significant": Q_between_p < 0.05,
        }
        logger.info(f"  Regression pooled g = {reg_meta['pooled_g']:+.3f} (p={reg_meta['p_value']:.4f})")
        logger.info(f"  Classification pooled g = {cls_meta['pooled_g']:+.3f} (p={cls_meta['p_value']:.4f})")
        logger.info(f"  Q_between = {Q_between:.3f}, p = {Q_between_p:.4f}")

    # ---- Step 5: Meta-regression ----
    logger.info("STEP 5: Meta-regression with moderators")
    task_map = {tr.task: tr for tr in all_tasks}
    mod_names_full = ["intercept", "fk_cardinality_norm", "cross_table_r2",
                      "card_x_r2_interaction", "task_type_reg", "is_saturated"]

    y_arr, X_rows, w_arr = [], [], []
    for te in primary:
        tr = task_map[te["task"]]
        card = tr.mean_fk_cardinality / 100.0
        r2 = tr.mean_cross_table_r2
        row = [1.0, card, r2, card * r2,
               1.0 if tr.task_type == "regression" else 0.0,
               1.0 if tr.is_saturated else 0.0]
        y_arr.append(te["hedges_g"])
        X_rows.append(row)
        se = te["g_se"]
        w_arr.append(1.0 / (se**2) if se > 0 and math.isfinite(se) else 1.0)

    y_np = np.array(y_arr)
    X_np = np.array(X_rows)
    w_np = np.array(w_arr)

    # Determine feasible model size
    k = len(y_arr)
    if k > len(mod_names_full):
        mod_names = mod_names_full
        X_use = X_np
    elif k > 3:
        # Reduced: intercept + interaction + task_type
        mod_names = ["intercept", "card_x_r2_interaction", "task_type_reg"]
        X_use = X_np[:, [0, 3, 4]]
    else:
        mod_names = ["intercept"]
        X_use = X_np[:, [0]]

    meta_reg = wls_regression(y_np, X_use, w_np)
    meta_regression_out = {
        "moderator_names": mod_names,
        "coefficients": {n: round(c, 6) for n, c in zip(mod_names, meta_reg["coefficients"])},
        "standard_errors": {n: round(s, 6) for n, s in zip(mod_names, meta_reg["standard_errors"])},
        "p_values": {n: round(p, 6) for n, p in zip(mod_names, meta_reg["p_values"])},
        "n_tasks": k,
        "n_predictors": len(mod_names),
    }
    for nm, coef, pv in zip(mod_names, meta_reg["coefficients"], meta_reg["p_values"]):
        logger.info(f"  {nm}: β = {coef:+.4f}, p = {pv:.4f}" + (" *" if pv < 0.05 else ""))

    # ---- Step 6: Ablation validation ----
    logger.info("STEP 6: Ablation validation (random predictions)")
    ablation_results = []
    for tr in all_tasks:
        if tr.random_mean is not None and tr.n_seeds >= 3:
            g_rb, se_rb, ci_lo_rb, ci_hi_rb, p_rb = compute_hedges_g(
                tr.random_mean, tr.random_std, tr.n_seeds,
                tr.baseline_mean, tr.baseline_std, tr.n_seeds,
                tr.higher_is_better,
            )
            g_pr, se_pr, ci_lo_pr, ci_hi_pr, p_pr = compute_hedges_g(
                tr.prmp_mean, tr.prmp_std, tr.n_seeds,
                tr.random_mean, tr.random_std, tr.n_seeds,
                tr.higher_is_better,
            )
            ablation_results.append({
                "task": tr.task,
                "random_vs_baseline_g": round(g_rb, 4),
                "random_vs_baseline_se": round(se_rb, 4),
                "random_vs_baseline_p": round(p_rb, 6),
                "random_vs_baseline_significant": p_rb < 0.05,
                "prmp_vs_random_g": round(g_pr, 4),
                "prmp_vs_random_se": round(se_pr, 4),
                "prmp_vs_random_p": round(p_pr, 6),
                "prmp_vs_random_significant": p_pr < 0.05,
                "learned_predictions_necessary": g_pr > g_rb,
            })
            logger.info(f"  {tr.task}: rand_vs_bl g={g_rb:+.3f}(p={p_rb:.3f}), "
                         f"prmp_vs_rand g={g_pr:+.3f}(p={p_pr:.3f})")

    # ---- Step 7: Win / Loss / Draw ----
    logger.info("STEP 7: Win/Loss/Draw tally")
    wins = losses = draws = 0
    wins_ns = losses_ns = draws_ns = 0
    for te in task_effects:
        if te["n_seeds"] < 3:
            continue
        if te["significant"]:
            if te["hedges_g"] > 0:
                wins += 1
                if not te["is_saturated"]:
                    wins_ns += 1
            else:
                losses += 1
                if not te["is_saturated"]:
                    losses_ns += 1
        else:
            draws += 1
            if not te["is_saturated"]:
                draws_ns += 1

    tally = {
        "all_tasks": {"wins": wins, "losses": losses, "draws": draws,
                      "total": wins + losses + draws},
        "non_saturated_only": {"wins": wins_ns, "losses": losses_ns, "draws": draws_ns,
                               "total": wins_ns + losses_ns + draws_ns},
    }
    logger.info(f"  All: {wins}W / {losses}L / {draws}D (of {wins+losses+draws})")
    logger.info(f"  Non-saturated: {wins_ns}W / {losses_ns}L / {draws_ns}D")

    # ---- Step 8-9: Plots ----
    logger.info("STEP 8-9: Creating plots")
    create_forest_plot(all_tasks, meta, WORKSPACE / "forest_plot.png")
    primary_objs = [tr for tr in all_tasks if tr.n_seeds >= 3]
    create_funnel_plot(primary_objs, WORKSPACE / "funnel_plot.png")
    create_summary_table_plot(task_effects, WORKSPACE / "summary_table.png")

    # ---- Step 10: Build output ----
    logger.info("STEP 10: Building eval_out.json")

    # Hypothesis conclusion
    pooled_pos = meta["pooled_g"] > 0
    pooled_sig = meta["p_value"] < 0.05
    interaction_p = meta_regression_out["p_values"].get("card_x_r2_interaction", 1.0)
    interaction_sig = interaction_p < 0.05
    ablation_ok = all(a["learned_predictions_necessary"] for a in ablation_results) if ablation_results else False

    if pooled_sig and pooled_pos and interaction_sig and ablation_ok:
        conclusion = "CONFIRMED"
        conclusion_text = (
            "Hypothesis CONFIRMED: PRMP provides systematic improvement "
            f"(pooled g={meta['pooled_g']:+.3f}, p={meta['p_value']:.4f}), "
            "cardinality×predictability interaction is significant, "
            "and learned predictions are necessary (ablation confirms)."
        )
    elif pooled_pos and (not pooled_sig or not interaction_sig):
        conclusion = "PARTIALLY_CONFIRMED"
        conclusion_text = (
            f"Hypothesis PARTIALLY CONFIRMED: PRMP shows positive pooled effect "
            f"(g={meta['pooled_g']:+.3f}, p={meta['p_value']:.4f}), "
            f"but {'pooled effect is not significant' if not pooled_sig else 'cardinality×predictability interaction is not significant'}. "
            "PRMP tends to help, especially on regression tasks, but the specific "
            "regime prediction (high cardinality AND high predictability) is not validated."
        )
    elif not pooled_pos:
        conclusion = "DISCONFIRMED"
        conclusion_text = (
            f"Hypothesis DISCONFIRMED: Pooled effect is non-positive "
            f"(g={meta['pooled_g']:+.3f}, p={meta['p_value']:.4f})."
        )
    else:
        conclusion = "INCONCLUSIVE"
        conclusion_text = "Results are inconclusive."

    logger.info(f"  Conclusion: {conclusion}")
    logger.info(f"  {conclusion_text}")

    # metrics_agg
    metrics_agg = {
        "pooled_hedges_g": round(meta["pooled_g"], 4),
        "pooled_g_ci_lower": round(meta["ci_lower"], 4),
        "pooled_g_ci_upper": round(meta["ci_upper"], 4),
        "pooled_p_value": round(meta["p_value"], 6),
        "pooled_z": round(meta["z"], 4),
        "I_squared": round(meta["I2"], 2),
        "tau_squared": round(meta["tau2"], 6),
        "Q_statistic": round(meta["Q"], 4),
        "Q_p_value": round(meta["Q_p_value"], 6),
        "n_tasks_in_meta": meta["k"],
        "n_tasks_total": len(all_tasks),
        "n_wins": tally["all_tasks"]["wins"],
        "n_losses": tally["all_tasks"]["losses"],
        "n_draws": tally["all_tasks"]["draws"],
        "n_wins_nonsat": tally["non_saturated_only"]["wins"],
        "n_losses_nonsat": tally["non_saturated_only"]["losses"],
        "n_draws_nonsat": tally["non_saturated_only"]["draws"],
    }
    if reg_meta:
        metrics_agg["regression_subgroup_g"] = round(reg_meta["pooled_g"], 4)
        metrics_agg["regression_subgroup_p"] = round(reg_meta["p_value"], 6)
    if cls_meta:
        metrics_agg["classification_subgroup_g"] = round(cls_meta["pooled_g"], 4)
        metrics_agg["classification_subgroup_p"] = round(cls_meta["p_value"], 6)

    # datasets
    datasets_out = []

    # Dataset 1: main per-task meta-analysis
    task_examples = []
    for te in task_effects:
        inp = json.dumps({
            "task": te["task"], "dataset": te["dataset"], "metric": te["metric"],
            "task_type": te["task_type"], "n_seeds": int(te["n_seeds"]),
            "higher_is_better": bool(te["higher_is_better"]), "is_saturated": bool(te["is_saturated"]),
        })
        out = json.dumps({
            "hedges_g": round(float(te["hedges_g"]), 4),
            "ci": [round(float(te["g_ci_lower"]), 4), round(float(te["g_ci_upper"]), 4)],
            "p_value": round(float(te["p_value"]), 6),
            "significant": bool(te["significant"]),
            "direction": "PRMP better" if te["hedges_g"] > 0 else "Baseline better",
        })
        task_examples.append({
            "input": inp, "output": out,
            "predict_baseline": f"{te['baseline_mean']:.6f}",
            "predict_prmp": f"{te['prmp_mean']:.6f}",
            "eval_hedges_g": round(te["hedges_g"], 4),
            "eval_g_se": round(te["g_se"], 4) if math.isfinite(te["g_se"]) else 999.0,
            "eval_g_ci_lower": round(te["g_ci_lower"], 4) if math.isfinite(te["g_ci_lower"]) else -999.0,
            "eval_g_ci_upper": round(te["g_ci_upper"], 4) if math.isfinite(te["g_ci_upper"]) else 999.0,
            "eval_p_value": round(te["p_value"], 6),
            "metadata_dataset": te["dataset"],
            "metadata_task": te["task"],
            "metadata_metric": te["metric"],
            "metadata_task_type": te["task_type"],
            "metadata_n_seeds": te["n_seeds"],
            "metadata_is_saturated": te["is_saturated"],
        })
    datasets_out.append({"dataset": "prmp_meta_analysis_tasks", "examples": task_examples})

    # Dataset 2: ablation validation
    if ablation_results:
        abl_examples = []
        for abl in ablation_results:
            inp = json.dumps({"task": abl["task"], "analysis": "ablation_random_predictions"})
            out = json.dumps({
                "random_vs_baseline_g": abl["random_vs_baseline_g"],
                "prmp_vs_random_g": abl["prmp_vs_random_g"],
                "learned_predictions_necessary": abl["learned_predictions_necessary"],
            })
            abl_examples.append({
                "input": inp, "output": out,
                "predict_random_vs_baseline": f"{abl['random_vs_baseline_g']:.4f}",
                "predict_prmp_vs_random": f"{abl['prmp_vs_random_g']:.4f}",
                "eval_random_vs_baseline_g": abl["random_vs_baseline_g"],
                "eval_random_vs_baseline_p": abl["random_vs_baseline_p"],
                "eval_prmp_vs_random_g": abl["prmp_vs_random_g"],
                "eval_prmp_vs_random_p": abl["prmp_vs_random_p"],
                "metadata_task": abl["task"],
            })
        datasets_out.append({"dataset": "ablation_validation", "examples": abl_examples})

    # Full output
    output = {
        "metadata": {
            "title": "Cross-Experiment PRMP Meta-Analysis: Effect Sizes, Forest Plots & Moderator Analysis",
            "description": (
                "Comprehensive meta-analysis pooling PRMP vs baseline results across 5 experiments "
                "(4 datasets, 8+ tasks). Includes Hedges' g effect sizes, DerSimonian-Laird "
                "random-effects pooling, subgroup analysis, meta-regression, and ablation validation."
            ),
            "conclusion": conclusion,
            "conclusion_text": conclusion_text,
            "random_effects_meta_analysis": meta,
            "subgroup_analysis": subgroup,
            "meta_regression": meta_regression_out,
            "ablation_validation": ablation_results,
            "win_loss_draw": tally,
            "figures": ["forest_plot.png", "funnel_plot.png", "summary_table.png"],
            "all_task_effects": task_effects,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"  eval_out.json written: {size_mb:.2f} MB")

    logger.info("=" * 60)
    logger.info(f"DONE — Conclusion: {conclusion}")
    logger.info(f"Pooled Hedges' g = {meta['pooled_g']:+.4f} "
                f"[{meta['ci_lower']:.4f}, {meta['ci_upper']:.4f}], p = {meta['p_value']:.6f}")
    logger.info(f"I² = {meta['I2']:.1f}%, τ² = {meta['tau2']:.4f}")
    logger.info(f"Win/Loss/Draw: {wins}W/{losses}L/{draws}D (non-sat: {wins_ns}W/{losses_ns}L/{draws_ns}D)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
