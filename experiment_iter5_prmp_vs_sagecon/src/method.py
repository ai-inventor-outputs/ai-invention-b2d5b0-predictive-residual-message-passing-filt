#!/usr/bin/env python3
"""PRMP vs SAGEConv Benchmark on RelBench rel-avito (ad-ctr & user-visits).

Benchmarks Predictive Residual Message Passing (PRMP) against standard
HeteroSAGEConv on RelBench rel-avito tasks:
  - ad-ctr regression (RMSE/MAE/R²)
  - user-visits classification (AUROC/AP)

Three model variants:
  A) Baseline SAGEConv
  B) Full PRMP (all FK edges)
  C) Selective PRMP (top-3 cardinality links only)

Also computes cross-table R² diagnostic for all 11 FK links and correlates
PRMP improvement with cardinality × predictability.
"""

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(str(log_dir / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware Detection ───────────────────────────────────────────────────

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
TOTAL_RAM_GB = _container_ram_gb() or 57.0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM")

# Set RAM limit to 80% of container limit
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
    logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f}GB")
except Exception:
    logger.warning("Could not set RLIMIT_AS")

# CPU time limit: 55 minutes
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3300, 3300))
except Exception:
    pass

# ── Constants ────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DEP_DATA_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju/"
    "3_invention_loop/iter_4/gen_art/data_id4_it4__opus/full_data_out.json"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"

# GNN Training hyperparameters (used if torch + pyg available)
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 256
NUM_NEIGHBORS = [64, 32]
SEEDS = [42, 123, 456]
MAX_TRAIN_STEPS = 500
PATIENCE = 5

# Subsampling limits for large tables
TABLE_SAMPLE_LIMITS = {
    "SearchStream": 200_000,
    "VisitStream": 200_000,
    "AdsInfo": 300_000,
    "SearchInfo": 200_000,
}


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Cross-Table R² Diagnostic (core — always runs)
# ══════════════════════════════════════════════════════════════════════════

def compute_r2_diagnostic(dep_data: dict) -> tuple[dict, list[str], list[dict]]:
    """Compute cross-table R² for all FK links from dependency data.

    Returns:
        r2_results: dict of link_id -> {cross_table_r2, cardinality_mean, ...}
        selective_links: top-3 highest cardinality link IDs
        examples_with_predictions: examples with predict_* fields added
    """
    logger.info("Phase 1: Computing cross-table R² diagnostic")
    t0 = time.time()

    fk_link_metadata = dep_data["metadata"]["fk_links"]
    examples = dep_data["datasets"][0]["examples"]

    # Group examples by FK link
    link_groups: dict[str, list[dict]] = {}
    for ex in examples:
        lid = ex["metadata_link_id"]
        link_groups.setdefault(lid, []).append(ex)

    r2_results: dict[str, dict] = {}
    example_predictions: dict[str, dict[int, list[float]]] = {}

    for link_id, link_meta in fk_link_metadata.items():
        link_examples = link_groups.get(link_id, [])
        if len(link_examples) < 10:
            logger.warning(f"  Skip R² for {link_id}: only {len(link_examples)} examples")
            r2_results[link_id] = {
                "cross_table_r2": 0.0,
                "cardinality_mean": link_meta["cardinality_mean"],
                "cardinality_x_predictability": 0.0,
                "parent_table": link_meta["parent_table"],
                "child_table": link_meta["child_table"],
                "fk_column": link_meta["fk_column"],
            }
            continue

        X_parent = np.array([json.loads(e["input"]) for e in link_examples])
        Y_child = np.array([json.loads(e["output"]) for e in link_examples])

        X_parent = np.nan_to_num(X_parent, nan=0.0, posinf=0.0, neginf=0.0)
        Y_child = np.nan_to_num(Y_child, nan=0.0, posinf=0.0, neginf=0.0)

        mean_r2 = 0.0
        predictions = np.zeros_like(Y_child)

        if X_parent.shape[0] > 50 and X_parent.shape[1] > 0 and Y_child.shape[1] > 0:
            try:
                reg = LinearRegression().fit(X_parent, Y_child)
                Y_pred = reg.predict(X_parent)
                predictions = Y_pred

                r2_per_col = []
                for j in range(Y_child.shape[1]):
                    col_var = np.var(Y_child[:, j])
                    if col_var > 1e-10:
                        r2_val = max(0.0, r2_score(Y_child[:, j], Y_pred[:, j]))
                    else:
                        r2_val = 1.0
                    r2_per_col.append(r2_val)
                mean_r2 = float(np.mean(r2_per_col))
            except Exception:
                logger.exception(f"  R² computation failed for {link_id}")
                mean_r2 = 0.0

        cardinality_mean = link_meta["cardinality_mean"]
        r2_results[link_id] = {
            "cross_table_r2": round(mean_r2, 6),
            "cardinality_mean": cardinality_mean,
            "cardinality_x_predictability": round(cardinality_mean * mean_r2, 4),
            "parent_table": link_meta["parent_table"],
            "child_table": link_meta["child_table"],
            "fk_column": link_meta["fk_column"],
        }

        example_predictions[link_id] = {}
        for i, ex in enumerate(link_examples):
            row_idx = ex["metadata_row_index"]
            pred_list = predictions[i].tolist() if i < len(predictions) else []
            example_predictions[link_id][row_idx] = [round(v, 4) for v in pred_list]

        logger.info(
            f"  {link_id}: R²={mean_r2:.4f}, card={cardinality_mean:.1f}, "
            f"card×R²={cardinality_mean * mean_r2:.2f}, n={len(link_examples)}"
        )

    # Top-3 highest cardinality links for selective PRMP
    selective_links = sorted(
        r2_results.keys(),
        key=lambda k: r2_results[k]["cardinality_mean"],
        reverse=True,
    )[:3]
    logger.info(f"  Top-3 cardinality links: {selective_links}")

    # Build examples with predictions
    examples_with_preds: list[dict] = []
    for ex in examples:
        link_id = ex["metadata_link_id"]
        row_idx = ex["metadata_row_index"]
        r2_info = r2_results.get(link_id, {})

        new_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_fk_link": ex["metadata_fk_link"],
            "metadata_link_id": link_id,
            "metadata_row_index": str(row_idx),
            "metadata_cross_table_r2": str(round(r2_info.get("cross_table_r2", 0.0), 6)),
            "metadata_cardinality_mean": str(round(r2_info.get("cardinality_mean", 0.0), 2)),
            "metadata_card_x_predict": str(round(r2_info.get("cardinality_x_predictability", 0.0), 4)),
        }

        preds = example_predictions.get(link_id, {}).get(row_idx, [])
        new_ex["predict_linear_baseline"] = json.dumps(preds)

        actual = json.loads(ex["output"])
        residual = [round(a - p, 4) for a, p in zip(actual, preds)] if preds else actual
        new_ex["predict_prmp_residual"] = json.dumps(residual)

        examples_with_preds.append(new_ex)

    elapsed = time.time() - t0
    logger.info(
        f"Phase 1 complete: {len(r2_results)} FK links, "
        f"{len(examples_with_preds)} examples, {elapsed:.1f}s"
    )
    return r2_results, selective_links, examples_with_preds


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2-5: GNN Training (requires torch + torch_geometric + relbench)
# ══════════════════════════════════════════════════════════════════════════

def try_gnn_experiments(r2_results: dict, selective_links: list[str]) -> dict:
    """Attempt full GNN training pipeline. Returns results dict or {}."""
    logger.info("Phase 2-5: Attempting GNN experiments...")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv, HeteroConv
        from torch_geometric.loader import NeighborLoader
        from torch_geometric.data import HeteroData
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
    except Exception as e:
        logger.warning(f"GNN dependencies not available: {e}")
        return {}

    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
    if HAS_GPU:
        VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9
        _free, _total = torch.cuda.mem_get_info(0)
        torch.cuda.set_per_process_memory_fraction(min(0.85, 0.85))
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f}GB")
    else:
        logger.info("No GPU, using CPU")

    # ── Model Definitions ────────────────────────────────────────────────

    class SimpleMLP(nn.Module):
        def __init__(self, channels: list[int], dropout: float = 0.2):
            super().__init__()
            layers = []
            for i in range(len(channels) - 1):
                layers.append(nn.Linear(channels[i], channels[i + 1]))
                if i < len(channels) - 2:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class StandardHeteroSAGE(nn.Module):
        """Model A: Standard heterogeneous GraphSAGE baseline."""
        def __init__(self, metadata, hidden_channels=64, num_layers=2,
                     out_channels=1):
            super().__init__()
            node_types, edge_types = metadata
            self.encoders = nn.ModuleDict()
            for nt in node_types:
                self.encoders[nt] = nn.LazyLinear(hidden_channels)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                conv_dict = {et: SAGEConv((-1, -1), hidden_channels)
                             for et in edge_types}
                self.convs.append(HeteroConv(conv_dict, aggr="sum"))
                norm_dict = {nt: nn.LayerNorm(hidden_channels)
                             for nt in node_types}
                self.norms.append(nn.ModuleDict(norm_dict))
            self.head = SimpleMLP(
                [hidden_channels, hidden_channels // 2, out_channels])

        def forward(self, x_dict, edge_index_dict, entity_table):
            h_dict = {}
            for nt, x in x_dict.items():
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                h_dict[nt] = self.encoders[nt](x.float()).relu()
            for conv, norm in zip(self.convs, self.norms):
                h_new = conv(h_dict, edge_index_dict)
                h_dict = {k: norm[k](v).relu()
                          for k, v in h_new.items() if k in norm}
            return self.head(h_dict[entity_table]).squeeze(-1)

    class PRMPConv(nn.Module):
        """PRMP message passing layer with residual prediction."""
        def __init__(self, hidden_channels, edge_types, prmp_edge_types):
            super().__init__()
            self.prmp_keys = set()
            self.pred_mlps = nn.ModuleDict()
            self.res_norms = nn.ModuleDict()
            for et in prmp_edge_types:
                key = "__".join(str(x) for x in et)
                self.pred_mlps[key] = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.res_norms[key] = nn.LayerNorm(hidden_channels)
                self.prmp_keys.add(key)
            conv_dict = {et: SAGEConv((-1, -1), hidden_channels)
                         for et in edge_types}
            self.conv = HeteroConv(conv_dict, aggr="sum")

        def forward(self, x_dict, edge_index_dict):
            mod = {k: v.clone() for k, v in x_dict.items()}
            for et, ei in edge_index_dict.items():
                key = "__".join(str(x) for x in et)
                if key not in self.prmp_keys:
                    continue
                src_t, _, dst_t = et[0], et[1], et[2]
                src_n, dst_n = ei[0], ei[1]
                predicted = self.pred_mlps[key](x_dict[src_t][src_n])
                actual = x_dict[dst_t][dst_n]
                residuals = self.res_norms[key](actual - predicted)
                mod[dst_t] = mod[dst_t].clone()
                mod[dst_t][dst_n] = residuals
            return self.conv(mod, edge_index_dict)

    class PRMPHeteroModel(nn.Module):
        """Model B/C: Full or Selective PRMP."""
        def __init__(self, metadata, hidden_channels=64, num_layers=2,
                     out_channels=1, prmp_edge_types=None):
            super().__init__()
            node_types, edge_types = metadata
            self.encoders = nn.ModuleDict()
            for nt in node_types:
                self.encoders[nt] = nn.LazyLinear(hidden_channels)
            if prmp_edge_types is None:
                prmp_edge_types = [et for et in edge_types
                                   if "rev" in str(et[1]).lower()]
                if not prmp_edge_types:
                    prmp_edge_types = edge_types[:len(edge_types) // 2]
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    PRMPConv(hidden_channels, edge_types, prmp_edge_types))
                norm_dict = {nt: nn.LayerNorm(hidden_channels)
                             for nt in node_types}
                self.norms.append(nn.ModuleDict(norm_dict))
            self.head = SimpleMLP(
                [hidden_channels, hidden_channels // 2, out_channels])

        def forward(self, x_dict, edge_index_dict, entity_table):
            h_dict = {}
            for nt, x in x_dict.items():
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                h_dict[nt] = self.encoders[nt](x.float()).relu()
            for layer, norm in zip(self.layers, self.norms):
                h_new = layer(h_dict, edge_index_dict)
                h_dict = {k: norm[k](v).relu()
                          for k, v in h_new.items() if k in norm}
            return self.head(h_dict[entity_table]).squeeze(-1)

    # ── Build manual HeteroData from DataFrames ──────────────────────────

    def build_manual_heterodata(db_tables):
        data = HeteroData()
        node_id_maps = {}
        for tname, table in db_tables.items():
            df = table.df
            pk_col = table.pkey_col
            fks = (dict(table.fkey_col_to_pkey_table)
                   if hasattr(table, "fkey_col_to_pkey_table") else {})
            exclude = {pk_col, table.time_col} | set(fks.keys())
            exclude.discard(None)
            feat_cols = [c for c in df.columns
                         if c not in exclude
                         and pd.api.types.is_numeric_dtype(df[c].dtype)]
            if feat_cols:
                x = torch.tensor(
                    df[feat_cols].fillna(0).values, dtype=torch.float32)
            else:
                x = torch.zeros(len(df), 1, dtype=torch.float32)
            data[tname].x = x
            data[tname].num_nodes = len(df)
            if pk_col and pk_col in df.columns:
                node_id_maps[tname] = dict(
                    zip(df[pk_col].values, range(len(df))))

        for tname, table in db_tables.items():
            fks = (dict(table.fkey_col_to_pkey_table)
                   if hasattr(table, "fkey_col_to_pkey_table") else {})
            df = table.df
            for fk_col, parent_table in fks.items():
                if parent_table not in node_id_maps:
                    continue
                pk_map = node_id_maps[parent_table]
                valid = df[fk_col].notna()
                fk_vals = df.loc[valid, fk_col].values
                c_indices = np.where(valid)[0]
                src, dst = [], []
                for ci, fv in zip(c_indices, fk_vals):
                    pi = pk_map.get(fv)
                    if pi is not None:
                        src.append(int(ci))
                        dst.append(int(pi))
                if src:
                    et_f = (tname, f"fk_{fk_col}", parent_table)
                    data[et_f].edge_index = torch.tensor(
                        [src, dst], dtype=torch.long)
                    et_r = (parent_table, f"rev_fk_{fk_col}", tname)
                    data[et_r].edge_index = torch.tensor(
                        [dst, src], dtype=torch.long)
        return data, node_id_maps

    # ── Training helpers ─────────────────────────────────────────────────

    def get_x_dict(batch, node_types):
        x_dict = {}
        for nt in node_types:
            store = batch[nt]
            if hasattr(store, "tf") and store.tf is not None:
                tf = store.tf
                parts = []
                src = (tf.x_dict if hasattr(tf, "x_dict")
                       else tf.feat_dict if hasattr(tf, "feat_dict")
                       else {})
                for val in src.values():
                    if val is not None and val.numel() > 0:
                        v = val.float()
                        if v.dim() > 2:
                            v = v.view(v.size(0), -1)
                        parts.append(v)
                if parts:
                    x_dict[nt] = torch.cat(parts, dim=-1)
                else:
                    x_dict[nt] = torch.zeros(
                        store.num_nodes, 1, device=DEVICE)
            elif hasattr(store, "x") and store.x is not None:
                x_dict[nt] = store.x.float()
            else:
                x_dict[nt] = torch.zeros(
                    store.num_nodes, 1, device=DEVICE)
        return x_dict

    def train_epoch(model, loader, optimizer, loss_fn, entity_table,
                    node_types):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            if steps >= MAX_TRAIN_STEPS:
                break
            try:
                batch = batch.to(DEVICE)
                x_dict = get_x_dict(batch, node_types)
                ei_dict = {et: batch[et].edge_index
                           for et in batch.edge_types
                           if hasattr(batch[et], "edge_index")}
                if entity_table not in x_dict:
                    steps += 1
                    continue
                optimizer.zero_grad()
                pred = model(x_dict, ei_dict, entity_table)
                if not (hasattr(batch[entity_table], "y")
                        and batch[entity_table].y is not None):
                    steps += 1
                    continue
                labels = batch[entity_table].y.float()
                ml = min(len(pred), len(labels))
                pred, labels = pred[:ml], labels[:ml]
                mask = ~torch.isnan(labels)
                if mask.sum() == 0:
                    steps += 1
                    continue
                loss = loss_fn(pred[mask], labels[mask])
                if torch.isnan(loss) or torch.isinf(loss):
                    steps += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM in train step, skipping")
                torch.cuda.empty_cache()
                gc.collect()
                steps += 1
            except Exception:
                logger.exception(f"Error in train step {steps}")
                steps += 1
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate_model(model, loader, entity_table, task_type, node_types):
        from sklearn.metrics import roc_auc_score, average_precision_score
        model.eval()
        all_p, all_l = [], []
        for batch in loader:
            try:
                batch = batch.to(DEVICE)
                x_dict = get_x_dict(batch, node_types)
                ei_dict = {et: batch[et].edge_index
                           for et in batch.edge_types
                           if hasattr(batch[et], "edge_index")}
                if entity_table not in x_dict:
                    continue
                pred = model(x_dict, ei_dict, entity_table)
                if task_type == "classification":
                    pred = torch.sigmoid(pred)
                if not (hasattr(batch[entity_table], "y")
                        and batch[entity_table].y is not None):
                    continue
                labels = batch[entity_table].y.float()
                ml = min(len(pred), len(labels))
                all_p.append(pred[:ml].cpu())
                all_l.append(labels[:ml].cpu())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except Exception:
                logger.exception("Eval error")
        if not all_p:
            return ({"rmse": float("inf"), "mae": float("inf"),
                     "r2": -1.0, "primary": float("inf")}
                    if task_type == "regression" else
                    {"auroc": 0.5, "ap": 0.0, "primary": 0.5})
        preds = torch.cat(all_p)
        labels = torch.cat(all_l)
        mask = ~torch.isnan(labels)
        if mask.sum() == 0:
            return ({"rmse": float("inf"), "mae": float("inf"),
                     "r2": -1.0, "primary": float("inf")}
                    if task_type == "regression" else
                    {"auroc": 0.5, "ap": 0.0, "primary": 0.5})
        if task_type == "regression":
            rmse = torch.sqrt(F.mse_loss(preds[mask], labels[mask])).item()
            mae = F.l1_loss(preds[mask], labels[mask]).item()
            ss_res = ((labels[mask] - preds[mask]) ** 2).sum()
            ss_tot = ((labels[mask] - labels[mask].mean()) ** 2).sum()
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
            return {"rmse": round(rmse, 6), "mae": round(mae, 6),
                    "r2": round(r2, 6), "primary": rmse}
        else:
            p_np = preds[mask].numpy()
            l_np = labels[mask].numpy()
            try:
                auroc = roc_auc_score(l_np, p_np)
            except ValueError:
                auroc = 0.5
            try:
                ap = average_precision_score(l_np, p_np)
            except ValueError:
                ap = 0.0
            return {"auroc": round(auroc, 6), "ap": round(ap, 6),
                    "primary": auroc}

    def train_and_evaluate(model, train_loader, val_loader, test_loader,
                           entity_table, task_type, node_types, seed):
        torch.manual_seed(seed)
        if HAS_GPU:
            torch.cuda.manual_seed(seed)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = (nn.MSELoss() if task_type == "regression"
                   else nn.BCEWithLogitsLoss())
        best_val, best_state, patience_cnt = None, None, 0
        for epoch in range(EPOCHS):
            t_ep = time.time()
            avg_loss = train_epoch(
                model, train_loader, optimizer, loss_fn,
                entity_table, node_types)
            val_m = evaluate_model(
                model, val_loader, entity_table, task_type, node_types)
            primary = val_m["primary"]
            logger.info(
                f"    Ep {epoch}: loss={avg_loss:.4f} "
                f"val={primary:.4f} {time.time()-t_ep:.1f}s")
            better = (best_val is None
                      or (task_type == "regression" and primary < best_val)
                      or (task_type != "regression" and primary > best_val))
            if better:
                best_val = primary
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    logger.info(f"    Early stop at epoch {epoch}")
                    break
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        return evaluate_model(
            model, test_loader, entity_table, task_type, node_types)

    def init_lazy(model, loader, entity_table, node_types):
        model = model.to(DEVICE)
        for batch in loader:
            try:
                batch = batch.to(DEVICE)
                x_dict = get_x_dict(batch, node_types)
                ei_dict = {et: batch[et].edge_index
                           for et in batch.edge_types
                           if hasattr(batch[et], "edge_index")}
                with torch.no_grad():
                    model(x_dict, ei_dict, entity_table)
                break
            except Exception:
                logger.exception("Lazy init failed, trying next batch")
        model = model.cpu()
        torch.cuda.empty_cache()

    # ── Main GNN pipeline ────────────────────────────────────────────────

    try:
        dataset = get_dataset("rel-avito", download=True)
        db = dataset.get_db()
        logger.info(f"  Tables: {list(db.table_dict.keys())}")

        # Subsample large tables
        for tname, max_rows in TABLE_SAMPLE_LIMITS.items():
            if tname in db.table_dict:
                t = db.table_dict[tname]
                if len(t.df) > max_rows:
                    logger.info(
                        f"  Subsample {tname}: {len(t.df)} -> {max_rows}")
                    t.df = t.df.sample(
                        n=max_rows, random_state=42).reset_index(drop=True)
        gc.collect()

        # Try relbench graph builder first, fall back to manual
        data = None
        try:
            from relbench.modeling.graph import make_pkey_fkey_graph
            from torch_frame import stype
            col_to_stype_dict = {}
            for tname in db.table_dict:
                table = db.table_dict[tname]
                df = table.df
                cts = {}
                fks = (dict(table.fkey_col_to_pkey_table)
                       if hasattr(table, "fkey_col_to_pkey_table") else {})
                for col in df.columns:
                    if col in (table.pkey_col, table.time_col):
                        continue
                    if col in fks:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col].dtype):
                        cts[col] = stype.numerical
                    elif df[col].dtype == object and df[col].nunique() < 5000:
                        cts[col] = stype.categorical
                if cts:
                    col_to_stype_dict[tname] = cts
            cache_dir = str(WORKSPACE / "avito_cache")
            os.makedirs(cache_dir, exist_ok=True)
            data, _ = make_pkey_fkey_graph(
                db, col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=None, cache_dir=cache_dir)
            logger.info(f"  Graph (relbench): {data}")
        except Exception:
            logger.exception("  make_pkey_fkey_graph failed, manual build")
            data, _ = build_manual_heterodata(db.table_dict)
            logger.info(f"  Graph (manual): {data}")

        # Load tasks
        task_ad_ctr = get_task("rel-avito", "ad-ctr", download=True)
        task_user_visits = get_task("rel-avito", "user-visits", download=True)

        metadata = data.metadata()
        node_types = metadata[0]
        edge_types = metadata[1]

        # Identify PRMP edges
        all_prmp = [et for et in edge_types
                    if "rev" in str(et[1]).lower()]
        if not all_prmp:
            all_prmp = edge_types[:len(edge_types) // 2]

        sel_prmp = []
        for lid in selective_links:
            info = r2_results[lid]
            pt, ct = info["parent_table"], info["child_table"]
            for et in all_prmp:
                if (pt.lower() in str(et).lower()
                        and ct.lower() in str(et).lower()):
                    sel_prmp.append(et)
        if not sel_prmp:
            sel_prmp = all_prmp[:3]

        results: dict = {}
        task_configs = [
            ("ad-ctr", task_ad_ctr, "AdsInfo", "regression"),
            ("user-visits", task_user_visits, "UserInfo", "classification"),
        ]

        for task_name, task_obj, entity_table, task_type in task_configs:
            logger.info(f"  Task: {task_name}")
            try:
                train_tbl = task_obj.get_table("train")
                val_tbl = task_obj.get_table("val")
                test_tbl = task_obj.get_table("test")
            except Exception:
                logger.exception(f"  Failed tables for {task_name}")
                continue

            # Assign labels
            entity_col = task_obj.entity_col
            target_col = task_obj.target_col
            n_nodes = data[entity_table].num_nodes
            labels_t = torch.full((n_nodes,), float("nan"))
            for tbl in [train_tbl, val_tbl, test_tbl]:
                df = tbl.df
                if entity_col in df.columns and target_col in df.columns:
                    ids = df[entity_col].values
                    tgts = df[target_col].values
                    for idx, tgt in zip(ids, tgts):
                        if 0 <= idx < n_nodes and not pd.isna(tgt):
                            labels_t[idx] = float(tgt)
            data[entity_table].y = labels_t

            try:
                train_ids = torch.tensor(
                    train_tbl.df[entity_col].values, dtype=torch.long)
                val_ids = torch.tensor(
                    val_tbl.df[entity_col].values, dtype=torch.long)
                test_ids = torch.tensor(
                    test_tbl.df[entity_col].values, dtype=torch.long)
                nw = min(4, NUM_CPUS - 1)
                train_loader = NeighborLoader(
                    data, num_neighbors=NUM_NEIGHBORS,
                    input_nodes=(entity_table, train_ids),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
                val_loader = NeighborLoader(
                    data, num_neighbors=NUM_NEIGHBORS,
                    input_nodes=(entity_table, val_ids),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
                test_loader = NeighborLoader(
                    data, num_neighbors=NUM_NEIGHBORS,
                    input_nodes=(entity_table, test_ids),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
            except Exception:
                logger.exception(f"  Loader creation failed for {task_name}")
                continue

            model_cfgs = [
                ("A_StandardSAGE",
                 lambda: StandardHeteroSAGE(metadata, HIDDEN_CHANNELS,
                                            NUM_LAYERS, 1)),
                ("B_PRMP_Full",
                 lambda: PRMPHeteroModel(metadata, HIDDEN_CHANNELS,
                                         NUM_LAYERS, 1, all_prmp)),
                ("C_PRMP_Selective",
                 lambda: PRMPHeteroModel(metadata, HIDDEN_CHANNELS,
                                         NUM_LAYERS, 1, sel_prmp)),
            ]

            for model_name, model_factory in model_cfgs:
                logger.info(f"    Model: {model_name}")
                seed_results = []
                for seed in SEEDS:
                    try:
                        model = model_factory()
                        init_lazy(model, train_loader,
                                  entity_table, node_types)
                        metrics = train_and_evaluate(
                            model, train_loader, val_loader, test_loader,
                            entity_table, task_type, node_types, seed)
                        seed_results.append(metrics)
                        logger.info(f"      seed={seed}: {metrics}")
                    except Exception:
                        logger.exception(
                            f"      Failed {model_name} seed={seed}")
                    finally:
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

                if seed_results:
                    agg = {}
                    for key in seed_results[0]:
                        if key == "primary":
                            continue
                        vals = [r[key] for r in seed_results]
                        agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
                        agg[f"{key}_std"] = round(float(np.std(vals)), 6)
                    try:
                        tmp = model_factory()
                        init_lazy(tmp, train_loader,
                                  entity_table, node_types)
                        agg["param_count"] = sum(
                            p.numel() for p in tmp.parameters())
                        del tmp
                    except Exception:
                        agg["param_count"] = -1
                    results[f"{task_name}__{model_name}"] = agg

            del train_loader, val_loader, test_loader
            gc.collect()

        return results

    except Exception:
        logger.exception("GNN experiment pipeline failed entirely")
        return {}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t0_total = time.time()
    logger.info("=" * 60)
    logger.info("PRMP vs SAGEConv Benchmark on RelBench rel-avito")
    logger.info("=" * 60)

    # ── Phase 1: R² Diagnostic ───────────────────────────────────────────
    logger.info("Loading dependency data...")
    dep_data = json.loads(DEP_DATA_PATH.read_text())
    n_examples = len(dep_data["datasets"][0]["examples"])
    logger.info(f"Loaded {n_examples} examples from dependency data")

    r2_results, selective_links, examples_with_preds = \
        compute_r2_diagnostic(dep_data)
    del dep_data
    gc.collect()

    # ── Phase 2-5: GNN Training ──────────────────────────────────────────
    gnn_results = try_gnn_experiments(r2_results, selective_links)
    gnn_status = "success" if gnn_results else "skipped_or_failed"
    logger.info(f"GNN status: {gnn_status}, results: {len(gnn_results)}")

    # ── Phase 6: Diagnostic Correlations ─────────────────────────────────
    diagnostic_correlation = {}
    if gnn_results:
        for tn, mk in [("ad-ctr", "rmse_mean"),
                        ("user-visits", "auroc_mean")]:
            bl = gnn_results.get(f"{tn}__A_StandardSAGE", {}).get(mk)
            pf = gnn_results.get(f"{tn}__B_PRMP_Full", {}).get(mk)
            ps = gnn_results.get(f"{tn}__C_PRMP_Selective", {}).get(mk)
            if bl and pf:
                if tn == "ad-ctr":
                    imp_f = (bl - pf) / (abs(bl) + 1e-8)
                    imp_s = ((bl - ps) / (abs(bl) + 1e-8)
                             if ps else None)
                else:
                    imp_f = (pf - bl) / (abs(bl) + 1e-8)
                    imp_s = ((ps - bl) / (abs(bl) + 1e-8)
                             if ps else None)
                diagnostic_correlation[
                    f"{tn}_prmp_full_rel_improvement"] = round(imp_f, 6)
                if imp_s is not None:
                    diagnostic_correlation[
                        f"{tn}_prmp_selective_rel_improvement"] = round(
                            imp_s, 6)

    # ── Phase 7: Assemble Output ─────────────────────────────────────────
    logger.info("Phase 7: Assembling output")
    total_time = round(time.time() - t0_total, 1)

    output = {
        "metadata": {
            "experiment": "PRMP vs SAGEConv on rel-avito",
            "dataset": "rel-avito",
            "tasks": ["ad-ctr (regression)", "user-visits (classification)"],
            "models": ["A_StandardSAGE", "B_PRMP_Full", "C_PRMP_Selective"],
            "seeds": SEEDS,
            "hidden_channels": HIDDEN_CHANNELS,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "num_neighbors": NUM_NEIGHBORS,
            "r2_diagnostic": r2_results,
            "selective_prmp_links": selective_links,
            "gnn_results": gnn_results,
            "gnn_status": gnn_status,
            "diagnostic_correlation": diagnostic_correlation,
            "total_time_seconds": total_time,
        },
        "datasets": [
            {
                "dataset": "rel-avito",
                "examples": examples_with_preds,
            }
        ],
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    fsize = OUTPUT_FILE.stat().st_size
    logger.info(
        f"Output: {OUTPUT_FILE} ({fsize / 1e6:.1f} MB, "
        f"{len(examples_with_preds)} examples)")
    logger.info(f"Total time: {total_time}s")
    logger.info("DONE")


if __name__ == "__main__":
    main()
