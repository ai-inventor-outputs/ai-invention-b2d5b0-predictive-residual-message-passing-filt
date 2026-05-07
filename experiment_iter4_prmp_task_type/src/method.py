#!/usr/bin/env python3
"""PRMP Task-Type Effect: Why PRMP Helps Regression but Not Classification.

Systematically investigates the task-type effect on PRMP using two RelBench datasets
(rel-f1 and rel-stack) as controlled natural experiments.

On rel-f1: train SAGEConv baseline and PRMP on 3 tasks (driver-position regression,
driver-dnf classification, driver-top3 classification) sharing the same graph.

On rel-stack: train on user-engagement (classification) and post-votes (regression).

For each training run, logs per-epoch: train/val loss, prediction MLP R²,
gradient norms through residual vs aggregation paths, embedding statistics
(variance, effective rank via SVD), and test metric learning curves.
"""

import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
import gc
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats as scipy_stats

# ── Logging ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware ──────────────────────────────────────────────────────────────
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


import psutil
NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(0.88, 0.88))
else:
    VRAM_GB = 0

RAM_BUDGET = int(min(38, TOTAL_RAM_GB * 0.8) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU}")
if HAS_GPU:
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM={VRAM_GB:.1f}GB")

# ── PyG + relbench imports ────────────────────────────────────────────────
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, MLP
from torch_geometric.nn import LayerNorm as PyGLayerNorm
from torch_geometric.nn.conv import MessagePassing

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.nn import HeteroEncoder
from torch_frame import stype as tf_stype


# ═══════════════════════════════════════════════════════════════════════════
# Instrumented PRMP Conv
# ═══════════════════════════════════════════════════════════════════════════

class InstrumentedPRMPConv(MessagePassing):
    """PRMPConv with per-forward-pass diagnostics collection."""
    def __init__(self, in_ch_src: int, in_ch_dst: int, out_ch: int):
        super().__init__(aggr='mean')
        hidden = min(in_ch_dst, in_ch_src)
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_ch_dst, hidden), nn.ReLU(),
            nn.Linear(hidden, in_ch_src),
        )
        # Zero-init so PRMP starts identical to standard aggregation
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)
        self.norm = nn.LayerNorm(in_ch_src)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_ch_dst + in_ch_src, out_ch), nn.ReLU(),
        )
        # Instrumentation storage
        self.last_r2 = None
        self.last_residual_var = None
        self.last_pred_var = None

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        predicted = self.pred_mlp(x_i.detach())
        residual = x_j - predicted
        # Compute and store R² diagnostic
        with torch.no_grad():
            ss_res = ((residual) ** 2).sum()
            ss_tot = ((x_j - x_j.mean(0)) ** 2).sum()
            self.last_r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
            self.last_residual_var = residual.var().item()
            self.last_pred_var = predicted.var().item()
        return self.norm(residual)

    def update(self, aggr_out, x):
        if isinstance(x, tuple):
            dst_feat = x[1]
        else:
            dst_feat = x
        # Handle 1D edge case (single node)
        if dst_feat.dim() == 1:
            dst_feat = dst_feat.unsqueeze(0)
        if aggr_out.dim() == 1:
            aggr_out = aggr_out.unsqueeze(0)
        return self.update_mlp(torch.cat([dst_feat, aggr_out], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# Model with gradient hooks and embedding stats
# ═══════════════════════════════════════════════════════════════════════════

class InstrumentedModel(nn.Module):
    """HeteroGNN supporting 'standard' (SAGEConv) and 'prmp' (InstrumentedPRMPConv).
    Includes backward hooks for gradient norm tracking."""

    def __init__(self, data, col_stats_dict, channels: int = 64,
                 out_channels: int = 1, num_layers: int = 2,
                 variant: str = "standard"):
        super().__init__()
        self.variant = variant
        self.channels = channels
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict
                for nt in data.node_types if hasattr(data[nt], 'tf')
            },
            node_to_col_stats=col_stats_dict,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in data.edge_types:
                if variant == "standard":
                    conv_dict[et] = SAGEConv((channels, channels), channels, aggr="mean")
                elif variant == "prmp":
                    conv_dict[et] = InstrumentedPRMPConv(channels, channels, channels)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                nt: PyGLayerNorm(channels, mode="node") for nt in data.node_types
            }))

        self.head = MLP(channels, out_channels=out_channels, num_layers=1)

        # Gradient tracking storage
        self.grad_norms = {}

    def register_gradient_hooks(self):
        """Register backward hooks on pred_mlp and update_mlp of PRMP convs."""
        hooks = []
        if self.variant != "prmp":
            return hooks
        for li, conv_layer in enumerate(self.convs):
            for et_key, conv in conv_layer.convs.items():
                if isinstance(conv, InstrumentedPRMPConv):
                    def make_hook(layer_idx, edge_type, path_name):
                        def hook_fn(module, grad_input, grad_output):
                            key = f"L{layer_idx}/{edge_type}/{path_name}"
                            if grad_output[0] is not None:
                                self.grad_norms[key] = grad_output[0].norm().item()
                        return hook_fn
                    hooks.append(conv.pred_mlp[-1].register_full_backward_hook(
                        make_hook(li, str(et_key), 'pred_mlp')))
                    hooks.append(conv.update_mlp[0].register_full_backward_hook(
                        make_hook(li, str(et_key), 'update_mlp')))
        return hooks

    def collect_prmp_r2(self) -> dict:
        """Collect R² from all InstrumentedPRMPConv layers after a forward pass."""
        r2s = {}
        for li, conv_layer in enumerate(self.convs):
            for et_key, conv in conv_layer.convs.items():
                if isinstance(conv, InstrumentedPRMPConv) and conv.last_r2 is not None:
                    r2s[f"L{li}/{et_key}"] = round(conv.last_r2, 6)
        return r2s

    def _run_encoder_and_convs(self, batch):
        """Run encoder + conv layers (no head). Returns x_dict."""
        x_dict = self.encoder({
            nt: batch[nt].tf for nt in batch.node_types if hasattr(batch[nt], 'tf')
        })
        for k in x_dict:
            x_dict[k] = torch.nan_to_num(x_dict[k], nan=0.0, posinf=1.0, neginf=-1.0)
        for conv, norm_dict in zip(self.convs, self.norms):
            x_new = conv(x_dict, batch.edge_index_dict)
            x_out = {}
            for k, v in x_new.items():
                v = torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
                if k in norm_dict and v.size(0) > 0:
                    x_out[k] = norm_dict[k](v).relu()
                else:
                    x_out[k] = v.relu()
            x_dict = x_out
        return x_dict

    def compute_embedding_stats(self, batch, entity_table: str) -> dict:
        """After forward, compute embedding variance and effective rank via SVD."""
        try:
            x_dict = self._run_encoder_and_convs(batch)
            if entity_table not in x_dict:
                return {}
            seed_size = batch[entity_table].batch_size
            embeddings = x_dict[entity_table][:seed_size]
            if embeddings.shape[0] < 2:
                return {}

            # Variance
            var_per_dim = embeddings.var(dim=0)
            mean_var = var_per_dim.mean().item()

            # Effective rank via SVD
            centered = embeddings - embeddings.mean(0)
            # Clamp to prevent SVD issues
            centered = torch.nan_to_num(centered, nan=0.0)
            try:
                U, S, V = torch.linalg.svd(centered, full_matrices=False)
                S_norm = S / S.sum().clamp(min=1e-10)
                entropy = -(S_norm * S_norm.clamp(min=1e-10).log()).sum()
                effective_rank = entropy.exp().item()
            except Exception:
                effective_rank = 0.0

            return {
                'mean_variance': round(mean_var, 6),
                'effective_rank': round(effective_rank, 4),
                'num_embeddings': int(embeddings.shape[0]),
                'embed_dim': int(embeddings.shape[1]),
            }
        except Exception as e:
            logger.debug(f"Embedding stats failed: {e}")
            return {}

    def forward(self, batch, entity_table: str):
        x_dict = self._run_encoder_and_convs(batch)
        seed_size = batch[entity_table].batch_size
        return self.head(x_dict[entity_table][:seed_size])


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════

CHANNELS = 64
NUM_LAYERS = 2
NUM_NEIGHBORS = [10, 5]
BATCH_SIZE = 512
MAX_EPOCHS = 20
PATIENCE = 7
LR = 0.001
MAX_TRAIN_BATCHES = 50
MAX_VAL_BATCHES = 30
SEEDS = [42]
VARIANTS = ["standard", "prmp"]

EXPERIMENTS = [
    # === rel-f1 (same graph, 3 tasks) ===
    {
        'dataset_name': 'rel-f1',
        'task_name': 'driver-position',
        'task_type': 'regression',
        'metric': 'mae',
        'higher_better': False,
    },
    {
        'dataset_name': 'rel-f1',
        'task_name': 'driver-dnf',
        'task_type': 'classification',
        'metric': 'auroc',
        'higher_better': True,
    },
    {
        'dataset_name': 'rel-f1',
        'task_name': 'driver-top3',
        'task_type': 'classification',
        'metric': 'auroc',
        'higher_better': True,
    },
    # === rel-stack (separate graph, 2 tasks) ===
    {
        'dataset_name': 'rel-stack',
        'task_name': 'user-engagement',
        'task_type': 'classification',
        'metric': 'auroc',
        'higher_better': True,
    },
    {
        'dataset_name': 'rel-stack',
        'task_name': 'post-votes',
        'task_type': 'regression',
        'metric': 'mae',
        'higher_better': False,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def _zero_nan_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            bad = ~torch.isfinite(p.grad)
            if bad.any():
                p.grad[bad] = 0.0


def _compute_metric(preds: np.ndarray, labels: np.ndarray, task_config: dict) -> float:
    if task_config['metric'] == 'auroc':
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(labels, preds)
        except Exception:
            return 0.5
    elif task_config['metric'] == 'mae':
        return float(np.mean(np.abs(labels - preds)))
    return float('nan')


def _compute_loss_val(preds: np.ndarray, labels: np.ndarray, task_config: dict) -> float:
    if task_config['task_type'] == 'classification':
        # BCE loss
        eps = 1e-7
        preds_clipped = np.clip(preds, eps, 1 - eps)
        return float(-np.mean(labels * np.log(preds_clipped) +
                              (1 - labels) * np.log(1 - preds_clipped)))
    else:
        return float(np.mean(np.abs(labels - preds)))


def build_loaders(data, task, batch_size: int = BATCH_SIZE) -> dict:
    loaders = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        ti = get_node_train_table_input(table=table, task=task)
        loaders[split] = NeighborLoader(
            data, num_neighbors=NUM_NEIGHBORS, time_attr="time",
            input_nodes=ti.nodes, input_time=ti.time, transform=ti.transform,
            batch_size=batch_size, temporal_strategy="last",
            shuffle=(split == "train"), num_workers=0,
        )
    return loaders


def instrumented_train(model: InstrumentedModel, loaders: dict,
                       task_config: dict, entity_table: str) -> tuple:
    """Train with per-epoch collection of all instrumentation signals."""
    if task_config['task_type'] == 'classification':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    hooks = model.register_gradient_hooks()

    epoch_records = []
    best_val = None
    best_state = copy.deepcopy(model.state_dict())
    patience_ctr = 0

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()

        # === TRAIN ===
        model.train()
        total_loss, count = 0.0, 0
        epoch_grad_norms = {}

        for bi, batch in enumerate(loaders['train']):
            if bi >= MAX_TRAIN_BATCHES:
                break
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            model.grad_norms = {}  # reset per batch

            pred = model(batch, entity_table).view(-1)

            if not hasattr(batch[entity_table], 'y') or batch[entity_table].y is None:
                continue
            y = batch[entity_table].y.float()
            valid = torch.isfinite(y)
            if valid.sum() == 0:
                continue
            pred_v, y_v = pred[valid], y[valid]

            loss = loss_fn(pred_v, y_v)
            if not torch.isfinite(loss):
                continue
            loss.backward()

            # Collect gradient norms from hooks
            for k, v in model.grad_norms.items():
                epoch_grad_norms.setdefault(k, []).append(v)

            _zero_nan_grads(model)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * pred_v.size(0)
            count += pred_v.size(0)

        avg_train_loss = total_loss / max(count, 1)

        # === COLLECT PRMP R² (from last train batch) ===
        prmp_r2 = model.collect_prmp_r2() if model.variant == "prmp" else {}

        # === VALIDATE ===
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for bi, batch in enumerate(loaders['val']):
                if bi >= MAX_VAL_BATCHES:
                    break
                batch = batch.to(DEVICE)
                pred = model(batch, entity_table).view(-1)
                if not hasattr(batch[entity_table], 'y') or batch[entity_table].y is None:
                    continue
                y = batch[entity_table].y.float()
                if task_config['task_type'] == 'classification':
                    pred = torch.sigmoid(pred)
                valid = torch.isfinite(y)
                if valid.sum() > 0:
                    val_preds.append(pred[valid].cpu())
                    val_labels.append(y[valid].cpu())

        if val_preds:
            vp = torch.cat(val_preds).numpy()
            vl = torch.cat(val_labels).numpy()
            vp = np.nan_to_num(vp, nan=0.5 if task_config['task_type'] == 'classification' else 0.0)
            val_metric = _compute_metric(vp, vl, task_config)
            val_loss = _compute_loss_val(vp, vl, task_config)
        else:
            val_metric = 0.5 if task_config['task_type'] == 'classification' else 999.0
            val_loss = 999.0

        # === TEST METRIC (per epoch, for learning curve) ===
        test_preds, test_labels = [], []
        with torch.no_grad():
            for bi, batch in enumerate(loaders['test']):
                if bi >= MAX_VAL_BATCHES:
                    break
                batch = batch.to(DEVICE)
                pred = model(batch, entity_table).view(-1)
                if not hasattr(batch[entity_table], 'y') or batch[entity_table].y is None:
                    continue
                y = batch[entity_table].y.float()
                if task_config['task_type'] == 'classification':
                    pred = torch.sigmoid(pred)
                valid = torch.isfinite(y)
                if valid.sum() > 0:
                    test_preds.append(pred[valid].cpu())
                    test_labels.append(y[valid].cpu())

        if test_preds:
            tp = torch.cat(test_preds).numpy()
            tl = torch.cat(test_labels).numpy()
            tp = np.nan_to_num(tp, nan=0.5 if task_config['task_type'] == 'classification' else 0.0)
            test_metric = _compute_metric(tp, tl, task_config)
        else:
            test_metric = 0.5 if task_config['task_type'] == 'classification' else 999.0

        # === EMBEDDING STATS (every 5 epochs + first + last) ===
        embed_stats = {}
        if epoch == 0 or epoch == MAX_EPOCHS - 1 or epoch % 5 == 0:
            with torch.no_grad():
                try:
                    sample_batch = next(iter(loaders['val'])).to(DEVICE)
                    embed_stats = model.compute_embedding_stats(sample_batch, entity_table)
                except Exception:
                    pass

        # === AGGREGATE GRADIENT NORMS ===
        avg_grad_norms = {k: round(float(np.mean(v)), 6) for k, v in epoch_grad_norms.items()}

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(avg_train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_metric': round(float(val_metric), 6),
            'test_metric': round(float(test_metric), 6),
            'prmp_r2': prmp_r2,
            'grad_norms': avg_grad_norms,
            'embed_stats': embed_stats,
            'time_s': round(time.time() - t0, 1),
        }
        epoch_records.append(epoch_record)

        logger.info(
            f"  ep{epoch:2d} loss={avg_train_loss:.4f} val_m={val_metric:.4f} "
            f"test_m={test_metric:.4f} {time.time()-t0:.1f}s"
        )

        # Early stopping on val_metric
        improved = False
        if best_val is None:
            improved = True
        elif task_config['higher_better'] and val_metric > best_val:
            improved = True
        elif not task_config['higher_better'] and val_metric < best_val:
            improved = True
        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    # Remove hooks
    for h in hooks:
        h.remove()

    # Restore best model
    model.load_state_dict(best_state)
    return model, best_val, epoch_records


@torch.no_grad()
def collect_test_predictions(model, loader, task_config, entity_table,
                             max_instances: int = 30) -> list:
    """Collect per-instance test predictions."""
    model.eval()
    instances = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        if not hasattr(batch[entity_table], 'y') or batch[entity_table].y is None:
            continue
        y = batch[entity_table].y.float()
        if task_config['task_type'] == 'classification':
            pred = torch.sigmoid(pred)
        valid = torch.isfinite(y)
        for i in range(valid.size(0)):
            if valid[i]:
                p = pred[i].cpu().item()
                l = y[i].cpu().item()
                if np.isfinite(p) and np.isfinite(l):
                    instances.append((p, l))
                    if len(instances) >= max_instances:
                        return instances
    return instances


def compute_cross_task_analysis(all_run_summaries: list) -> dict:
    """Compute cross-task analysis: PRMP delta per task type."""
    analysis = {
        'per_task': {},
        'regression_deltas': [],
        'classification_deltas': [],
        'regression_tasks': [],
        'classification_tasks': [],
    }

    # Group summaries by (dataset, task)
    task_results = {}
    for s in all_run_summaries:
        key = (s['dataset'], s['task'])
        if key not in task_results:
            task_results[key] = {'standard': None, 'prmp': None,
                                 'task_type': s['task_type'],
                                 'metric': s.get('metric', ''),
                                 'higher_better': s.get('higher_better', True)}
        task_results[key][s['variant']] = s['best_val']

    for (ds, task_name), info in task_results.items():
        std_val = info['standard']
        prmp_val = info['prmp']
        if std_val is not None and prmp_val is not None:
            # Delta: positive means PRMP is better
            if info['higher_better']:
                delta = prmp_val - std_val
            else:
                delta = std_val - prmp_val  # Lower is better, so std-prmp = improvement

            task_key = f"{ds}/{task_name}"
            analysis['per_task'][task_key] = {
                'task_type': info['task_type'],
                'metric': info['metric'],
                'standard': round(std_val, 6),
                'prmp': round(prmp_val, 6),
                'delta': round(delta, 6),
                'prmp_better': delta > 0,
            }

            if info['task_type'] == 'regression':
                analysis['regression_deltas'].append(delta)
                analysis['regression_tasks'].append(task_key)
            else:
                analysis['classification_deltas'].append(delta)
                analysis['classification_tasks'].append(task_key)

    # Compute summary statistics
    reg_d = analysis['regression_deltas']
    cls_d = analysis['classification_deltas']

    analysis['regression_mean_delta'] = round(float(np.mean(reg_d)), 6) if reg_d else None
    analysis['classification_mean_delta'] = round(float(np.mean(cls_d)), 6) if cls_d else None

    # Test if regression delta > classification delta
    if len(reg_d) >= 1 and len(cls_d) >= 1:
        analysis['reg_vs_cls_delta_diff'] = round(
            float(np.mean(reg_d)) - float(np.mean(cls_d)), 6)
    else:
        analysis['reg_vs_cls_delta_diff'] = None

    # Clean up arrays for JSON (convert numpy)
    analysis['regression_deltas'] = [round(d, 6) for d in reg_d]
    analysis['classification_deltas'] = [round(d, 6) for d in cls_d]

    return analysis


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PRMP Task-Type Effect Experiment")
    logger.info("=" * 60)

    all_examples = []
    all_run_summaries = []  # For cross-task analysis
    dataset_cache = {}

    for exp_idx, exp_config in enumerate(EXPERIMENTS):
        elapsed = time.time() - start_time
        if elapsed > 6000:  # 100 min guard
            logger.warning(f"Time budget exceeded ({elapsed:.0f}s), stopping")
            break

        ds_name = exp_config['dataset_name']
        task_name = exp_config['task_name']
        logger.info(f"\n{'='*60}")
        logger.info(f"[{exp_idx+1}/{len(EXPERIMENTS)}] {ds_name}/{task_name} ({exp_config['task_type']})")

        # Load/cache dataset + graph
        if ds_name not in dataset_cache:
            logger.info(f"Loading {ds_name} dataset...")
            t_load = time.time()
            try:
                dataset = get_dataset(ds_name, download=True)
                db = dataset.get_db()
                logger.info(f"  Tables: {list(db.table_dict.keys())}")

                col_to_stype = get_stype_proposal(db)
                # Remove text columns to avoid GloVe dependency
                for tbl in list(col_to_stype.keys()):
                    text_cols = [c for c, s in col_to_stype[tbl].items()
                                 if s in (tf_stype.text_embedded, tf_stype.text_tokenized)]
                    for c in text_cols:
                        del col_to_stype[tbl][c]
                        logger.debug(f"  Removed text col: {tbl}.{c}")

                cache_dir = str(WORKSPACE / "cache" / ds_name)
                logger.info(f"  Building heterogeneous graph (cache: {cache_dir})...")
                data, col_stats = make_pkey_fkey_graph(
                    db, col_to_stype_dict=col_to_stype,
                    text_embedder_cfg=None, cache_dir=cache_dir,
                )
                dataset_cache[ds_name] = {
                    'data': data, 'col_stats': col_stats, 'dataset': dataset,
                }
                logger.info(f"  Dataset loaded in {time.time()-t_load:.1f}s")
                logger.info(f"  Node types: {list(data.node_types)}")
                logger.info(f"  Edge types: {len(data.edge_types)}")
            except Exception:
                logger.exception(f"Failed to load {ds_name}, skipping all its tasks")
                # Mark remaining tasks for this dataset as skipped
                continue

        if ds_name not in dataset_cache:
            logger.warning(f"Skipping {ds_name}/{task_name} — dataset not loaded")
            continue

        cached = dataset_cache[ds_name]
        data, col_stats = cached['data'], cached['col_stats']

        # Load task + build loaders
        try:
            task = get_task(ds_name, task_name, download=True)
            entity_table = task.entity_table
            logger.info(f"  Entity table: {entity_table}")
            loaders = build_loaders(data, task, BATCH_SIZE)
        except Exception:
            logger.exception(f"Failed to load task {task_name}")
            continue

        for variant in VARIANTS:
            elapsed = time.time() - start_time
            if elapsed > 6000:
                logger.warning("Time budget exceeded, stopping")
                break

            logger.info(f"\n--- {ds_name}/{task_name}/{variant} ---")
            t_run = time.time()

            try:
                torch.manual_seed(42)
                np.random.seed(42)
                if HAS_GPU:
                    torch.cuda.manual_seed(42)

                model = InstrumentedModel(
                    data=data, col_stats_dict=col_stats,
                    channels=CHANNELS, out_channels=1,
                    num_layers=NUM_LAYERS, variant=variant,
                ).to(DEVICE)

                model, best_val, epoch_records = instrumented_train(
                    model, loaders, exp_config, entity_table,
                )
                run_time = time.time() - t_run
                logger.info(f"  {variant}: best_val={best_val:.4f} in {run_time:.1f}s")

                # Collect per-instance test predictions
                test_instances = collect_test_predictions(
                    model, loaders['test'], exp_config, entity_table,
                    max_instances=15,
                )
                if not test_instances:
                    test_instances = collect_test_predictions(
                        model, loaders['val'], exp_config, entity_table,
                        max_instances=15,
                    )

                # Build per-epoch examples
                for rec in epoch_records:
                    example = {
                        'input': json.dumps({
                            'dataset': ds_name, 'task': task_name,
                            'variant': variant, 'task_type': exp_config['task_type'],
                            'metric': exp_config['metric'], 'epoch': rec['epoch'],
                        }),
                        'output': json.dumps({
                            'train_loss': rec['train_loss'],
                            'val_loss': rec['val_loss'],
                            'val_metric': rec['val_metric'],
                            'test_metric': rec['test_metric'],
                            'prmp_r2': rec['prmp_r2'],
                            'grad_norms': rec['grad_norms'],
                            'embed_stats': rec['embed_stats'],
                        }),
                        'metadata_dataset': ds_name,
                        'metadata_task': task_name,
                        'metadata_variant': variant,
                        'metadata_task_type': exp_config['task_type'],
                        'metadata_epoch': rec['epoch'],
                        'predict_baseline': str(rec['val_metric']) if variant == 'standard' else '',
                        'predict_our_method': str(rec['val_metric']) if variant == 'prmp' else '',
                    }
                    all_examples.append(example)

                # Per-instance test examples
                for inst_idx, (pred_val, label_val) in enumerate(test_instances):
                    inst_example = {
                        'input': json.dumps({
                            'dataset': ds_name, 'task': task_name,
                            'variant': variant, 'task_type': exp_config['task_type'],
                            'metric': exp_config['metric'],
                            'type': 'test_instance', 'instance_idx': inst_idx,
                        }),
                        'output': json.dumps({
                            'prediction': round(pred_val, 6),
                            'label': round(label_val, 6),
                        }),
                        'metadata_dataset': ds_name,
                        'metadata_task': task_name,
                        'metadata_variant': variant,
                        'metadata_task_type': exp_config['task_type'],
                        'metadata_epoch': -1,
                        'predict_baseline': str(round(pred_val, 6)) if variant == 'standard' else '',
                        'predict_our_method': str(round(pred_val, 6)) if variant == 'prmp' else '',
                    }
                    all_examples.append(inst_example)

                # Summary example per task-variant
                summary = {
                    'input': json.dumps({
                        'type': 'run_summary', 'dataset': ds_name,
                        'task': task_name, 'variant': variant,
                        'task_type': exp_config['task_type'],
                    }),
                    'output': json.dumps({
                        'best_val': round(float(best_val), 6) if best_val is not None else None,
                        'final_test': round(float(epoch_records[-1]['test_metric']), 6) if epoch_records else None,
                        'num_epochs': len(epoch_records),
                        'final_prmp_r2': epoch_records[-1]['prmp_r2'] if epoch_records else {},
                        'final_embed_stats': epoch_records[-1].get('embed_stats', {}),
                        'training_time_s': round(run_time, 1),
                    }),
                    'metadata_dataset': ds_name,
                    'metadata_task': task_name,
                    'metadata_variant': variant,
                    'metadata_task_type': exp_config['task_type'],
                    'metadata_epoch': -1,
                    'predict_baseline': str(round(float(best_val), 6)) if variant == 'standard' and best_val is not None else '',
                    'predict_our_method': str(round(float(best_val), 6)) if variant == 'prmp' and best_val is not None else '',
                }
                all_examples.append(summary)

                # Store for cross-task analysis
                all_run_summaries.append({
                    'dataset': ds_name, 'task': task_name,
                    'variant': variant, 'task_type': exp_config['task_type'],
                    'metric': exp_config['metric'],
                    'higher_better': exp_config['higher_better'],
                    'best_val': float(best_val) if best_val is not None else None,
                })

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  OOM for {variant}, skipping")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                logger.exception(f"  Failed: {ds_name}/{task_name}/{variant}")

            # Cleanup
            if 'model' in dir():
                try:
                    del model
                except Exception:
                    pass
            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

    # === CROSS-TASK ANALYSIS ===
    logger.info(f"\n{'='*60}")
    logger.info("Computing cross-task analysis...")

    analysis = compute_cross_task_analysis(all_run_summaries)

    # Log analysis results
    logger.info(f"Per-task results:")
    for tk, tv in analysis.get('per_task', {}).items():
        logger.info(f"  {tk}: std={tv['standard']:.4f} prmp={tv['prmp']:.4f} "
                     f"delta={tv['delta']:.4f} better={tv['prmp_better']}")
    logger.info(f"Regression mean delta: {analysis.get('regression_mean_delta')}")
    logger.info(f"Classification mean delta: {analysis.get('classification_mean_delta')}")
    logger.info(f"Reg vs Cls delta diff: {analysis.get('reg_vs_cls_delta_diff')}")

    # Add analysis summary example
    all_examples.append({
        'input': json.dumps({'type': 'cross_task_analysis'}),
        'output': json.dumps(analysis),
        'metadata_dataset': 'all',
        'metadata_task': 'analysis',
        'metadata_variant': 'comparison',
        'metadata_task_type': 'analysis',
        'metadata_epoch': -1,
        'predict_baseline': str(analysis.get('classification_mean_delta', '')),
        'predict_our_method': str(analysis.get('regression_mean_delta', '')),
    })

    # === Save output ===
    output = {
        'metadata': {
            'method_name': 'PRMP_Task_Type_Effect',
            'description': (
                'Systematic investigation of PRMP task-type effect using RelBench '
                'rel-f1 and rel-stack datasets. Compares regression vs classification '
                'to test if PRMP residual subtraction creates a de-noising effect '
                'benefiting continuous targets.'
            ),
            'datasets': ['rel-f1', 'rel-stack'],
            'tasks': [e['task_name'] for e in EXPERIMENTS],
            'variants': VARIANTS,
            'seeds': SEEDS,
            'hyperparameters': {
                'channels': CHANNELS, 'num_layers': NUM_LAYERS,
                'batch_size': BATCH_SIZE, 'num_neighbors': NUM_NEIGHBORS,
                'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'lr': LR,
            },
            'device': str(DEVICE),
            'gpu_name': torch.cuda.get_device_name(0) if HAS_GPU else 'CPU',
        },
        'datasets': [{'dataset': 'prmp_task_type_analysis', 'examples': all_examples}],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nSaved {len(all_examples)} examples to {out_path}")

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
