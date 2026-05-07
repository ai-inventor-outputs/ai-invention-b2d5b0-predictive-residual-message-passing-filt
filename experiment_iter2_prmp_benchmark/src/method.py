#!/usr/bin/env python3
"""PRMP Benchmark on RelBench rel-hm: PRMPConv vs Standard Aggregation with Ablations.

Benchmarks Predictive Residual Message Passing (PRMPConv) against standard SAGEConv
(mean aggregation) on 2 FK links from the H&M Fashion (rel-hm) relational dataset:
  - customer -> transaction (mean cardinality ~4.17, low)
  - article  -> transaction (mean cardinality ~20.05, moderate)

Includes 4 variants:
  1. Standard: SAGEConv baseline (mean aggregation)
  2. PRMP:     PRMPConv with learned predictions
  3. Random:   PRMPConv with random frozen predictions (ablation)
  4. No-subtract: PRMPConv with concatenation instead of subtraction (ablation)
"""

import sys
import os
import json
import time
import gc
import math
import resource
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psutil

# ─── Logging ──────────────────────────────────────────────────────────────────
from loguru import logger

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ─── Hardware Detection ──────────────────────────────────────────────────────

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

def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# ─── Memory Limits ────────────────────────────────────────────────────────────
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.7, _avail / 1e9) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.85)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f} GB")

resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ─── Imports: PyG ─────────────────────────────────────────────────────────────
from torch_geometric.nn import MessagePassing, HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__prmp_residual_passing_aju"
                "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
CUSTOMER_PARQUET = DATA_DIR / "supplementary_customer_transaction_aligned_features.parquet"
ARTICLE_PARQUET = DATA_DIR / "supplementary_article_transaction_aligned_features.parquet"
DATA_JSON = DATA_DIR / "full_data_out.json"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
SEED = 42
VARIANTS = ['standard', 'prmp', 'random', 'no_subtract']


# ══════════════════════════════════════════════════════════════════════════════
#  PRMPConv Implementation
# ══════════════════════════════════════════════════════════════════════════════

class PRMPConv(MessagePassing):
    """Predictive Residual Message Passing convolution.

    Per the PRMP architectural spec (Sections 3, 5, 7):
      - 2-layer prediction MLP: parent_feat -> predicted_child_feat
      - Residual = child_feat - pred_mlp(parent_feat.detach())
      - LayerNorm on residuals
      - Mean aggregation
      - GraphSAGE-style concat+linear update
    """

    def __init__(self, in_channels_src: int, in_channels_dst: int,
                 out_channels: int, mode: str = 'prmp'):
        super().__init__(aggr='mean')
        self.mode = mode
        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels
        hidden = min(in_channels_dst, in_channels_src)

        # Prediction MLP: parent -> predicted child (Section 5.1)
        self.pred_mlp = nn.Sequential(
            nn.Linear(in_channels_dst, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels_src),
        )
        # Zero-initialize final layer (Section 5.3)
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)

        # Random ablation: freeze with random weights
        if mode == 'random':
            nn.init.kaiming_normal_(self.pred_mlp[-1].weight)
            for p in self.pred_mlp.parameters():
                p.requires_grad = False

        # LayerNorm on residuals (Section 5.2, Option B)
        if mode != 'no_subtract':
            self.norm = nn.LayerNorm(in_channels_src)

        # GraphSAGE-style update (Section 3.4)
        if mode == 'no_subtract':
            self.update_mlp = nn.Linear(in_channels_dst + in_channels_src * 2, out_channels)
        else:
            self.update_mlp = nn.Linear(in_channels_dst + in_channels_src, out_channels)

    def forward(self, x, edge_index):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        x_src, x_dst = x
        # propagate handles message() + aggregate()
        aggr_out = self.propagate(edge_index, x=x)
        # GraphSAGE-style: concat destination features with aggregated messages
        out = self.update_mlp(torch.cat([x_dst, aggr_out], dim=-1))
        return out

    def message(self, x_j, x_i):
        """Compute residual messages.
        x_j = source (child) features, x_i = destination (parent) features.
        """
        # Detach parent input (Section 5.4)
        predicted = self.pred_mlp(x_i.detach())

        if self.mode == 'no_subtract':
            return torch.cat([x_j, predicted], dim=-1)
        else:
            residual = x_j - predicted
            return self.norm(residual)


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class BipartiteGNN(nn.Module):
    """Heterogeneous GNN for bipartite parent-child graphs.

    4 variants: 'standard', 'prmp', 'random', 'no_subtract'.
    Uses HeteroConv with child->parent edges (PRMPConv or SAGEConv)
    and parent->child reverse edges (always SAGEConv).
    """

    def __init__(self, parent_feat_dim: int, child_feat_dim: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 conv_type: str = 'prmp'):
        super().__init__()
        self.conv_type = conv_type
        self.num_layers = num_layers

        # Feature encoders
        self.parent_enc = nn.Sequential(
            nn.Linear(parent_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.child_enc = nn.Sequential(
            nn.Linear(child_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == 'standard':
                fwd_conv = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
            else:
                fwd_conv = PRMPConv(hidden_dim, hidden_dim, hidden_dim, mode=conv_type)

            rev_conv = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

            conv = HeteroConv({
                ('child', 'fk_to', 'parent'): fwd_conv,
                ('parent', 'rev_fk_to', 'child'): rev_conv,
            }, aggr='sum')

            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                'parent': nn.LayerNorm(hidden_dim),
                'child': nn.LayerNorm(hidden_dim),
            }))

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> torch.Tensor:
        h_dict = {
            'parent': self.parent_enc(x_dict['parent']),
            'child': self.child_enc(x_dict['child']),
        }

        for i, conv in enumerate(self.convs):
            new_h = conv(h_dict, edge_index_dict)
            for node_type in h_dict:
                if node_type in new_h:
                    new_h[node_type] = self.norms[i][node_type](new_h[node_type])
                    new_h[node_type] = F.relu(new_h[node_type])
                    if i > 0:
                        new_h[node_type] = new_h[node_type] + h_dict[node_type]
                else:
                    new_h[node_type] = h_dict[node_type]
            h_dict = new_h

        return self.head(h_dict['parent']).squeeze(-1)

    @torch.no_grad()
    def get_prmp_diagnostics(self, x_dict, edge_index_dict):
        """Get PRMP-specific diagnostics: prediction R^2, residual stats."""
        if self.conv_type not in ('prmp', 'random', 'no_subtract'):
            return {}

        diagnostics = {}
        h_dict = {
            'parent': self.parent_enc(x_dict['parent']),
            'child': self.child_enc(x_dict['child']),
        }

        for layer_idx, conv in enumerate(self.convs):
            edge_type = ('child', 'fk_to', 'parent')
            if edge_type in conv.convs:
                prmp_conv = conv.convs[edge_type]
                if hasattr(prmp_conv, 'pred_mlp'):
                    ei = edge_index_dict[edge_type]
                    child_h = h_dict['child'][ei[0]]
                    parent_h = h_dict['parent'][ei[1]]
                    predicted = prmp_conv.pred_mlp(parent_h)
                    residual = child_h - predicted

                    ss_res = ((child_h - predicted) ** 2).sum().item()
                    ss_tot = ((child_h - child_h.mean(0)) ** 2).sum().item()
                    r2 = 1 - ss_res / max(ss_tot, 1e-8)

                    diagnostics[f'layer_{layer_idx}'] = {
                        'prediction_r2': round(r2, 6),
                        'residual_mean': round(residual.mean().item(), 6),
                        'residual_std': round(residual.std().item(), 6),
                        'residual_abs_mean': round(residual.abs().mean().item(), 6),
                        'pred_weight_norm': round(
                            sum(p.norm().item() for p in prmp_conv.pred_mlp.parameters()), 6),
                    }

            new_h = conv(h_dict, edge_index_dict)
            for nt in h_dict:
                if nt in new_h:
                    new_h[nt] = F.relu(new_h[nt])
                else:
                    new_h[nt] = h_dict[nt]
            h_dict = new_h

        return diagnostics


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Graph Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_from_parquet(parquet_path: Path, link_name: str) -> dict:
    """Build a bipartite HeteroData graph from aligned parent-child parquet."""
    logger.info(f"Building graph for {link_name} from {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    parent_cols = sorted([c for c in df.columns if c.startswith('parent__')])
    child_cols = sorted([c for c in df.columns if c.startswith('child__')])
    logger.info(f"  Parent features ({len(parent_cols)}): {parent_cols}")
    logger.info(f"  Child features ({len(child_cols)}): {child_cols}")

    df = df.fillna(0.0)

    # Identify unique parents by feature vector
    parent_df = df[parent_cols].drop_duplicates().reset_index(drop=True)
    parent_df['_parent_id'] = parent_df.index
    logger.info(f"  Unique parents: {len(parent_df)}")

    merged = df.merge(parent_df, on=parent_cols, how='left')
    parent_indices = merged['_parent_id'].values.astype(np.int64)

    num_parents = len(parent_df)
    num_children = len(df)

    # Feature tensors
    parent_features = torch.tensor(parent_df[parent_cols].values, dtype=torch.float32)
    child_features = torch.tensor(df[child_cols].values, dtype=torch.float32)

    # Normalize (standardize per column)
    for feat in [parent_features, child_features]:
        mu = feat.mean(0, keepdim=True)
        std = feat.std(0, keepdim=True).clamp(min=1e-8)
        feat.sub_(mu).div_(std)

    # Edge indices
    child_idx = np.arange(num_children, dtype=np.int64)
    edge_index_fwd = torch.from_numpy(np.stack([child_idx, parent_indices]))
    edge_index_rev = torch.from_numpy(np.stack([parent_indices, child_idx]))

    # Target: binary classification — mean child price above median
    price_col = 'child__price'
    if price_col in df.columns:
        raw_prices = df[price_col].values
    else:
        raw_prices = df[child_cols[0]].values

    parent_mean_price = pd.Series(raw_prices).groupby(parent_indices).mean()
    median_price = parent_mean_price.median()
    targets = torch.zeros(num_parents, dtype=torch.float32)
    for pid, mp in parent_mean_price.items():
        targets[pid] = 1.0 if mp > median_price else 0.0

    # Cardinality stats
    card_counts = np.bincount(parent_indices, minlength=num_parents)
    mean_card = float(card_counts.mean())
    median_card = float(np.median(card_counts[card_counts > 0]))
    max_card = int(card_counts.max())

    # Train/val/test split (70/15/15 on parent nodes)
    np.random.seed(SEED)
    pids = np.arange(num_parents)
    train_ids, test_val_ids = train_test_split(pids, test_size=0.3, random_state=SEED)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=SEED)

    train_mask = torch.zeros(num_parents, dtype=torch.bool)
    val_mask = torch.zeros(num_parents, dtype=torch.bool)
    test_mask = torch.zeros(num_parents, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True

    # Build HeteroData
    data = HeteroData()
    data['parent'].x = parent_features
    data['parent'].y = targets
    data['parent'].train_mask = train_mask
    data['parent'].val_mask = val_mask
    data['parent'].test_mask = test_mask
    data['child'].x = child_features
    data[('child', 'fk_to', 'parent')].edge_index = edge_index_fwd
    data[('parent', 'rev_fk_to', 'child')].edge_index = edge_index_rev

    # Cross-table predictability (R^2 from parent features to child features)
    sample_n = min(50000, num_children)
    sample_idx = np.random.choice(num_children, sample_n, replace=False)
    X_p = parent_features[parent_indices[sample_idx]].numpy()
    Y_c = child_features[sample_idx].numpy()
    lr = LinearRegression()
    lr.fit(X_p, Y_c)
    r2_cross = round(lr.score(X_p, Y_c), 6)

    info = {
        'link_name': link_name,
        'num_parents': num_parents,
        'num_children': num_children,
        'num_edges': num_children,
        'parent_feat_dim': len(parent_cols),
        'child_feat_dim': len(child_cols),
        'mean_cardinality': round(mean_card, 4),
        'median_cardinality': round(median_card, 4),
        'max_cardinality': max_card,
        'parent_cols': parent_cols,
        'child_cols': child_cols,
        'target': 'mean_child_price_above_median',
        'positive_rate': round(float(targets.mean()), 4),
        'cross_table_r2': r2_cross,
        'train_size': int(train_mask.sum()),
        'val_size': int(val_mask.sum()),
        'test_size': int(test_mask.sum()),
    }
    logger.info(f"  Graph: {num_parents} parents, {num_children} children, "
                f"mean_card={mean_card:.2f}, R2={r2_cross:.4f}, pos_rate={targets.mean():.3f}")

    return {'data': data, 'info': info}


# ══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _get_graph_inputs(data: HeteroData):
    x_dict = {k: data[k].x for k in ['parent', 'child']}
    edge_index_dict = {
        ('child', 'fk_to', 'parent'): data[('child', 'fk_to', 'parent')].edge_index,
        ('parent', 'rev_fk_to', 'child'): data[('parent', 'rev_fk_to', 'child')].edge_index,
    }
    return x_dict, edge_index_dict


def train_epoch(model: nn.Module, data: HeteroData,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module) -> float:
    model.train()
    optimizer.zero_grad()
    x_dict, edge_index_dict = _get_graph_inputs(data)
    out = model(x_dict, edge_index_dict)
    mask = data['parent'].train_mask
    loss = criterion(out[mask], data['parent'].y[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data: HeteroData,
             mask: torch.Tensor) -> dict:
    model.eval()
    x_dict, edge_index_dict = _get_graph_inputs(data)
    out = model(x_dict, edge_index_dict)
    logits = out[mask].cpu()
    targets = data['parent'].y[mask].cpu()

    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_true = targets.numpy()

    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = 0.5

    acc = accuracy_score(y_true, preds)
    loss = F.binary_cross_entropy_with_logits(logits, targets).item()

    return {
        'auroc': round(float(auroc), 6),
        'accuracy': round(float(acc), 6),
        'loss': round(float(loss), 6),
    }


def run_experiment(graph_data: dict, variant: str, epochs: int = EPOCHS) -> dict:
    """Train and evaluate one variant on one FK link."""
    data = graph_data['data'].to(DEVICE)
    info = graph_data['info']

    logger.info(f"  Training {variant} on {info['link_name']} for {epochs} epochs")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = BipartiteGNN(
        parent_feat_dim=info['parent_feat_dim'],
        child_feat_dim=info['child_feat_dim'],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        conv_type=variant,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Params: {num_params:,} total, {trainable_params:,} trainable")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_metrics_history = []
    best_val_auroc = 0.0
    best_epoch = 0
    best_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, optimizer, criterion)
        train_losses.append(round(train_loss, 6))

        val_m = evaluate(model, data, data['parent'].val_mask)
        val_metrics_history.append(val_m)

        if val_m['auroc'] > best_val_auroc:
            best_val_auroc = val_m['auroc']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            logger.info(f"    Epoch {epoch:3d}: loss={train_loss:.4f}, "
                        f"val_auroc={val_m['auroc']:.4f}, val_acc={val_m['accuracy']:.4f}")

        # Sanity check: warn if loss is NaN
        if math.isnan(train_loss):
            logger.warning(f"    NaN loss at epoch {epoch}, stopping early")
            break

    train_time = time.time() - t0
    logger.info(f"    Done in {train_time:.1f}s, best val AUROC={best_val_auroc:.4f} @ epoch {best_epoch}")

    # Restore best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    test_m = evaluate(model, data, data['parent'].test_mask)
    logger.info(f"    Test: auroc={test_m['auroc']:.4f}, acc={test_m['accuracy']:.4f}")

    # PRMP diagnostics
    diagnostics = {}
    if variant in ('prmp', 'random', 'no_subtract'):
        try:
            x_dict, edge_index_dict = _get_graph_inputs(data)
            diagnostics = model.get_prmp_diagnostics(x_dict, edge_index_dict)
        except Exception:
            logger.exception("Failed to compute PRMP diagnostics")

    return {
        'variant': variant,
        'link_name': info['link_name'],
        'test_metrics': test_m,
        'best_val_metrics': {'auroc': round(best_val_auroc, 6), 'epoch': best_epoch},
        'training_curve': {
            'train_losses': train_losses,
            'val_aurocs': [round(m['auroc'], 6) for m in val_metrics_history],
        },
        'diagnostics': diagnostics,
        'num_params': num_params,
        'trainable_params': trainable_params,
        'train_time_seconds': round(train_time, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("PRMP Benchmark on RelBench rel-hm")
    logger.info("=" * 60)

    # Load FK link metadata from data dependency
    try:
        data_json = json.loads(DATA_JSON.read_text())
        fk_links_meta = {}
        for ds in data_json['datasets']:
            for ex in ds['examples']:
                inp = json.loads(ex['input'])
                out_data = json.loads(ex['output'])
                key = f"{inp['parent_table']}_to_{inp['child_table']}"
                fk_links_meta[key] = {'input': inp, 'output': out_data}
    except Exception:
        logger.exception("Could not load data dependency metadata, using defaults")
        fk_links_meta = {}

    # FK link configs
    fk_configs = [
        {'name': 'customer_to_transaction', 'parquet': CUSTOMER_PARQUET},
        {'name': 'article_to_transaction', 'parquet': ARTICLE_PARQUET},
    ]

    all_results = {}
    examples = []

    for fk_cfg in fk_configs:
        link_name = fk_cfg['name']
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FK Link: {link_name}")
        logger.info(f"{'=' * 60}")

        try:
            graph_data = build_graph_from_parquet(fk_cfg['parquet'], link_name)
        except Exception:
            logger.exception(f"Failed to build graph for {link_name}")
            continue

        link_results = {}

        for variant in VARIANTS:
            try:
                result = run_experiment(graph_data, variant)
                link_results[variant] = result
            except Exception:
                logger.exception(f"Failed {variant} on {link_name}")
                link_results[variant] = {
                    'variant': variant,
                    'link_name': link_name,
                    'error': True,
                    'test_metrics': {'auroc': 0.5, 'accuracy': 0.5, 'loss': 999.0},
                    'best_val_metrics': {'auroc': 0.5, 'epoch': 0},
                    'training_curve': {'train_losses': [], 'val_aurocs': []},
                    'diagnostics': {},
                    'num_params': 0,
                    'trainable_params': 0,
                    'train_time_seconds': 0.0,
                }

            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

        all_results[link_name] = link_results

        # Compute relative improvements
        std_auroc = link_results.get('standard', {}).get('test_metrics', {}).get('auroc', 0.5)
        regime_analysis = {}
        for vn, vr in link_results.items():
            if vn != 'standard':
                va = vr.get('test_metrics', {}).get('auroc', 0.5)
                ri = (va - std_auroc) / max(abs(std_auroc), 0.001)
                regime_analysis[vn] = {
                    'auroc': round(va, 6),
                    'relative_improvement': round(ri, 6),
                }

        # Build example
        example_input = json.dumps({
            'fk_link': link_name,
            'parent_table': link_name.split('_to_')[0],
            'child_table': '_'.join(link_name.split('_to_')[1:]),
            'num_parents': graph_data['info']['num_parents'],
            'num_children': graph_data['info']['num_children'],
            'parent_feat_dim': graph_data['info']['parent_feat_dim'],
            'child_feat_dim': graph_data['info']['child_feat_dim'],
            'mean_cardinality': graph_data['info']['mean_cardinality'],
            'median_cardinality': graph_data['info']['median_cardinality'],
            'max_cardinality': graph_data['info']['max_cardinality'],
            'cross_table_r2': graph_data['info']['cross_table_r2'],
            'task': 'binary_classification',
            'target': 'mean_child_price_above_median',
        })

        example_output = json.dumps({
            'task': 'binary_classification',
            'target': 'mean_child_price_above_median',
            'metric': 'AUROC',
            'positive_rate': graph_data['info']['positive_rate'],
            'fk_link_stats': {
                'mean_cardinality': graph_data['info']['mean_cardinality'],
                'median_cardinality': graph_data['info']['median_cardinality'],
                'max_cardinality': graph_data['info']['max_cardinality'],
                'cross_table_r2': graph_data['info']['cross_table_r2'],
            },
            'regime_analysis': regime_analysis,
            'all_variant_results': {
                v: r.get('test_metrics', {}) for v, r in link_results.items()
            },
            'ablation_summary': _ablation_summary(link_results),
        })

        example = {
            'input': example_input,
            'output': example_output,
            'metadata_fk_link': link_name,
            'metadata_mean_cardinality': graph_data['info']['mean_cardinality'],
            'metadata_cross_table_r2': graph_data['info']['cross_table_r2'],
        }
        for variant in VARIANTS:
            if variant in link_results:
                example[f'predict_{variant}'] = json.dumps(link_results[variant])

        examples.append(example)

        del graph_data
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    # ── Final Output ──────────────────────────────────────────────────────
    output = {
        'metadata': {
            'method_name': 'PRMP Benchmark on rel-hm',
            'description': ('Benchmarks PRMPConv vs SAGEConv on RelBench rel-hm '
                            'FK links with random-prediction and no-subtraction ablations'),
            'variants': VARIANTS,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'seed': SEED,
        },
        'datasets': [{
            'dataset': 'rel_hm_fashion',
            'examples': examples,
        }],
    }

    output_path = WORKSPACE / 'method_out.json'
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for lname, results in all_results.items():
        logger.info(f"\n  {lname}:")
        for vn, vr in results.items():
            tm = vr.get('test_metrics', {})
            logger.info(f"    {vn:20s}: AUROC={tm.get('auroc', 'N/A')}, "
                        f"Acc={tm.get('accuracy', 'N/A')}")

    logger.info("\nDone!")


def _ablation_summary(link_results: dict) -> str:
    """Generate a textual ablation summary."""
    parts = []
    std = link_results.get('standard', {}).get('test_metrics', {}).get('auroc', 0.5)
    prmp = link_results.get('prmp', {}).get('test_metrics', {}).get('auroc', 0.5)
    rand = link_results.get('random', {}).get('test_metrics', {}).get('auroc', 0.5)
    nosub = link_results.get('no_subtract', {}).get('test_metrics', {}).get('auroc', 0.5)

    if prmp > std:
        parts.append(f"PRMP improves over standard by {(prmp-std)*100:.2f}pp AUROC.")
    else:
        parts.append(f"PRMP does not improve over standard (delta={(prmp-std)*100:.2f}pp).")

    if rand < std:
        parts.append("Random ablation hurts, confirming learned predictions matter.")
    elif rand >= std:
        parts.append("Random ablation matches or exceeds standard (unexpected).")

    if nosub < prmp:
        parts.append("No-subtract ablation < PRMP: subtraction mechanism matters.")
    else:
        parts.append("No-subtract ablation >= PRMP: subtraction may not be critical.")

    return " ".join(parts)


if __name__ == "__main__":
    main()
