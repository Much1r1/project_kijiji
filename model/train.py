"""
model/train.py
────────────────────────────────────────────────────────────────────────────
Project Kijiji — Temporal Walk-Forward Training Loop

Handles the full training lifecycle for KijijiGNN:
    1. Temporal train/val/test split (no data leakage)
    2. Walk-forward cross-validation across time windows
    3. Early stopping with best-model checkpointing
    4. Results rescaled to real milliseconds (interpretable output)
    5. Final peering simulation on the test-period graph

Why temporal splitting matters here:
    BGP data is a time series. A random 80/20 split would leak future routing
    state into training — the model would "know" about peering agreements that
    hadn't been signed yet. Walk-forward validation mirrors real deployment:
    train on the past, predict the future.

    Train  : synthetic events tagged 2022-01 → 2023-12  (24 months)
    Val    : synthetic events tagged 2024-01 → 2024-06   (6 months)
    Test   : synthetic events tagged 2024-07 → 2024-12   (6 months)

Usage:
    python model/train.py
    python model/train.py --epochs 200 --lr 5e-4 --hidden-dim 128
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

# Local imports — assumes train.py lives alongside graph_sage.py in model/
from graph_sage import (
    AFRICAN_NODES,
    CityNode,
    KijijiGNN,
    _COORDS,
    _haversine_km,
    build_node_features,
    evaluate,
    get_gdp_weights,
    weighted_latency_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Optimiser
    epochs:       int   = 150
    lr:           float = 1e-3
    weight_decay: float = 1e-4

    # Architecture (must match graph_sage.py if overriding)
    hidden_dim:   int   = 64

    # Early stopping
    patience:     int   = 20       # stop if val loss doesn't improve for N epochs
    min_delta:    float = 1e-4     # minimum improvement to count as progress

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "kijiji_best.pt"

    # Reproducibility
    seed:         int   = 42

    # Output scaling: latency values are normalised during training.
    # This scalar rescales predictions back to milliseconds for reporting.
    # Set to the max observed latency in your dataset (89ms for KIN in synth).
    latency_scale_ms: float = 89.0


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL GRAPH BUILDER
# Generates train / val / test graphs from synthetic timestamped BGP events.
# Replace generate_temporal_events() with a ClickHouse query when live.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BGPEvent:
    """One synthetic BGP routing observation between two cities."""
    timestamp:      datetime
    src_idx:        int
    dst_idx:        int
    detour_ratio:   float
    observed_latency_ms: float


def generate_temporal_events(
    nodes:      list[CityNode],
    start_date: datetime,
    end_date:   datetime,
    n_events:   int = 2000,
    seed:       int = 42,
) -> list[BGPEvent]:
    """
    Generates synthetic BGP events spread across a time window.

    Each event represents one observed routing measurement between two
    cities. The detour ratio degrades slightly over time for low-IXP
    nodes (simulating infrastructure stagnation) and improves slightly
    for high-IXP nodes (simulating peering maturation).

    Args:
        nodes:      List of CityNode objects.
        start_date: Window start (inclusive).
        end_date:   Window end (inclusive).
        n_events:   Total events to generate.
        seed:       RNG seed for reproducibility.

    Returns:
        List of BGPEvent sorted by timestamp.
    """
    random.seed(seed)
    N = len(nodes)
    total_seconds = int((end_date - start_date).total_seconds())
    events = []

    for _ in range(n_events):
        # Random timestamp within window
        offset_s  = random.randint(0, total_seconds)
        ts        = start_date + timedelta(seconds=offset_s)

        # Random city pair
        src_idx, dst_idx = random.sample(range(N), 2)
        src, dst = nodes[src_idx], nodes[dst_idx]

        lat_s, lon_s = _COORDS[src.city_id]
        lat_d, lon_d = _COORDS[dst.city_id]
        geodesic_km  = _haversine_km(lat_s, lon_s, lat_d, lon_d)

        # Infrastructure maturity modulates detour probability over time.
        # Cities with no IXPs see no improvement; cities with IXPs improve.
        months_elapsed  = (ts - start_date).days / 30.0
        ixp_maturity    = min(1.0, (src.ixp_count + dst.ixp_count) / 4.0)
        improvement     = ixp_maturity * months_elapsed * 0.003  # slow drift

        via_europe      = random.random() < max(0.1, 0.40 - improvement)
        observed_km     = geodesic_km * (
            random.uniform(1.55, 1.85) if via_europe
            else random.uniform(1.0, 1.35)
        )
        detour_ratio    = observed_km / geodesic_km

        # Latency: baseline + detour penalty + noise
        base_ms         = (src.mean_latency_ms + dst.mean_latency_ms) / 2
        penalty_ms      = (detour_ratio - 1.0) * base_ms * 3.5
        noise_ms        = random.gauss(0, 2.0)
        observed_ms     = max(1.0, base_ms + penalty_ms + noise_ms)

        events.append(BGPEvent(
            timestamp=ts,
            src_idx=src_idx,
            dst_idx=dst_idx,
            detour_ratio=detour_ratio,
            observed_latency_ms=observed_ms,
        ))

    return sorted(events, key=lambda e: e.timestamp)


def events_to_graph(
    events:  list[BGPEvent],
    nodes:   list[CityNode],
) -> Data:
    """
    Aggregates a list of BGPEvents into a single PyG Data graph.

    Multiple events between the same (src, dst) pair are averaged —
    the graph represents the mean routing state over the time window.
    This is the right aggregation for walk-forward validation because
    each window represents a "snapshot" of the network state.
    """
    N = len(nodes)
    x = build_node_features(nodes)

    # Aggregate: mean detour ratio and mean latency per (src, dst) pair
    pair_ratios:   dict[tuple[int,int], list[float]] = {}
    pair_latencies: dict[tuple[int,int], list[float]] = {}

    for e in events:
        key = (e.src_idx, e.dst_idx)
        pair_ratios.setdefault(key, []).append(e.detour_ratio)
        pair_latencies.setdefault(key, []).append(e.observed_latency_ms)

    src_list, dst_list, attr_list = [], [], []
    for (src, dst), ratios in pair_ratios.items():
        src_list.append(src)
        dst_list.append(dst)
        attr_list.append([sum(ratios) / len(ratios)])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)

    # Node-level latency label: mean observed latency across all outgoing edges
    node_latencies = torch.zeros(N)
    node_counts    = torch.zeros(N)
    for (src, _), lats in pair_latencies.items():
        node_latencies[src] += sum(lats) / len(lats)
        node_counts[src] += 1
    mask = node_counts > 0
    node_latencies[mask] = node_latencies[mask] / node_counts[mask]

    # Normalise to [0, 1]
    lo, hi = node_latencies.min(), node_latencies.max()
    y = (node_latencies - lo) / (hi - lo + 1e-8)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_ids=[n.city_id for n in nodes],
        num_nodes=N,
    )


def build_temporal_splits(
    nodes: list[CityNode],
    seed:  int = 42,
) -> tuple[Data, Data, Data]:
    """
    Builds train / val / test graphs using a temporal walk-forward split.

    Timeline:
        Train : 2022-01-01 → 2023-12-31  (2 years, 1400 events)
        Val   : 2024-01-01 → 2024-06-30  (6 months, 300 events)
        Test  : 2024-07-01 → 2024-12-31  (6 months, 300 events)

    The val and test windows are always AFTER the train window.
    There is no overlap — no future leakage into training.

    Returns:
        (train_data, val_data, test_data) as PyG Data objects
    """
    train_events = generate_temporal_events(
        nodes,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        n_events=1400,
        seed=seed,
    )
    val_events = generate_temporal_events(
        nodes,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        n_events=300,
        seed=seed + 1,
    )
    test_events = generate_temporal_events(
        nodes,
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 12, 31),
        n_events=300,
        seed=seed + 2,
    )

    return (
        events_to_graph(train_events, nodes),
        events_to_graph(val_events,   nodes),
        events_to_graph(test_events,  nodes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val loss stops improving.
    Saves the best model weights for final evaluation.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.best_state = None
        self.counter    = 0
        self.stopped_epoch = 0

    def step(self, val_loss: float, model: KijijiGNN) -> bool:
        """
        Returns True if training should stop.
        Saves model state when val loss improves.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: KijijiGNN) -> None:
        """Restore best weights after training ends."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:   KijijiGNN,
    config:  TrainConfig,
    epoch:   int,
    metrics: dict,
) -> str:
    """Saves model weights + metadata to disk."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "metrics":      metrics,
        "config":       config.__dict__,
        "saved_at":     datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
    }, path)
    return path


def load_checkpoint(path: str, model: KijijiGNN) -> dict:
    """Loads model weights from a checkpoint file."""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    nodes:  list[CityNode],
    config: TrainConfig,
    n_folds: int = 3,
) -> list[dict]:
    """
    Walk-forward cross-validation across multiple time windows.

    Each fold expands the training window and advances the validation window:
        Fold 1:  Train 2022-01 → 2022-12,  Val 2023-01 → 2023-06
        Fold 2:  Train 2022-01 → 2023-06,  Val 2023-07 → 2023-12
        Fold 3:  Train 2022-01 → 2023-12,  Val 2024-01 → 2024-06

    This validates that model performance is stable across time, not just
    lucky on one particular 6-month window.

    Returns:
        List of metric dicts, one per fold.
    """
    fold_results = []

    # Define fold boundaries
    fold_defs = [
        # (train_end, val_start, val_end)
        (datetime(2022, 12, 31), datetime(2023, 1, 1),  datetime(2023, 6, 30)),
        (datetime(2023, 6,  30), datetime(2023, 7, 1),  datetime(2023, 12, 31)),
        (datetime(2023, 12, 31), datetime(2024, 1, 1),  datetime(2024, 6, 30)),
    ][:n_folds]

    for fold_idx, (train_end, val_start, val_end) in enumerate(fold_defs):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}  "
              f"train→{train_end.strftime('%Y-%m')}, "
              f"val {val_start.strftime('%Y-%m')}→{val_end.strftime('%Y-%m')}")

        train_events = generate_temporal_events(
            nodes,
            start_date=datetime(2022, 1, 1),
            end_date=train_end,
            n_events=800 + fold_idx * 300,
            seed=config.seed + fold_idx,
        )
        val_events = generate_temporal_events(
            nodes,
            start_date=val_start,
            end_date=val_end,
            n_events=200,
            seed=config.seed + fold_idx + 10,
        )

        train_data = events_to_graph(train_events, nodes)
        val_data   = events_to_graph(val_events,   nodes)

        model     = KijijiGNN()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        stopper   = EarlyStopping(patience=config.patience)

        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pos_scores, neg_scores = model(train_data)
            loss = weighted_latency_loss(
                pos_scores, neg_scores, get_gdp_weights(train_data)
            )
            loss.backward()
            optimizer.step()

            val_metrics = evaluate(model, val_data)
            if stopper.step(val_metrics["loss"], model):
                break

        stopper.restore_best(model)
        final = evaluate(model, val_data)
        final["fold"] = fold_idx + 1
        final["stopped_epoch"] = epoch
        fold_results.append(final)
        print(f"    val_loss={final['loss']:.4f}  sep={final['separation']:.4f}  "
              f"stopped at epoch {epoch}")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────

def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    t0 = time.time()

    print("=" * 62)
    print("  Project Kijiji — Temporal Walk-Forward Training")
    print("=" * 62)
    print(f"\n  Config:")
    print(f"    epochs       : {config.epochs}")
    print(f"    lr           : {config.lr}")
    print(f"    patience     : {config.patience}")
    print(f"    latency_scale: {config.latency_scale_ms} ms")
    print(f"    seed         : {config.seed}")

    # ── 1. Build temporal splits ──────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Building temporal splits...")
    train_data, val_data, test_data = build_temporal_splits(
        AFRICAN_NODES, seed=config.seed
    )
    print(f"  Train edges : {train_data.edge_index.size(1)}")
    print(f"  Val edges   : {val_data.edge_index.size(1)}")
    print(f"  Test edges  : {test_data.edge_index.size(1)}")

    # ── 2. Walk-forward cross-validation ─────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Walk-Forward Cross-Validation (3 folds)")
    print(f"{'─'*62}")
    cv_results = walk_forward_cv(AFRICAN_NODES, config, n_folds=3)

    cv_losses = [r["loss"] for r in cv_results]
    print(f"\n  CV summary:")
    print(f"    Mean val loss : {sum(cv_losses)/len(cv_losses):.4f}")
    print(f"    Best fold loss: {min(cv_losses):.4f}")
    print(f"    Worst fold loss: {max(cv_losses):.4f}")

    # ── 3. Final training on full train split ─────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Final training on full train split (2022–2023)")
    print(f"{'─'*62}")
    print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Sep':>7}  {'Status'}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}")

    model     = KijijiGNN()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    stopper   = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        # Train step
        model.train()
        optimizer.zero_grad()
        pos_scores, neg_scores = model(train_data)
        train_loss = weighted_latency_loss(
            pos_scores, neg_scores, get_gdp_weights(train_data)
        )
        train_loss.backward()
        optimizer.step()

        # Val step
        val_metrics = evaluate(model, val_data)

        should_stop = stopper.step(val_metrics["loss"], model)

        if epoch % 10 == 0 or should_stop:
            status = "← best" if stopper.counter == 0 else f"patience {stopper.counter}/{config.patience}"
            if should_stop:
                status = "EARLY STOP"
            print(f"  {epoch:>6}  {float(train_loss.detach()):>8.4f}  "
                  f"{val_metrics['loss']:>8.4f}  {val_metrics['separation']:>7.4f}  "
                  f"{status}")

        if should_stop:
            best_epoch = epoch - config.patience
            break

    stopper.restore_best(model)
    best_epoch = best_epoch or config.epochs

    # ── 4. Test evaluation ────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Test evaluation (2024-H2 — held-out window)")
    print(f"{'─'*62}")

    test_metrics = evaluate(model, test_data)
    print(f"  Test loss      : {test_metrics['loss']:.4f}")
    print(f"  Pos mean score : {test_metrics['pos_mean']:.4f}")
    print(f"  Neg mean score : {test_metrics['neg_mean']:.4f}")
    print(f"  Separation     : {test_metrics['separation']:.4f}")

    # Baseline comparison
    print(f"\n  Baseline comparison:")
    print(f"  {'Model':<28}  {'Val Loss':>10}  {'Sep':>8}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*8}")
    print(f"  {'Random (no training)':<28}  {'~0.6931':>10}  {'~0.0000':>8}   (ln 2 BCE)")
    print(f"  {'GraphSAGE (ours)':<28}  {test_metrics['loss']:>10.4f}  {test_metrics['separation']:>8.4f}")

    # ── 5. Peering simulation on test graph ───────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Peering Simulation on Test Graph")
    print(f"{'─'*62}")

    simulations = [
        ("ADD", "NBO", "Addis Ababa → Nairobi"),      # No IXP → regional hub
        ("KIN", "LOS", "Kinshasa → Lagos"),            # Two low-GDP, low-IXP nodes
        ("KIN", "NBO", "Kinshasa → Nairobi"),          # DRC → East Africa hub
    ]

    node_ids = test_data.node_ids
    scale    = config.latency_scale_ms

    print(f"\n  {'Proposed link':<28}  {'Regional Δ (ms)':>16}  {'Top beneficiary'}")
    print(f"  {'─'*28}  {'─'*16}  {'─'*20}")

    for src_id, dst_id, label in simulations:
        src_idx = node_ids.index(src_id)
        dst_idx = node_ids.index(dst_id)
        result  = model.simulate_peering(test_data, src_idx, dst_idx)

        # Rescale from normalised [0,1] to real milliseconds
        regional_ms = result["regional_dividend"] * scale
        top_node    = result["most_improved_node"]
        top_ms      = result["most_improved_ms"] * scale
        print(f"  {label:<28}  {regional_ms:>+14.2f}ms  {top_node} ({top_ms:+.2f}ms)")

    # ── 6. Checkpoint ─────────────────────────────────────────────────────
    ckpt_path = save_checkpoint(model, config, best_epoch, test_metrics)
    elapsed   = time.time() - t0

    print(f"\n{'─'*62}")
    print(f"  Checkpoint saved → {ckpt_path}")
    print(f"  Total training time : {elapsed:.1f}s")
    print(f"\n✅ Training complete.")
    print(f"   Next: model/evaluate.py for Pillar 3 (IXP/GDP correlation)")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Project Kijiji — Train the GraphSAGE peering recommender"
    )
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--latency-scale", type=float, default=89.0,
                        help="Max latency in ms for rescaling outputs (default: 89ms = KIN baseline)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        latency_scale_ms=args.latency_scale,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)