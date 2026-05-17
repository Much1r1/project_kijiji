"""
model/results_logger.py
────────────────────────────────────────────────────────────────────────────
Project Kijiji — Training Results Logger

Saves every training run to:
    model/results/runs.jsonl         — machine-readable, one JSON per line
    model/results/summary.md         — human-readable, for DAAD proposal
    model/results/latest.json        — always the most recent run

Usage (from train.py):
    from results_logger import log_run
    log_run(config, cv_results, test_metrics, sim_results, elapsed)
────────────────────────────────────────────────────────────────────────────
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import asdict


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def log_run(
    config,
    cv_results:   list[dict],
    test_metrics: dict,
    sim_results:  list[dict],
    elapsed_s:    float,
) -> str:
    """
    Persists one training run to disk.

    Args:
        config:       TrainConfig dataclass
        cv_results:   list of fold metric dicts from walk_forward_cv()
        test_metrics: dict from evaluate() on test split
        sim_results:  list of simulate_peering() result dicts
        elapsed_s:    total training time in seconds

    Returns:
        Path to the saved run file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cv_losses = [r["loss"] for r in cv_results]
    cv_seps   = [r["separation"] for r in cv_results]

    run = {
        "run_id":    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "epochs":           config.epochs,
            "lr":               config.lr,
            "patience":         config.patience,
            "hidden_dim":       config.hidden_dim,
            "seed":             config.seed,
            "latency_scale_ms": config.latency_scale_ms,
        },
        "cv": {
            "folds":          cv_results,
            "mean_val_loss":  round(sum(cv_losses) / len(cv_losses), 4),
            "best_fold_loss": round(min(cv_losses), 4),
            "mean_sep":       round(sum(cv_seps) / len(cv_seps), 4),
        },
        "test": {
            "loss":       round(test_metrics["loss"],       4),
            "pos_mean":   round(test_metrics["pos_mean"],   4),
            "neg_mean":   round(test_metrics["neg_mean"],   4),
            "separation": round(test_metrics["separation"], 4),
        },
        "baselines": {
            "random_bce":     0.6931,
            "improvement_vs_random": round(
                (0.6931 - test_metrics["loss"]) / 0.6931 * 100, 1
            ),
        },
        "simulations": sim_results,
        "elapsed_s": round(elapsed_s, 1),
    }

    # ── Append to runs.jsonl ──
    jsonl_path = os.path.join(RESULTS_DIR, "runs.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")

    # ── Overwrite latest.json ──
    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    # ── Regenerate summary.md ──
    _write_summary_md(run)

    print(f"  Results saved → {RESULTS_DIR}/")
    return latest_path


def _write_summary_md(run: dict) -> None:
    """
    Appends a human-readable summary block to summary.md.
    Suitable for copy-pasting into the DAAD proposal.
    """
    md_path = os.path.join(RESULTS_DIR, "summary.md")

    block = f"""
## Run {run['run_id']}
*{run['timestamp']}*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | {run['config']['epochs']} |
| Learning rate | {run['config']['lr']} |
| Early stopping patience | {run['config']['patience']} |
| Hidden dim | {run['config']['hidden_dim']} |
| Seed | {run['config']['seed']} |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
"""
    for fold in run["cv"]["folds"]:
        block += (f"| {fold['fold']} | {fold['loss']:.4f} | "
                  f"{fold['separation']:.4f} | {fold['stopped_epoch']} |\n")

    block += f"""
**Mean CV val loss**: {run['cv']['mean_val_loss']}
**Mean CV separation**: {run['cv']['mean_sep']}

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | {run['test']['loss']} |
| Pos mean score | {run['test']['pos_mean']} |
| Neg mean score | {run['test']['neg_mean']} |
| Separation | {run['test']['separation']} |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | {run['test']['loss']} | **{run['baselines']['improvement_vs_random']}% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
"""
    for sim in run["simulations"]:
        block += (f"| {sim['label']} | {sim['regional_ms']:+.2f}ms | "
                  f"{sim['top_node']} ({sim['top_ms']:+.2f}ms) |\n")

    block += f"\n**Training time**: {run['elapsed_s']}s\n\n---\n"

    with open(md_path, "a", encoding="utf-8") as f:
        if os.path.getsize(md_path) == 0 if os.path.exists(md_path) else False:
            f.write("# Project Kijiji — Training Run Log\n\n")
        f.write(block)


def load_all_runs() -> list[dict]:
    """Load all historical runs from runs.jsonl."""
    path = os.path.join(RESULTS_DIR, "runs.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_run_history() -> None:
    """Print a compact table of all historical runs."""
    runs = load_all_runs()
    if not runs:
        print("No runs logged yet.")
        return

    print(f"\n{'─'*70}")
    print(f"  Project Kijiji — Run History ({len(runs)} runs)")
    print(f"{'─'*70}")
    print(f"  {'Run ID':<18}  {'Test Loss':>10}  {'Sep':>8}  {'CV Loss':>8}  {'Time':>6}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")
    for r in runs:
        print(f"  {r['run_id']:<18}  {r['test']['loss']:>10.4f}  "
              f"{r['test']['separation']:>8.4f}  "
              f"{r['cv']['mean_val_loss']:>8.4f}  "
              f"{r['elapsed_s']:>5.1f}s")
    print(f"{'─'*70}\n")