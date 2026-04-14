# -*- coding: utf-8 -*-
"""
benchmark.plot
==============

Generate training loss curve plots from the benchmark database.

Usage::

    # From Python
    from benchmark.plot import plot_loss_curves
    from benchmark.db import BenchmarkDB
    db = BenchmarkDB("benchmark_results/benchmark.db")
    plot_loss_curves(db, "benchmark_results", dataset="book")

    # From CLI
    python3 -m benchmark.plot --output_dir benchmark_results
    python3 -m benchmark.plot --output_dir benchmark_results --dataset book --models BPR LightGCN
"""

import argparse
import os
import sys
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Optional

import numpy as np

logger = getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_loss_curves(
    db,
    output_dir: str,
    dataset: Optional[str] = None,
    models: Optional[List[str]] = None,
    benchmark_rounds: Optional[int] = None,
    train_splits: Optional[int] = None,
):
    """Generate loss curve plots from the benchmark database.

    Produces two types of plots per dataset:

    1. **Per-run plots**: one plot per ``(model, round_idx, seed)`` showing
       the raw epoch-by-epoch loss curve.  Saved under each run directory.

    2. **Comparison plot**: all models on one figure, with mean ± std
       across rounds and seeds.  Saved at
       ``<output_dir>/<dataset>_loss_curves.png``.
    """
    if not HAS_MPL:
        logger.warning(
            "matplotlib is not installed — skipping loss curve plots. "
            "Install with: pip install matplotlib"
        )
        return

    datasets = [dataset] if dataset else db.all_done_datasets()

    for ds in datasets:
        ds_models = models or sorted(db.all_done_models(ds))
        if not ds_models:
            continue

        # ---- Comparison plot: mean ± std across runs ----
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        has_data = False

        for mdl in ds_models:
            rows = db.get_losses(
                ds, mdl,
                benchmark_rounds=benchmark_rounds,
                train_splits=train_splits,
            )
            if not rows:
                continue

            # Group by epoch across all (round_idx, seed)
            epoch_vals: Dict[int, List[float]] = defaultdict(list)
            for _round, _seed, epoch, loss in rows:
                epoch_vals[epoch].append(loss)

            epochs = sorted(epoch_vals.keys())
            means = np.array([np.mean(epoch_vals[e]) for e in epochs])
            stds = np.array([np.std(epoch_vals[e]) for e in epochs])

            ax.plot(epochs, means, label=mdl, linewidth=1.5)
            ax.fill_between(
                epochs, means - stds, means + stds, alpha=0.15
            )
            has_data = True

        if has_data:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title(f"Training Loss Curves — {ds}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig_path = os.path.join(output_dir, f"{ds}_loss_curves.png")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=150)
            logger.info("Wrote comparison loss plot: %s", fig_path)
        plt.close(fig)

        # ---- Per-model individual plots ----
        for mdl in ds_models:
            rows = db.get_losses(
                ds, mdl,
                benchmark_rounds=benchmark_rounds,
                train_splits=train_splits,
            )
            if not rows:
                continue

            # Group by (round_idx, seed)
            run_curves: Dict[tuple, List[tuple]] = defaultdict(list)
            for rnd, seed, epoch, loss in rows:
                run_curves[(rnd, seed)].append((epoch, loss))

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            for (rnd, seed), curve in sorted(run_curves.items()):
                curve.sort()
                epochs = [e for e, _ in curve]
                losses = [l for _, l in curve]
                ax.plot(
                    epochs, losses,
                    label=f"r{rnd}/s{seed}",
                    linewidth=0.8, alpha=0.7,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title(f"{mdl} — {ds}")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            fig_path = os.path.join(output_dir, f"{ds}_{mdl}_loss_curves.png")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=150)
            logger.info("Wrote per-model loss plot: %s", fig_path)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point:  python3 -m benchmark.plot
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate loss curve plots from benchmark results"
    )
    parser.add_argument(
        "--output_dir", default="benchmark_results",
        help="Benchmark output directory (default: benchmark_results)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name (default: all datasets in DB)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names (default: all models in DB)",
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from benchmark.db import BenchmarkDB

    db_path = os.path.join(args.output_dir, "benchmark.db")
    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        return

    with BenchmarkDB(db_path) as db:
        plot_loss_curves(
            db, args.output_dir,
            dataset=args.dataset,
            models=args.models,
        )


if __name__ == "__main__":
    main()
