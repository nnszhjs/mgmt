# -*- coding: utf-8 -*-
"""
benchmark.report
================

Aggregate benchmark results from the SQLite database and generate:

1. A *grand CSV* – rows = models, columns = dataset × metric  (mean±std).
2. *Per-dataset CSVs* – rows = models, columns = metrics.
3. Both "seed-first" and "flat" aggregation views.
"""

import os
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmark.db import BenchmarkDB

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    db: BenchmarkDB,
    dataset: str,
    model: str,
    phase: str = "test",
) -> Dict[str, dict]:
    """Compute aggregated metrics for a (dataset, model) pair.

    Returns a dict keyed by metric name, each value being::

        {
            "flat_mean": float,     # mean over all (round, seed) runs
            "flat_std":  float,
            "per_round_mean": float,  # mean of per-round means
            "per_round_std":  float,  # std  of per-round means
            "n_runs": int,
            "per_round": {round_idx: {"mean": float, "std": float}},
        }
    """
    rows = db.get_metrics(dataset, model, phase=phase)
    if not rows:
        return {}

    # Organise: metric -> {round -> [values across seeds]}
    metric_round_vals: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rnd, seed, name, value in rows:
        metric_round_vals[name][rnd].append(value)

    result = {}
    for metric, round_dict in metric_round_vals.items():
        # Flat aggregation
        all_vals = [v for vals in round_dict.values() for v in vals]
        flat_mean = float(np.mean(all_vals))
        flat_std = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0

        # Per-round aggregation
        round_means = []
        per_round_info = {}
        for rnd in sorted(round_dict):
            vals = round_dict[rnd]
            rm = float(np.mean(vals))
            rs = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            round_means.append(rm)
            per_round_info[rnd] = {"mean": rm, "std": rs, "n": len(vals)}

        pr_mean = float(np.mean(round_means))
        pr_std = float(np.std(round_means, ddof=1)) if len(round_means) > 1 else 0.0

        result[metric] = {
            "flat_mean": flat_mean,
            "flat_std": flat_std,
            "per_round_mean": pr_mean,
            "per_round_std": pr_std,
            "n_runs": len(all_vals),
            "per_round": per_round_info,
        }
    return result


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def generate_reports(
    db: BenchmarkDB,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    phase: str = "test",
):
    """Generate all CSV reports from the database.

    Writes:
        ``<output_dir>/benchmark_report_flat.csv``   – grand table (flat agg)
        ``<output_dir>/benchmark_report_round.csv``  – grand table (per-round agg)
        ``<output_dir>/<dataset>_results_flat.csv``   – per-dataset (flat)
        ``<output_dir>/<dataset>_results_round.csv``  – per-dataset (per-round)
    """
    if datasets is None:
        datasets = sorted(db.all_done_datasets())
    if not datasets:
        logger.warning("No completed runs found – nothing to report.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect all data
    # agg_data[dataset][model] = {metric: agg_dict}
    agg_data: Dict[str, Dict[str, dict]] = {}
    all_metrics_set: set = set()

    for ds in datasets:
        agg_data[ds] = {}
        ds_models = models or sorted(db.all_done_models(ds))
        for mdl in ds_models:
            agg = _aggregate(db, ds, mdl, phase)
            if agg:
                agg_data[ds][mdl] = agg
                all_metrics_set.update(agg.keys())

    all_metrics = sorted(all_metrics_set)
    all_models = sorted(
        {mdl for ds_dict in agg_data.values() for mdl in ds_dict}
    )

    # ---- Grand tables ----
    for agg_key, label in [("flat", "flat"), ("per_round", "round")]:
        lines = []
        header = ["model"]
        for ds in datasets:
            for m in all_metrics:
                header.append(f"{ds}_{m}")
        lines.append(",".join(header))

        for mdl in all_models:
            row = [mdl]
            for ds in datasets:
                mdl_agg = agg_data.get(ds, {}).get(mdl, {})
                for m in all_metrics:
                    if m in mdl_agg:
                        info = mdl_agg[m]
                        mean = info[f"{agg_key}_mean"]
                        std = info[f"{agg_key}_std"]
                        row.append(_fmt(mean, std))
                    else:
                        row.append("")
            lines.append(",".join(row))

        path = os.path.join(output_dir, f"benchmark_report_{label}.csv")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Wrote %s", path)

    # ---- Per-dataset tables ----
    for ds in datasets:
        ds_models = sorted(agg_data.get(ds, {}).keys())
        ds_metrics = sorted(
            {m for mdl_agg in agg_data.get(ds, {}).values() for m in mdl_agg}
        )
        for agg_key, label in [("flat", "flat"), ("per_round", "round")]:
            lines = []
            header = ["model"] + ds_metrics
            lines.append(",".join(header))
            for mdl in ds_models:
                row = [mdl]
                mdl_agg = agg_data[ds][mdl]
                for m in ds_metrics:
                    if m in mdl_agg:
                        info = mdl_agg[m]
                        mean = info[f"{agg_key}_mean"]
                        std = info[f"{agg_key}_std"]
                        row.append(_fmt(mean, std))
                    else:
                        row.append("")
                lines.append(",".join(row))
            path = os.path.join(output_dir, f"{ds}_results_{label}.csv")
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            logger.info("Wrote %s", path)


def print_summary(db: BenchmarkDB):
    """Print a quick summary of the DB to the logger."""
    logger.info("\n%s", db.summary())
