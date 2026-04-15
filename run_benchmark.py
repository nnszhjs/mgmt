#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_benchmark.py
===================

Sliding window benchmark runner for RecBole.

Usage:
    python3 run_benchmark_v2.py --config benchmark_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml

# Ensure the repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.config import BenchmarkConfig
from benchmark.db import BenchmarkDB
from benchmark.splitter import generate_sliding_windows
from benchmark.runner import run_single_window
from benchmark.report import generate_reports, print_summary

def parse_args():
    p = argparse.ArgumentParser(
        description="RecBole Benchmark Runner v2 – sliding window temporal evaluation"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark config YAML file (e.g. benchmark_config.yaml)",
    )
    p.add_argument(
        "--report_only",
        action="store_true",
        help="Skip training; only regenerate reports from existing DB.",
    )
    p.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If a single run fails, log the error and continue with the next.",
    )
    p.add_argument(
        "--data_path",
        default="dataset/",
        help="Path to the RecBole dataset directory (default: dataset/).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run experiments, overwriting existing results.",
    )
    p.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force regenerate sliding windows even if they already exist.",
    )
    return p.parse_args()


def load_config(config_path: str, args) -> BenchmarkConfig:
    """Load benchmark config from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract parameters
    tw = config_dict.get('temporal_sliding_window', {})

    # Command-line args override config file
    overwrite = args.overwrite if hasattr(args, 'overwrite') else config_dict.get('overwrite', False)
    force_resplit = args.force_resplit if hasattr(args, 'force_resplit') else config_dict.get('force_resplit', False)

    # Keys managed by BenchmarkConfig — everything else goes to recbole_config
    _BENCHMARK_KEYS = {
        'temporal_sliding_window', 'eval_args', 'seeds', 'models', 'datasets',
        'output_dir', 'base_config_file', 'skip_existing', 'overwrite',
        'force_resplit', 'nproc_per_node', 'max_run_timeout', 'db_name',
    }
    recbole_config = {k: v for k, v in config_dict.items() if k not in _BENCHMARK_KEYS}

    return BenchmarkConfig(
        models=config_dict.get('models', []),
        datasets=config_dict.get('datasets', []),
        window_size=tw.get('window_size', 0.4),
        rounds=tw.get('rounds', 4),
        eval_args=config_dict.get('eval_args', {
            'split': {'TS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',
        }),
        seeds=config_dict.get('seeds', [2022, 2023, 2024, 2025, 2026]),
        output_dir=config_dict.get('output_dir', 'benchmark_results'),
        base_config_file=config_dict.get('base_config_file'),
        skip_existing=config_dict.get('skip_existing', True),
        overwrite=overwrite,
        force_resplit=force_resplit,
        nproc_per_node=config_dict.get('nproc_per_node', 1),
        max_run_timeout=config_dict.get('max_run_timeout', 7200),
        recbole_config=recbole_config,
    )


def main():
    args = parse_args()

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("benchmark")

    # ---- load config ----
    cfg = load_config(args.config, args)
    cfg.validate()

    log.info("=" * 70)
    log.info("Benchmark Configuration")
    log.info("=" * 70)
    log.info("Models: %s", cfg.models)
    log.info("Datasets: %s", cfg.datasets)
    log.info("Window size: %.2f (%.0f%% of data)", cfg.window_size, cfg.window_size * 100)
    log.info("Rounds: %d", cfg.rounds)
    log.info("Stride: %.2f (auto-calculated)", cfg.stride)
    log.info("Overlap: %.2f (%.0f%%)", cfg.overlap, cfg.overlap * 100)
    log.info("Seeds: %s", cfg.seeds)
    log.info("Eval args: %s", cfg.eval_args)
    log.info("GPU processes: %d", cfg.nproc_per_node)
    log.info("Overwrite mode: %s", cfg.overwrite)
    log.info("Force resplit: %s", cfg.force_resplit)
    log.info("=" * 70)

    total_experiments = len(cfg.models) * len(cfg.datasets) * cfg.rounds * len(cfg.seeds)
    log.info(
        "Total experiments: %d models × %d datasets × %d windows × %d seeds = %d",
        len(cfg.models), len(cfg.datasets), cfg.rounds, len(cfg.seeds), total_experiments,
    )

    db_path = os.path.join(cfg.output_dir, cfg.db_name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- report-only mode ----
    if args.report_only:
        with BenchmarkDB(db_path) as db:
            print_summary(db)
            generate_reports(db, cfg.output_dir)
        return

    # Resolve paths
    data_path = os.path.abspath(args.data_path)
    repo_root = _REPO_ROOT

    with BenchmarkDB(db_path) as db:
        # Reset interrupted runs from previous session
        db.reset_stale_running()

        # ---- Step 1: Generate sliding windows for each dataset ----
        windows_map = {}
        for ds in cfg.datasets:
            log.info("")
            log.info("=" * 70)
            log.info("Generating sliding windows for dataset: %s", ds)
            log.info("=" * 70)
            windows_dir, windows_info = generate_sliding_windows(
                dataset_name=ds,
                data_path=data_path,
                output_dir=cfg.output_dir,
                window_size=cfg.window_size,
                rounds=cfg.rounds,
                force=cfg.force_resplit,
            )
            windows_map[ds] = (windows_dir, windows_info)
            log.info("Generated %d windows for %s", len(windows_info), ds)

        # ---- Step 2: Run experiments ----
        log.info("")
        log.info("=" * 70)
        log.info("Starting experiments")
        log.info("=" * 70)

        n_done = 0
        n_skip = 0
        n_fail = 0

        for ds in cfg.datasets:
            windows_dir, windows_info = windows_map[ds]

            for model in cfg.models:
                for window_idx, window_info in enumerate(windows_info):
                    for seed in cfg.seeds:
                        progress = n_done + n_skip + n_fail + 1

                        try:
                            # Check if experiment should be run
                            status = db.run_status(ds, model, cfg.window_size, cfg.rounds, window_idx, seed)

                            if cfg.overwrite and status is not None:
                                # Overwrite mode: delete existing record
                                db.delete_run(ds, model, cfg.window_size, cfg.rounds, window_idx, seed)
                                log.info(
                                    "[%d/%d] OVERWRITE %s/%s/window%d/seed=%d (was: %s)",
                                    progress, total_experiments, model, ds, window_idx, seed, status,
                                )
                            elif status == "done":
                                # Skip completed experiments
                                n_skip += 1
                                log.info(
                                    "[%d/%d] SKIP %s/%s/window%d/seed=%d (already done)",
                                    progress, total_experiments, model, ds, window_idx, seed,
                                )
                                continue
                            elif status in ("failed", "pending"):
                                # Re-run failed or pending experiments
                                log.info(
                                    "[%d/%d] RETRY %s/%s/window%d/seed=%d (was: %s)",
                                    progress, total_experiments, model, ds, window_idx, seed, status,
                                )

                            log.info(
                                "[%d/%d] RUN  %s/%s/window%d (%.1f%%-%.1f%%)/seed=%d",
                                progress, total_experiments, model, ds, window_idx,
                                window_info['start_ratio'] * 100,
                                window_info['end_ratio'] * 100,
                                seed,
                            )

                            run_single_window(
                                cfg=cfg,
                                db=db,
                                model_name=model,
                                dataset_name=ds,
                                window_idx=window_idx,
                                window_info=window_info,
                                seed=seed,
                                windows_dir=windows_dir,
                                repo_root=repo_root,
                            )
                            n_done += 1

                        except Exception as exc:
                            n_fail += 1
                            if args.continue_on_error:
                                log.error("Error (continuing): %s", exc)
                            else:
                                raise

        log.info("")
        log.info("=" * 70)
        log.info(
            "Benchmark finished: %d done, %d skipped, %d failed out of %d total.",
            n_done, n_skip, n_fail, total_experiments,
        )
        log.info("=" * 70)

        # ---- Step 3: Generate reports ----
        print_summary(db)
        generate_reports(db, cfg.output_dir, datasets=cfg.datasets, models=cfg.models)

    log.info("All reports written to %s/", cfg.output_dir)


if __name__ == "__main__":
    main()
