#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_benchmark.py
================

One-click benchmark runner for RecBole.

Usage examples
--------------
::

    # Minimal – 2 models, 1 dataset, 2 rounds, 3 train splits, 3 seeds
    python run_benchmark.py \\
        --models BPR LightGCN \\
        --datasets book \\
        --rounds 2 --train_splits 3 \\
        --seeds 42 2023 12345

    # With a base config for shared hyper-parameters
    python run_benchmark.py \\
        --models BPR LightGCN SASRec \\
        --datasets book amazon \\
        --rounds 3 --train_splits 4 \\
        --seeds 42 2023 \\
        --base_config my_config.yaml \\
        --output_dir benchmark_results

    # Re-run failures only (already-done runs are skipped by default)
    python run_benchmark.py \\
        --models BPR \\
        --datasets book \\
        --rounds 2 --train_splits 3 \\
        --seeds 42

    # Force re-run everything
    python run_benchmark.py \\
        --models BPR \\
        --datasets book \\
        --rounds 2 --train_splits 3 \\
        --seeds 42 \\
        --no_skip

    # Generate reports only (no training)
    python run_benchmark.py --report_only --output_dir benchmark_results
"""

import argparse
import logging
import os
import sys

# Ensure the repo root is on sys.path so that ``import benchmark`` works
# regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.config import BenchmarkConfig
from benchmark.db import BenchmarkDB
from benchmark.report import generate_reports, print_summary
from benchmark.runner import run_single
from benchmark.splitter import split_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="RecBole Benchmark Runner – sliding-window temporal evaluation"
    )

    # Core
    p.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Model names to benchmark (e.g. BPR LightGCN SASRec).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset names (e.g. book amazon).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of sliding-window test rounds m (default: 2).",
    )
    p.add_argument(
        "--train_splits",
        type=int,
        default=3,
        help="Number of chunks for training + validation n (default: 3).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 2023, 12345],
        help="Random seeds for repeated runs (default: 42 2023 12345).",
    )

    # Paths
    p.add_argument(
        "--output_dir",
        default="benchmark_results",
        help="Root output directory (default: benchmark_results).",
    )
    p.add_argument(
        "--data_path",
        default="dataset/",
        help="Path to the RecBole dataset directory (default: dataset/).",
    )
    p.add_argument(
        "--base_config",
        default=None,
        help="Path to a RecBole YAML config for shared defaults.",
    )

    # Behaviour
    p.add_argument(
        "--no_skip",
        action="store_true",
        help="Do NOT skip already-completed runs.",
    )
    p.add_argument(
        "--train_valid_ratio",
        type=float,
        default=0.9,
        help="Chronological train/valid split ratio within the training chunks (default: 0.9).",
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

    return p.parse_args()


def main():
    args = parse_args()

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("benchmark")

    # ---- config ----
    cfg = BenchmarkConfig(
        models=args.models,
        datasets=args.datasets,
        benchmark_rounds=args.rounds,
        train_splits=args.train_splits,
        seeds=args.seeds,
        output_dir=args.output_dir,
        base_config_file=args.base_config,
        skip_existing=not args.no_skip,
        train_valid_ratio=args.train_valid_ratio,
    )

    db_path = os.path.join(cfg.output_dir, cfg.db_name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- report-only mode ----
    if args.report_only:
        with BenchmarkDB(db_path) as db:
            print_summary(db)
            generate_reports(db, cfg.output_dir)
        return

    # ---- validate ----
    cfg.validate()
    log.info("Benchmark config: %s", cfg)
    log.info(
        "Total experiments: %d models × %d datasets × %d rounds × %d seeds = %d",
        len(cfg.models),
        len(cfg.datasets),
        cfg.benchmark_rounds,
        len(cfg.seeds),
        len(cfg.models) * len(cfg.datasets) * cfg.benchmark_rounds * len(cfg.seeds),
    )

    # Resolve data_path to absolute
    data_path = os.path.abspath(args.data_path)

    with BenchmarkDB(db_path) as db:
        # ---- Step 1: Split datasets ----
        splits = {}
        for ds in cfg.datasets:
            splits[ds] = split_dataset(
                dataset_name=ds,
                data_path=data_path,
                output_dir=cfg.output_dir,
                total_chunks=cfg.total_chunks,
            )

        # ---- Step 2: Run experiments ----
        n_done = 0
        n_skip = 0
        n_fail = 0
        total = (
            len(cfg.models)
            * len(cfg.datasets)
            * cfg.benchmark_rounds
            * len(cfg.seeds)
        )

        for ds in cfg.datasets:
            for model in cfg.models:
                for r in range(cfg.benchmark_rounds):
                    for seed in cfg.seeds:
                        try:
                            if cfg.skip_existing and db.is_done(ds, model, r, seed):
                                n_skip += 1
                                log.info(
                                    "[%d/%d] SKIP %s/%s/r%d/s%d",
                                    n_done + n_skip + n_fail,
                                    total,
                                    model,
                                    ds,
                                    r,
                                    seed,
                                )
                                continue

                            log.info(
                                "[%d/%d] RUN  %s/%s/r%d/s%d",
                                n_done + n_skip + n_fail + 1,
                                total,
                                model,
                                ds,
                                r,
                                seed,
                            )
                            run_single(
                                cfg=cfg,
                                db=db,
                                model_name=model,
                                dataset_name=ds,
                                round_idx=r,
                                seed=seed,
                                splits_dir=splits[ds],
                                data_path=data_path,
                            )
                            n_done += 1

                        except Exception as exc:
                            n_fail += 1
                            if args.continue_on_error:
                                log.error("Error (continuing): %s", exc)
                            else:
                                raise

        log.info(
            "Benchmark finished: %d done, %d skipped, %d failed out of %d total.",
            n_done,
            n_skip,
            n_fail,
            total,
        )

        # ---- Step 3: Generate reports ----
        print_summary(db)
        generate_reports(
            db,
            cfg.output_dir,
            datasets=cfg.datasets,
            models=cfg.models,
        )

    log.info("All reports written to %s/", cfg.output_dir)


if __name__ == "__main__":
    main()
