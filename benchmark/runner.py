# -*- coding: utf-8 -*-
"""
benchmark.runner
================

Run a single (model, dataset, round, seed) experiment by assembling RecBole
config and calling the standard training + evaluation pipeline.
"""

import os
import traceback
from logging import getLogger

from benchmark.config import BenchmarkConfig
from benchmark.db import BenchmarkDB
from benchmark.splitter import prepare_round_files

logger = getLogger(__name__)


def run_single(
    cfg: BenchmarkConfig,
    db: BenchmarkDB,
    model_name: str,
    dataset_name: str,
    round_idx: int,
    seed: int,
    splits_dir: str,
    data_path: str,
):
    """Train and evaluate a single (model, dataset, round, seed) combination.

    The function:
    1. Checks the DB – skips if already ``done`` and ``skip_existing`` is set.
    2. Assembles per-round train/valid/test ``.inter`` files from the
       pre-computed chunks (sliding window).
    3. Builds a RecBole ``Config`` that points at the assembled files via
       ``benchmark_filename``.
    4. Runs the standard ``run_recbole`` pipeline.
    5. Writes results into the SQLite DB.
    """
    # ---- skip check ----
    if cfg.skip_existing and db.is_done(dataset_name, model_name, round_idx, seed):
        logger.info(
            "SKIP %s / %s / round=%d / seed=%d  (already done)",
            model_name,
            dataset_name,
            round_idx,
            seed,
        )
        return

    run_id = db.get_or_create_run(dataset_name, model_name, round_idx, seed)
    db.mark_running(run_id)
    logger.info(
        "START %s / %s / round=%d / seed=%d  (run_id=%d)",
        model_name,
        dataset_name,
        round_idx,
        seed,
        run_id,
    )

    try:
        # ---- assemble round files ----
        round_dir = os.path.join(
            cfg.output_dir, dataset_name, model_name, f"round_{round_idx}", f"seed_{seed}"
        )
        round_data_dir = prepare_round_files(
            splits_dir=splits_dir,
            dataset_name=dataset_name,
            round_idx=round_idx,
            train_splits=cfg.train_splits,
            train_valid_ratio=cfg.train_valid_ratio,
            round_dir=round_dir,
        )
        # round_data_dir looks like:
        #   benchmark_results/book/BPR/round_0/seed_42/book/
        # containing  book.train.inter, book.valid.inter, book.test.inter

        # ---- build RecBole config ----
        config_dict = _build_config_dict(
            cfg, model_name, dataset_name, round_data_dir, round_dir, seed
        )
        config_file_list = [cfg.base_config_file] if cfg.base_config_file else None

        # ---- run ----
        from recbole.quick_start import run_recbole

        result = run_recbole(
            model=model_name,
            dataset=dataset_name,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=True,
        )

        # ---- persist results ----
        valid_metrics = _flatten_result(result.get("best_valid_result", {}))
        test_metrics = _flatten_result(result.get("test_result", {}))
        db.mark_done(run_id, valid_metrics, test_metrics)

        logger.info(
            "DONE  %s / %s / round=%d / seed=%d  test=%s",
            model_name,
            dataset_name,
            round_idx,
            seed,
            _brief(test_metrics),
        )

    except Exception as exc:
        db.mark_failed(run_id, traceback.format_exc())
        logger.error(
            "FAIL  %s / %s / round=%d / seed=%d  error=%s",
            model_name,
            dataset_name,
            round_idx,
            seed,
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config_dict(
    cfg: BenchmarkConfig,
    model_name: str,
    dataset_name: str,
    round_data_dir: str,
    round_dir: str,
    seed: int,
) -> dict:
    """Build the ``config_dict`` passed to ``run_recbole``."""
    # The data_path should be the *parent* of the directory named <dataset>,
    # because RecBole constructs the path as  data_path / dataset / <files>.
    data_path = os.path.dirname(round_data_dir)

    config = {
        "seed": seed,
        "reproducibility": True,
        # Use pre-split files
        "benchmark_filename": ["train", "valid", "test"],
        "data_path": data_path,
        # Save checkpoints inside the round dir
        "checkpoint_dir": os.path.join(round_dir, "saved"),
        # Evaluation
        "eval_args": {
            "split": None,         # Already split
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
    }
    return config


def _flatten_result(result) -> dict:
    """Convert RecBole's OrderedDict result (which may contain tensors) to
    a plain ``{str: float}`` dict."""
    flat = {}
    if result is None:
        return flat
    for k, v in result.items():
        try:
            flat[str(k)] = float(v)
        except (TypeError, ValueError):
            flat[str(k)] = 0.0
    return flat


def _brief(metrics: dict, top_n: int = 3) -> str:
    """Return a short string showing the top-N metrics."""
    items = list(metrics.items())[:top_n]
    return ", ".join(f"{k}={v:.4f}" for k, v in items)
