# -*- coding: utf-8 -*-
"""
benchmark.runner
================

Run a single (model, dataset, window, seed) experiment using RecBole's standard API.

New approach (v2):
- Each window is a standalone .inter file
- Use RecBole's standard eval_args for train/valid/test split
- No custom sequential feature building
"""

import json
import os
import re
import traceback
from logging import getLogger

from benchmark.config import BenchmarkConfig
from benchmark.db import BenchmarkDB
from benchmark.launcher import launch_torchrun

logger = getLogger(__name__)


def run_single_window(
    cfg: BenchmarkConfig,
    db: BenchmarkDB,
    model_name: str,
    dataset_name: str,
    window_idx: int,
    window_info: dict,
    seed: int,
    windows_dir: str,
    repo_root: str,
):
    """Train and evaluate on a single temporal window.

    Args:
        cfg: Benchmark configuration.
        db: Database handle.
        model_name: Model to train (e.g. "LightGCN").
        dataset_name: Dataset name (e.g. "amazon").
        window_idx: Window index (0, 1, 2, ...).
        window_info: Dict with window metadata (start_ratio, end_ratio, size).
        seed: Random seed.
        windows_dir: Directory containing window .inter files.
        repo_root: Repository root path.
    """
    run_id = db.get_or_create_run(
        dataset_name,
        model_name,
        window_idx,
        seed,
        window_size=cfg.window_size,
        window_start=window_info.get('start_ratio'),
        window_end=window_info.get('end_ratio'),
    )
    db.mark_running(run_id)
    logger.info(
        "START %s / %s / window%d (%.1f%%-%.1f%%) / seed=%d  (run_id=%d)",
        model_name,
        dataset_name,
        window_idx,
        window_info.get('start_ratio', 0) * 100,
        window_info.get('end_ratio', 0) * 100,
        seed,
        run_id,
    )

    try:
        # Create run directory
        run_dir = os.path.join(
            cfg.output_dir,
            dataset_name,
            model_name,
            f"window_{window_idx}",
            f"seed_{seed}",
        )
        os.makedirs(run_dir, exist_ok=True)

        # Copy window .inter file to a temporary dataset directory
        temp_dataset_name = f"{dataset_name}_w{window_idx}"
        temp_dataset_dir = os.path.join(cfg.output_dir, "temp", temp_dataset_name)
        os.makedirs(temp_dataset_dir, exist_ok=True)

        window_file = os.path.join(windows_dir, f"window_{window_idx}.inter")
        temp_inter_file = os.path.join(temp_dataset_dir, f"{temp_dataset_name}.inter")

        # Copy window data
        import shutil
        shutil.copy(window_file, temp_inter_file)

        # Build config dict
        config_dict = {
            'seed': seed,
            'eval_args': cfg.eval_args,
            'checkpoint_dir': run_dir,
        }

        # Merge with base config if provided
        if cfg.base_config_file:
            config_files = [os.path.abspath(cfg.base_config_file)]
        else:
            config_files = []

        # Write run-specific config
        run_config_path = os.path.join(run_dir, "run_config.yaml")
        import yaml
        with open(run_config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config_files.append(run_config_path)

        # Launch training via torchrun
        result = launch_torchrun(
            model_name=model_name,
            dataset_name=temp_dataset_name,
            config_files=config_files,
            nproc_per_node=cfg.nproc_per_node,
            run_dir=run_dir,
            repo_root=repo_root,
            timeout=cfg.max_run_timeout,
            data_path=os.path.join(cfg.output_dir, "temp"),
        )

        if result.returncode != 0:
            stderr_path = getattr(result, "stderr_path", "")
            error_tail = ""
            if stderr_path and os.path.exists(stderr_path):
                with open(stderr_path) as f:
                    lines = f.readlines()
                error_tail = "".join(lines[-50:])
            db.mark_failed(run_id, f"exit code {result.returncode}\n{error_tail}")
            raise RuntimeError(
                f"torchrun failed with exit code {result.returncode}. "
                f"See {stderr_path}"
            )

        # Parse results from logs
        stdout_path = getattr(result, "stdout_path", "")
        stderr_path = getattr(result, "stderr_path", "")
        valid_metrics, test_metrics = {}, {}
        for log_path in (stderr_path, stdout_path):
            v, t = _parse_recbole_log(log_path)
            if v:
                valid_metrics = v
            if t:
                test_metrics = t

        # Validate parsed metrics
        if not valid_metrics and not test_metrics:
            error_msg = (
                "Log parsing returned no metrics. "
                f"Check logs at: {stderr_path}, {stdout_path}"
            )
            logger.error(error_msg)
            db.mark_failed(run_id, error_msg)
            raise RuntimeError(error_msg)

        if not test_metrics:
            logger.warning(
                "No test metrics parsed for %s/%s/window%d/seed=%d, "
                "only valid metrics found.",
                model_name, dataset_name, window_idx, seed,
            )

        db.mark_done(run_id, valid_metrics, test_metrics)
        logger.info(
            "DONE  %s / %s / window%d / seed=%d  test=%s",
            model_name, dataset_name, window_idx, seed,
            _brief(test_metrics),
        )

    except Exception as exc:
        if db.run_status(dataset_name, model_name, window_idx, seed) != "done":
            db.mark_failed(run_id, traceback.format_exc())
        logger.error(
            "FAIL  %s / %s / window%d / seed=%d  error=%s",
            model_name, dataset_name, window_idx, seed, exc,
        )
        raise


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(
    r"(?:best valid |test result)\s*:\s*(?:OrderedDict\(\[|\{)(.+?)(?:\]\)|\})",
    re.DOTALL,
)
_METRIC_RE = re.compile(r"['\"]([^'\"]+)['\"](?:,|:)\s*([\d.eE+-]+)")


def _parse_recbole_log(log_path: str):
    """Extract valid and test metrics from a RecBole log."""
    valid_metrics: dict = {}
    test_metrics: dict = {}

    if not os.path.exists(log_path):
        logger.warning("Log file not found: %s", log_path)
        return valid_metrics, test_metrics

    with open(log_path) as f:
        content = f.read()

    for match in _RESULT_RE.finditer(content):
        inner = match.group(1)
        metrics = {}
        for m in _METRIC_RE.finditer(inner):
            metrics[m.group(1)] = float(m.group(2))

        line = match.group(0)
        if "best valid" in line:
            valid_metrics = metrics
        elif "test result" in line:
            test_metrics = metrics

    return valid_metrics, test_metrics


def _brief(metrics: dict, top_n: int = 3) -> str:
    items = list(metrics.items())[:top_n]
    return ", ".join(f"{k}={v:.4f}" for k, v in items)
