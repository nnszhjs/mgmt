# -*- coding: utf-8 -*-
"""
benchmark.launcher
==================

Launch a single RecBole training run via ``torchrun``.

This module is the **only** place that constructs the ``torchrun`` command.
Both single-GPU and multi-GPU runs go through here — single-GPU simply uses
``--nproc_per_node=1``.
"""

import os
import json
import random
import socket
import subprocess
from logging import getLogger
from typing import Dict, List, Optional

import yaml

logger = getLogger(__name__)


def _free_port(max_retries: int = 5) -> int:
    """Find a free TCP port on localhost with retry.

    Uses a random port in the 29500-39999 range to reduce collision
    probability when multiple experiments launch concurrently.
    """
    for _ in range(max_retries):
        port = random.randint(29500, 39999)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    # Fallback: let OS assign
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def write_run_config(
    run_dir: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    round_data_dir: str,
    extra: Optional[Dict] = None,
) -> str:
    """Write a per-run RecBole YAML config file.

    Returns:
        Path to the written config file.
    """
    # data_path should be the parent of the directory named <dataset>,
    # because RecBole constructs the path as  data_path / dataset / <files>.
    data_path = os.path.dirname(round_data_dir)

    config = {
        "seed": seed,
        "reproducibility": True,
        # Use pre-split files
        "benchmark_filename": ["train", "valid", "test"],
        "data_path": data_path,
        # Save checkpoints inside the run dir
        "checkpoint_dir": os.path.join(run_dir, "saved"),
        # Evaluation — already split, don't re-split
        "eval_args": {
            "split": None,
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        # Allow loading the dynamically generated sequence columns
        "load_col": None,
    }
    if extra:
        config.update(extra)

    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "run_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.debug("Wrote run config: %s", config_path)
    return config_path


def launch_torchrun(
    model_name: str,
    dataset_name: str,
    config_files: List[str],
    nproc_per_node: int = 1,
    run_dir: str = ".",
    repo_root: str = ".",
    timeout: Optional[int] = None,
    data_path: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Launch a ``torchrun`` process for a single benchmark run.

    Args:
        model_name: RecBole model class name.
        dataset_name: Dataset name.
        config_files: List of config file paths.
        nproc_per_node: Number of GPU processes.
        run_dir: Directory for stdout/stderr logs.
        repo_root: Path to the RecBole repository root (where run_recbole.py lives).
        timeout: Maximum wall-clock seconds for the subprocess (None = no limit).
        data_path: Optional data_path to pass to RecBole (for temp datasets).

    Returns:
        ``subprocess.CompletedProcess`` with returncode and log paths.
    """
    master_port = _free_port()

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        os.path.join(repo_root, "run_recbole.py"),
        f"--model={model_name}",
        f"--dataset={dataset_name}",
    ]
    if config_files:
        cmd.append("--config_files")
        cmd.extend(config_files)
    if data_path:
        # Add data_path as environment variable or config
        # RecBole will read it from config files
        pass

    os.makedirs(run_dir, exist_ok=True)
    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    logger.info("CMD: %s", " ".join(cmd))
    logger.info("  stdout → %s", stdout_path)
    logger.info("  stderr → %s", stderr_path)

    try:
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            result = subprocess.run(
                cmd,
                stdout=out,
                stderr=err,
                cwd=repo_root,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "torchrun timed out after %d seconds — see %s",
            timeout, stderr_path,
        )
        result = subprocess.CompletedProcess(cmd, returncode=-1)
        result.stdout_path = stdout_path  # type: ignore[attr-defined]
        result.stderr_path = stderr_path  # type: ignore[attr-defined]
        return result

    result.stdout_path = stdout_path  # type: ignore[attr-defined]
    result.stderr_path = stderr_path  # type: ignore[attr-defined]

    if result.returncode != 0:
        logger.error(
            "torchrun exited with code %d — see %s", result.returncode, stderr_path
        )

    return result
