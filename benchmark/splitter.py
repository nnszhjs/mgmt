# -*- coding: utf-8 -*-
"""
benchmark.splitter
==================

Generate sliding temporal windows for robust evaluation.

Version History:
- v2 (2024-04): Rewritten to use fixed-length sliding windows (window_size + rounds).
                Each window uses RecBole's standard eval_args.
- v1 (deprecated): Split into m+n chunks with custom sequential features.
                   Old functions (split_dataset, prepare_round_files) kept at bottom
                   for backward compatibility but not used in new workflow.
"""

import os
from logging import getLogger

import numpy as np
import pandas as pd

from benchmark.config import BenchmarkConfig

logger = getLogger(__name__)

_MIN_WINDOW_SIZE = 100  # warn if a window has fewer interactions than this


def _read_inter(path: str) -> pd.DataFrame:
    """Read a RecBole ``.inter`` file (tab-separated with typed header)."""
    df = pd.read_csv(path, sep="\t")
    return df


def _timestamp_col(df: pd.DataFrame) -> str:
    """Return the name of the timestamp column (includes :type suffix)."""
    for col in df.columns:
        if col.startswith("timestamp"):
            return col
    raise ValueError(
        "No timestamp column found in the .inter file.  "
        "Benchmark mode requires a timestamp field."
    )


def generate_sliding_windows(
    dataset_name: str,
    data_path: str,
    output_dir: str,
    window_size: float,
    rounds: int,
    *,
    force: bool = False,
) -> tuple:
    """Generate fixed-length sliding windows for temporal evaluation.

    Args:
        dataset_name: Name of the dataset (e.g. "amazon").
        data_path: Root directory containing <dataset_name>/<dataset_name>.inter.
        output_dir: Benchmark results root (e.g. "benchmark_results").
        window_size: Size of each window as a fraction of total data (e.g. 0.4 = 40%).
        rounds: Number of sliding windows to generate.
        force: If True, regenerate windows even if they already exist.

    Returns:
        (windows_dir, windows_info)
        - windows_dir: Path to directory containing window .inter files
        - windows_info: List of dicts with window metadata

    Example:
        window_size=0.4, rounds=4
        → stride = (1.0 - 0.4) / (4 - 1) = 0.2
        → windows: [0.0-0.4], [0.2-0.6], [0.4-0.8], [0.6-1.0]
    """
    inter_file = os.path.join(data_path, dataset_name, f"{dataset_name}.inter")
    windows_dir = os.path.join(output_dir, dataset_name, "_windows")
    marker = os.path.join(windows_dir, f".done_w{window_size}_r{rounds}")

    # Skip if already generated
    if not force and os.path.exists(marker):
        logger.info(
            "Windows already exist for %s (window_size=%.2f, rounds=%d), skipping.",
            dataset_name, window_size, rounds,
        )
        # Load window info from marker
        windows_info = _load_window_info(marker)
        return windows_dir, windows_info

    os.makedirs(windows_dir, exist_ok=True)

    logger.info("Reading %s …", inter_file)
    df = _read_inter(inter_file)
    ts_col = _timestamp_col(df)
    logger.info(
        "Dataset %s: %d interactions, timestamp column = %s",
        dataset_name, len(df), ts_col,
    )

    # Sort by timestamp globally
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    total_len = len(df)

    # Calculate stride to evenly cover the full time range
    # Last window should end at 1.0
    # (rounds - 1) * stride + window_size = 1.0
    stride = (1.0 - window_size) / (rounds - 1) if rounds > 1 else 0.0

    logger.info(
        "Generating %d sliding windows: window_size=%.2f, stride=%.2f (overlap=%.2f)",
        rounds, window_size, stride, window_size - stride,
    )

    windows_info = []
    for i in range(rounds):
        start_ratio = i * stride
        end_ratio = min(start_ratio + window_size, 1.0)

        start_idx = int(start_ratio * total_len)
        end_idx = int(end_ratio * total_len)

        # Extract window data
        window_df = df.iloc[start_idx:end_idx].copy()

        if len(window_df) < _MIN_WINDOW_SIZE:
            logger.warning(
                "  Window %d has only %d interactions (%.1f%% - %.1f%%) — results may be unreliable.",
                i, len(window_df), start_ratio * 100, end_ratio * 100,
            )

        # Save window as a standalone .inter file
        window_file = os.path.join(windows_dir, f"window_{i}.inter")
        window_df.to_csv(window_file, sep="\t", index=False)

        window_info = {
            "window_idx": i,
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "size": len(window_df),
            "file": window_file,
        }
        windows_info.append(window_info)

        logger.info(
            "  Window %d: %.1f%% - %.1f%% (%d interactions)",
            i, start_ratio * 100, end_ratio * 100, len(window_df),
        )

    # Write marker with window info
    _save_window_info(marker, windows_info, window_size, rounds, stride)

    logger.info("Finished generating %d windows for %s.", rounds, dataset_name)
    return windows_dir, windows_info


def _save_window_info(marker_path: str, windows_info: list, window_size: float, rounds: int, stride: float):
    """Save window metadata to marker file."""
    with open(marker_path, "w") as f:
        f.write(f"window_size={window_size}\n")
        f.write(f"rounds={rounds}\n")
        f.write(f"stride={stride}\n")
        f.write(f"overlap={window_size - stride}\n")
        f.write(f"\n")
        for w in windows_info:
            f.write(
                f"window_{w['window_idx']}: "
                f"{w['start_ratio']:.4f}-{w['end_ratio']:.4f} "
                f"({w['size']} interactions)\n"
            )


def _load_window_info(marker_path: str) -> list:
    """Load window metadata from marker file."""
    windows_info = []
    with open(marker_path) as f:
        for line in f:
            if line.startswith("window_") and ":" in line.split()[0]:
                # Parse: window_0: 0.0000-0.4000 (12345 interactions)
                parts = line.strip().split()
                idx = int(parts[0].split("_")[1].rstrip(":"))
                ratios = parts[1].split("-")
                start_ratio = float(ratios[0])
                end_ratio = float(ratios[1])
                size = int(parts[2].strip("()"))

                windows_info.append({
                    "window_idx": idx,
                    "start_ratio": start_ratio,
                    "end_ratio": end_ratio,
                    "size": size,
                })
    return windows_info


# ============================================================================
# Deprecated functions (old chunk-based approach)
# Kept for backward compatibility, but not used in new sliding window approach
# ============================================================================

# The old approach used split_dataset() and prepare_round_files()
# New approach uses generate_sliding_windows() + RecBole's standard eval_args
