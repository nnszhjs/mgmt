# -*- coding: utf-8 -*-
"""
benchmark.splitter
==================

Split a RecBole ``.inter`` file into *m + n* temporal chunks using global
timestamp percentiles.  Each chunk is written as a standalone ``.inter`` file
that can later be concatenated and fed via RecBole's ``benchmark_filename``
mechanism.
"""

import os
from logging import getLogger

import numpy as np
import pandas as pd

from benchmark.config import BenchmarkConfig

logger = getLogger(__name__)


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


def split_dataset(
    dataset_name: str,
    data_path: str,
    output_dir: str,
    total_chunks: int,
    *,
    force: bool = False,
) -> str:
    """Split *dataset_name* into *total_chunks* temporal chunks.

    Chunks are written to ``<output_dir>/<dataset_name>/_splits/chunk_<i>.inter``.

    Args:
        dataset_name: Name of the dataset (e.g. ``"book"``).
        data_path: Root directory that contains ``<dataset_name>/<dataset_name>.inter``.
        output_dir: Benchmark results root (e.g. ``"benchmark_results"``).
        total_chunks: Number of temporal chunks (*m + n*).
        force: If ``True``, re-split even when chunks already exist.

    Returns:
        Path to the ``_splits`` directory.
    """
    inter_file = os.path.join(data_path, dataset_name, f"{dataset_name}.inter")
    splits_dir = os.path.join(output_dir, dataset_name, "_splits")
    marker = os.path.join(splits_dir, f".done_{total_chunks}")

    # Skip if already split
    if not force and os.path.exists(marker):
        logger.info(
            "Splits already exist for %s (chunks=%d), skipping.",
            dataset_name,
            total_chunks,
        )
        return splits_dir

    os.makedirs(splits_dir, exist_ok=True)

    logger.info("Reading %s …", inter_file)
    df = _read_inter(inter_file)
    ts_col = _timestamp_col(df)
    logger.info(
        "Dataset %s: %d interactions, timestamp column = %s",
        dataset_name,
        len(df),
        ts_col,
    )

    # Sort by timestamp globally
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    # Compute percentile boundaries  (total_chunks equal-size bins)
    timestamps = df[ts_col].values.astype(np.float64)
    boundaries = np.percentile(
        timestamps, np.linspace(0, 100, total_chunks + 1)
    )
    # First boundary is -inf so the first chunk captures everything up to p1
    boundaries[0] = -np.inf

    header_line = "\t".join(df.columns)

    for i in range(total_chunks):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i < total_chunks - 1:
            mask = (timestamps > lo) & (timestamps <= hi)
        else:
            # Last chunk: include the upper boundary
            mask = timestamps > lo
        chunk = df[mask]
        out_path = os.path.join(splits_dir, f"chunk_{i}.inter")
        # Write in RecBole .inter format (tab-separated, first line = typed header)
        chunk.to_csv(out_path, sep="\t", index=False)
        logger.info(
            "  chunk_%d: %d interactions  [%.0f, %.0f]",
            i,
            len(chunk),
            lo if np.isfinite(lo) else timestamps.min(),
            hi,
        )

    # Write marker
    with open(marker, "w") as f:
        f.write(f"chunks={total_chunks}\n")

    logger.info("Finished splitting %s into %d chunks.", dataset_name, total_chunks)
    return splits_dir


def prepare_round_files(
    splits_dir: str,
    dataset_name: str,
    round_idx: int,
    train_splits: int,
    train_valid_ratio: float,
    round_dir: str,
) -> str:
    """Assemble the train / valid / test ``.inter`` files for a single round.

    For sliding-window round *r* with *n = train_splits*:
        train + valid = chunks [r, r+1, …, r+n-1]   (split by train_valid_ratio)
        test          = chunk  [r+n]

    Files are written to ``<round_dir>/`` and a symlink-style dataset directory
    is returned that can be passed to RecBole via ``data_path``.

    Returns:
        Path to the directory containing the assembled .inter files.
    """
    round_data_dir = os.path.join(round_dir, dataset_name)
    os.makedirs(round_data_dir, exist_ok=True)

    # Concatenate training chunks
    train_valid_frames = []
    for i in range(round_idx, round_idx + train_splits):
        chunk_path = os.path.join(splits_dir, f"chunk_{i}.inter")
        train_valid_frames.append(pd.read_csv(chunk_path, sep="\t"))
    train_valid_df = pd.concat(train_valid_frames, ignore_index=True)

    # Detect timestamp column and sort
    ts_col = _timestamp_col(train_valid_df)
    train_valid_df = train_valid_df.sort_values(ts_col, kind="mergesort").reset_index(
        drop=True
    )

    # Split into train / valid by ratio (chronological)
    split_point = int(len(train_valid_df) * train_valid_ratio)
    train_df = train_valid_df.iloc[:split_point]
    valid_df = train_valid_df.iloc[split_point:]

    # Test chunk
    test_chunk_path = os.path.join(splits_dir, f"chunk_{round_idx + train_splits}.inter")
    test_df = pd.read_csv(test_chunk_path, sep="\t")

    # Write
    train_df.to_csv(
        os.path.join(round_data_dir, f"{dataset_name}.train.inter"),
        sep="\t",
        index=False,
    )
    valid_df.to_csv(
        os.path.join(round_data_dir, f"{dataset_name}.valid.inter"),
        sep="\t",
        index=False,
    )
    test_df.to_csv(
        os.path.join(round_data_dir, f"{dataset_name}.test.inter"),
        sep="\t",
        index=False,
    )

    logger.info(
        "Round %d: train=%d, valid=%d, test=%d",
        round_idx,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    return round_data_dir
