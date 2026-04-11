# -*- coding: utf-8 -*-
"""
benchmark.config
================

BenchmarkConfig: centralised configuration for the benchmark runner.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """All knobs for a benchmark run.

    Attributes:
        models: List of model names to evaluate (e.g. ["BPR", "LightGCN"]).
        datasets: List of dataset names (e.g. ["book", "amazon"]).
        benchmark_rounds: Number of sliding-window test rounds (*m*).
        train_splits: Number of chunks used for training + validation (*n*).
        seeds: List of random seeds for repeated runs.
        output_dir: Root directory for benchmark artifacts.
        base_config_file: Path to a RecBole YAML config that provides shared
            defaults (metrics, topk, gpu_id, …).
        skip_existing: If True, skip (model, dataset, round, seed) combos
            whose status in the DB is already 'done'.
        train_valid_ratio: Ratio of the training portion within the
            train+valid chunks (the rest becomes validation).
        db_name: Filename of the SQLite database inside *output_dir*.
    """

    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    benchmark_rounds: int = 2
    train_splits: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 2023, 12345])
    output_dir: str = "benchmark_results"
    base_config_file: Optional[str] = None
    skip_existing: bool = True
    train_valid_ratio: float = 0.9
    db_name: str = "benchmark.db"

    # ---- derived helpers ------------------------------------------------

    @property
    def total_chunks(self) -> int:
        """Total number of temporal chunks the dataset is partitioned into."""
        return self.benchmark_rounds + self.train_splits

    def validate(self):
        """Raise on obviously invalid combinations."""
        if not self.models:
            raise ValueError("At least one model must be specified.")
        if not self.datasets:
            raise ValueError("At least one dataset must be specified.")
        if self.benchmark_rounds < 1:
            raise ValueError("benchmark_rounds must be >= 1.")
        if self.train_splits < 2:
            raise ValueError(
                "train_splits must be >= 2 (need at least train + valid)."
            )
        if not self.seeds:
            raise ValueError("At least one seed must be specified.")
        if not 0 < self.train_valid_ratio < 1:
            raise ValueError("train_valid_ratio must be in (0, 1).")
