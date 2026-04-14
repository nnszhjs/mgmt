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

    Version History:
    - v2 (2024-04): Uses window_size + rounds + eval_args for sliding windows.
    - v1 (deprecated): Used benchmark_rounds + train_splits for chunk-based approach.

    Attributes:
        models: List of model names to evaluate (e.g. ["BPR", "LightGCN"]).
        datasets: List of dataset names (e.g. ["book", "amazon"]).
        window_size: Size of each sliding window as a fraction of total data (0.0-1.0).
        rounds: Number of sliding windows to generate.
        eval_args: RecBole's eval_args dict (split, group_by, order, mode).
        seeds: List of random seeds for repeated runs.
        output_dir: Root directory for benchmark artifacts.
        base_config_file: Path to a RecBole YAML config that provides shared
            defaults (metrics, topk, gpu_id, …).
        skip_existing: If True, skip (model, dataset, window, seed) combos
            whose status in the DB is already 'done'.
        db_name: Filename of the SQLite database inside *output_dir*.
        nproc_per_node: Number of GPU processes per training run.
        max_run_timeout: Maximum wall-clock seconds per training run.
    """

    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)

    # New sliding window parameters
    window_size: float = 0.4
    rounds: int = 4
    eval_args: dict = field(default_factory=lambda: {
        'split': {'TS': [0.8, 0.1, 0.1]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full',
    })

    seeds: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025, 2026])
    output_dir: str = "benchmark_results"
    base_config_file: Optional[str] = None
    skip_existing: bool = True
    db_name: str = "benchmark.db"
    nproc_per_node: int = 1
    max_run_timeout: int = 7200  # seconds (2 hours default)

    # ---- derived helpers ------------------------------------------------

    @property
    def stride(self) -> float:
        """Calculate stride from window_size and rounds."""
        if self.rounds <= 1:
            return 0.0
        return (1.0 - self.window_size) / (self.rounds - 1)

    @property
    def overlap(self) -> float:
        """Calculate overlap from window_size and stride."""
        return self.window_size - self.stride

    def validate(self):
        """Raise on obviously invalid combinations."""
        if not self.models:
            raise ValueError("At least one model must be specified.")
        if not self.datasets:
            raise ValueError("At least one dataset must be specified.")
        if not 0.1 <= self.window_size <= 1.0:
            raise ValueError("window_size must be in [0.1, 1.0].")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1.")
        if not self.seeds:
            raise ValueError("At least one seed must be specified.")
        if not isinstance(self.eval_args, dict):
            raise ValueError("eval_args must be a dict.")
        if 'split' not in self.eval_args:
            raise ValueError("eval_args must contain 'split' key.")
