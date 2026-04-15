# Benchmark Module

Sliding window temporal evaluation framework for RecBole.

## Overview

This module provides a robust benchmark framework for evaluating recommendation models across multiple temporal windows. It uses **fixed-length sliding windows** combined with RecBole's standard `eval_args` for data splitting.

**Key Features**:
- ✅ Fixed-length sliding windows (`window_size + rounds`)
- ✅ RecBole standard `eval_args` for train/valid/test split
- ✅ Automatic stride calculation for even coverage
- ✅ Multi-seed evaluation for statistical significance
- ✅ SQLite database for result tracking

---

## Quick Start

```bash
# 1. Create config from example
cp benchmark_config.yaml.example benchmark_config.yaml

# 2. Edit configuration
vim benchmark_config.yaml

# 3. Run benchmark
./benchmark.sh

# Or use Python directly
python3 run_benchmark.py --config benchmark_config.yaml
```

### Command-line Options

```bash
# Default: skip done, retry failed
./benchmark.sh

# Force re-run all experiments
./benchmark.sh --overwrite

# Regenerate sliding windows
./benchmark.sh --force-resplit

# Complete fresh start
./benchmark.sh --overwrite --force-resplit

# Generate reports only
./benchmark.sh --report
```

---

## Configuration

### Basic Structure

```yaml
temporal_sliding_window:
  window_size: 0.4      # Window size (40% of data)
  rounds: 4             # Number of windows

eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO

seeds: [2022, 2023, 2024, 2025, 2026]
models: [BPR, LightGCN, SASRec, MECoDGNN]
datasets: [amazon]
nproc_per_node: 2
```

### Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `window_size` | float | Window size as fraction (0.0-1.0) | `0.4` (40%) |
| `rounds` | int | Number of sliding windows | `4` |
| `eval_args` | dict | RecBole's standard eval_args | See below |
| `seeds` | list[int] | Random seeds for repeated runs | `[2022, 2023]` |
| `models` | list[str] | Model names to evaluate | `[BPR, LightGCN]` |
| `datasets` | list[str] | Dataset names | `[amazon]` |
| `nproc_per_node` | int | GPUs per training run | `2` |

**Auto-calculated**:
- `stride = (1.0 - window_size) / (rounds - 1)`
- `overlap = window_size - stride`

### eval_args Options

**Time-based Splitting (TS)** - Recommended for general models:
```yaml
eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO
  mode: full
```

**Leave-one-out (LS)** - Recommended for sequential models:
```yaml
eval_args:
  split: {'LS': 'valid_and_test'}
  group_by: user
  order: TO
  mode: full
```

---

## Module Structure

```
benchmark/
├── __init__.py
├── config.py          # BenchmarkConfig dataclass
├── splitter.py        # Sliding window generation
├── runner.py          # Single experiment runner
├── launcher.py        # torchrun wrapper
├── db.py              # SQLite database interface
├── report.py          # Result aggregation
└── plot.py            # Visualization utilities
```

### Core Components

#### 1. `config.py` - Configuration

```python
from benchmark.config import BenchmarkConfig

cfg = BenchmarkConfig(
    window_size=0.4,
    rounds=4,
    eval_args={'split': {'TS': [0.8, 0.1, 0.1]}, ...},
    seeds=[2022, 2023],
    models=['BPR', 'LightGCN'],
    datasets=['amazon'],
)
```

#### 2. `splitter.py` - Window Generation

```python
from benchmark.splitter import generate_sliding_windows

windows_dir, windows_info = generate_sliding_windows(
    dataset_name='amazon',
    data_path='dataset/',
    output_dir='benchmark_results',
    window_size=0.4,
    rounds=4,
)
```

**Output**:
```
benchmark_results/amazon/_windows/
├── window_0.inter  # [0-40%]
├── window_1.inter  # [20-60%]
├── window_2.inter  # [40-80%]
├── window_3.inter  # [60-100%]
└── .done_w0.4_r4   # Marker file
```

#### 3. `runner.py` - Experiment Execution

```python
from benchmark.runner import run_single_window

run_single_window(
    cfg=cfg,
    db=db,
    model_name='LightGCN',
    dataset_name='amazon',
    window_idx=0,
    window_info={'start_ratio': 0.0, 'end_ratio': 0.4, ...},
    seed=2022,
    windows_dir='benchmark_results/amazon/_windows',
    repo_root='.',
)
```

#### 4. `db.py` - Database Interface

```python
from benchmark.db import BenchmarkDB

with BenchmarkDB('benchmark_results/benchmark.db') as db:
    # Create run
    run_id = db.get_or_create_run(
        dataset='amazon',
        model='LightGCN',
        window_idx=0,
        seed=2022,
    )
    
    # Mark running
    db.mark_running(run_id)
    
    # Save results
    db.mark_done(run_id, valid_metrics, test_metrics)
    
    # Check status
    is_done = db.is_done('amazon', 'LightGCN', 0, 2022)
```

---

## Database Schema

```sql
CREATE TABLE runs (
    id                INTEGER PRIMARY KEY,
    dataset           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    window_size       REAL    NOT NULL,
    rounds            INTEGER NOT NULL,
    window_idx        INTEGER NOT NULL,
    window_start      REAL,
    window_end        REAL,
    seed              INTEGER NOT NULL,
    status            TEXT    NOT NULL,  -- pending | running | done | failed
    started_at        TEXT,
    finished_at       TEXT,
    error_msg         TEXT,
    UNIQUE(dataset, model, window_size, rounds, window_idx, seed)
);

CREATE TABLE metrics (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    phase   TEXT    NOT NULL,  -- 'valid' or 'test'
    name    TEXT    NOT NULL,
    value   REAL    NOT NULL,
    UNIQUE(run_id, phase, name)
);
```

**Unique Key**: `(dataset, model, window_size, rounds, window_idx, seed)`

This allows you to run the same model/dataset/seed with different window configurations, and results will be stored separately.

---

## Run Modes

### Default Mode

```bash
./benchmark.sh
```

**Behavior**:
- Skip experiments with status = `done`
- Re-run experiments with status = `failed` or `pending`
- Reuse existing window files

**Use case**: Continue interrupted runs, retry failed experiments

### Overwrite Mode

```bash
./benchmark.sh --overwrite
```

**Behavior**:
- Delete existing records for matching experiments
- Force re-run all experiments
- Reuse existing window files

**Use case**: Code changed, need to re-validate results

### Force Resplit Mode

```bash
./benchmark.sh --force-resplit
```

**Behavior**:
- Regenerate window files even if they exist
- Skip experiments with status = `done`

**Use case**: Original dataset changed, need to regenerate windows

### Combined Mode

```bash
./benchmark.sh --overwrite --force-resplit
```

**Behavior**:
- Regenerate window files
- Delete existing records
- Force re-run all experiments

**Use case**: Complete fresh start

---

## Usage Examples

### Example 1: Quick Test

```yaml
# test_config.yaml
temporal_sliding_window:
  window_size: 0.5
  rounds: 2

eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO

seeds: [2022]
models: [BPR]
datasets: [ml-100k]
nproc_per_node: 1
```

```bash
python3 run_benchmark.py --config test_config.yaml
# Cost: 1 model × 1 dataset × 2 windows × 1 seed = 2 training runs
```

### Example 2: Full Evaluation

```yaml
# full_config.yaml
temporal_sliding_window:
  window_size: 0.4
  rounds: 4

eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO

seeds: [2022, 2023, 2024, 2025, 2026]
models: [BPR, LightGCN, SASRec, MECoDGNN]
datasets: [amazon]
nproc_per_node: 2
```

```bash
python3 run_benchmark.py --config full_config.yaml
# Cost: 4 models × 1 dataset × 4 windows × 5 seeds = 80 training runs
```

### Example 3: Sequential Models

```yaml
# sequential_config.yaml
temporal_sliding_window:
  window_size: 0.4
  rounds: 4

eval_args:
  split: {'LS': 'valid_and_test'}  # Leave-one-out
  group_by: user
  order: TO

seeds: [2022, 2023, 2024]
models: [SASRec, GRU4Rec, BERT4Rec]
datasets: [amazon]
nproc_per_node: 2
```

---

## API Reference

### BenchmarkConfig

```python
@dataclass
class BenchmarkConfig:
    models: List[str]
    datasets: List[str]
    window_size: float = 0.4
    rounds: int = 4
    eval_args: dict = field(default_factory=lambda: {...})
    seeds: List[int] = field(default_factory=lambda: [2022, ...])
    output_dir: str = "benchmark_results"
    skip_existing: bool = True
    nproc_per_node: int = 1
    max_run_timeout: int = 7200
    
    @property
    def stride(self) -> float:
        """Auto-calculated stride"""
        
    @property
    def overlap(self) -> float:
        """Auto-calculated overlap"""
```

### generate_sliding_windows()

```python
def generate_sliding_windows(
    dataset_name: str,
    data_path: str,
    output_dir: str,
    window_size: float,
    rounds: int,
    *,
    force: bool = False,
) -> tuple[str, list[dict]]:
    """Generate fixed-length sliding windows.
    
    Returns:
        (windows_dir, windows_info)
    """
```

### run_single_window()

```python
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
    """Train and evaluate on a single temporal window."""
```

---

## Version History

- **v2 (2024-04)**: Current version
  - Uses `window_size + rounds` parameters
  - Each window uses RecBole's standard `eval_args`
  - Simpler, more maintainable code
  
- **v1 (deprecated)**: Old version
  - Used `benchmark_rounds + train_splits` parameters
  - Custom chunk-based splitting with sequential features
  - Code preserved in `*_v1_backup.py` files

---

## See Also

- **Main README**: `../BENCHMARK_README.md`
- **Data Splitting Guide**: `../data_splitting_explained.md`
- **Design Document**: `../benchmark_sliding_window_design.md`
- **Quick Reference**: `../QUICK_REFERENCE.txt`
