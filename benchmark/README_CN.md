# Benchmark 模块

RecBole的滑动窗口时序评估框架。

## 概述

本模块提供了一个鲁棒的benchmark框架，用于在多个时间窗口上评估推荐模型。它使用**固定长度滑动窗口**结合RecBole的标准`eval_args`进行数据划分。

**核心特性**:
- ✅ 固定长度滑动窗口 (`window_size + rounds`)
- ✅ 使用RecBole标准 `eval_args` 进行train/valid/test划分
- ✅ 自动计算stride确保均匀覆盖
- ✅ 多种子评估提供统计显著性
- ✅ SQLite数据库跟踪结果

---

## 快速开始

```bash
# 1. 从示例创建配置
cp benchmark_config.yaml.example benchmark_config.yaml

# 2. 编辑配置
vim benchmark_config.yaml

# 3. 运行benchmark
./benchmark.sh

# 或直接使用Python
python3 run_benchmark.py --config benchmark_config.yaml
```

---

## 配置说明

### 基本结构

```yaml
temporal_sliding_window:
  window_size: 0.4      # 窗口大小 (40%数据)
  rounds: 4             # 窗口数量

eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO

seeds: [2022, 2023, 2024, 2025, 2026]
models: [BPR, LightGCN, SASRec, MECoDGNN]
datasets: [amazon]
nproc_per_node: 2
```

### 参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `window_size` | float | 窗口大小，占总数据的比例 (0.0-1.0) | `0.4` (40%) |
| `rounds` | int | 滑动窗口数量 | `4` |
| `eval_args` | dict | RecBole标准eval_args | 见下文 |
| `seeds` | list[int] | 随机种子列表 | `[2022, 2023]` |
| `models` | list[str] | 模型名称列表 | `[BPR, LightGCN]` |
| `datasets` | list[str] | 数据集名称列表 | `[amazon]` |
| `nproc_per_node` | int | 每次训练使用的GPU数量 | `2` |

**自动计算**:
- `stride = (1.0 - window_size) / (rounds - 1)`
- `overlap = window_size - stride`

### eval_args 选项

**时间划分 (TS)** - 推荐用于一般模型:
```yaml
eval_args:
  split: {'TS': [0.8, 0.1, 0.1]}
  group_by: user
  order: TO
  mode: full
```

**留一法 (LS)** - 推荐用于序列模型:
```yaml
eval_args:
  split: {'LS': 'valid_and_test'}
  group_by: user
  order: TO
  mode: full
```

---

## 模块结构

```
benchmark/
├── __init__.py
├── config.py          # BenchmarkConfig 配置类
├── splitter.py        # 滑动窗口生成
├── runner.py          # 单个实验运行器
├── launcher.py        # torchrun 包装器
├── db.py              # SQLite 数据库接口
├── report.py          # 结果聚合
└── plot.py            # 可视化工具
```

### 核心组件

#### 1. `config.py` - 配置管理

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

#### 2. `splitter.py` - 窗口生成

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

**输出**:
```
benchmark_results/amazon/_windows/
├── window_0.inter  # [0-40%]
├── window_1.inter  # [20-60%]
├── window_2.inter  # [40-80%]
├── window_3.inter  # [60-100%]
└── .done_w0.4_r4   # 标记文件
```

#### 3. `runner.py` - 实验执行

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

#### 4. `db.py` - 数据库接口

```python
from benchmark.db import BenchmarkDB

with BenchmarkDB('benchmark_results/benchmark.db') as db:
    # 创建运行记录
    run_id = db.get_or_create_run(
        dataset='amazon',
        model='LightGCN',
        window_idx=0,
        seed=2022,
    )
    
    # 标记为运行中
    db.mark_running(run_id)
    
    # 保存结果
    db.mark_done(run_id, valid_metrics, test_metrics)
    
    # 检查状态
    is_done = db.is_done('amazon', 'LightGCN', 0, 2022)
```

---

## 数据库Schema

```sql
CREATE TABLE runs (
    id                INTEGER PRIMARY KEY,
    dataset           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    window_idx        INTEGER NOT NULL,
    window_size       REAL,
    window_start      REAL,
    window_end        REAL,
    seed              INTEGER NOT NULL,
    status            TEXT    NOT NULL,  -- pending | running | done | failed
    started_at        TEXT,
    finished_at       TEXT,
    error_msg         TEXT,
    UNIQUE(dataset, model, window_idx, seed)
);

CREATE TABLE metrics (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    phase   TEXT    NOT NULL,  -- 'valid' 或 'test'
    name    TEXT    NOT NULL,
    value   REAL    NOT NULL,
    UNIQUE(run_id, phase, name)
);
```

---

## 使用示例

### 示例1: 快速测试

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
# 成本: 1 模型 × 1 数据集 × 2 窗口 × 1 种子 = 2次训练
```

### 示例2: 完整评估

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
# 成本: 4 模型 × 1 数据集 × 4 窗口 × 5 种子 = 80次训练
```

### 示例3: 序列模型

```yaml
# sequential_config.yaml
temporal_sliding_window:
  window_size: 0.4
  rounds: 4

eval_args:
  split: {'LS': 'valid_and_test'}  # 留一法
  group_by: user
  order: TO

seeds: [2022, 2023, 2024]
models: [SASRec, GRU4Rec, BERT4Rec]
datasets: [amazon]
nproc_per_node: 2
```

---

## API参考

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
        """自动计算的stride"""
        
    @property
    def overlap(self) -> float:
        """自动计算的overlap"""
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
    """生成固定长度的滑动窗口。
    
    返回:
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
    """在单个时间窗口上训练和评估。"""
```

---

## 版本历史

- **v2 (2024-04)**: 当前版本
  - 使用 `window_size + rounds` 参数
  - 每个窗口使用RecBole标准 `eval_args`
  - 代码更简单、更易维护
  
- **v1 (已弃用)**: 旧版本
  - 使用 `benchmark_rounds + train_splits` 参数
  - 自定义chunk划分和序列特征构建
  - 代码保留在 `*_v1_backup.py` 文件中

---

## 相关文档

- **主README**: `../BENCHMARK_README.md`
- **数据划分指南**: `../data_splitting_explained.md`
- **设计文档**: `../benchmark_sliding_window_design.md`
- **快速参考**: `../QUICK_REFERENCE.txt`
