# Benchmark Module / Benchmark 模块

## English

### What it does

This module adds a sliding-window benchmark runner for RecBole.

Given:
- multiple models
- multiple datasets
- `m` benchmark rounds
- `n` train splits
- `k` random seeds

it will:
1. split each dataset into `m + n` global time-ordered chunks,
2. for round `r`, use chunks `[r, ..., r + n - 1]` as the train+valid window,
3. use chunk `[r + n]` as the test window,
4. split the train+valid window again by time into train / valid,
5. run RecBole training and evaluation for every `(dataset, model, round, seed)` combination,
6. store per-run results in SQLite,
7. generate aggregated CSV reports.

This is designed for reproducible benchmark comparisons across models and datasets.

### Current implementation scope

Implemented:
- temporal chunking by global timestamp,
- sliding-window benchmark rounds,
- per-run result storage in SQLite,
- skip completed runs,
- preserve per-run checkpoints and logs,
- generate summary CSV reports.

Not yet implemented:
- automatic resume from an existing RecBole checkpoint in the middle of a failed run.

Current behavior is:
- if one `(dataset, model, round, seed)` run is already marked `done`, it will be skipped by default;
- if a run failed or was interrupted before completion, it will be re-run;
- checkpoints remain on disk under the corresponding run directory for inspection or future extension.

---

### Files

- `benchmark/config.py` — benchmark configuration dataclass
- `benchmark/splitter.py` — temporal splitting and per-round train/valid/test assembly
- `benchmark/db.py` — SQLite schema and DB helpers
- `benchmark/runner.py` — single-run execution wrapper around `run_recbole`
- `benchmark/report.py` — aggregation and CSV report generation
- `run_benchmark.py` — CLI entrypoint

---

### Directory layout

A typical output tree looks like this:

```text
benchmark_results/
├── benchmark.db
├── benchmark_report_flat.csv
├── benchmark_report_round.csv
├── book_results_flat.csv
├── book_results_round.csv
├── amazon_results_flat.csv
├── amazon_results_round.csv
├── book/
│   ├── _splits/
│   │   ├── chunk_0.inter
│   │   ├── chunk_1.inter
│   │   └── ...
│   └── BPR/
│       └── round_0/
│           └── seed_42/
│               ├── book/
│               │   ├── book.train.inter
│               │   ├── book.valid.inter
│               │   └── book.test.inter
│               └── saved/
└── amazon/
    └── ...
```

### SQLite schema

The benchmark database uses two tables:

#### `runs`
One row per benchmark run.

Fields:
- `dataset`
- `model`
- `round`
- `seed`
- `status` (`pending`, `running`, `done`, `failed`)
- `started_at`
- `finished_at`
- `error_msg`

#### `metrics`
One row per metric value.

Fields:
- `run_id`
- `phase` (`valid` or `test`)
- `name`
- `value`

---

### Aggregation modes

The report module computes two views for each `(dataset, model)` pair:

1. `flat`
   - treat all `m × k` runs as equally weighted samples,
   - report `mean ± std`.

2. `round`
   - first average over seeds inside each round,
   - then average across rounds,
   - also report `mean ± std`.

This produces:
- one global CSV across all selected datasets,
- one per-dataset CSV for easier comparison.

---

### CLI usage

Run from the repository root:

```bash
python3 run_benchmark.py --models BPR LightGCN --datasets book --rounds 2 --train_splits 3 --seeds 42 2023 12345
```

#### Main arguments

- `--models`
  - list of model names, e.g. `BPR LightGCN SASRec`
- `--datasets`
  - list of dataset names, e.g. `book amazon`
- `--rounds`
  - number of benchmark rounds `m`
- `--train_splits`
  - number of train+valid chunks `n`
- `--seeds`
  - list of random seeds
- `--output_dir`
  - output root, default `benchmark_results`
- `--data_path`
  - dataset root, default `dataset/`
- `--base_config`
  - optional shared RecBole config file, e.g. `my_config.yaml`
- `--train_valid_ratio`
  - chronological split ratio inside the train window, default `0.9`
- `--no_skip`
  - re-run completed jobs instead of skipping them
- `--report_only`
  - only rebuild reports from the existing SQLite database
- `--continue_on_error`
  - keep running later jobs even if one job fails

---

### Example commands

#### 1) Minimal benchmark

```bash
python3 run_benchmark.py \
  --models BPR LightGCN \
  --datasets book \
  --rounds 2 \
  --train_splits 3 \
  --seeds 42 2023
```

#### 2) Multiple datasets + shared config

```bash
python3 run_benchmark.py \
  --models BPR LightGCN SASRec \
  --datasets book amazon \
  --rounds 3 \
  --train_splits 4 \
  --seeds 42 2023 12345 \
  --base_config my_config.yaml
```

#### 3) Rebuild reports only

```bash
python3 run_benchmark.py --report_only --output_dir benchmark_results
```

#### 4) Force re-run completed results

```bash
python3 run_benchmark.py \
  --models BPR \
  --datasets book \
  --rounds 2 \
  --train_splits 3 \
  --seeds 42 \
  --no_skip
```

---

### Skip / resume behavior

By default, completed runs are skipped.

That means if SQLite already contains:
- `dataset = book`
- `model = BPR`
- `round = 0`
- `seed = 42`
- `status = done`

then this run will not be executed again.

This is useful when:
- a long benchmark is interrupted,
- you add more models later,
- you add more seeds later,
- only part of the benchmark has finished.

Current resume semantics:
- `done` → skip by default
- `failed` / `pending` / missing → run again
- RecBole checkpoint files are kept, but automatic checkpoint-based recovery is not yet enabled

---

### Output files

#### Database
- `benchmark_results/benchmark.db`
  - all run states and metric values

#### Global reports
- `benchmark_results/benchmark_report_flat.csv`
- `benchmark_results/benchmark_report_round.csv`

These are matrix-style reports:
- rows = models
- columns = `dataset_metric`

Example columns:
- `book_ndcg@10`
- `book_recall@10`
- `amazon_ndcg@10`

#### Per-dataset reports
- `benchmark_results/book_results_flat.csv`
- `benchmark_results/book_results_round.csv`
- `benchmark_results/amazon_results_flat.csv`
- ...

These are easier to read when comparing many metrics inside one dataset.

---

### Notes

1. Benchmark mode relies on the dataset having a timestamp column.
2. Dataset chunking is global by time, not per-user.
3. Within each round, the train/valid split is also chronological.
4. The runner uses RecBole's `benchmark_filename = ["train", "valid", "test"]` mechanism for pre-split input files.
5. If you want consistent shared hyperparameters across models, pass a base YAML with `--base_config`.

---

## 中文

### 功能说明

这个模块为 RecBole 增加了一个滑动时间窗口的 benchmark 运行器。

给定：
- 多个模型
- 多个数据集
- `m` 轮 benchmark
- `n` 个训练时间分片
- `k` 个随机种子

它会：
1. 把每个数据集按全局时间戳切成 `m + n` 份；
2. 对于第 `r` 轮，使用 `[r, ..., r + n - 1]` 这些分片组成 train+valid 窗口；
3. 使用第 `[r + n]` 个分片作为测试集；
4. 再把 train+valid 按时间顺序切成 train / valid；
5. 对每个 `(dataset, model, round, seed)` 组合运行 RecBole 的训练与评估；
6. 把每轮结果写入 SQLite；
7. 最后生成聚合后的 CSV 报表。

它适合做不同模型、不同数据集之间的可复现实验对比。

### 当前实现范围

已实现：
- 基于全局时间戳的分块；
- sliding window benchmark 轮次；
- 基于 SQLite 的单轮结果存储；
- 已完成任务自动跳过；
- 每轮 checkpoint 和中间文件保留；
- 聚合结果并生成 CSV 报表。

尚未实现：
- 在单轮实验失败后，自动从已有 RecBole checkpoint 中间恢复继续训练。

当前行为是：
- 某个 `(dataset, model, round, seed)` 如果已经标记为 `done`，默认直接跳过；
- 如果某轮失败或者中途中断，则下次会重新跑该轮；
- 该轮产生的 checkpoint 会保留在对应目录中，方便后续人工检查或后续扩展自动恢复逻辑。

---

### 模块文件

- `benchmark/config.py` — benchmark 配置 dataclass
- `benchmark/splitter.py` — 时序切分与每轮 train/valid/test 文件拼装
- `benchmark/db.py` — SQLite 表结构和数据库接口
- `benchmark/runner.py` — 单轮实验运行器，封装 `run_recbole`
- `benchmark/report.py` — 结果聚合与 CSV 报表生成
- `run_benchmark.py` — 命令行入口脚本

---

### 目录结构

典型输出目录如下：

```text
benchmark_results/
├── benchmark.db
├── benchmark_report_flat.csv
├── benchmark_report_round.csv
├── book_results_flat.csv
├── book_results_round.csv
├── amazon_results_flat.csv
├── amazon_results_round.csv
├── book/
│   ├── _splits/
│   │   ├── chunk_0.inter
│   │   ├── chunk_1.inter
│   │   └── ...
│   └── BPR/
│       └── round_0/
│           └── seed_42/
│               ├── book/
│               │   ├── book.train.inter
│               │   ├── book.valid.inter
│               │   └── book.test.inter
│               └── saved/
└── amazon/
    └── ...
```

### SQLite 表结构

benchmark 数据库包含两张表：

#### `runs`
每个 benchmark 子任务一行。

字段：
- `dataset`
- `model`
- `round`
- `seed`
- `status`（`pending`、`running`、`done`、`failed`）
- `started_at`
- `finished_at`
- `error_msg`

#### `metrics`
每个指标一行。

字段：
- `run_id`
- `phase`（`valid` 或 `test`）
- `name`
- `value`

---

### 聚合方式

对每个 `(dataset, model)`，当前会计算两种结果：

1. `flat`
   - 把所有 `m × k` 次实验都看成等权样本；
   - 输出 `mean ± std`。

2. `round`
   - 先对每一轮内部的多个 seed 求平均；
   - 再对不同轮次求平均；
   - 同样输出 `mean ± std`。

因此最终会生成：
- 一张跨所有数据集的总表；
- 每个数据集各自一张对比表。

---

### 命令行用法

在仓库根目录执行：

```bash
python3 run_benchmark.py --models BPR LightGCN --datasets book --rounds 2 --train_splits 3 --seeds 42 2023 12345
```

#### 主要参数

- `--models`
  - 模型名列表，例如 `BPR LightGCN SASRec`
- `--datasets`
  - 数据集名列表，例如 `book amazon`
- `--rounds`
  - benchmark 轮数 `m`
- `--train_splits`
  - 训练+验证窗口所用分片数 `n`
- `--seeds`
  - 随机种子列表
- `--output_dir`
  - 输出根目录，默认 `benchmark_results`
- `--data_path`
  - 数据集根目录，默认 `dataset/`
- `--base_config`
  - 可选，共享的 RecBole 配置文件，例如 `my_config.yaml`
- `--train_valid_ratio`
  - 训练窗口内部按时间切分 train/valid 的比例，默认 `0.9`
- `--no_skip`
  - 不跳过已完成任务，而是强制重跑
- `--report_only`
  - 只根据已有 SQLite 数据库重新生成报表
- `--continue_on_error`
  - 某个子任务失败后继续跑后面的任务

---

### 示例命令

#### 1）最小 benchmark

```bash
python3 run_benchmark.py \
  --models BPR LightGCN \
  --datasets book \
  --rounds 2 \
  --train_splits 3 \
  --seeds 42 2023
```

#### 2）多个数据集 + 共享配置

```bash
python3 run_benchmark.py \
  --models BPR LightGCN SASRec \
  --datasets book amazon \
  --rounds 3 \
  --train_splits 4 \
  --seeds 42 2023 12345 \
  --base_config my_config.yaml
```

#### 3）只重建报表

```bash
python3 run_benchmark.py --report_only --output_dir benchmark_results
```

#### 4）强制重跑已完成实验

```bash
python3 run_benchmark.py \
  --models BPR \
  --datasets book \
  --rounds 2 \
  --train_splits 3 \
  --seeds 42 \
  --no_skip
```

---

### 跳过 / 续跑行为

默认情况下，已完成的实验会被跳过。

也就是说，如果 SQLite 里已经有：
- `dataset = book`
- `model = BPR`
- `round = 0`
- `seed = 42`
- `status = done`

那么下次运行时，这一轮不会重复执行。

这在以下场景很有用：
- 长时间 benchmark 被中断；
- 后续新增模型；
- 后续新增种子；
- 只完成了一部分实验。

当前的续跑语义是：
- `done` → 默认跳过
- `failed` / `pending` / 不存在 → 重新运行
- RecBole checkpoint 会保留，但还没有接上“自动从 checkpoint 恢复训练”

---

### 输出文件说明

#### 数据库
- `benchmark_results/benchmark.db`
  - 保存所有运行状态和指标结果

#### 总报表
- `benchmark_results/benchmark_report_flat.csv`
- `benchmark_results/benchmark_report_round.csv`

这两张表是矩阵形式：
- 行 = 模型
- 列 = `dataset_metric`

例如：
- `book_ndcg@10`
- `book_recall@10`
- `amazon_ndcg@10`

#### 分数据集报表
- `benchmark_results/book_results_flat.csv`
- `benchmark_results/book_results_round.csv`
- `benchmark_results/amazon_results_flat.csv`
- ...

当你想在单个数据集内比较很多指标时，这种表更直观。

---

### 说明

1. benchmark 模式依赖数据集存在时间戳列。
2. 数据分块是按全局时间顺序切分，不是按用户分别切分。
3. 每轮内部的 train/valid 划分也同样保持时间顺序。
4. 运行器底层依赖 RecBole 的 `benchmark_filename = ["train", "valid", "test"]` 机制读取预切分文件。
5. 如果你希望不同模型共用同一组训练超参数，建议使用 `--base_config` 传入公共 YAML。
