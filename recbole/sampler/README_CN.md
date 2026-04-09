# recbole.sampler

RecBole 的采样模块。提供多种负采样策略，用于在模型训练期间生成负样本（用户未交互过的物品），以支持成对和逐点学习目标。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出 `Sampler`、`KGSampler`、`RepeatableSampler` 和 `SeqSampler`。 |
| `sampler.py` | 实现采样类：`AbstractSampler`（基类，支持均匀分布和基于流行度的别名表采样）、`Sampler`（标准用户-物品交互负采样器，支持动态负采样）、`KGSampler`（知识图谱三元组负采样器，通过替换尾实体生成负样本）、`RepeatableSampler`（确保评估过程中负样本可复现）和 `SeqSampler`（为序列模型如 DIEN 生成负物品序列）。 |
