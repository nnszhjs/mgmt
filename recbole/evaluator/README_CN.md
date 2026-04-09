# recbole.evaluator

RecBole 的评估框架。提供模块化的指标系统、用于收集预测结果的收集器，以及编排指标计算的评估器。支持基于排名的指标（如 Hit、NDCG、MRR、Recall、Precision、MAP）和基于数值的指标（如 AUC、RMSE、MAE、LogLoss）。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出所有指标类、评估器、注册器和收集器组件。 |
| `base_metric.py` | 定义抽象基类：`AbstractMetric`（所有指标的基类）、`TopkMetric`（top-k 排名指标的基类，需要 `rec.topk` 数据）和 `LossMetric`（基于损失/数值的指标基类，需要 `rec.score` 数据）。 |
| `metrics.py` | 实现具体指标类，包括 `Hit`、`MRR`、`MAP`、`Recall`、`NDCG`、`Precision`、`GiniIndex`、`GAUC`、`AUC`、`MAE`、`RMSE`、`LogLoss`、`ItemCoverage`、`AveragePopularity`、`ShannonEntropy` 和 `TailPercentage`。 |
| `register.py` | 提供 `cluster_info()` 函数，通过内省自动发现所有指标类，收集其数据需求、类型和"越小越好"标志。导出 `metrics_dict`、`metric_types`、`metric_need` 和 `smaller_metrics`。 |
| `collector.py` | 实现 `DataStruct`（类似字典的评估数据容器）和 `Collector`（在评估过程中将模型预测、标签和物品信息收集到 `DataStruct` 中）。 |
| `utils.py` | 评估工具函数，包括 `pad_sequence`（用于填充分数序列）和 `_binary_clf_curve`（用于计算二分类曲线）。 |
