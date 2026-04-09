# recbole.trainer

RecBole 的训练编排模块。提供训练器类来管理完整的训练循环、评估、早停、检查点保存，以及适用于各种推荐模型的超参数调优。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出 `Trainer`、`KGTrainer`、`KGATTrainer`、`S3RecTrainer` 和 `HyperTuning`。 |
| `trainer.py` | 实现训练器类：`AbstractTrainer`（带 fit/evaluate 接口的基类）、`Trainer`（标准训练器，包含训练循环、验证、早停、学习率调度和检查点管理）、`KGTrainer`（在知识图谱和推荐系统训练之间交替）、`KGATTrainer`（为 KGAT 添加注意力分数更新）、`PretrainTrainer` / `S3RecTrainer`（无评估的预训练）、`MKRTrainer`（多任务 KG+RS 训练）、`TraditionalTrainer`（用于非神经网络模型）、`DecisionTreeTrainer` / `XGBoostTrainer` / `LightGBMTrainer`（用于树模型）、`RaCTTrainer`、`RecVAETrainer` 和 `NCLTrainer`。 |
| `hyper_tuning.py` | 实现 `HyperTuning` 类，使用 HyperOpt（树结构 Parzen 估计器）进行自动超参数搜索。支持通过 YAML 文件定义搜索空间和导出结果。 |
