# recbole.quick_start

RecBole 的快速启动入口。该模块提供高级 API，将配置、数据集创建、模型实例化和训练整合为简单的函数调用，便于快速实验。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出 `run`、`run_recbole`、`objective_function` 和 `load_data_and_model`。 |
| `quick_start.py` | 实现主要入口函数：`run()`（启动训练，支持通过 `torch.multiprocessing` 或 `torchrun` 进行多 GPU/分布式训练）、`run_recbole()`（端到端流程：配置加载、数据集创建、数据准备、模型创建、训练和评估）、`objective_function()`（用于超参数调优的单次运行目标函数）和 `load_data_and_model()`（从检查点文件恢复已训练的模型及其数据拆分）。 |
