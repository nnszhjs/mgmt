# recbole.utils

RecBole 的工具模块，提供整个框架共用的辅助函数、枚举类型、参数定义、日志记录和第三方集成。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；聚合并导出关键工具，包括 `init_logger`、`set_color`、`get_model`、`get_trainer`、`init_seed`、`early_stopping`、`calculate_valid_score`、`dict2str`、所有枚举类型、参数列表和 `WandbLogger`。 |
| `utils.py` | 核心工具函数：`get_local_time`、`ensure_dir`、`get_model`（根据名称自动选择模型类）、`get_trainer`（自动选择训练器类）、`get_environment`（收集系统信息）、`early_stopping`、`calculate_valid_score`、`dict2str`、`init_seed`（设置随机种子以确保可复现性）、`get_tensorboard`、`get_gpu_usage`、`get_flops` 和 `list_to_latex`。 |
| `enum_type.py` | RecBole 中使用的枚举定义：`ModelType`（GENERAL、SEQUENTIAL、CONTEXT、KNOWLEDGE、TRADITIONAL、DECISIONTREE）、`KGDataLoaderState`（RSKG、RS、KG）、`EvaluatorType`（RANKING、VALUE）、`InputType`（POINTWISE、PAIRWISE）、`FeatureType` 和 `FeatureSource`。 |
| `argument_list.py` | 预定义参数名称列表：`general_arguments`、`training_arguments`、`evaluation_arguments` 和 `dataset_arguments`，供配置器对参数进行分类。 |
| `logger.py` | 日志设置，包含 `init_logger`（配置彩色控制台输出和文件日志）和 `set_color`（为日志消息添加 ANSI 颜色代码）。包含 `RemoveColorFilter` 用于生成干净的文件日志输出。 |
| `case_study.py` | 案例研究工具：`full_sort_scores`（给定用户 ID 计算所有物品的分数）和 `full_sort_topk`（获取 top-k 物品推荐），适用于训练后的推理和分析。 |
| `url.py` | URL 和下载工具：`decide_download`、`download_url`、`extract_zip`、`makedirs` 和 `rename_atomic_files`，用于自动下载数据集。 |
| `wandblogger.py` | 实现 `WandbLogger`，用于将训练指标、评估结果和最佳指标记录到 Weights & Biases。 |
