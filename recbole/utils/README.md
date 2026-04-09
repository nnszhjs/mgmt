# recbole.utils

Utility module for RecBole providing shared helper functions, enum types, argument definitions, logging, and third-party integrations used across the entire framework.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; aggregates and exports key utilities including `init_logger`, `set_color`, `get_model`, `get_trainer`, `init_seed`, `early_stopping`, `calculate_valid_score`, `dict2str`, all enum types, argument lists, and `WandbLogger`. |
| `utils.py` | Core utility functions: `get_local_time`, `ensure_dir`, `get_model` (auto-select model class by name), `get_trainer` (auto-select trainer class), `get_environment` (collect system info), `early_stopping`, `calculate_valid_score`, `dict2str`, `init_seed` (set random seeds for reproducibility), `get_tensorboard`, `get_gpu_usage`, `get_flops`, and `list_to_latex`. |
| `enum_type.py` | Enum definitions used throughout RecBole: `ModelType` (GENERAL, SEQUENTIAL, CONTEXT, KNOWLEDGE, TRADITIONAL, DECISIONTREE), `KGDataLoaderState` (RSKG, RS, KG), `EvaluatorType` (RANKING, VALUE), `InputType` (POINTWISE, PAIRWISE), `FeatureType`, and `FeatureSource`. |
| `argument_list.py` | Predefined argument name lists: `general_arguments`, `training_arguments`, `evaluation_arguments`, and `dataset_arguments`, used by the configurator to categorize parameters. |
| `logger.py` | Logging setup with `init_logger` (configures colored console output and file logging) and `set_color` (applies ANSI color codes to log messages). Includes `RemoveColorFilter` for clean file log output. |
| `case_study.py` | Case study utilities: `full_sort_scores` (compute scores for all items given user IDs) and `full_sort_topk` (get top-k item recommendations), useful for inference and analysis after training. |
| `url.py` | URL and download utilities: `decide_download`, `download_url`, `extract_zip`, `makedirs`, and `rename_atomic_files`, used for automatic dataset downloading. |
| `wandblogger.py` | Implements `WandbLogger` for logging training metrics, evaluation results, and best metrics to Weights & Biases. |
