# recbole

The top-level package of RecBole, an open-source recommendation system framework built on PyTorch. It provides unified interfaces for data processing, model training, evaluation, and hyperparameter tuning across various recommendation paradigms.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; defines the version number (`__version__`). |

## Sub-directories

| Directory | Description |
|-----------|-------------|
| `config/` | Configuration module for loading and managing parameters from YAML files, command-line arguments, and dictionaries. |
| `data/` | Data processing package including dataset loading, interaction representation, data transformation, and dataloaders. |
| `evaluator/` | Evaluation framework with ranking-based and value-based metrics (e.g., Hit, NDCG, MRR, AUC). |
| `model/` | Recommendation model implementations covering general, sequential, context-aware, knowledge-aware, and external library models. |
| `trainer/` | Training orchestration module including trainers for various model types and hyperparameter tuning. |
| `sampler/` | Negative sampling strategies including uniform, popularity-based, and knowledge-graph sampling. |
| `utils/` | Utility functions for logging, enum types, argument lists, seed initialization, and Weights & Biases integration. |
| `quick_start/` | Quick start entry points providing high-level APIs such as `run()` and `run_recbole()` for easy experimentation. |
| `properties/` | Default YAML configuration files for overall settings, individual models, and datasets. |
