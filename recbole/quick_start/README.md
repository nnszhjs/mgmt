# recbole.quick_start

Quick start entry points for RecBole. This module provides high-level APIs that wire together configuration, dataset creation, model instantiation, and training into simple function calls for rapid experimentation.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports `run`, `run_recbole`, `objective_function`, and `load_data_and_model`. |
| `quick_start.py` | Implements the main entry-point functions: `run()` (launches training with optional multi-GPU/distributed support via `torch.multiprocessing` or `torchrun`), `run_recbole()` (end-to-end pipeline: config loading, dataset creation, data preparation, model creation, training, and evaluation), `objective_function()` (single-run objective for hyperparameter tuning), and `load_data_and_model()` (restores a trained model and its data splits from a checkpoint file). |
