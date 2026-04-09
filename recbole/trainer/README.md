# recbole.trainer

The training orchestration module for RecBole. It provides trainer classes that manage the full training loop, evaluation, early stopping, checkpoint saving, and hyperparameter tuning for various types of recommendation models.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports `Trainer`, `KGTrainer`, `KGATTrainer`, `S3RecTrainer`, and `HyperTuning`. |
| `trainer.py` | Implements trainer classes: `AbstractTrainer` (base class with fit/evaluate interface), `Trainer` (standard trainer with training loop, validation, early stopping, learning rate scheduling, and checkpoint management), `KGTrainer` (alternates between KG and RS training), `KGATTrainer` (adds attention score updates for KGAT), `PretrainTrainer` / `S3RecTrainer` (pre-training without evaluation), `MKRTrainer` (multi-task KG+RS training), `TraditionalTrainer` (for non-neural models), `DecisionTreeTrainer` / `XGBoostTrainer` / `LightGBMTrainer` (for tree-based models), `RaCTTrainer`, `RecVAETrainer`, and `NCLTrainer`. |
| `hyper_tuning.py` | Implements the `HyperTuning` class for automated hyperparameter search using HyperOpt (Tree-structured Parzen Estimators). Supports defining search spaces via YAML files and exporting results. |
