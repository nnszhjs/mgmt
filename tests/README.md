# Tests

This directory contains the test suites for the RecBole framework, covering configuration, data processing, evaluation settings, hyperparameter tuning, evaluation metrics, and model correctness.

## Subdirectories

| Directory | Description |
|---|---|
| `config/` | Tests for the configuration module |
| `data/` | Tests for data processing, dataset construction, and dataloaders |
| `evaluation_setting/` | Tests for evaluation setting combinations |
| `hyper_tuning/` | Tests for hyperparameter tuning functionality |
| `metrics/` | Tests for evaluation metric calculations |
| `model/` | Tests for recommendation model training and inference |
| `test_data/` | Shared test dataset used across multiple test modules |

## Detailed Description

### `config/`

Tests for RecBole's configuration system.

| File | Description |
|---|---|
| `test_config.py` | Tests the `Config` class: verifying default settings for general, context-aware, and sequential models; loading configurations from YAML files; overriding with config dictionaries; and testing priority resolution when multiple config sources conflict |
| `test_command_line.py` | Tests command-line argument parsing for configuration overrides (e.g., `use_gpu`, `valid_metric`, `epochs`, `learning_rate`) |
| `test_overall.py` | End-to-end integration tests that run `run_recbole` with various configuration parameters including GPU settings, reproducibility, batch sizes, learning rates, negative sampling, evaluation settings, metrics, split ratios, and mixed precision training |
| `test_config_example.yaml` | Example YAML configuration file used by the config tests |
| `run.sh` | Shell script to run the config test suite via `unittest` and `test_command_line.py` with sample CLI arguments |

### `data/`

Tests for data processing pipelines and dataloader behavior. Contains both test scripts and test data subdirectories.

| File | Description |
|---|---|
| `test_dataset.py` | Comprehensive tests for dataset preprocessing: NaN filtering, duplicate removal, field value filtering, user/item interaction count filtering, ID remapping (including alias support), user/item feature preparation, NaN filling, label thresholding, normalization, and various dataset splitting strategies (TO/RO with RS/LS) for general, sequential, and knowledge-graph datasets |
| `test_dataloader.py` | Tests for dataloaders: verifying general dataloaders, negative sampling in pair-wise and point-wise modes, full-sort evaluation dataloaders, uni100 sampling dataloaders with different batch sizes, and support for different eval modes in validation/test phases |
| `test_transform.py` | Tests for data transformation operations: mask item sequences (for BERT4Rec), inverse item sequences (for SHAN), crop item sequences, and reorder item sequences -- used for data augmentation in sequential recommendation |

The `data/` directory also contains 22+ test dataset subdirectories (e.g., `build_dataset/`, `filter_by_field_value/`, `remap_id/`, `seq_dataset/`, `kg_remap_id/`, etc.) with `.inter`, `.item`, `.user`, `.kg`, and `.link` files used as fixtures for the above tests.

### `evaluation_setting/`

| File | Description |
|---|---|
| `test_evaluation_setting.py` | Tests various combinations of evaluation settings including split methods (leave-one-out LS vs. ratio-based RS), ordering (random RO vs. temporal TO), and evaluation modes (full ranking vs. uni100 sampling) for general, context-aware, and sequential recommenders |

### `hyper_tuning/`

| File | Description |
|---|---|
| `test_hyper_tuning.py` | Tests the `HyperTuning` class with three search algorithms: exhaustive grid search, random search, and Bayesian optimization |
| `test_hyper_tuning_config.yaml` | YAML configuration file specifying base settings for hyper-tuning tests |
| `test_hyper_tuning_params.yaml` | YAML file defining the hyperparameter search space for tuning tests |

### `metrics/`

| File | Description |
|---|---|
| `test_loss_metrics.py` | Tests loss-based evaluation metrics: AUC, RMSE, LogLoss, and MAE with known input/output pairs |
| `test_rank_metrics.py` | Tests rank-based evaluation metrics: GAUC (Group AUC) calculation with known input/output pairs |
| `test_topk_metrics.py` | Tests top-K evaluation metrics: Hit, NDCG, MRR, MAP, Recall, Precision, ItemCoverage, AveragePopularity, GiniIndex, ShannonEntropy, and TailPercentage with known input/output pairs |

### `model/`

| File | Description |
|---|---|
| `test_model_auto.py` | Automated tests for 80+ recommendation models across four categories: General (Pop, BPR, NeuMF, LightGCN, SGL, DiffRec, etc.), Context-aware (LR, FM, DeepFM, xDeepFM, DCN, XGBoost, LightGBM, etc.), Sequential (DIN, GRU4Rec, SASRec, BERT4Rec, SRGNN, etc.), and Knowledge-based (CKE, KGAT, RippleNet, KGCN, etc.) |
| `test_model_manual.py` | Manual tests for models requiring special handling, such as S3Rec with its pretrain/finetune two-stage training process |
| `test_model.yaml` | YAML configuration file defining shared test settings (dataset, epochs, field mappings, loading columns, normalization) for model tests |

### `test_data/`

| File | Description |
|---|---|
| `test/test.inter` | Interaction data for the shared test dataset |
| `test/test.item` | Item feature data for the shared test dataset |
| `test/test.user` | User feature data for the shared test dataset |
| `test/test.kg` | Knowledge graph triple data for the shared test dataset |
| `test/test.link` | Item-entity linking data for the shared test dataset |
