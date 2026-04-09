# Tests (测试)

本目录包含 RecBole 框架的测试套件，涵盖配置、数据处理、评估设置、超参数调优、评估指标和模型正确性验证。

## 子目录

| 目录 | 描述 |
|---|---|
| `config/` | 配置模块的测试 |
| `data/` | 数据处理、数据集构建和数据加载器的测试 |
| `evaluation_setting/` | 评估设置组合的测试 |
| `hyper_tuning/` | 超参数调优功能的测试 |
| `metrics/` | 评估指标计算的测试 |
| `model/` | 推荐模型训练和推理的测试 |
| `test_data/` | 多个测试模块共用的测试数据集 |

## 详细描述

### `config/`

RecBole 配置系统的测试。

| 文件 | 描述 |
|---|---|
| `test_config.py` | 测试 `Config` 类：验证通用、上下文感知和序列模型的默认设置；从 YAML 文件加载配置；使用配置字典覆盖设置；以及测试多个配置源冲突时的优先级解析 |
| `test_command_line.py` | 测试命令行参数解析，用于配置覆盖（如 `use_gpu`、`valid_metric`、`epochs`、`learning_rate`） |
| `test_overall.py` | 端到端集成测试，使用各种配置参数运行 `run_recbole`，包括 GPU 设置、可复现性、批大小、学习率、负采样、评估设置、指标、数据分割比例和混合精度训练 |
| `test_config_example.yaml` | 配置测试使用的示例 YAML 配置文件 |
| `run.sh` | 通过 `unittest` 和 `test_command_line.py` 运行配置测试套件的 Shell 脚本，包含示例命令行参数 |

### `data/`

数据处理流程和数据加载器行为的测试。包含测试脚本和测试数据子目录。

| 文件 | 描述 |
|---|---|
| `test_dataset.py` | 数据集预处理的全面测试：NaN 过滤、去重、字段值过滤、用户/物品交互数量过滤、ID 重映射（包括别名支持）、用户/物品特征准备、NaN 填充、标签阈值化、归一化，以及各种数据集分割策略（TO/RO 与 RS/LS 组合），涵盖通用、序列和知识图谱数据集 |
| `test_dataloader.py` | 数据加载器测试：验证通用数据加载器、成对和逐点负采样模式、全排序评估数据加载器、不同批大小的 uni100 采样数据加载器，以及验证/测试阶段不同评估模式的支持 |
| `test_transform.py` | 数据变换操作的测试：掩码物品序列（用于 BERT4Rec）、逆序物品序列（用于 SHAN）、裁剪物品序列和重排物品序列——用于序列推荐中的数据增强 |

`data/` 目录还包含 22 个以上的测试数据集子目录（如 `build_dataset/`、`filter_by_field_value/`、`remap_id/`、`seq_dataset/`、`kg_remap_id/` 等），其中包含 `.inter`、`.item`、`.user`、`.kg` 和 `.link` 文件，作为上述测试的测试数据。

### `evaluation_setting/`

| 文件 | 描述 |
|---|---|
| `test_evaluation_setting.py` | 测试各种评估设置组合，包括分割方式（留一法 LS 与 比例分割 RS）、排序方式（随机排序 RO 与 时间排序 TO）以及评估模式（全排序 full 与 uni100 采样），覆盖通用推荐、上下文感知推荐和序列推荐模型 |

### `hyper_tuning/`

| 文件 | 描述 |
|---|---|
| `test_hyper_tuning.py` | 测试 `HyperTuning` 类的三种搜索算法：穷举网格搜索、随机搜索和贝叶斯优化 |
| `test_hyper_tuning_config.yaml` | 指定超参数调优测试基本设置的 YAML 配置文件 |
| `test_hyper_tuning_params.yaml` | 定义调优测试超参数搜索空间的 YAML 文件 |

### `metrics/`

| 文件 | 描述 |
|---|---|
| `test_loss_metrics.py` | 基于损失的评估指标测试：使用已知输入/输出对验证 AUC、RMSE、LogLoss 和 MAE |
| `test_rank_metrics.py` | 基于排名的评估指标测试：使用已知输入/输出对验证 GAUC（分组 AUC）计算 |
| `test_topk_metrics.py` | Top-K 评估指标测试：使用已知输入/输出对验证 Hit、NDCG、MRR、MAP、Recall、Precision、ItemCoverage、AveragePopularity、GiniIndex、ShannonEntropy 和 TailPercentage |

### `model/`

| 文件 | 描述 |
|---|---|
| `test_model_auto.py` | 80 多个推荐模型的自动化测试，涵盖四大类别：通用推荐（Pop、BPR、NeuMF、LightGCN、SGL、DiffRec 等）、上下文感知推荐（LR、FM、DeepFM、xDeepFM、DCN、XGBoost、LightGBM 等）、序列推荐（DIN、GRU4Rec、SASRec、BERT4Rec、SRGNN 等）和基于知识图谱的推荐（CKE、KGAT、RippleNet、KGCN 等） |
| `test_model_manual.py` | 需要特殊处理的模型的手动测试，例如 S3Rec 的预训练/微调两阶段训练流程 |
| `test_model.yaml` | 定义模型测试共享设置（数据集、训练轮数、字段映射、加载列、归一化）的 YAML 配置文件 |

### `test_data/`

| 文件 | 描述 |
|---|---|
| `test/test.inter` | 共享测试数据集的交互数据 |
| `test/test.item` | 共享测试数据集的物品特征数据 |
| `test/test.user` | 共享测试数据集的用户特征数据 |
| `test/test.kg` | 共享测试数据集的知识图谱三元组数据 |
| `test/test.link` | 共享测试数据集的物品-实体链接数据 |
