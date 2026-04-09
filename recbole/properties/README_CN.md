# recbole.properties

RecBole 的默认 YAML 配置文件。这些文件定义了框架全局设置、各模型超参数、数据集特定配置和快速启动预设的默认参数值。

## 文件

| 文件 | 描述 |
|------|------|
| `overall.yaml` | 全局默认配置，涵盖环境设置（GPU、随机种子、日志）、训练设置（训练轮数、批大小、优化器、学习率、负采样）和评估设置（指标、top-k、数据拆分策略、批大小）。 |

## 子目录

| 目录 | 描述 |
|------|------|
| `model/` | 各模型的 YAML 配置文件（94 个文件），定义每个支持模型的默认超参数（如 `BPR.yaml`、`LightGCN.yaml`、`SASRec.yaml`、`DeepFM.yaml`、`KGAT.yaml` 等）。 |
| `dataset/` | 数据集特定的 YAML 配置文件（4 个文件），包括 `ml-100k.yaml`、`sample.yaml`、`url.yaml` 和 `kg_url.yaml`，包含数据集路径和字段定义。 |
| `quick_start_config/` | 常见实验场景的预设配置文件（如 `context-aware.yaml`、`sequential.yaml`、`knowledge_base.yaml`、`sequential_DIN.yaml` 等），将模型、数据集和训练设置组合在一起，支持一键运行实验。 |
