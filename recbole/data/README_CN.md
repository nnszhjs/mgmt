# recbole.data

RecBole 的数据处理包。负责数据集加载、交互表示、数据转换，并提供数据加载器用于在训练和评估过程中向模型传递批量数据。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出 `create_dataset`、`data_preparation`、`save_split_dataloaders` 和 `load_split_dataloaders`。 |
| `utils.py` | 工具函数，包括数据集创建（根据模型类型自动选择数据集类）、数据准备（拆分和构建数据加载器）以及数据加载器的序列化。 |
| `transform.py` | 批量数据转换类，包括 `MaskItemSequence`、`InverseItemSequence`、`CropItemSequence`、`ReorderItemSequence` 和 `UserDefinedTransform`，用于训练时的数据增强。 |
| `interaction.py` | 定义 `Interaction` 类，这是表示一批交互记录（用户-物品对）的核心数据结构，支持张量运算、设备转移和拼接操作。 |

## 子目录

| 目录 | 描述 |
|------|------|
| `dataset/` | 适用于不同推荐范式（通用、序列、知识图谱、决策树）的数据集类。 |
| `dataloader/` | 用于训练和评估的数据加载器类，支持负采样和知识图谱数据。 |
