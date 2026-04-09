# recbole.data.dataset

RecBole 的数据集类，用于在内存中存储和预处理原始数据集。每个类为不同的推荐范式提供专门的功能，包括数据过滤、特征工程和数据增强。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；导出 `Dataset`、`SequentialDataset`、`KnowledgeBasedDataset`、`KGSeqDataset`、`DecisionTreeDataset` 和自定义数据集。 |
| `dataset.py` | 实现基础 `Dataset` 类（继承自 `torch.utils.data.Dataset`），适用于通用和上下文感知模型。提供 k-core 过滤、缺失值填充、DataFrame 格式的特征存储和自动数据集下载。 |
| `sequential_dataset.py` | 实现 `SequentialDataset`，扩展 `Dataset` 以支持序列推荐的数据增强，管理历史物品列表和序列长度字段。 |
| `kg_dataset.py` | 实现 `KnowledgeBasedDataset`，扩展 `Dataset` 以加载 `.kg` 和 `.link` 文件，将实体与物品 ID 重新映射，并将知识图谱特征转换为稀疏矩阵或图结构（DGL/PyG）。 |
| `kg_seq_dataset.py` | 实现 `KGSeqDataset`，通过多重继承结合 `SequentialDataset` 和 `KnowledgeBasedDataset`，适用于同时需要序列和知识图谱处理的模型。 |
| `decisiontree_dataset.py` | 实现 `DecisionTreeDataset`，扩展 `Dataset` 以支持决策树模型（如 XGBoost、LightGBM）的 token 到数值的转换。 |
| `customized_dataset.py` | 模型特定的数据集类（如 `GRU4RecKGDataset`、`KSRDataset`、`DIENDataset`），通过命名规则 `[模型名]Dataset` 自动加载。 |
