# recbole.data.dataloader

RecBole 的数据加载器类，负责在训练和评估过程中处理批处理、负采样和数据迭代。所有数据加载器继承自 PyTorch 的 `torch.utils.data.DataLoader`，并支持分布式训练。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件；从子模块导出所有数据加载器类。 |
| `abstract_dataloader.py` | 定义 `AbstractDataLoader`（所有数据加载器的基类，提供批大小管理、随机打乱和数据转换）和 `NegSampleDataLoader`（为逐点和成对训练添加负采样支持）。 |
| `general_dataloader.py` | 实现 `TrainDataLoader`（带负采样的训练数据加载器）、`NegSampleEvalDataLoader`（带采样负样本的评估加载器）和 `FullSortEvalDataLoader`（对所有物品进行全排序评估）。 |
| `knowledge_dataloader.py` | 实现 `KGDataLoader`（加载带负尾实体的知识图谱三元组）和 `KnowledgeBasedDataLoader`（在知识图谱三元组加载和用户-物品交互加载之间交替）。 |
| `user_dataloader.py` | 实现 `UserDataLoader`，按批次迭代用户 ID，用于用户级别的预测任务。 |
