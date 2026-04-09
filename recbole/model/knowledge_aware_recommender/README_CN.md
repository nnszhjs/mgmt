# 知识感知推荐模型

本目录包含知识感知推荐模型。这些模型融合知识图谱（KG）信息来丰富物品和用户表示，捕捉实体之间的语义关系，从而超越纯协同过滤方法提升推荐质量。

## 模型列表

| 文件 | 模型 | 参考文献 | 描述 |
|------|------|----------|------|
| `cfkg.py` | CFKG | Qingyao Ai et al., "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation." MDPI 2018 | 将知识图谱和用户-物品交互图合并为统一图，用户、物品和属性均视为实体。 |
| `cke.py` | CKE | Fuzheng Zhang et al., "Collaborative Knowledge Base Embedding for Recommender Systems." SIGKDD 2016 | 融合KG结构知识来丰富物品表示，增强协同过滤效果。 |
| `kgat.py` | KGAT | Xiang Wang et al., "KGAT: Knowledge Graph Attention Network for Recommendation." SIGKDD 2019 | 知识图谱注意力网络，在KG增强的用户-物品图上进行注意力嵌入传播。 |
| `kgcn.py` | KGCN | Hongwei Wang et al., "Knowledge Graph Convolution Networks for Recommender Systems." WWW 2019 | 知识图谱卷积网络，通过挖掘KG上的关联属性捕捉物品间的关联性。 |
| `kgin.py` | KGIN | Xiang Wang et al., "Learning Intents behind Interactions with Knowledge Graph for Recommendation." WWW 2021 | 利用知识图谱信息学习交互背后的用户意图，实现意图感知推荐。 |
| `kgnnls.py` | KGNNLS | Hongwei Wang et al., "Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems." KDD 2019 | 带标签平滑正则化的知识感知图神经网络推荐模型。 |
| `ktup.py` | KTUP | Yixin Cao et al., "Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences." WWW 2019 | 多任务学习模型，联合学习推荐和KG任务，使用注意力机制聚合偏好。 |
| `mcclk.py` | MCCLK | Ding Zou et al., "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." SIGIR 2022 | 多层次跨视图对比学习框架，用于知识感知推荐。 |
| `mkr.py` | MKR | Hongwei Wang et al., "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation." WWW 2019 | 多任务框架，使用交叉压缩单元在推荐和KG嵌入任务之间共享潜在特征。 |
| `ripplenet.py` | RippleNet | Hongwei Wang et al., "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems." CIKM 2018 | 在知识图谱上使用涟漪集传播用户偏好，实现知识增强的矩阵分解。 |
