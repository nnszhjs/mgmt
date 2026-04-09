# 上下文感知推荐模型

本目录包含上下文感知推荐模型（也称为特征交互模型）。这些模型利用丰富的上下文特征（如用户画像、物品属性、上下文字段），而不仅仅依赖简单的用户-物品ID。它们广泛应用于点击率（CTR）预测以及特征交互至关重要的场景。

## 模型列表

| 文件 | 模型 | 参考文献 | 描述 |
|------|------|----------|------|
| `afm.py` | AFM | Jun Xiao et al., "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks." IJCAI 2017 | 基于注意力的因子分解机，通过注意力机制学习特征交互的重要性。 |
| `autoint.py` | AutoInt | Weiping Song et al., "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." CIKM 2018 | 基于自注意力的CTR模型，自动显式学习高阶特征交互。 |
| `dcn.py` | DCN | Ruoxi Wang et al., "Deep & Cross Network for Ad Click Predictions." ADKDD 2017 | 深度交叉网络，结合交叉网络的显式特征交叉和深度网络。 |
| `dcnv2.py` | DCNV2 | Ruoxi Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems." WWW 2021 | 改进的深度交叉网络，将交叉权重向量扩展为矩阵，引入MoE和低秩技术。 |
| `deepfm.py` | DeepFM | Huifeng Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017 | DNN增强的FM，结合因子分解机和深度神经网络进行特征交互。 |
| `dssm.py` | DSSM | PS Huang et al., "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data." CIKM 2013 | 深度结构化语义模型，将用户和物品表示为低维向量并使用余弦相似度。 |
| `eulernet.py` | EulerNet | Zhen Tian et al., "EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction." SIGIR 2023 | 利用欧拉公式在复数向量空间中学习任意阶特征交互。 |
| `ffm.py` | FFM | Yuchin Juan et al., "Field-aware Factorization Machines for CTR Prediction." RecSys 2016 | 域感知因子分解机，每个特征对每个域有独立的潜向量。 |
| `fignn.py` | FiGNN | Li, Zekun et al., "Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction." CIKM 2019 | 将特征交互建模为图结构，应用图神经网络进行CTR预测。 |
| `fm.py` | FM | Steffen Rendle et al., "Factorization Machines." ICDM 2010 | 因子分解机，建模二阶特征交互用于预测。 |
| `fnn.py` | FNN | Weinan Zhang et al., "Deep Learning over Multi-field Categorical Data." ECIR 2016 | 基于多域特征的深度神经网络CTR预测模型（也称DNN）。 |
| `fwfm.py` | FwFM | Junwei Pan et al., "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising." WWW 2018 | 域加权FM，使用域对权重矩阵捕捉异构域交互。 |
| `kd_dagfm.py` | KD_DAGFM | Zhen Tian et al., "Directed Acyclic Graph Factorization Machines for CTR Prediction via Knowledge Distillation." WSDM 2023 | 基于有向无环图的FM，通过知识蒸馏从复杂教师网络学习任意特征交互。 |
| `lr.py` | LR | Matthew Richardson et al., "Predicting Clicks: Estimating the Click-Through Rate for New Ads." WWW 2007 | 逻辑回归CTR预测基线模型，使用加权特征求和。 |
| `nfm.py` | NFM | He X, Chua T S, "Neural Factorization Machines for Sparse Predictive Analytics." SIGIR 2017 | 神经因子分解机，用MLP替代FM交互层实现更深层的特征交互。 |
| `pnn.py` | PNN | Qu Y et al., "Product-based Neural Networks for User Response Prediction." ICDM 2016 | 基于乘积的神经网络，使用特征嵌入的内积和外积进行交互建模。 |
| `widedeep.py` | WideDeep | Heng-Tze Cheng et al., "Wide & Deep Learning for Recommender Systems." RecSys 2016 | 联合训练宽线性模型和深度神经网络，兼顾记忆能力和泛化能力。 |
| `xdeepfm.py` | xDeepFM | Jianxun Lian et al., "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems." SIGKDD 2018 | 结合显式压缩交互网络（CIN）和隐式DNN进行特征交互。 |
