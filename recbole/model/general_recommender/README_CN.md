# 通用推荐模型

本目录包含通用推荐模型（也称为协同过滤模型）。这些模型主要依赖用户-物品交互数据（如评分、点击等），无需额外的辅助信息（如物品属性、知识图谱或序列行为）。它们是推荐系统的基础核心模型。

## 模型列表

| 文件 | 模型 | 参考文献 | 描述 |
|------|------|----------|------|
| `admmslim.py` | ADMMSLIM | Steck et al., "ADMM SLIM: Sparse Recommendations for Many Users." WSDM 2020 | 使用ADMM优化的稀疏线性推荐模型，适用于大规模Top-N推荐。 |
| `asymknn.py` | AsymKNN | - | 基于非对称余弦相似度的K近邻协同过滤模型。 |
| `bpr.py` | BPR | Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009 | 使用贝叶斯个性化排序损失训练的基础矩阵分解模型。 |
| `cdae.py` | CDAE | Yao Wu et al., "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems." WSDM 2016 | 协同去噪自编码器，利用用户特定的去噪方式进行Top-N推荐。 |
| `convncf.py` | ConvNCF | Xiangnan He et al., "Outer Product-based Neural Collaborative Filtering." IJCAI 2018 | 使用外积和CNN捕捉交互模式的神经协同过滤模型。 |
| `dgcf.py` | DGCF | Xiang Wang et al., "Disentangled Graph Collaborative Filtering." SIGIR 2020 | 解耦图协同过滤模型，在交互图上学习解耦的用户/物品表示。 |
| `diffrec.py` | DiffRec | Wenjie Wang et al., "Diffusion Recommender Model." SIGIR 2023 | 基于扩散过程的生成式推荐模型。 |
| `dmf.py` | DMF | Hong-Jian Xue et al., "Deep Matrix Factorization Models for Recommender Systems." IJCAI 2017 | 以原始交互矩阵为输入的深度神经网络增强矩阵分解模型。 |
| `ease.py` | EASE | Harald Steck, "Embarrassingly Shallow Autoencoders for Sparse Data." WWW 2019 | 结合自编码器和近邻方法优势的简单线性协同过滤模型。 |
| `enmf.py` | ENMF | Chong Chen et al., "Efficient Neural Matrix Factorization without Sampling for Recommendation." TOIS 2020 | 高效的无采样神经矩阵分解通用推荐模型。 |
| `fism.py` | FISM | S. Kabbur et al., "FISM: Factored Item Similarity Models for Top-N Recommender Systems." KDD 2013 | 基于物品的模型，将物品-物品相似度矩阵分解为两个低维潜因子矩阵的乘积。 |
| `gcmc.py` | GCMC | van den Berg et al., "Graph Convolutional Matrix Completion." SIGKDD 2018 | 在二部用户-物品图上使用图卷积进行矩阵补全的模型。 |
| `itemknn.py` | ItemKNN | Aiolli, F et al., "Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets." RecSys 2013 | 基于余弦相似度的物品K近邻协同过滤模型。 |
| `ldiffrec.py` | LDiffRec | Wenjie Wang et al., "Diffusion Recommender Model." SIGIR 2023 | DiffRec的潜空间变体，结合VAE降维实现大规模扩散推荐。 |
| `lightgcn.py` | LightGCN | Xiangnan He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020 | 简化的图卷积网络，通过在用户-物品图上线性传播学习嵌入表示。 |
| `line.py` | LINE | Jian Tang et al., "LINE: Large-scale Information Network Embedding." WWW 2015 | 保留一阶和二阶近邻关系的大规模网络嵌入模型。 |
| `macridvae.py` | MacridVAE | Jianxin Ma et al., "Learning Disentangled Representations for Recommendation." NeurIPS 2019 | 宏观解耦变分自编码器，学习解耦表示用于推荐。 |
| `mecodgnn.py` | MECoDGNN | "Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach" | 动态图神经网络框架，通过阶段度数和基尼系数正则化控制马太效应，实现推荐多样化。 |
| `multidae.py` | MultiDAE | Dawen Liang et al., "Variational Autoencoders for Collaborative Filtering." WWW 2018 | 去噪自编码器，同时对每个用户的所有物品进行排序的协同过滤模型。 |
| `multivae.py` | MultiVAE | Dawen Liang et al., "Variational Autoencoders for Collaborative Filtering." WWW 2018 | 变分自编码器，同时对每个用户的所有物品进行排序的协同过滤模型。 |
| `nais.py` | NAIS | Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." TKDE 2018 | 神经注意力物品相似度模型，使用注意力机制区分物品重要性。 |
| `nceplrec.py` | NCEPLRec | Ga Wu et al., "Noise Contrastive Estimation for One-Class Collaborative Filtering." SIGIR 2019 | 使用噪声对比估计训练的投影线性推荐模型，用于单类协同过滤。 |
| `ncl.py` | NCL | Zihan Lin et al., "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." WWW 2022 | 通过结构和语义邻居增强的对比学习改进图协同过滤。 |
| `neumf.py` | NeuMF | Xiangnan He et al., "Neural Collaborative Filtering." WWW 2017 | 用MLP替代点积的神经矩阵分解，实现更精确的用户-物品交互建模。 |
| `ngcf.py` | NGCF | Xiang Wang et al., "Neural Graph Collaborative Filtering." SIGIR 2019 | 通过用户-物品图上的嵌入传播显式编码协同信号的神经图协同过滤。 |
| `nncf.py` | NNCF | Ting Bai et al., "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." CIKM 2017 | 结合交互邻域信息增强的神经协同过滤模型。 |
| `pop.py` | Pop | - | 始终推荐最热门物品的基线模型。 |
| `ract.py` | RaCT | Sam Lobel et al., "RaCT: Towards Amortized Ranking-Critical Training for Collaborative Filtering." ICLR 2020 | 基于演员-评论家强化学习的协同过滤模型，采用排序关键训练策略。 |
| `random.py` | Random | - | 随机推荐物品的基线模型。 |
| `recvae.py` | RecVAE | Shenbin, Ilya et al., "RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback." WSDM 2020 | 改进的变分自编码器，采用复合先验和交替训练进行Top-N推荐。 |
| `sgl.py` | SGL | Jiancan Wu et al., "SGL: Self-supervised Graph Learning for Recommendation." SIGIR 2021 | 通过图数据增强的自监督对比学习增强的GCN推荐模型。 |
| `simplex.py` | SimpleX | Kelong Mao et al., "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." CIKM 2021 | 简单而强大的协同过滤基线，使用大量负采样和余弦对比损失。 |
| `slimelastic.py` | SLIMElastic | Xia Ning et al., "SLIM: Sparse Linear Methods for Top-N Recommender Systems." ICDM 2011 | 使用L1+L2正则化学习聚合系数矩阵的稀疏线性Top-N推荐方法。 |
| `spectralcf.py` | SpectralCF | Lei Zheng et al., "Spectral Collaborative Filtering." RecSys 2018 | 从频谱域学习潜因子进行推荐的谱卷积模型。 |
