# 序列推荐模型

本目录包含序列推荐和基于会话的推荐模型。这些模型利用用户交互的时间顺序来捕捉动态用户偏好，预测用户下一个可能交互的物品。它们广泛应用于用户行为随时间演变的场景。

## 模型列表

| 文件 | 模型 | 参考文献 | 描述 |
|------|------|----------|------|
| `bert4rec.py` | BERT4Rec | Fei Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." CIKM 2019 | 使用双向Transformer编码器和掩码物品预测的序列推荐模型。 |
| `caser.py` | Caser | Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding." WSDM 2018 | 基于CNN的序列模型，使用水平和垂直卷积滤波器捕捉序列模式。 |
| `core.py` | CORE | Yupeng Hou et al., "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." SIGIR 2022 | 在一致表示空间中使用Transformer编码器的会话推荐模型。 |
| `dien.py` | DIEN | Guorui Zhou et al., "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019 | 深度兴趣演化网络，建模用户兴趣的动态演化过程用于CTR预测。 |
| `din.py` | DIN | Guorui Zhou et al., "Deep Interest Network for Click-Through Rate Prediction." ACM SIGKDD 2018 | 深度兴趣网络，使用注意力机制从历史行为中自适应学习用户兴趣表示。 |
| `fdsa.py` | FDSA | Tingting Zhang et al., "Feature-level Deeper Self-Attention Network for Sequential Recommendation." IJCAI 2019 | 特征级自注意力网络，使用两个Transformer编码器分别处理物品和特征序列。 |
| `fearec.py` | FEARec | Xinyu Du et al., "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation." SIGIR 2023 | 频率增强混合注意力网络，结合时域和频域进行序列推荐。 |
| `fossil.py` | FOSSIL | Ruining He et al., "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation." ICDM 2016 | 融合物品相似度模型和高阶马尔可夫链的稀疏序列推荐模型。 |
| `fpmc.py` | FPMC | Steffen Rendle et al., "Factorizing Personalized Markov Chains for Next-Basket Recommendation." WWW 2010 | 因子化个性化马尔可夫链模型，结合矩阵分解和马尔可夫链。 |
| `gcsan.py` | GCSAN | Chengfeng Xu et al., "Graph Contextualized Self-Attention Network for Session-based Recommendation." IJCAI 2019 | 图上下文自注意力网络，结合GNN和Transformer用于会话推荐。 |
| `gru4rec.py` | GRU4Rec | Yong Kiam Tan et al., "Improved Recurrent Neural Networks for Session-based Recommendations." DLRS 2016 | 基于GRU的循环神经网络会话推荐模型。 |
| `gru4reccpr.py` | GRU4RecCPR | Tan et al. (DLRS 2016) + Chang et al., "To Copy, or not to Copy..." WSDM 2024 | GRU4Rec扩展版，加入Softmax-CPR（复制或不复制机制）改进输出层。 |
| `gru4recf.py` | GRU4RecF | Balazs Hidasi et al., "Parallel Recurrent Neural Network Architectures for Feature-rich Session-based Recommendations." RecSys 2016 | GRU4Rec的特征增强版，将物品特征嵌入与物品序列结合。 |
| `gru4reckg.py` | GRU4RecKG | - | GRU4Rec的扩展版，将物品嵌入与预训练知识图谱嵌入拼接作为输入。 |
| `hgn.py` | HGN | Chen Ma et al., "Hierarchical Gating Networks for Sequential Recommendation." SIGKDD 2019 | 层次门控网络，通过特征门控和实例门控进行序列推荐。 |
| `hrm.py` | HRM | Pengfei Wang et al., "Learning Hierarchical Representation Model for Next Basket Recommendation." SIGIR 2015 | 层次表示模型，捕捉序列行为和用户通用偏好用于下一篮推荐。 |
| `ksr.py` | KSR | Jin Huang et al., "Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks." SIGIR 2018 | 将RNN与知识库增强的键值记忆网络相结合的序列推荐模型。 |
| `lightsans.py` | LightSANs | Xin-Yan Fan et al., "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." SIGIR 2021 | 低秩分解自注意力网络，实现高效的下一物品推荐。 |
| `narm.py` | NARM | Jing Li et al., "Neural Attentive Session-based Recommendation." CIKM 2017 | 神经注意力会话推荐模型，结合RNN和注意力机制。 |
| `nextitnet.py` | NextItNet | Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation." WSDM 2019 | 膨胀卷积生成网络，使用堆叠空洞卷积进行下一物品推荐。 |
| `npe.py` | NPE | ThaiBinh Nguyen et al., "NPE: Neural Personalized Embedding for Collaborative Filtering." IJCAI 2018 | 神经个性化嵌入，从用户点击历史建模个人偏好和物品关系。 |
| `repeatnet.py` | RepeatNet | Pengjie Ren et al., "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation." AAAI 2019 | 重复感知神经推荐模型，具有重复/探索模式的会话推荐。 |
| `s3rec.py` | S3Rec | Kun Zhou, Hui Wang et al., "S^3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization." CIKM 2020 | 基于互信息最大化的自监督序列推荐模型，支持预训练。 |
| `sasrec.py` | SASRec | Wang-Cheng Kang et al., "Self-Attentive Sequential Recommendation." ICDM 2018 | 首个基于自注意力的序列推荐模型，使用单向Transformer编码器。 |
| `sasreccpr.py` | SASRecCPR | Kang et al. (ICDM 2018) + Chang et al., "To Copy, or not to Copy..." WSDM 2024 | SASRec扩展版，加入Softmax-CPR（复制或不复制机制）改进输出层。 |
| `sasrecf.py` | SASRecF | - | SASRec的扩展版，将物品表示与物品属性表示拼接作为输入。 |
| `shan.py` | SHAN | Ying, H et al., "Sequential Recommender System based on Hierarchical Attention Network." IJCAI 2018 | 层次注意力网络，融合长期和短期用户偏好进行序列推荐。 |
| `sine.py` | SINE | Qiaoyu Tan et al., "Sparse-Interest Network for Sequential Recommendation." WSDM 2021 | 稀疏兴趣网络，提取多样化用户兴趣用于序列推荐。 |
| `srgnn.py` | SRGNN | Shu Wu et al., "Session-based Recommendation with Graph Neural Networks." AAAI 2019 | 基于图神经网络的会话推荐模型，将会话序列建模为图结构。 |
| `stamp.py` | STAMP | Qiao Liu et al., "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation." KDD 2018 | 从长期会话上下文捕捉通用兴趣，从最近点击捕捉当前兴趣。 |
| `transrec.py` | TransRec | Ruining He et al., "Translation-based Recommendation." RecSys 2017 | 基于翻译的序列模型，假设嵌入空间中"上一物品 + 用户 = 下一物品"。 |
