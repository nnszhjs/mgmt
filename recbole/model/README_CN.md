# recbole.model

RecBole 的模型包，包含抽象基类、通用神经网络层、损失函数、权重初始化工具，以及按推荐范式组织的 90 多个推荐模型实现。

## 文件

| 文件 | 描述 |
|------|------|
| `__init__.py` | 包初始化文件。 |
| `abstract_recommender.py` | 所有模型的抽象基类：`AbstractRecommender`（基类）、`GeneralRecommender`（带用户/物品嵌入的通用协同过滤模型）、`SequentialRecommender`（处理物品序列的序列模型）、`KnowledgeRecommender`（带实体/关系嵌入的知识感知模型）、`ContextRecommender`（带特征嵌入的上下文感知模型）和 `AutoEncoderMixin`（自编码器模型的混入类）。 |
| `layers.py` | 跨模型共用的通用神经网络层，包括 `MLPLayers`、`FMEmbedding`、`FLEmbedding`、`BaseFactorizationMachine`、`BiGNNLayer`、`AttLayer`、`Dice`、`SequenceAttLayer`、`VanillaAttention`、`MultiHeadAttention`、`TransformerEncoder`、`LightTransformerEncoder`、`CNNLayers`、`FMFirstOrderLinear`、`ContextSeqEmbLayer` 和 `SparseDropout`。 |
| `loss.py` | 通用损失函数：`BPRLoss`（贝叶斯个性化排名损失）、`RegLoss`（L2 正则化）、`EmbLoss`（嵌入 L2 正则化）和 `EmbMarginLoss`（嵌入间距损失）。 |
| `init.py` | 权重初始化函数：`xavier_normal_initialization` 和 `xavier_uniform_initialization`，适用于 `nn.Embedding` 和 `nn.Linear` 层。 |

## 子目录

| 目录 | 描述 |
|------|------|
| `general_recommender/` | 通用协同过滤模型（如 BPR、CDAE、ConvNCF、DGCF、DMF、DSSM、LightGCN、NeuMF、NGCF 等）。 |
| `sequential_recommender/` | 序列推荐模型（如 BERT4Rec、Caser、GRU4Rec、SASRec、S3Rec、CORE、DIEN、DIN 等）。 |
| `context_aware_recommender/` | 上下文感知推荐模型（如 AFM、AutoInt、DCN、DCNv2、DeepFM、FM、FFM、WideDeep、xDeepFM 等）。 |
| `knowledge_aware_recommender/` | 知识感知推荐模型（如 CKE、CFKG、KGAT、KGCN、KGIN、KGNN-LS、KTUP、MKR、RippleNet 等）。 |
| `exlib_recommender/` | 基于外部库的模型，封装第三方实现（如 `LightGBM`、`XGBoost`）。 |
