# recbole.model

The model package for RecBole, containing abstract base classes, common neural network layers, loss functions, weight initialization utilities, and implementations of 90+ recommendation models organized by paradigm.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer. |
| `abstract_recommender.py` | Abstract base classes for all models: `AbstractRecommender` (base), `GeneralRecommender` (general CF models with user/item embeddings), `SequentialRecommender` (sequential models with item sequence handling), `KnowledgeRecommender` (knowledge-aware models with entity/relation embeddings), `ContextRecommender` (context-aware models with feature embeddings), and `AutoEncoderMixin` (mixin for autoencoder-based models). |
| `layers.py` | Common neural network layers used across models, including `MLPLayers`, `FMEmbedding`, `FLEmbedding`, `BaseFactorizationMachine`, `BiGNNLayer`, `AttLayer`, `Dice`, `SequenceAttLayer`, `VanillaAttention`, `MultiHeadAttention`, `TransformerEncoder`, `LightTransformerEncoder`, `CNNLayers`, `FMFirstOrderLinear`, `ContextSeqEmbLayer`, and `SparseDropout`. |
| `loss.py` | Common loss functions: `BPRLoss` (Bayesian Personalized Ranking loss), `RegLoss` (L2 regularization), `EmbLoss` (embedding L2 regularization), and `EmbMarginLoss` (embedding margin loss). |
| `init.py` | Weight initialization functions: `xavier_normal_initialization` and `xavier_uniform_initialization` for `nn.Embedding` and `nn.Linear` layers. |

## Sub-directories

| Directory | Description |
|-----------|-------------|
| `general_recommender/` | General collaborative filtering models (e.g., BPR, CDAE, ConvNCF, DGCF, DMF, DSSM, LightGCN, NeuMF, NGCF, etc.). |
| `sequential_recommender/` | Sequential recommendation models (e.g., BERT4Rec, Caser, GRU4Rec, SASRec, S3Rec, CORE, DIEN, DIN, etc.). |
| `context_aware_recommender/` | Context-aware recommendation models (e.g., AFM, AutoInt, DCN, DCNv2, DeepFM, FM, FFM, WideDeep, xDeepFM, etc.). |
| `knowledge_aware_recommender/` | Knowledge-aware recommendation models (e.g., CKE, CFKG, KGAT, KGCN, KGIN, KGNN-LS, KTUP, MKR, RippleNet, etc.). |
| `exlib_recommender/` | External library-based models wrapping third-party implementations (e.g., `LightGBM`, `XGBoost`). |
