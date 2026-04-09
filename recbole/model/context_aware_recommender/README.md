# Context-Aware Recommender Models

This directory contains context-aware recommendation models (also known as feature-interaction models). These models leverage rich contextual features (e.g., user profiles, item attributes, contextual fields) beyond simple user-item IDs. They are widely used in click-through rate (CTR) prediction and scenarios where feature interactions are critical.

## Model List

| File | Model | Reference | Description |
|------|-------|-----------|-------------|
| `afm.py` | AFM | Jun Xiao et al., "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks." IJCAI 2017 | Attention-based factorization machine that learns the importance of feature interactions via attention. |
| `autoint.py` | AutoInt | Weiping Song et al., "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." CIKM 2018 | Self-attention based CTR model that automatically learns high-order feature interactions explicitly. |
| `dcn.py` | DCN | Ruoxi Wang et al., "Deep & Cross Network for Ad Click Predictions." ADKDD 2017 | Deep and cross network combining a cross network for explicit feature crossing with a deep network. |
| `dcnv2.py` | DCNV2 | Ruoxi Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems." WWW 2021 | Improved deep and cross network extending the cross weight vector to a matrix with MoE and low-rank techniques. |
| `deepfm.py` | DeepFM | Huifeng Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017 | DNN-enhanced FM combining a factorization machine and deep neural network for feature interaction. |
| `dssm.py` | DSSM | PS Huang et al., "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data." CIKM 2013 | Deep structured semantic model representing user and item as low-dimensional vectors with cosine similarity. |
| `eulernet.py` | EulerNet | Zhen Tian et al., "EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction." SIGIR 2023 | Learns arbitrary-order feature interactions in complex vector space using Euler's formula. |
| `ffm.py` | FFM | Yuchin Juan et al., "Field-aware Factorization Machines for CTR Prediction." RecSys 2016 | Field-aware factorization machine where each feature has multiple latent vectors, one for each field. |
| `fignn.py` | FiGNN | Li, Zekun et al., "Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction." CIKM 2019 | Models feature interactions as a graph and applies graph neural networks for CTR prediction. |
| `fm.py` | FM | Steffen Rendle et al., "Factorization Machines." ICDM 2010 | Factorization machine modeling second-order feature interactions for prediction. |
| `fnn.py` | FNN | Weinan Zhang et al., "Deep Learning over Multi-field Categorical Data." ECIR 2016 | Deep neural network over field features for CTR prediction (also known as DNN). |
| `fwfm.py` | FwFM | Junwei Pan et al., "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising." WWW 2018 | Field-weighted FM using a field pair weight matrix to capture heterogeneous field interactions. |
| `kd_dagfm.py` | KD_DAGFM | Zhen Tian et al., "Directed Acyclic Graph Factorization Machines for CTR Prediction via Knowledge Distillation." WSDM 2023 | DAG-based FM with knowledge distillation learning arbitrary feature interactions from complex teacher networks. |
| `lr.py` | LR | Matthew Richardson et al., "Predicting Clicks: Estimating the Click-Through Rate for New Ads." WWW 2007 | Logistic regression baseline for CTR prediction using weighted feature sums. |
| `nfm.py` | NFM | He X, Chua T S, "Neural Factorization Machines for Sparse Predictive Analytics." SIGIR 2017 | Neural factorization machine replacing the FM interaction layer with MLP for deeper feature interaction. |
| `pnn.py` | PNN | Qu Y et al., "Product-based Neural Networks for User Response Prediction." ICDM 2016 | Product-based neural network using inner and outer products of feature embeddings for interaction modeling. |
| `widedeep.py` | WideDeep | Heng-Tze Cheng et al., "Wide & Deep Learning for Recommender Systems." RecSys 2016 | Jointly trains wide linear models and deep neural networks for memorization and generalization. |
| `xdeepfm.py` | xDeepFM | Jianxun Lian et al., "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems." SIGKDD 2018 | Combines explicit compressed interaction network (CIN) with implicit DNN for feature interactions. |
