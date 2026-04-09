# Knowledge-Aware Recommender Models

This directory contains knowledge-aware recommendation models. These models incorporate knowledge graph (KG) information to enrich item and user representations, capturing semantic relationships between entities and improving recommendation quality beyond what pure collaborative filtering can achieve.

## Model List

| File | Model | Reference | Description |
|------|-------|-----------|-------------|
| `cfkg.py` | CFKG | Qingyao Ai et al., "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation." MDPI 2018 | Combines knowledge graph and user-item interaction graph into a unified graph where users, items, and attributes are entities. |
| `cke.py` | CKE | Fuzheng Zhang et al., "Collaborative Knowledge Base Embedding for Recommender Systems." SIGKDD 2016 | Incorporates KG structural knowledge to enrich item representations for collaborative filtering. |
| `kgat.py` | KGAT | Xiang Wang et al., "KGAT: Knowledge Graph Attention Network for Recommendation." SIGKDD 2019 | Knowledge graph attention network using attentive embedding propagation on the KG-enhanced user-item graph. |
| `kgcn.py` | KGCN | Hongwei Wang et al., "Knowledge Graph Convolution Networks for Recommender Systems." WWW 2019 | Knowledge graph convolution network capturing inter-item relatedness by mining associated attributes on the KG. |
| `kgin.py` | KGIN | Xiang Wang et al., "Learning Intents behind Interactions with Knowledge Graph for Recommendation." WWW 2021 | Learns user intents behind interactions using knowledge graph information for intent-aware recommendation. |
| `kgnnls.py` | KGNNLS | Hongwei Wang et al., "Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems." KDD 2019 | Knowledge-aware GNN with label smoothness regularization for improved recommendation. |
| `ktup.py` | KTUP | Yixin Cao et al., "Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences." WWW 2019 | Multi-task learning model jointly learning recommendation and KG tasks with attention-based preference aggregation. |
| `mcclk.py` | MCCLK | Ding Zou et al., "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." SIGIR 2022 | Multi-level cross-view contrastive learning framework for knowledge-aware recommendation. |
| `mkr.py` | MKR | Hongwei Wang et al., "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation." WWW 2019 | Multi-task framework using cross-and-compress units to share latent features between recommendation and KG embedding tasks. |
| `ripplenet.py` | RippleNet | Hongwei Wang et al., "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems." CIKM 2018 | Propagates user preferences on the knowledge graph using ripple sets for knowledge-enhanced matrix factorization. |
