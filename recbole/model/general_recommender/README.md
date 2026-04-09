# General Recommender Models

This directory contains general recommendation models (also known as collaborative filtering models). These models rely primarily on user-item interaction data (e.g., ratings, clicks) without requiring side information such as item attributes, knowledge graphs, or sequential behavior. They form the foundational building blocks of recommender systems.

## Model List

| File | Model | Reference | Description |
|------|-------|-----------|-------------|
| `admmslim.py` | ADMMSLIM | Steck et al., "ADMM SLIM: Sparse Recommendations for Many Users." WSDM 2020 | Sparse linear recommendation model using ADMM optimization for scalable top-N recommendation. |
| `asymknn.py` | AsymKNN | - | Asymmetric cosine similarity-based k-nearest neighbor model for collaborative filtering. |
| `bpr.py` | BPR | Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009 | Basic matrix factorization model trained with pairwise Bayesian personalized ranking loss. |
| `cdae.py` | CDAE | Yao Wu et al., "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems." WSDM 2016 | Collaborative denoising auto-encoder for top-N recommendation using user-specific denoising. |
| `convncf.py` | ConvNCF | Xiangnan He et al., "Outer Product-based Neural Collaborative Filtering." IJCAI 2018 | Neural collaborative filtering model that uses outer product and CNN to capture interaction patterns. |
| `dgcf.py` | DGCF | Xiang Wang et al., "Disentangled Graph Collaborative Filtering." SIGIR 2020 | Disentangled graph collaborative filtering that learns disentangled user/item representations on the interaction graph. |
| `diffrec.py` | DiffRec | Wenjie Wang et al., "Diffusion Recommender Model." SIGIR 2023 | Diffusion-based generative recommender model that applies diffusion processes for recommendation. |
| `dmf.py` | DMF | Hong-Jian Xue et al., "Deep Matrix Factorization Models for Recommender Systems." IJCAI 2017 | Deep neural network enhanced matrix factorization using the original interaction matrix as input. |
| `ease.py` | EASE | Harald Steck, "Embarrassingly Shallow Autoencoders for Sparse Data." WWW 2019 | Simple linear model combining strengths of auto-encoders and neighborhood-based approaches. |
| `enmf.py` | ENMF | Chong Chen et al., "Efficient Neural Matrix Factorization without Sampling for Recommendation." TOIS 2020 | Efficient non-sampling neural matrix factorization for general recommendation. |
| `fism.py` | FISM | S. Kabbur et al., "FISM: Factored Item Similarity Models for Top-N Recommender Systems." KDD 2013 | Item-based model learning item-item similarity as the product of two low-dimensional latent factor matrices. |
| `gcmc.py` | GCMC | van den Berg et al., "Graph Convolutional Matrix Completion." SIGKDD 2018 | Graph convolutional matrix completion model using graph convolutions on the bipartite user-item graph. |
| `itemknn.py` | ItemKNN | Aiolli, F et al., "Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets." RecSys 2013 | Item-based k-nearest neighbor collaborative filtering using cosine similarity. |
| `ldiffrec.py` | LDiffRec | Wenjie Wang et al., "Diffusion Recommender Model." SIGIR 2023 | Latent space variant of DiffRec with VAE-based dimensionality reduction for large-scale diffusion recommendation. |
| `lightgcn.py` | LightGCN | Xiangnan He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020 | Simplified graph convolution network that learns embeddings by linearly propagating on the user-item graph. |
| `line.py` | LINE | Jian Tang et al., "LINE: Large-scale Information Network Embedding." WWW 2015 | Large-scale network embedding model preserving first-order and second-order proximity. |
| `macridvae.py` | MacridVAE | Jianxin Ma et al., "Learning Disentangled Representations for Recommendation." NeurIPS 2019 | Macro-disentangled variational auto-encoder that learns disentangled representations for recommendation. |
| `mecodgnn.py` | MECoDGNN | "Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach" | Dynamic GNN framework with Matthew Effect control for diversifying recommendations via stage degree and Gini regularization. |
| `multidae.py` | MultiDAE | Dawen Liang et al., "Variational Autoencoders for Collaborative Filtering." WWW 2018 | Denoising auto-encoder that simultaneously ranks all items for each user in collaborative filtering. |
| `multivae.py` | MultiVAE | Dawen Liang et al., "Variational Autoencoders for Collaborative Filtering." WWW 2018 | Variational auto-encoder that simultaneously ranks all items for each user in collaborative filtering. |
| `nais.py` | NAIS | Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." TKDE 2018 | Neural attentive item similarity model that distinguishes item importance using attention mechanisms. |
| `nceplrec.py` | NCEPLRec | Ga Wu et al., "Noise Contrastive Estimation for One-Class Collaborative Filtering." SIGIR 2019 | Projected linear recommendation model trained with noise contrastive estimation for one-class CF. |
| `ncl.py` | NCL | Zihan Lin et al., "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." WWW 2022 | Graph collaborative filtering enhanced with neighborhood-enriched contrastive learning using structural and semantic neighbors. |
| `neumf.py` | NeuMF | Xiangnan He et al., "Neural Collaborative Filtering." WWW 2017 | Neural matrix factorization replacing dot product with MLP for more precise user-item interaction modeling. |
| `ngcf.py` | NGCF | Xiang Wang et al., "Neural Graph Collaborative Filtering." SIGIR 2019 | Neural graph collaborative filtering that explicitly encodes collaborative signal via embedding propagation on the user-item graph. |
| `nncf.py` | NNCF | Ting Bai et al., "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." CIKM 2017 | Neural collaborative filtering enhanced with interaction-based neighborhood information. |
| `pop.py` | Pop | - | Baseline model that always recommends the most popular items. |
| `ract.py` | RaCT | Sam Lobel et al., "RaCT: Towards Amortized Ranking-Critical Training for Collaborative Filtering." ICLR 2020 | Actor-critic reinforcement learning based collaborative filtering model with ranking-critical training. |
| `random.py` | Random | - | Baseline model that recommends items randomly. |
| `recvae.py` | RecVAE | Shenbin, Ilya et al., "RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback." WSDM 2020 | Improved variational auto-encoder for top-N recommendation with composite prior and alternating training. |
| `sgl.py` | SGL | Jiancan Wu et al., "SGL: Self-supervised Graph Learning for Recommendation." SIGIR 2021 | GCN-based recommender supplemented with self-supervised contrastive learning via graph augmentations. |
| `simplex.py` | SimpleX | Kelong Mao et al., "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." CIKM 2021 | Simple yet strong collaborative filtering baseline using large negative sampling with cosine contrastive loss. |
| `slimelastic.py` | SLIMElastic | Xia Ning et al., "SLIM: Sparse Linear Methods for Top-N Recommender Systems." ICDM 2011 | Sparse linear method learning an aggregation coefficient matrix with L1+L2 regularization for top-N recommendation. |
| `spectralcf.py` | SpectralCF | Lei Zheng et al., "Spectral Collaborative Filtering." RecSys 2018 | Spectral convolution model learning latent factors from the spectral domain for recommendation. |
