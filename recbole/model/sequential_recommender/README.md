# Sequential Recommender Models

This directory contains sequential and session-based recommendation models. These models leverage the temporal order of user interactions to capture dynamic user preferences and predict the next item a user is likely to interact with. They are widely used in scenarios where user behavior evolves over time.

## Model List

| File | Model | Reference | Description |
|------|-------|-----------|-------------|
| `bert4rec.py` | BERT4Rec | Fei Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." CIKM 2019 | Bidirectional Transformer encoder for sequential recommendation using masked item prediction. |
| `caser.py` | Caser | Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding." WSDM 2018 | CNN-based sequential model using horizontal and vertical convolutional filters to capture sequential patterns. |
| `core.py` | CORE | Yupeng Hou et al., "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." SIGIR 2022 | Session-based recommendation model operating within a consistent representation space using Transformer encoders. |
| `dien.py` | DIEN | Guorui Zhou et al., "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019 | Deep interest evolution network that models the evolution of user interests for CTR prediction. |
| `din.py` | DIN | Guorui Zhou et al., "Deep Interest Network for Click-Through Rate Prediction." ACM SIGKDD 2018 | Deep interest network using attention to adaptively learn user interest representations from historical behaviors. |
| `fdsa.py` | FDSA | Tingting Zhang et al., "Feature-level Deeper Self-Attention Network for Sequential Recommendation." IJCAI 2019 | Feature-level self-attention network using two Transformer encoders for item and feature sequences. |
| `fearec.py` | FEARec | Xinyu Du et al., "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation." SIGIR 2023 | Frequency enhanced hybrid attention network combining time and frequency domain for sequential recommendation. |
| `fossil.py` | FOSSIL | Ruining He et al., "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation." ICDM 2016 | Fuses item similarity models with high-order Markov chains for sparse sequential recommendation. |
| `fpmc.py` | FPMC | Steffen Rendle et al., "Factorizing Personalized Markov Chains for Next-Basket Recommendation." WWW 2010 | Factorized personalized Markov chain model combining matrix factorization and Markov chains. |
| `gcsan.py` | GCSAN | Chengfeng Xu et al., "Graph Contextualized Self-Attention Network for Session-based Recommendation." IJCAI 2019 | Graph contextualized self-attention network combining GNN and Transformer for session-based recommendation. |
| `gru4rec.py` | GRU4Rec | Yong Kiam Tan et al., "Improved Recurrent Neural Networks for Session-based Recommendations." DLRS 2016 | GRU-based recurrent neural network for session-based recommendation. |
| `gru4reccpr.py` | GRU4RecCPR | Tan et al. (DLRS 2016) + Chang et al., "To Copy, or not to Copy..." WSDM 2024 | GRU4Rec extended with Softmax-CPR (Copy-or-not mechanism) for improved output layer. |
| `gru4recf.py` | GRU4RecF | Balazs Hidasi et al., "Parallel Recurrent Neural Network Architectures for Feature-rich Session-based Recommendations." RecSys 2016 | Feature-rich extension of GRU4Rec incorporating item feature embeddings alongside item sequences. |
| `gru4reckg.py` | GRU4RecKG | - | Extension of GRU4Rec that concatenates item embeddings with pre-trained knowledge graph embeddings as input. |
| `hgn.py` | HGN | Chen Ma et al., "Hierarchical Gating Networks for Sequential Recommendation." SIGKDD 2019 | Hierarchical gating network with feature gating and instance gating for sequential recommendation. |
| `hrm.py` | HRM | Pengfei Wang et al., "Learning Hierarchical Representation Model for Next Basket Recommendation." SIGIR 2015 | Hierarchical representation model capturing sequential behavior and general user taste for next-basket recommendation. |
| `ksr.py` | KSR | Jin Huang et al., "Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks." SIGIR 2018 | Integrates RNN with key-value memory networks enhanced by knowledge base information. |
| `lightsans.py` | LightSANs | Xin-Yan Fan et al., "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." SIGIR 2021 | Low-rank decomposed self-attention network for efficient next-item recommendation. |
| `narm.py` | NARM | Jing Li et al., "Neural Attentive Session-based Recommendation." CIKM 2017 | Neural attentive session-based recommendation model combining RNN with attention mechanism. |
| `nextitnet.py` | NextItNet | Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation." WSDM 2019 | Dilated convolutional generative network using stacked holed convolutions for next-item recommendation. |
| `npe.py` | NPE | ThaiBinh Nguyen et al., "NPE: Neural Personalized Embedding for Collaborative Filtering." IJCAI 2018 | Neural personalized embedding modeling personal preference and item relationships from user click history. |
| `repeatnet.py` | RepeatNet | Pengjie Ren et al., "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation." AAAI 2019 | Repeat-aware neural recommendation model for session-based recommendation with repeat/explore modes. |
| `s3rec.py` | S3Rec | Kun Zhou, Hui Wang et al., "S^3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization." CIKM 2020 | Self-supervised sequential recommendation using mutual information maximization with pre-training. |
| `sasrec.py` | SASRec | Wang-Cheng Kang et al., "Self-Attentive Sequential Recommendation." ICDM 2018 | First self-attention based sequential recommender using unidirectional Transformer encoder. |
| `sasreccpr.py` | SASRecCPR | Kang et al. (ICDM 2018) + Chang et al., "To Copy, or not to Copy..." WSDM 2024 | SASRec extended with Softmax-CPR (Copy-or-not mechanism) for improved output layer. |
| `sasrecf.py` | SASRecF | - | Extension of SASRec that concatenates item representations with item attribute representations as input. |
| `shan.py` | SHAN | Ying, H et al., "Sequential Recommender System based on Hierarchical Attention Network." IJCAI 2018 | Hierarchical attention network fusing long-term and short-term user preferences for sequential recommendation. |
| `sine.py` | SINE | Qiaoyu Tan et al., "Sparse-Interest Network for Sequential Recommendation." WSDM 2021 | Sparse-interest network that extracts diverse user interests for sequential recommendation. |
| `srgnn.py` | SRGNN | Shu Wu et al., "Session-based Recommendation with Graph Neural Networks." AAAI 2019 | Graph neural network for session-based recommendation modeling session sequences as graphs. |
| `stamp.py` | STAMP | Qiao Liu et al., "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation." KDD 2018 | Captures general interests from long-term session context and current interests from last clicks. |
| `transrec.py` | TransRec | Ruining He et al., "Translation-based Recommendation." RecSys 2017 | Translation-based sequential model assuming prev_item + user = next_item in embedding space. |
