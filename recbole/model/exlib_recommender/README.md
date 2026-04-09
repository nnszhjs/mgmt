# External Library Recommender Models

This directory contains wrapper models for external machine learning libraries. These models provide a RecBole-compatible interface around popular gradient boosting frameworks, enabling their use within the RecBole pipeline for recommendation tasks such as CTR prediction.

## Model List

| File | Model | Reference | Description |
|------|-------|-----------|-------------|
| `lightgbm.py` | LightGBM | Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017 | RecBole wrapper for the LightGBM gradient boosting framework, inherited from `lgb.Booster`. Supports decision tree-based recommendation with pointwise input. |
| `xgboost.py` | XGBoost | Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System." KDD 2016 | RecBole wrapper for the XGBoost gradient boosting framework, inherited from `xgb.Booster`. Supports decision tree-based recommendation with pointwise input. |
