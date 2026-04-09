# 外部库推荐模型

本目录包含外部机器学习库的封装模型。这些模型为流行的梯度提升框架提供RecBole兼容接口，使其能够在RecBole流水线中用于推荐任务（如CTR预测）。

## 模型列表

| 文件 | 模型 | 参考文献 | 描述 |
|------|------|----------|------|
| `lightgbm.py` | LightGBM | Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017 | LightGBM梯度提升框架的RecBole封装，继承自 `lgb.Booster`。支持基于决策树的逐点输入推荐。 |
| `xgboost.py` | XGBoost | Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System." KDD 2016 | XGBoost梯度提升框架的RecBole封装，继承自 `xgb.Booster`。支持基于决策树的逐点输入推荐。 |
