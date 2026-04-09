# recbole.evaluator

The evaluation framework for RecBole. It provides a modular system of metrics, a collector for gathering prediction results, and an evaluator that orchestrates metric computation. Supports both ranking-based metrics (e.g., Hit, NDCG, MRR, Recall, Precision, MAP) and value-based metrics (e.g., AUC, RMSE, MAE, LogLoss).

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports all metric classes, evaluator, register, and collector components. |
| `base_metric.py` | Defines abstract base classes: `AbstractMetric` (base for all metrics), `TopkMetric` (base for top-k ranking metrics with `rec.topk` data need), and `LossMetric` (base for loss/value-based metrics with `rec.score` data need). |
| `metrics.py` | Implements concrete metric classes including `Hit`, `MRR`, `MAP`, `Recall`, `NDCG`, `Precision`, `GiniIndex`, `GAUC`, `AUC`, `MAE`, `RMSE`, `LogLoss`, `ItemCoverage`, `AveragePopularity`, `ShannonEntropy`, and `TailPercentage`. |
| `register.py` | Provides `cluster_info()` to auto-discover all metric classes via introspection, collecting their data needs, types, and smaller-is-better flags. Exports `metrics_dict`, `metric_types`, `metric_need`, and `smaller_metrics`. |
| `collector.py` | Implements `DataStruct` (a dictionary-like container for evaluation data) and `Collector` (gathers model predictions, labels, and item information during evaluation into a `DataStruct`). |
| `utils.py` | Utility functions for evaluation, including `pad_sequence` for padding score sequences and `_binary_clf_curve` for computing binary classification curves. |
