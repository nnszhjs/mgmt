# recbole.data.dataset

Dataset classes for RecBole that store and preprocess the original dataset in memory. Each class provides specialized functionality for different recommendation paradigms, including data filtering, feature engineering, and data augmentation.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports `Dataset`, `SequentialDataset`, `KnowledgeBasedDataset`, `KGSeqDataset`, `DecisionTreeDataset`, and customized datasets. |
| `dataset.py` | Implements the base `Dataset` class (extends `torch.utils.data.Dataset`) for general and context-aware models. Provides k-core filtering, missing value imputation, feature storage as DataFrames, and automatic dataset download. |
| `sequential_dataset.py` | Implements `SequentialDataset` which extends `Dataset` with data augmentation for sequential recommendation, managing historical item lists and sequence length fields. |
| `kg_dataset.py` | Implements `KnowledgeBasedDataset` which extends `Dataset` to load `.kg` and `.link` files, remap entities with item IDs, and convert KG features to sparse matrices or graph structures (DGL/PyG). |
| `kg_seq_dataset.py` | Implements `KGSeqDataset` combining both `SequentialDataset` and `KnowledgeBasedDataset` via multiple inheritance for models that need both sequential and knowledge graph processing. |
| `decisiontree_dataset.py` | Implements `DecisionTreeDataset` which extends `Dataset` with token-to-numeric conversion for decision tree models (e.g., XGBoost, LightGBM). |
| `customized_dataset.py` | Model-specific dataset classes (e.g., `GRU4RecKGDataset`, `KSRDataset`, `DIENDataset`) that are automatically loaded by naming convention `[ModelName]Dataset`. |
