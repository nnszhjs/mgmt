# recbole.data

The data processing package for RecBole. It handles dataset loading, interaction representation, data transformation, and provides dataloaders for feeding batches to models during training and evaluation.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports `create_dataset`, `data_preparation`, `save_split_dataloaders`, and `load_split_dataloaders`. |
| `utils.py` | Utility functions for dataset creation (auto-selecting dataset class by model type), data preparation (splitting and building dataloaders), and serialization of dataloaders. |
| `transform.py` | Batch data transformation classes including `MaskItemSequence`, `InverseItemSequence`, `CropItemSequence`, `ReorderItemSequence`, and `UserDefinedTransform` for data augmentation during training. |
| `interaction.py` | Defines the `Interaction` class, the core data structure representing a batch of interaction records (user-item pairs), with support for tensor operations, device transfer, and concatenation. |

## Sub-directories

| Directory | Description |
|-----------|-------------|
| `dataset/` | Dataset classes for different recommendation paradigms (general, sequential, knowledge-based, decision tree). |
| `dataloader/` | DataLoader classes for training and evaluation with support for negative sampling and knowledge graph data. |
