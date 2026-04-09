# recbole.data.dataloader

DataLoader classes for RecBole that handle batching, negative sampling, and data iteration during training and evaluation. All dataloaders extend PyTorch's `torch.utils.data.DataLoader` and support distributed training.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports all dataloader classes from the sub-modules. |
| `abstract_dataloader.py` | Defines `AbstractDataLoader` (base class for all dataloaders with batch size management, shuffling, and data transformation) and `NegSampleDataLoader` (adds negative sampling support for pointwise and pairwise training). |
| `general_dataloader.py` | Implements `TrainDataLoader` (training dataloader with negative sampling), `NegSampleEvalDataLoader` (evaluation with sampled negative items), and `FullSortEvalDataLoader` (full-ranking evaluation over all items). |
| `knowledge_dataloader.py` | Implements `KGDataLoader` (loads knowledge graph triplets with negative tail entities) and `KnowledgeBasedDataLoader` (alternates between KG triplet loading and user-item interaction loading). |
| `user_dataloader.py` | Implements `UserDataLoader` which iterates over user IDs in batches, used for user-level prediction tasks. |
