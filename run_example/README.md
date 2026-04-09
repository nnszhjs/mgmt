# Run Example

This directory contains example scripts and Jupyter notebooks that demonstrate how to use various features of RecBole. These examples serve as practical guides for users to get started quickly.

## Files

| File | Description |
|---|---|
| `case_study_example.py` | Demonstrates how to perform case studies in RecBole, including loading a saved model, performing full-sort top-K prediction for specific users, and retrieving item scores using `full_sort_topk` and `full_sort_scores` utilities |
| `save_and_load_example.py` | Demonstrates how to save and load models, datasets, and dataloaders in RecBole. Shows saving checkpoints during training and reloading them with `load_data_and_model`, supporting model files, dataset files, and dataloader files |
| `session_based_rec_example.py` | Demonstrates how to run session-based recommendation benchmarks (e.g., Diginetica, Tmall, NowPlaying) with RecBole, including custom configuration, dataset splitting, model initialization, and training/evaluation workflow |
| `lstm-model-with-item-infor-fix-missing-last-item.ipynb` | Jupyter notebook demonstrating an LSTM-based recommendation model with item information, addressing the missing-last-item issue |
| `recbole-using-all-items-for-prediction.ipynb` | Jupyter notebook demonstrating how to use all items for prediction in RecBole |
| `sequential-model-fixed-missing-last-item.ipynb` | Jupyter notebook demonstrating a sequential recommendation model with a fix for the missing-last-item problem |
