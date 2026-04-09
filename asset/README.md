# Asset

This directory contains static assets used by the RecBole project, including logos, images, configuration files for hyperparameter tuning, and benchmark timing results.

## Files

| File / Directory | Description |
|---|---|
| `logo.png` | RecBole project logo image |
| `framework.png` | RecBole framework architecture diagram |
| `new.gif` | Animated "new" badge icon |
| `dataset_list.json` | JSON file listing all supported datasets and their metadata |
| `model_list.json` | JSON file listing all supported recommendation models and their metadata |
| `questionnaire.xlsx` | Questionnaire spreadsheet for user feedback or survey purposes |
| `hyper_tune_configs/` | Hyperparameter tuning configuration files organized by recommendation type |
| `time_test_result/` | Benchmark timing results for different recommendation model categories |

## Subdirectories

### `hyper_tune_configs/`

Contains markdown files with optimal hyperparameter configurations for various datasets and recommendation scenarios, organized into subdirectories:

| Subdirectory | Description | Files |
|---|---|---|
| `context/` | Context-aware recommendation configs | `avazu-2m_context.md`, `criteo-4m_context.md`, `ml-1m_context.md` |
| `general/` | General recommendation configs | `amazon-books_general.md`, `ml-1m_general.md`, `yelp_general.md` |
| `knowledge/` | Knowledge-based recommendation configs | `amazon-books_kg.md`, `lastfm-track_kg.md`, `ml-1m_kg.md` |
| `sequential/` | Sequential recommendation configs | `amazon-books_seq.md`, `ml-1m_seq.md`, `yelp_seq.md` |

### `time_test_result/`

Contains benchmark timing results in markdown format for different recommendation model categories:

| File | Description |
|---|---|
| `General_recommendation.md` | Timing benchmarks for general recommendation models |
| `Sequential_recommendation.md` | Timing benchmarks for sequential recommendation models |
| `Context-aware_recommendation.md` | Timing benchmarks for context-aware recommendation models |
| `Knowledge-based_recommendation.md` | Timing benchmarks for knowledge-based recommendation models |
