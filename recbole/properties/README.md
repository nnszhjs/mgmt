# recbole.properties

Default YAML configuration files for RecBole. These files define the default parameter values for the framework's overall settings, individual model hyperparameters, dataset-specific configurations, and quick-start presets.

## Files

| File | Description |
|------|-------------|
| `overall.yaml` | Global default configuration covering environment settings (GPU, seed, logging), training settings (epochs, batch size, optimizer, learning rate, negative sampling), and evaluation settings (metrics, top-k, splitting strategy, batch size). |

## Sub-directories

| Directory | Description |
|-----------|-------------|
| `model/` | Per-model YAML configuration files (94 files) defining default hyperparameters for each supported model (e.g., `BPR.yaml`, `LightGCN.yaml`, `SASRec.yaml`, `DeepFM.yaml`, `KGAT.yaml`, etc.). |
| `dataset/` | Dataset-specific YAML configuration files (4 files) including `ml-100k.yaml`, `sample.yaml`, `url.yaml`, and `kg_url.yaml` with dataset paths and field definitions. |
| `quick_start_config/` | Preset configuration files for common experimental scenarios (e.g., `context-aware.yaml`, `sequential.yaml`, `knowledge_base.yaml`, `sequential_DIN.yaml`, etc.) that combine model, dataset, and training settings for one-command experiments. |
