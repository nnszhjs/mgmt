# recbole.config

The configuration module for RecBole. It provides the `Config` class that loads default parameters from built-in YAML properties and merges them with external input from config files, command-line arguments, and parameter dictionaries, with a clear priority order: command line > parameter dictionaries > config file.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports the `Config` class. |
| `configurator.py` | Implements the `Config` class which handles parameter loading, merging, validation, and compatibility checks across different input sources. Automatically configures evaluation settings, model types, and device assignments. |
