---
sidebar_label: portfolio
title: default.portfolio
---

#### config\_predictor\_tuple

```python
def config_predictor_tuple(tasks, configs, meta_features, regret_matrix)
```

Config predictor represented in tuple.

The returned tuple consists of (meta_features, preferences, proc).

**Returns**:

- `meta_features_norm` - A dataframe of normalized meta features, each column for a task.
- `preferences` - A dataframe of sorted configuration indicies by their performance per task (column).
- `regret_matrix` - A dataframe of the configuration(row)-task(column) regret matrix.

#### build\_portfolio

```python
def build_portfolio(meta_features, regret, strategy)
```

Build a portfolio from meta features and regret matrix.

**Arguments**:

- `meta_features` - A dataframe of metafeatures matrix.
- `regret` - A dataframe of regret matrix.
- `strategy` - A str of the strategy, one of ("greedy", "greedy-feedback").

#### load\_json

```python
def load_json(filename)
```

Returns the contents of json file filename.

#### serialize

```python
def serialize(configs, regret, meta_features, output_file, config_path)
```

Store to disk all information FLAML-metalearn needs at runtime.

configs: names of model configs
regret: regret matrix
meta_features: task metafeatures
output_file: filename
config_path: path containing config json files

