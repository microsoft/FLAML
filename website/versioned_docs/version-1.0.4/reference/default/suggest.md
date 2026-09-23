---
sidebar_label: suggest
title: default.suggest
---

#### suggest\_config

```python
def suggest_config(task, X, y, estimator_or_predictor, location=None, k=None)
```

Suggest a list of configs for the given task and training data.

The returned configs can be used as starting points for AutoML.fit().
`FLAML_sample_size` is removed from the configs.

#### suggest\_learner

```python
def suggest_learner(task, X, y, estimator_or_predictor="all", estimator_list=None, location=None)
```

Suggest best learner within estimator_list.

#### suggest\_hyperparams

```python
def suggest_hyperparams(task, X, y, estimator_or_predictor, location=None)
```

Suggest hyperparameter configurations and an estimator class.

The configurations can be used to initialize the estimator class like lightgbm.LGBMRegressor.

**Example**:

  
```python
hyperparams, estimator_class = suggest_hyperparams("regression", X_train, y_train, "lgbm")
model = estimator_class(**hyperparams)  # estimator_class is LGBMRegressor
model.fit(X_train, y_train)
```
  

**Arguments**:

- `task` - A string of the task type, e.g.,
  'classification', 'regression', 'ts_forecast', 'rank',
  'seq-classification', 'seq-regression'.
- `X` - A dataframe of training data in shape n*m.
  For 'ts_forecast' task, the first column of X_train
  must be the timestamp column (datetime type). Other
  columns in the dataframe are assumed to be exogenous
  variables (categorical or numeric).
- `y` - A series of labels in shape n*1.
- `estimator_or_predictor` - A str of the learner name or a dict of the learned config predictor.
  If a dict, it contains:
  - "version": a str of the version number.
  - "preprocessing": a dictionary containing:
  * "center": a list of meta feature value offsets for normalization.
  * "scale": a list of meta feature scales to normalize each dimension.
  - "neighbors": a list of dictionaries. Each dictionary contains:
  * "features": a list of the normalized meta features for a neighbor.
  * "choice": an integer of the configuration id in the portfolio.
  - "portfolio": a list of dictionaries, each corresponding to a configuration:
  * "class": a str of the learner name.
  * "hyperparameters": a dict of the config. The key "FLAML_sample_size" will be ignored.
- `location` - (Optional) A str of the location containing mined portfolio file.
  Only valid when the portfolio is a str, by default the location is flaml/default.
  

**Returns**:

- `hyperparams` - A dict of the hyperparameter configurations.
- `estiamtor_class` - A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.

#### preprocess\_and\_suggest\_hyperparams

```python
def preprocess_and_suggest_hyperparams(task, X, y, estimator_or_predictor, location=None)
```

Preprocess the data and suggest hyperparameters.

**Example**:

  
```python
hyperparams, estimator_class, X, y, feature_transformer, label_transformer = \
    preprocess_and_suggest_hyperparams("classification", X_train, y_train, "xgb_limitdepth")
model = estimator_class(**hyperparams)  # estimator_class is XGBClassifier
model.fit(X, y)
X_test = feature_transformer.transform(X_test)
y_pred = label_transformer.inverse_transform(pd.Series(model.predict(X_test).astype(int)))
```
  

**Arguments**:

- `task` - A string of the task type, e.g.,
  'classification', 'regression', 'ts_forecast', 'rank',
  'seq-classification', 'seq-regression'.
- `X` - A dataframe of training data in shape n*m.
  For 'ts_forecast' task, the first column of X_train
  must be the timestamp column (datetime type). Other
  columns in the dataframe are assumed to be exogenous
  variables (categorical or numeric).
- `y` - A series of labels in shape n*1.
- `estimator_or_predictor` - A str of the learner name or a dict of the learned config predictor.
  "choose_xgb" means choosing between xgb_limitdepth and xgboost.
  If a dict, it contains:
  - "version": a str of the version number.
  - "preprocessing": a dictionary containing:
  * "center": a list of meta feature value offsets for normalization.
  * "scale": a list of meta feature scales to normalize each dimension.
  - "neighbors": a list of dictionaries. Each dictionary contains:
  * "features": a list of the normalized meta features for a neighbor.
  * "choice": a integer of the configuration id in the portfolio.
  - "portfolio": a list of dictionaries, each corresponding to a configuration:
  * "class": a str of the learner name.
  * "hyperparameters": a dict of the config. They key "FLAML_sample_size" will be ignored.
- `location` - (Optional) A str of the location containing mined portfolio file.
  Only valid when the portfolio is a str, by default the location is flaml/default.
  

**Returns**:

- `hyperparams` - A dict of the hyperparameter configurations.
- `estiamtor_class` - A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.
- `X` - the preprocessed X.
- `y` - the preprocessed y.
- `feature_transformer` - a data transformer that can be applied to X_test.
- `label_transformer` - a label transformer that can be applied to y_test.

