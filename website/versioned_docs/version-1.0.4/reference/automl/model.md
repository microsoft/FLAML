---
sidebar_label: model
title: automl.model
---

## BaseEstimator Objects

```python
class BaseEstimator()
```

The abstract class for all learners.

Typical examples:
* XGBoostEstimator: for regression.
* XGBoostSklearnEstimator: for classification.
* LGBMEstimator, RandomForestEstimator, LRL1Classifier, LRL2Classifier:
    for both regression and classification.

#### \_\_init\_\_

```python
def __init__(task="binary", **config)
```

Constructor.

**Arguments**:

- `task` - A string of the task type, one of
  'binary', 'multiclass', 'regression', 'rank', 'seq-classification',
  'seq-regression', 'token-classification', 'multichoice-classification',
  'summarization', 'ts_forecast', 'ts_forecast_classification'.
- `config` - A dictionary containing the hyperparameter names, 'n_jobs' as keys.
  n_jobs is the number of parallel threads.

#### model

```python
@property
def model()
```

Trained model after fit() is called, or None before fit() is called.

#### estimator

```python
@property
def estimator()
```

Trained model after fit() is called, or None before fit() is called.

#### feature\_names\_in\_

```python
@property
def feature_names_in_()
```

if self._model has attribute feature_names_in_, return it.
otherwise, if self._model has attribute feature_name_, return it.
otherwise, if self._model has attribute feature_names, return it.
otherwise, if self._model has method get_booster, return the feature names.
otherwise, return None.

#### feature\_importances\_

```python
@property
def feature_importances_()
```

if self._model has attribute feature_importances_, return it.
otherwise, if self._model has attribute coef_, return it.
otherwise, return None.

#### fit

```python
def fit(X_train, y_train, budget=None, free_mem_ratio=0, **kwargs)
```

Train the model from given training data.

**Arguments**:

- `X_train` - A numpy array or a dataframe of training data in shape n*m.
- `y_train` - A numpy array or a series of labels in shape n*1.
- `budget` - A float of the time budget in seconds.
- `free_mem_ratio` - A float between 0 and 1 for the free memory ratio to keep during training.
  

**Returns**:

- `train_time` - A float of the training time in seconds.

#### predict

```python
def predict(X, **kwargs)
```

Predict label from features.

**Arguments**:

- `X` - A numpy array or a dataframe of featurized instances, shape n*m.
  

**Returns**:

  A numpy array of shape n*1.
  Each element is the label for a instance.

#### predict\_proba

```python
def predict_proba(X, **kwargs)
```

Predict the probability of each class from features.

Only works for classification problems

**Arguments**:

- `X` - A numpy array of featurized instances, shape n*m.
  

**Returns**:

  A numpy array of shape n*c. c is the # classes.
  Each element at (i,j) is the probability for instance i to be in
  class j.

#### score

```python
def score(X_val: DataFrame, y_val: Series, **kwargs)
```

Report the evaluation score of a trained estimator.


**Arguments**:

- `X_val` - A pandas dataframe of the validation input data.
- `y_val` - A pandas series of the validation label.
- `kwargs` - keyword argument of the evaluation function, for example:
  - metric: A string of the metric name or a function
  e.g., 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
  'f1', 'micro_f1', 'macro_f1', 'log_loss', 'mae', 'mse', 'r2',
  'mape'. Default is 'auto'.
  If metric is given, the score will report the user specified metric.
  If metric is not given, the metric is set to accuracy for classification and r2
  for regression.
  You can also pass a customized metric function, for examples on how to pass a
  customized metric function, please check
  [test/nlp/test_autohf_custom_metric.py](https://github.com/microsoft/FLAML/blob/main/test/nlp/test_autohf_custom_metric.py) and
  [test/automl/test_multiclass.py](https://github.com/microsoft/FLAML/blob/main/test/automl/test_multiclass.py).
  

**Returns**:

  The evaluation score on the validation dataset.

#### search\_space

```python
@classmethod
def search_space(cls, data_size, task, **params)
```

[required method] search space.

**Arguments**:

- `data_size` - A tuple of two integers, number of rows and columns.
- `task` - A str of the task type, e.g., "binary", "multiclass", "regression".
  

**Returns**:

  A dictionary of the search space.
  Each key is the name of a hyperparameter, and value is a dict with
  its domain (required) and low_cost_init_value, init_value,
  cat_hp_cost (if applicable).
  e.g., ```{'domain': tune.randint(lower=1, upper=10), 'init_value': 1}```.

#### size

```python
@classmethod
def size(cls, config: dict) -> float
```

[optional method] memory size of the estimator in bytes.

**Arguments**:

- `config` - A dict of the hyperparameter config.
  

**Returns**:

  A float of the memory size required by the estimator to train the
  given config.

#### cost\_relative2lgbm

```python
@classmethod
def cost_relative2lgbm(cls) -> float
```

[optional method] relative cost compared to lightgbm.

#### init

```python
@classmethod
def init(cls)
```

[optional method] initialize the class.

#### config2params

```python
def config2params(config: dict) -> dict
```

[optional method] config dict to params dict

**Arguments**:

- `config` - A dict of the hyperparameter config.
  

**Returns**:

  A dict that will be passed to self.estimator_class's constructor.

## TransformersEstimator Objects

```python
class TransformersEstimator(BaseEstimator)
```

The class for fine-tuning language models, using huggingface transformers API.

## SKLearnEstimator Objects

```python
class SKLearnEstimator(BaseEstimator)
```

The base class for tuning scikit-learn estimators.

Subclasses can modify the function signature of ``__init__`` to
ignore the values in ``config`` that are not relevant to the constructor
of their underlying estimator. For example, some regressors in ``scikit-learn``
don't accept the ``n_jobs`` parameter contained in ``config``. For these,
one can add ``n_jobs=None,`` before ``**config`` to make sure ``config`` doesn't
contain an ``n_jobs`` key.

## LGBMEstimator Objects

```python
class LGBMEstimator(BaseEstimator)
```

The class for tuning LGBM, using sklearn API.

## XGBoostEstimator Objects

```python
class XGBoostEstimator(SKLearnEstimator)
```

The class for tuning XGBoost regressor, not using sklearn API.

## XGBoostSklearnEstimator Objects

```python
class XGBoostSklearnEstimator(SKLearnEstimator,  LGBMEstimator)
```

The class for tuning XGBoost with unlimited depth, using sklearn API.

## XGBoostLimitDepthEstimator Objects

```python
class XGBoostLimitDepthEstimator(XGBoostSklearnEstimator)
```

The class for tuning XGBoost with limited depth, using sklearn API.

## RandomForestEstimator Objects

```python
class RandomForestEstimator(SKLearnEstimator,  LGBMEstimator)
```

The class for tuning Random Forest.

## ExtraTreesEstimator Objects

```python
class ExtraTreesEstimator(RandomForestEstimator)
```

The class for tuning Extra Trees.

## LRL1Classifier Objects

```python
class LRL1Classifier(SKLearnEstimator)
```

The class for tuning Logistic Regression with L1 regularization.

## LRL2Classifier Objects

```python
class LRL2Classifier(SKLearnEstimator)
```

The class for tuning Logistic Regression with L2 regularization.

## CatBoostEstimator Objects

```python
class CatBoostEstimator(BaseEstimator)
```

The class for tuning CatBoost.

## Prophet Objects

```python
class Prophet(SKLearnEstimator)
```

The class for tuning Prophet.

## ARIMA Objects

```python
class ARIMA(Prophet)
```

The class for tuning ARIMA.

## SARIMAX Objects

```python
class SARIMAX(ARIMA)
```

The class for tuning SARIMA.

## TS\_SKLearn Objects

```python
class TS_SKLearn(SKLearnEstimator)
```

The class for tuning SKLearn Regressors for time-series forecasting, using hcrystalball

## LGBM\_TS Objects

```python
class LGBM_TS(TS_SKLearn)
```

The class for tuning LGBM Regressor for time-series forecasting

## XGBoost\_TS Objects

```python
class XGBoost_TS(TS_SKLearn)
```

The class for tuning XGBoost Regressor for time-series forecasting

## RF\_TS Objects

```python
class RF_TS(TS_SKLearn)
```

The class for tuning Random Forest Regressor for time-series forecasting

## ExtraTrees\_TS Objects

```python
class ExtraTrees_TS(TS_SKLearn)
```

The class for tuning Extra Trees Regressor for time-series forecasting

## XGBoostLimitDepth\_TS Objects

```python
class XGBoostLimitDepth_TS(TS_SKLearn)
```

The class for tuning XGBoost Regressor with unlimited depth for time-series forecasting

## TemporalFusionTransformerEstimator Objects

```python
class TemporalFusionTransformerEstimator(SKLearnEstimator)
```

The class for tuning Temporal Fusion Transformer

