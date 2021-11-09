---
sidebar_label: model
title: model
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
  &#x27;binary&#x27;, &#x27;multi&#x27;, &#x27;regression&#x27;, &#x27;rank&#x27;, &#x27;forecast&#x27;
- `config` - A dictionary containing the hyperparameter names
  and &#x27;n_jobs&#x27; as keys. n_jobs is the number of parallel threads.

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

#### fit

```python
def fit(X_train, y_train, budget=None, **kwargs)
```

Train the model from given training data.

**Arguments**:

- `X_train` - A numpy array or a dataframe of training data in shape n*m.
- `y_train` - A numpy array or a series of labels in shape n*1.
- `budget` - A float of the time budget in seconds.
  

**Returns**:

- `train_time` - A float of the training time in seconds.

#### predict

```python
def predict(X_test)
```

Predict label from features.

**Arguments**:

- `X_test` - A numpy array or a dataframe of featurized instances, shape n*m.
  

**Returns**:

  A numpy array of shape n*1.
  Each element is the label for a instance.

#### predict\_proba

```python
def predict_proba(X_test)
```

Predict the probability of each class from features.

Only works for classification problems

**Arguments**:

- `X_test` - A numpy array of featurized instances, shape n*m.
  

**Returns**:

  A numpy array of shape n*c. c is the # classes.
  Each element at (i,j) is the probability for instance i to be in
  class j.

#### search\_space

```python
@classmethod
def search_space(cls, **params)
```

[required method] search space.

**Returns**:

  A dictionary of the search space.
  Each key is the name of a hyperparameter, and value is a dict with
  its domain (required) and low_cost_init_value, init_value,
  cat_hp_cost (if applicable).
  e.g.,
- ``{&#x27;domain&#x27;` - tune.randint(lower=1, upper=10), &#x27;init_value&#x27;: 1}.`

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

  A dict that will be passed to self.estimator_class&#x27;s constructor.

## SKLearnEstimator Objects

```python
class SKLearnEstimator(BaseEstimator)
```

The base class for tuning scikit-learn estimators.

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

The class for tuning XGBoost (for classification), using sklearn API.

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

