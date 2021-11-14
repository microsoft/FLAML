# Task-oriented AutoML

## Overview

`flaml.AutoML` is a class for task-oriented AutoML. It can be used as a scikit-learn style estimator with the standard `fit` and `predict` functions. The minimal input from users is the training data and the task type.

* Training data:
    - numpy array. When the input data are stored in numpy array, they are passed to `fit()` as `X_train` and `y_train`.
    - pandas dataframe. When the input data are stored in pandas dataframe, they are passed to `fit()` either as `X_train` and `y_train`, or as `dataframe` and `label`.
* Tasks (specified via `task`):
    - 'classification': classification
    - 'regression': regression
    - 'ts_forecast': time series forecasting
    - 'rank': learning to rank

An optional input is `time_budget` for searching models and hyperparameters. When not specified, a default budget of 60 seconds will be used.

A typical way to use `flaml.AutoML`:

```python
# Prepare training data
# ...
from flaml import AutoML
automl = AutoML()
automl.fit(X_train, y_train, task="regression", time_budget=60, **other_settings)
# Save the model
with open("automl.pkl", "wb") as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

# At prediction time
with open("automl.pkl", "rb") as f:
    automl = pickle.load(f)
pred = automl.predict(X_test)
```

If users provide the minimal input only, `AutoML` uses the default settings for time budget, optimization metric, estimator list etc.

## Customize AutoML.fit()

### Optimization metric

The optimization metric is specified via the `metric` argument. It can be either a string which refers to a built-in metric, or a user-defined function.

* Built-in metric
    - 'accuracy': 1 - accuracy as the corresponding metric to minimize.
* User-defined function

### Estimator and search space

The estimator list can contain one or more estimator names, each corresponding to a built-in estimator or a custom estimator. Each estimator has a search space for hyperparameter configurations.

#### Estimator
* Built-in estimator
    - 'lgbm': LightGBM.
* Custom estimator. Use custom estimator for:
    - tuning an estimator that is not built-in
    - customizing search space for a built-in estimator

To tune a custom estimator that is not built-in, inherit `flaml.model.BaseEstimator` or a derived class.
For example, if you have a estimator class with scikit-learn style `fit()` and `predict()` functions, you only need to set `self.estimator_class`to be that class in your constructor.

```python
from flaml.model import SKLearnEstimator
# SKLearnEstimator is derived from BaseEstimator
import rgf


class MyRegularizedGreedyForest(SKLearnEstimator):
    def __init__(self, task="binary", **config):
        super().__init__(task, **config)

        if task in CLASSIFICATION:
            from rgf.sklearn import RGFClassifier

            self.estimator_class = RGFClassifier
        else:
            from rgf.sklearn import RGFRegressor

            self.estimator_class = RGFRegressor

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            "max_leaf": {
                "domain": tune.lograndint(lower=4, upper=data_size),
                "low_cost_init_value": 4,
            },
            "n_iter": {
                "domain": tune.lograndint(lower=1, upper=data_size),
                "low_cost_init_value": 1,
            },
            "learning_rate": {"domain": tune.loguniform(lower=0.01, upper=20.0)},
            "min_samples_leaf": {
                "domain": tune.lograndint(lower=1, upper=20),
                "init_value": 20,
            },
        }
        return space
```

In the constructor, we set `self.estimator_class` as `RGFClassifier` or `RGFRegressor` according to the task type. If the estimator you want to tune does not have a scikit-learn style `fit()` and `predict()` API, you can override the `fit()` and `predict()` function of `flaml.model.BaseEstimator`, like [XGBoostEstimator](https://github.com/microsoft/FLAML/blob/59083fbdcb95c15819a0063a355969203022271c/flaml/model.py#L511).

#### Search space

Each estimator class, built-in or not, must have a `search_space` function. In the `search_space` function, we return a dictionary about the hyperparameters, the keys of which are the names of the hyperparameters to tune, and each value is a set of detailed search configurations about the corresponding hyperparameters represented in a dictionary. A search configuration dictionary includes the following fields:
* `domain`, which specifies the possible values of the hyperparameter and their distribution;
* `init_value` (optional), which specifies the initial value of the hyperparameter;
* `low_cost_init_value`(optional), which specifies the value of the hyperparameter that is associated with low computation cost. 

In the example above, we tune four hyperparameters, three integers and one float. They all follow a log-uniform distribution. "max_leaf" and "n_iter" have "low_cost_init_value" specified as their values heavily influence the training cost.

For a complete guide about how to set the domain, please refer to [search space](Tune-User-Defined-Function#search-space).


To customize the search space for a built-in estimator, use a similar approach to define a class that inherits the existing estimator. For example,

```python
from flaml.model import XGBoostEstimator


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


class MyXGB1(XGBoostEstimator):
    """XGBoostEstimator with logregobj as the objective function"""

    def __init__(self, **config):
        super().__init__(objective=logregobj, **config)
```

We override the constructor and set the training objective as a custom function `logregobj`. The hyperparameters and their search range do not change. For another example,

```python
class XGBoost2D(XGBoostSklearnEstimator):
    @classmethod
    def search_space(cls, data_size, task):
        upper = min(32768, int(data_size))
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "low_cost_init_value": 4,
            },
            "max_leaves": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "low_cost_init_value": 4,
            },
        }
```

We override the `search_space` function to tune two hyperparameters only, "n_estimators" and "max_leaves". They are both random integers in the log space, ranging from 4 to data-dependent upper bound. The lower bound for each corresponds to low training cost, hence the "low_cost_init_value" for each is set to 4. 

### How to set time budget

* If you have an exact constraint for the total search time, set it as the time budget.
* If you have flexible time constraints, for example, your desirable time budget is t1=60s, and the longest time budget you can tolerate is t2=3600s, you can try the following two ways:
1. set t1 as the time budget, and check the message in the console log in the end. If the budget is too small, you will see a warning like 
> WARNING - Time taken to find the best model is 91% of the provided time budget and not all estimators' hyperparameter search converged. Consider increasing the time budget.
2. set t2 as the time budget, and also set `early_stop=True`. If the early stopping is triggered, you will see a warning like
    > WARNING - All estimator hyperparameters local search has converged at least once, and the total search time exceeds 10 times the time taken to find the best model.

    > WARNING - Stopping search as early_stop is set to True.

### How much time is needed to find the best model

If you want to get a sense of how much time is needed to find the best model, you can use `max_iter=2` to perform two trials first. The message will be like:
> INFO - iteration 0, current learner lgbm

> INFO - Estimated sufficient time budget=145194s. Estimated necessary time budget=2118s.

> INFO -  at 2.6s,  estimator lgbm's best error=0.4459,     best estimator lgbm's best error=0.4459

You will see that the time to finish the first and cheapest trial is 2.6 seconds. The estimated necessary time budget is 2118 seconds, and the estimated sufficient time budget is 145194 seconds. Note that this is only an estimated range to help you decide your budget.

### Constraint

Besides the time budget for model search, users can set other constraints such as the maximal number of models to try, limit on training time and prediction time per model.
* `max_iter`: maximal number of models to try.
* `train_time_limit`: training time in seconds.
* `pred_time_limit`: prediction time per instance in seconds.

For example,
```python
automl.fit(X_train, y_train, max_iter=100, train_time_limit=1, pred_time_limit=1e-3)
```

### Ensemble

To use stacked ensemble after the model search, set `ensemble=True` or a dict. When `ensemble=True`, the final estimator and `passthrough` in the stacker will be automatically chosen. You can specify customized final estimator or passthrough option:
* "final_estimator": an instance of the final estimator in the stacker.
* "passthrough": True (default) or False, whether to pass the original features to the stacker.

For example,
```python
automl.fit(
    X_train, y_train, task="classification",
    "ensemble": {
        "final_estimator": LogisticRegression(),
        "passthrough": False,
    },
)
```

### Resampling strategy

By default, flaml decides the resampling automatically according to the data size and the time budget. If you would like to enforce a certain resampling strategy, you can set `eval_method` to be "holdout" or "cv" for holdout or cross-validation.

For holdout, you can also set:
* `split_ratio`: the fraction for validation data, 0.1 by default.
* `X_val`, `y_val`: a separate validation dataset. When they are passed, the validation metrics will be computed against this given validation dataset. If they are not passed, then a validation dataset will be split from the training data and held out from training during the model search. After the model search, flaml will retrain the model with best configuration on the full training data.
You can set`retrain_full` to be `False` to skip the final retraining or "budget" to ask flaml to do its best to retrain within the time budget.

For cross validation, you can also set `n_splits` of the number of folds. By default it is 5.

#### Data split method

By default, flaml uses the following method to split the data:
* stratified split for classification;
* uniform split for regression;
* time-based split for time series forecasting;
* group-based split for learning to rank.

The data split method for classification can be changed into uniform split by setting `split_type="uniform"`. For both classification and regression, time-based split can be enforced if the data are sorted by timestamps, by setting `split_type="time"`.

### Parallel tuning

When you have parallel resources, you can either spend them in training and keep the model search sequential, or perform parallel search. Following scikit-learn, the parameter `n_jobs` specifies how many CPU cores to use for each training job. The number of parallel trials is specified via the parameter `n_concurrent_trials`. By default, `n_jobs=-1, n_concurrent_trials=1`. That is, all the CPU cores (in a single compute node) are used for training a single model and the search is sequential. When you have more resources than what each single training job needs, you can consider increasing `n_concurrent_trials`.

To do parallel tuning, install the `ray` and `blendsearch` options:
```bash
pip install flaml[ray,blendsearch]
```

`ray` is used to manage the resources. For example,
```python
ray.init(n_cpus=16)
```
allocates 16 CPU cores. Then, when you run:
```python
automl.fit(X_train, y_train, n_jobs=4, n_concurrent_trials=4)
```
flaml will perform 4 trials in parallel, each consuming 4 CPU cores. The parallel tuning uses the [BlendSearch](Tune-User-Defined-Function##blendsearch-economical-hyperparameter-optimization-with-blended-search-strategy) algorithm.


### Warm start

### Log the trials

### Extra fit arguments



## Retrieve and analyze the outcomes of AutoML.fit()

### Get best model

* Feature importance

### Get best configuration

### Plot learning curve




