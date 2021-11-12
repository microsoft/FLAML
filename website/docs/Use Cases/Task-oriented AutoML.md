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

### Estimator list

The estimator list can contain one or more estimator names, each corresponding to a built-in estimator or a custom estimator.

* Build-in estimator
    - 'lgbm': LightGBM.
* Custom estimator

### How to set time budget

* If you have an exact constraint for the total search time, set it as the time budget.
* If you have flexible time constraints, for example, your desirable time budget is t1=60s, and the longest time budget you can tolerate is t2=3600s, you can try the following two ways:
1. set t1 as the time budget, and check the message in the console log in the end. If the budget is too small, you will see a warning like 
> WARNING - Time taken to find the best model is 91% of the provided time budget and not all estimators' hyperparameter search converged. Consider increasing the time budget.
2. set t2 as the time budget, and also set `early_stop=True`. If the early stopping is triggered, you will see a warning like
    > WARNING - All estimator hyperparameters local search has converged at least once, and the total search time exceeds 10 times the time taken to find the best model.

    > WARNING - Stopping search as early_stop is set to True.

### How long is required to find the best model
If you want to get a sense of how long is required to find the best model, you can use `max_iter=2` to perform two trials first. The message will be like:
> INFO - iteration 0, current learner lgbm

> INFO - Estimated sufficient time budget=145194s. Estimated necessary time budget=2118s.

> INFO -  at 2.6s,  estimator lgbm's best error=0.4459,     best estimator lgbm's best error=0.4459

You will see that the time to finish the first and cheapest trial is 2.6 seconds. The estimated necessary time budget is 2118 seconds, and the estimated sufficient time budget is 145194 seconds. Note that this is only an estimated range to help you decide your budget.

### Constraint

### Ensemble

### Resampling strategy

### Data split method

### Extra fit arguments

### Parallel tuning

### Warm start


## After AutoML.fit()

### Get best model

* Feature importance

### Get best configuration

### Plot learning curve




