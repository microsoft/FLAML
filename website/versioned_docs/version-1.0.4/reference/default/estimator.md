---
sidebar_label: estimator
title: default.estimator
---

#### flamlize\_estimator

```python
def flamlize_estimator(super_class, name: str, task: str, alternatives=None)
```

Enhance an estimator class with flaml's data-dependent default hyperparameter settings.

**Example**:

  
```python
import sklearn.ensemble as ensemble
RandomForestRegressor = flamlize_estimator(
    ensemble.RandomForestRegressor, "rf", "regression"
)
```
  

**Arguments**:

- `super_class` - an scikit-learn compatible estimator class.
- `name` - a str of the estimator's name.
- `task` - a str of the task type.
- `alternatives` - (Optional) a list for alternative estimator names. For example,
  ```[("max_depth", 0, "xgboost")]``` means if the "max_depth" is set to 0
  in the constructor, then look for the learned defaults for estimator "xgboost".

