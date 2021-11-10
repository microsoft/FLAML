# Getting Started

### Welcome to FLAML, a Fast and Lightweight AutoML library!
FLAML is a lightweight Python library that finds accurate machine
learning models automatically, efficiently and economically. It frees users from selecting
learners and hyperparameters for each learner. It is fast and economical.
The simple and lightweight design makes it easy to extend, such as
adding customized learners or metrics. FLAML is powered by a new, [cost-effective
hyperparameter optimization](https://github.com/microsoft/FLAML/tree/main/flaml/tune)
and learner selection method invented by Microsoft Research.
FLAML leverages the structure of the search space to choose a search order optimized for both cost and error. For example, the system tends to propose cheap configurations at the beginning stage of the search,
but quickly moves to configurations with high model complexity and large sample size when needed in the later stage of the search. For another example, it favors cheap learners in the beginning but penalizes them later if the error improvement is slow. The cost-bounded search and cost-based prioritization make a big difference in the search efficiency under budget constraints.

### Quickstart
* With three lines of code, you can start using this economical and fast
AutoML engine as a scikit-learn style estimator.

```python
from flaml import AutoML
automl = AutoML()
automl.fit(X_train, y_train, task="classification")
```

* You can restrict the learners and use FLAML as a fast hyperparameter tuning
tool for XGBoost, LightGBM, Random Forest etc. or a customized learner.

```python
automl.fit(X_train, y_train, task="classification", estimator_list=["lgbm"])
```

* You can also run generic hyperparameter tuning for a custom function (machine learning or beyond).

```python
from flaml import tune
tune.run(func_to_tune, config={…}, low_cost_partial_config={…}, time_budget_s=3600)
```

## Advantages

* For common machine learning tasks like classification and regression, find quality models with small computational resources.
* Users can choose their desired customizability: minimal customization (computational resource budget), medium customization (e.g., scikit-style learner, search space and metric), full customization (arbitrary training and evaluation code).
* Allow human guidance in hyperparameter tuning to respect prior on certain subspaces but also able to explore other subspaces. Read more about the
hyperparameter optimization methods
in FLAML [here](https://github.com/microsoft/FLAML/tree/main/flaml/tune). They can be used beyond the AutoML context.
And they can be used in distributed HPO frameworks such as ray tune or nni.
* Support online AutoML: automatic hyperparameter tuning for online learning algorithms. Read more about the online AutoML method in FLAML [here](https://github.com/microsoft/FLAML/tree/main/flaml/onlineml).