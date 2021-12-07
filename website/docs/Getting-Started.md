# Getting Started

### Welcome to FLAML, a Fast Library for Automated Machine Learning & Tuning!
FLAML is a lightweight Python library that finds accurate machine
learning models automatically, efficiently and economically. It frees users from selecting learners and hyperparameters for each learner. FLAML has the following uniques properties comparing with other AutoML and hyperparameter tuning libraries,

1. It is fast and economical. FLAML is powered by a new, [cost-effective
hyperparameter optimization](https://github.com/microsoft/FLAML/tree/main/flaml/tune)
and learner selection method invented by Microsoft Research.

2. It is easy to customize or extend. Users can choose their desired customizability: minimal customization (computational resource budget), medium customization (e.g., scikit-style learner, search space and metric), or full customization (arbitrary training and evaluation code).

3. It supports fast automatic tuning, capable of handling complex constraints/guidance/early stopping.

### Installation

FLAML requires **Python version >= 3.6**. It can be installed from pip:

```bash
pip install flaml
```

To run the [`notebook example`](https://github.com/microsoft/FLAML/tree/main/notebook),
install flaml with the [notebook] option:

```bash
pip install flaml[notebook]
```

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
automl.fit(X_train, y_train, task="regression", estimator_list=["lgbm"])
```

* You can also run generic hyperparameter tuning for a custom function (machine learning or beyond).

```python
from flaml import tune
tune.run(func_to_tune, config={…}, low_cost_partial_config={…}, time_budget_s=3600)
```
