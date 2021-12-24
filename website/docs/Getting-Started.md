# Getting Started

<!-- ### Welcome to FLAML, a Fast Library for Automated Machine Learning & Tuning! -->

FLAML is a lightweight Python library that finds accurate machine
learning models automatically, efficiently and economically. It frees users from selecting learners and hyperparameters for each learner.

### Main Features

1. For common machine learning tasks like classification and regression, it quickly finds quality models for user-provided data with low computational resources. It supports both classifcal machine learning models and deep neural networks.

2. It is easy to customize or extend. Users can choose their desired customizability: minimal customization (computational resource budget), medium customization (e.g., scikit-style learner, search space and metric), or full customization (arbitrary training and evaluation code).

3. It supports fast and economical automatic tuning, capable of handling large search space with heterogeneous evaluation cost and complex constraints/guidance/early stopping. FLAML is powered by a new, [cost-effective
hyperparameter optimization](Use-Cases/Tune-User-Defined-Function#hyperparameter-optimization-algorithm)
and learner selection method invented by Microsoft Research.

### Quickstart

Install FLAML from pip: `pip install flaml`. Find more options in [Installation](Installation).

There are two ways of using flaml:

#### [Task-oriented AutoML](Use-Cases/task-oriented-automl)

For example, with three lines of code, you can start using this economical and fast AutoML engine as a scikit-learn style estimator.

```python
from flaml import AutoML
automl = AutoML()
automl.fit(X_train, y_train, task="classification")
```

It automatically tunes the hyparparameters and selects the best model from default learners such as LightGBM, XGBoost, random forest etc. [Customizing](Use-Cases/task-oriented-automl#customize-automlfit) the optimization metrics, learners and search spaces etc. is very easy. For example,

```python
automl.add_learner("mylgbm", MyLGBMEstimator)
automl.fit(X_train, y_train, task="classification", metric=custom_metric, estimator_list=["mylgbm"])
```

#### [Tune user-defined function](Use-Cases/Tune-User-Defined-Function)

You can run generic hyperparameter tuning for a custom function (machine learning or beyond). For example,

```python
from flaml import tune
from flaml.model import LGBMEstimator

def train_lgbm(config: dict) -> dict:
    # convert config dict to lgbm params
    params = LGBMEstimator(**config).params
    # train the model
    train_set = lightgbm.Dataset(X_train, y_train)
    model = lightgbm.train(params, train_set)
    # evaluate the model
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    # return eval results as a dictionary
    return {"mse": mse}

# load a built-in search space from flaml
flaml_lgbm_search_space = LGBMEstimator.search_space(X_train.shape)
# specify the search space as a dict from hp name to domain; you can define your own search space same way
config_search_space = {hp: space["domain"] for hp, space in flaml_lgbm_search_space.items()}
# give guidance about hp values corresponding to low training cost, i.e., {"n_estimators": 4, "num_leaves": 4}
low_cost_partial_config = {
    hp: space["low_cost_init_value"]
    for hp, space in flaml_lgbm_search_space.items()
    if "low_cost_init_value" in space
}
# run the tuning, minimizing mse, with total time budget 3 seconds
analysis = tune.run(
    train_lgbm, metric="mse", mode="min", config=config_search_space,
    low_cost_partial_config=low_cost_partial_config, time_budget_s=3, num_samples=-1,
)
```
Please see this [script](https://github.com/microsoft/FLAML/blob/main/test/tune_example.py) for the complete version of the above example.

### Where to Go Next?

* Understand the use cases for [Task-oriented AutoML](Use-Cases/task-oriented-automl) and [Tune user-defined function](Use-Cases/Tune-User-Defined-Function).
* Find code examples under "Examples": from [AutoML - Classification](Examples/AutoML-Classification) to [Tune - PyTorch](Examples/Tune-PyTorch).
* Watch [video tutorials](https://www.youtube.com/channel/UCfU0zfFXHXdAd5x-WvFBk5A).
* Learn about [research](Research) around FLAML.
* Refer to [SDK](reference/automl) and [FAQ](FAQ).

If you like our project, please give it a [star](https://github.com/microsoft/FLAML/stargazers) on GitHub. If you are interested in contributing, please read [Contributor's Guide](Contribute).