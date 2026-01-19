# Frequently Asked Questions

### [Guidelines on how to set a hyperparameter search space](Use-Cases/Tune-User-Defined-Function#details-and-guidelines-on-hyperparameter-search-space)

### [Guidelines on parallel vs seqential tuning](Use-Cases/Task-Oriented-AutoML#guidelines-on-parallel-vs-sequential-tuning)

### [Guidelines on creating and tuning a custom estimator](Use-Cases/Task-Oriented-AutoML#guidelines-on-tuning-a-custom-estimator)

### About `low_cost_partial_config` in `tune`.

- Definition and purpose: The `low_cost_partial_config` is a dictionary of subset of the hyperparameter coordinates whose value corresponds to a configuration with known low-cost (i.e., low computation cost for training the corresponding model). The concept of low/high-cost is meaningful in the case where a subset of the hyperparameters to tune directly affects the computation cost for training the model. For example, `n_estimators` and `max_leaves` are known to affect the training cost of tree-based learners. We call this subset of hyperparameters, *cost-related hyperparameters*. In such scenarios, if you are aware of low-cost configurations for the cost-related hyperparameters, you are recommended to set them as the `low_cost_partial_config`. Using the tree-based method example again, since we know that small `n_estimators` and `max_leaves` generally correspond to simpler models and thus lower cost, we set `{'n_estimators': 4, 'max_leaves': 4}` as the `low_cost_partial_config` by default (note that `4` is the lower bound of search space for these two hyperparameters), e.g., in [LGBM](https://github.com/microsoft/FLAML/blob/main/flaml/model.py#L215). Configuring `low_cost_partial_config` helps the search algorithms make more cost-efficient choices.
  In AutoML, the `low_cost_init_value` in `search_space()` function for each estimator serves the same role.

- Usage in practice: It is recommended to configure it if there are cost-related hyperparameters in your tuning task and you happen to know the low-cost values for them, but it is not required (It is fine to leave it the default value, i.e., `None`).

- How does it work: `low_cost_partial_config` if configured, will be used as an initial point of the search. It also affects the search trajectory. For more details about how does it play a role in the search algorithms, please refer to the papers about the search algorithms used: Section 2 of [Frugal Optimization for Cost-related Hyperparameters (CFO)](https://arxiv.org/pdf/2005.01571.pdf) and Section 3 of [Economical Hyperparameter Optimization with Blended Search Strategy (BlendSearch)](https://openreview.net/pdf?id=VbLH04pRA3).

### How does FLAML handle missing values?

FLAML does not perform any preprocessing to handle missing values (NaN, null, etc.) in the input data. Instead, it delegates the handling of missing values to the underlying estimators. This means that the behavior depends on which estimator is being used:

**Estimators that handle missing values natively:**

These estimators can work with missing values without any preprocessing:

- **`lgbm`** (LightGBM): Handles missing values natively by assigning them to the side that reduces the loss the most in each split. Works for both categorical and continuous features.
- **`xgboost`** (XGBoost): Handles missing values natively by learning the best direction to handle missing values during training. Works for both categorical and continuous features.
- **`xgb_limitdepth`** (XGBoost with depth limit): Same as `xgboost`, handles missing values natively.
- **`catboost`** (CatBoost): Handles missing values natively with multiple strategies for both categorical and continuous features. See [CatBoost documentation](https://catboost.ai/en/docs/concepts/algorithm-missing-values-processing) for details.
- **`histgb`** (HistGradientBoosting): sklearn's HistGradientBoosting handles missing values natively by learning the best direction during training.

**Estimators that require preprocessing for missing values:**

These estimators will raise an error if they encounter missing values in the data:

- **`rf`** (RandomForest): sklearn's RandomForest does not accept missing values.
- **`extra_tree`** (ExtraTrees): sklearn's ExtraTrees does not accept missing values.
- **`lrl1`**, **`lrl2`** (LogisticRegression): sklearn's LogisticRegression does not accept missing values.
- **`kneighbor`** (KNeighbors): sklearn's KNeighbors does not accept missing values.
- **`sgd`** (SGDClassifier/Regressor): sklearn's SGD estimators do not accept missing values.

**Recommendations for handling missing values:**

1. **Use estimators that handle missing values natively**: If your data contains missing values, consider restricting the estimator list to those that can handle them:

```python
from flaml import AutoML

automl = AutoML()
automl_settings = {
    "time_budget": 60,
    "metric": "accuracy",
    "task": "classification",
    "estimator_list": [
        "lgbm",
        "xgboost",
        "catboost",
        "histgb",
    ],  # Only estimators that handle NaN
}
automl.fit(X_train, y_train, **automl_settings)
```

2. **Preprocess the data before passing to FLAML**: If you want to use estimators that don't handle missing values, preprocess your data using techniques such as:
   - Imputation (e.g., using `sklearn.impute.SimpleImputer` or `sklearn.impute.KNNImputer`)
   - Dropping samples or features with missing values
   - Using a scikit-learn `Pipeline` with an imputer (see [integration with scikit-learn Pipeline](Examples/Integrate%20-%20Scikit-learn%20Pipeline))

Example with imputation:

```python
from flaml import AutoML
from sklearn.impute import SimpleImputer
import numpy as np

# Preprocess data to handle missing values
imputer = SimpleImputer(strategy="mean")  # for continuous features
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Now you can use any estimator
automl = AutoML()
automl.fit(X_train_imputed, y_train, task="classification", time_budget=60)
```

3. **Use sklearn Pipeline for integrated preprocessing**: You can integrate FLAML with sklearn's Pipeline to automatically handle missing values:

```python
from flaml import AutoML
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Create a pipeline with imputation
pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("automl", AutoML())]
)

# The pipeline will automatically impute missing values before AutoML
pipeline.fit(X_train, y_train)
```

**Note on time series forecasting**: For time series tasks (`ts_forecast`, `ts_forecast_panel`), missing values handling may differ depending on the specific forecasting model used. Refer to the documentation of the specific time series model for details.

### How does FLAML handle imbalanced data (unequal distribution of target classes in classification task)?

Currently FLAML does several things for imbalanced data.

1. When a class contains fewer than 20 examples, we repeatedly add these examples to the training data until the count is at least 20.
1. We use stratified sampling when doing holdout and kf.
1. We make sure no class is empty in both training and holdout data.
1. We allow users to pass `sample_weight` to `AutoML.fit()`.
1. User can customize the weight of each class by setting the `custom_hp` or `fit_kwargs_by_estimator` arguments. For example, the following code sets the weight for pos vs. neg as 2:1 for the RandomForest estimator:

```python
from flaml import AutoML
from sklearn.datasets import load_iris

X_train, y_train = load_iris(return_X_y=True)
automl = AutoML()
automl_settings = {
    "time_budget": 2,
    "task": "classification",
    "log_file_name": "test/iris.log",
    "estimator_list": ["rf", "xgboost"],
}

automl_settings["custom_hp"] = {
    "xgboost": {
        "scale_pos_weight": {
            "domain": 0.5,
            "init_value": 0.5,
        }
    },
    "rf": {"class_weight": {"domain": "balanced", "init_value": "balanced"}},
}
print(automl.model)
```

### How to interpret model performance? Is it possible for me to visualize feature importance, SHAP values, optimization history?

You can use `automl.model.estimator.feature_importances_` to get the `feature_importances_` for the best model found by automl. See an [example](Examples/AutoML-for-XGBoost#plot-feature-importance).

Packages such as `azureml-interpret` and `sklearn.inspection.permutation_importance` can be used on `automl.model.estimator` to explain the selected model.
Model explanation is frequently asked and adding a native support may be a good feature. Suggestions/contributions are welcome.

Optimization history can be checked from the [log](Use-Cases/Task-Oriented-AutoML#log-the-trials). You can also [retrieve the log and plot the learning curve](Use-Cases/Task-Oriented-AutoML#plot-learning-curve).

### How to resolve out-of-memory error in `AutoML.fit()`

- Set `free_mem_ratio` a float between 0 and 1. For example, 0.2 means try to keep free memory above 20% of total memory. Training may be early stopped for memory consumption reason when this is set.
- Set `model_history` False.
- If your data are already preprocessed, set `skip_transform` False. If you can preprocess the data before the fit starts, this setting can save memory needed for preprocessing in `fit`.
- If the OOM error only happens for some particular trials:
  - set `use_ray` True. This will increase the overhead per trial but can keep the AutoML process running when a single trial fails due to OOM error.
  - provide a more accurate [`size`](reference/automl/model#size) function for the memory bytes consumption of each config for the estimator causing this error.
  - modify the [search space](Use-Cases/Task-Oriented-AutoML#a-shortcut-to-override-the-search-space) for the estimators causing this error.
  - or remove this estimator from the `estimator_list`.
- If the OOM error happens when ensembling, consider disabling ensemble, or use a cheaper ensemble option. ([Example](Use-Cases/Task-Oriented-AutoML#ensemble)).

### How to get the best config of an estimator and use it to train the original model outside FLAML?

When you finished training an AutoML estimator, you may want to use it in other code w/o depending on FLAML. You can get the `automl.best_config` and convert it to the parameters of the original model with below code:

```python
from flaml import AutoML
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
settings = {"time_budget": 3}
automl = AutoML(**settings)
automl.fit(X, y)

print(f"{automl.best_estimator=}")
print(f"{automl.best_config=}")
print(f"params for best estimator: {automl.model.config2params(automl.best_config)}")
```

If the automl instance is not accessible and you've the `best_config`. You can also convert it with below code:

```python
from flaml.automl.task.factory import task_factory

task = "classification"
best_estimator = "rf"
best_config = {
    "n_estimators": 15,
    "max_features": 0.35807183923834934,
    "max_leaves": 12,
    "criterion": "gini",
}

model_class = task_factory(task).estimator_class_from_str(best_estimator)(task=task)
best_params = model_class.config2params(best_config)
```

Then you can use it to train the sklearn estimators directly:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(**best_params)
model.fit(X, y)
```

### How to save and load an AutoML object? (`pickle` / `load_pickle`)

FLAML provides `AutoML.pickle()` / `AutoML.load_pickle()` as a convenient and robust way to persist an AutoML run.

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=60)

# Save
automl.pickle("automl.pkl")

# Load
automl_loaded = AutoML.load_pickle("automl.pkl")
pred = automl_loaded.predict(X_test)
```

Notes:

- If you used Spark estimators, `AutoML.pickle()` externalizes Spark ML models into an adjacent artifact folder and keeps
  the pickle itself lightweight.
- If you want to skip re-loading externalized Spark models (e.g., in an environment without Spark), use:

```python
automl_loaded = AutoML.load_pickle("automl.pkl", load_spark_models=False)
```

### How to list all available estimators for a task?

The available estimator set is task-dependent and can vary with optional dependencies. You can list the estimator keys
that FLAML currently has registered in your environment:

```python
from flaml.automl.task.factory import task_factory

print(sorted(task_factory("classification").estimators.keys()))
print(sorted(task_factory("regression").estimators.keys()))
print(sorted(task_factory("forecast").estimators.keys()))
print(sorted(task_factory("rank").estimators.keys()))
```

### How to list supported built-in metrics?

```python
from flaml import AutoML

automl = AutoML()
sklearn_metrics, hf_metrics, spark_metrics = automl.supported_metrics
print(sorted(sklearn_metrics))
print(sorted(hf_metrics))
print(spark_metrics)
```
