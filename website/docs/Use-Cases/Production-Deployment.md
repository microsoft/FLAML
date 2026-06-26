# Production Deployment

This page walks through the **train → save → reload → predict on new data** lifecycle for FLAML models, with a focus on the gotchas that surface in production but not in the quick-start tutorials. Each section is a self-contained pattern with runnable code and a pointer to the issue or PR that motivated it.

## Scope

You have called `AutoML.fit(...)` once on training data and now need to:

- Serialize the trained model so that a separate process can load and use it.
- Score new (unseen) input rows that may contain categorical features, new categorical values, or a slightly different class distribution.
- Reach into individual ensemble component models (`automl.model.estimators_[i]`).
- Pass sample weights at training time, and understand what `predict()` does (and does not) accept at inference time.
- Avoid the common silent-correctness bugs reported in #1101 (categorical encoding drift) and #1136 (ensemble component prediction).

What this page does **not** cover: training-time configuration (see [Task-Oriented AutoML](Task-Oriented-AutoML)), zero-shot estimators (see [Zero-Shot AutoML](Zero-Shot-AutoML)), or distributed/Spark deployment.

## 1. Save and reload the trained model

### 1.1 `automl.pickle()` — recommended default

`automl.pickle()` writes the entire `AutoML` instance, including the data transformer, the best estimator, and the search history. `AutoML.load_pickle()` restores it in another process. This is the simplest reliable path for FLAML.

```python
import numpy as np
import pandas as pd
from flaml import AutoML

X = pd.DataFrame(
    {
        "age": np.random.randint(20, 70, 400),
        "income": np.random.normal(50000, 15000, 400),
        "gender": np.random.choice(["M", "F"], 400),
        "education": np.random.choice(["HS", "BS", "MS", "PhD"], 400),
    }
)
y = (X["age"] > 40).astype(int)

automl = AutoML()
automl.fit(X, y, task="classification", time_budget=5, estimator_list=["lgbm"])
automl.pickle("automl.pkl")

# In a different process:
loaded = AutoML.load_pickle("automl.pkl")
assert np.array_equal(automl.predict(X), loaded.predict(X))
```

Use `automl.pickle()` whenever possible. It is the only path that preserves *everything* needed at inference time (data transformer included), so the categorical-encoding behavior described in section 3 is reproduced correctly.

### 1.2 MLflow logging — for MLflow-managed deployments

If your serving stack is built around MLflow, log the trained `AutoML` instance explicitly via the sklearn flavor. This works because the `AutoML` object exposes a sklearn-compatible `predict`/`predict_proba` API.

```python
import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from flaml import AutoML

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("flaml_prod")
automl = AutoML()
with mlflow.start_run() as run:
    automl.fit(
        X_train, y_train, task="classification", time_budget=5, mlflow_logging=False
    )
    mlflow.sklearn.log_model(automl, artifact_path="flaml_model")
    run_id = run.info.run_id

loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/flaml_model")
assert np.array_equal(automl.predict(X_test), loaded.predict(X_test))
```

Two practical notes:

- `mlflow_logging=False` disables FLAML's built-in MLflow autologging path inside `fit`. With it enabled, MLflow auto-saves an artifact under `runs:/{run_id}/model`, but on recent MLflow versions reloading that artifact via `mlflow.sklearn.load_model` can return an unfitted `Pipeline`. The explicit `mlflow.sklearn.log_model(automl, ...)` call above sidesteps that issue.
- The argument is `artifact_path=` (not `name=`) in MLflow 2.x.

### 1.3 Pickling just the best estimator — lean serving

If you do not need the data transformer (because your serving pipeline preprocesses upstream and only needs to call the bare ML model), you can pickle `automl.model` instead of the whole `AutoML`. **Use this only if you can guarantee** that inference-time inputs match what FLAML produced *after* its data transformer ran — otherwise you will hit the categorical and ensemble issues in sections 3 and 4.

## 2. The public `automl.preprocess(X)` API

FLAML applies two layers of preprocessing inside `automl.predict(X)`:

1. **Task-level preprocessing** — handled by the internal `DataTransformer`: type coercions, NaN handling, categorical encoding, datetime expansion.
1. **Estimator-level preprocessing** — handled by the estimator wrapper itself (e.g., `Normalizer` for the `SGDEstimator`, sparse-input conversion for XGBoost).

Calling `automl.predict(X)` chains both layers automatically. When you need to reach a single ensemble component or write a custom inference pipeline, call them explicitly:

```python
# Task-level preprocessing, accessible since #1497
X_pre = automl.preprocess(X_test)

# Estimator-level preprocessing on top of the task-level output
X_full = automl.model.preprocess(X_pre)
```

For most consumers, `automl.preprocess(X_test)` is all you need before delegating to a single estimator. Section 4 walks through the canonical use case.

## 3. Categorical features at inference time

This section is the answer to issue #1101 and the silent-correctness bug fixed in PR #1561.

### 3.1 What FLAML does at fit time

When `X` is a pandas DataFrame containing `object`, `string`, or `category` columns, `DataTransformer.fit_transform` records the per-column category list seen at fit time and pins it on the transformer instance. Each known category gets a stable integer code; an extra reserved slot is held for the `"__NAN__"` sentinel that future inference batches may need.

### 3.2 What `transform` does at predict time

`DataTransformer.transform` re-uses the pinned category list, so the integer code assigned to each known category at predict time is identical to the one assigned at fit time — regardless of which values happen to appear in the predict-time DataFrame.

```python
import pandas as pd
import numpy as np
from flaml.automl.data import DataTransformer
from flaml.automl.task.factory import task_factory

rng = np.random.RandomState(0)
fit_df = pd.DataFrame(
    {
        "a": rng.randn(120),
        "gender": rng.choice(["M", "F"], 120),
    }
)
fit_y = pd.Series(rng.randn(120))

transformer = DataTransformer()
transformer.fit_transform(
    fit_df.copy(), fit_y, task_factory("regression", fit_df, fit_y)
)

# Predict-time DataFrame contains only the "M" category
predict_df = pd.DataFrame({"a": np.zeros(20), "gender": ["M"] * 20})
X_pred = transformer.transform(predict_df.copy())

# The integer code assigned to "M" is the same as at fit time — no drift.
```

### 3.3 Unseen categories

If predict-time data contains values that were not seen at fit time, FLAML now emits a `UserWarning` and encodes those rows as the `"__NAN__"` sentinel. Consume the warning category in your serving code and decide how to react (log, alert, reject the batch, etc.).

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    predict_df = pd.DataFrame({"a": np.zeros(5), "gender": ["M", "F", "X", "Y", "M"]})
    X_pred = transformer.transform(predict_df.copy())

unseen = [
    w
    for w in caught
    if issubclass(w.category, UserWarning) and "unseen at fit time" in str(w.message)
]
if unseen:
    # In production this is where you raise an alert / reject the batch /
    # fall back to a default category.
    print(unseen[0].message)
```

The model still produces a prediction for rows mapped to `"__NAN__"`, but those predictions are unreliable: the model was not trained on that category. Treat unseen-category warnings as a deployment health signal, not background noise.

### 3.4 Recommended workflow

If your production data may legitimately introduce new categorical values over time (a new product code, a new geography), pin the category list upstream of FLAML using sklearn's `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` or an equivalent component, and pass the encoded DataFrame into `AutoML.fit`. This makes encoding consistency an explicit part of your pipeline rather than relying on FLAML's defensive sentinel.

## 4. Ensemble component access

This is the canonical pattern for issue #1136 (closed by PR #1558).

When `AutoML.fit(..., ensemble=True)` is used, `automl.model` is a sklearn `StackingClassifier`/`StackingRegressor` whose `estimators_` were trained on data that has already passed through FLAML's task-level preprocessing. As a result, calling `automl.model.estimators_[i].predict(X_raw)` directly raises a confusing error from the underlying estimator (`LightGBM: train and valid dataset categorical_feature do not match`, `XGBoost: DataFrame.dtypes must be int/float/bool/category`, etc.).

The fix is to preprocess raw input via `automl.preprocess(X)` first:

```python
automl = AutoML()
automl.fit(
    X,
    y,
    task="classification",
    ensemble=True,
    estimator_list=["lgbm", "xgboost", "rf"],
    time_budget=10,
)

# Direct call on raw input — DOES NOT WORK:
# automl.model.estimators_[0].predict(X)   # raises ValueError on categorical input

# Correct pattern — preprocess first:
X_pre = automl.preprocess(X)
component_preds = [est.predict(X_pre) for est in automl.model.estimators_]
```

This is intentionally a two-step process. `automl.predict(X)` does both steps for you; component-level access is for cases where you need per-component scores, predictions, or feature attributions, in which case you handle the preprocessing call site explicitly.

## 5. Sample weights and cost-sensitive learning

Pass `sample_weight` at fit time to perform cost-sensitive training. FLAML honors the weight inside both the holdout and CV evaluation paths.

```python
import numpy as np
from flaml import AutoML

# 5x weight on the minority (positive) class
sample_weight = np.where(y_train == 1, 5.0, 1.0)
automl = AutoML()
automl.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    task="classification",
    time_budget=5,
)
```

Compatibility notes:

- `split_type="time"` + `sample_weight` works correctly after PR #1554 (closes #887).
- `predict()` does not take a `sample_weight` argument — weights apply only during training. For weighted evaluation on new data, compute the metric outside FLAML (e.g., `sklearn.metrics.f1_score(y_test, automl.predict(X_test), sample_weight=test_weight)`).
- `class_weight` is passed through to the underlying estimator unchanged if your chosen estimator accepts it (e.g., LightGBM, XGBoost sklearn API).

For severe class imbalance, see also [issue #1200](https://github.com/microsoft/FLAML/issues/1200) on adding a `resampler=` integration. The current recommendation is to apply SMOTE (or your resampler of choice) upstream of `AutoML.fit`; see the imbalanced-learn documentation for the canonical pattern.

## 6. Multi-output regression

For multi-target regression today, wrap a fresh `AutoML(task="regression", ...)` per target with sklearn's `MultiOutputRegressor` or `RegressorChain`:

```python
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from flaml import AutoML

X, y = make_regression(n_samples=200, n_targets=3, random_state=42)
model = MultiOutputRegressor(
    AutoML(task="regression", time_budget=1, estimator_list=["lgbm"])
)
model.fit(X[:150], y[:150])
preds = model.predict(X[150:])
```

Known limitation: passing `X_val` and `y_val` through `MultiOutputRegressor` does not flow into each inner `AutoML.fit` ([#1115](https://github.com/microsoft/FLAML/issues/1115)). Workaround: concatenate train + val into a single dataset and use a custom splitter, or call `AutoML` per target manually.

Native multi-target support is being tracked in [#1301](https://github.com/microsoft/FLAML/issues/1301); when it lands, prefer the native path.

## 7. Versioning and reproducibility

Two pieces matter for reproducible predictions in production:

1. **The FLAML `random_seed`** — pass it via `automl.fit(..., seed=N)` to make the search deterministic. The 2026-05 reproducibility audit (closes #1540) standardized how every audited estimator honors this seed; see #1541 (SGD), #1546 (LRL1), #1547 (RandomForest/ExtraTrees), #1549 (XGBoost sklearn), #1551 (XGBoost native), #1552 (LRL2), #1556 (LRL `penalty`/`n_jobs` deprecations).
1. **Pinned library versions** — `flaml`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pandas`, and `numpy` should all be pinned in your serving environment. Mismatches between training-time and serving-time versions of any of these can produce silently divergent predictions even with the same `random_seed`.

A minimal training-environment `requirements.txt` snippet:

```text
flaml==2.6.0
scikit-learn==1.8.0
lightgbm>=4.0,<5.0
xgboost>=2.0,<3.0
pandas>=2.0,<3.0
numpy>=1.26,<3.0
```

When you ship a model, ship the corresponding `requirements.txt` (or `conda-lock.yml`) alongside the pickle/MLflow artifact and use the same versions to instantiate the serving environment.

## 8. Common gotchas — quick reference

| Symptom                                                                                     | Cause                                                                          | Fix                                                                                    |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `predict()` on a DataFrame returns different codes than at fit time                         | Predict-time DataFrame had a different subset of categorical values            | Use FLAML ≥ post-#1561; or pin categories upstream via `OrdinalEncoder`                |
| `UserWarning: Column '...' contains values unseen at fit time`                              | New category at inference time                                                 | Decide policy: alert, retrain, or fall back to default                                 |
| `automl.model.estimators_[i].predict(X)` raises on categorical input                        | Component model expects preprocessed input                                     | Call `automl.preprocess(X)` first                                                      |
| `MultiOutputRegressor(AutoML(...))` ignores `X_val`                                         | Per-target inner `AutoML.fit` doesn't see validation kwargs                    | Use a custom splitter on the concatenated dataset                                      |
| `AttributeError: 'AutoMLState' has no attribute 'sample_weight_all'` on `retrain_full=True` | Pre-#1554 bug                                                                  | Upgrade FLAML past #1554                                                               |
| MLflow autolog'd model loads as an unfitted `Pipeline`                                      | Older example assumed an autolog artifact path that no longer reliably reloads | Use the explicit `mlflow.sklearn.log_model(automl, artifact_path=...)` pattern in §1.2 |

See also: [Best-Practices](../Best-Practices), [Task-Oriented AutoML](Task-Oriented-AutoML), [FAQ](../FAQ).
