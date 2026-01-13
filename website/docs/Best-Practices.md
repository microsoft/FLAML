````markdown
# Best Practices

This page collects practical guidance for using FLAML effectively across common tasks.

## General tips

- Start simple: set `task`, `time_budget`, and keep `metric="auto"` unless you have a strong reason to override.
- Prefer correct splits: ensure your evaluation strategy matches your data (time series vs i.i.d., grouped data, etc.).
- Keep estimator lists explicit when debugging: start with a small `estimator_list` and expand.
- Use built-in discovery helpers to avoid stale hardcoded lists:

```python
from flaml import AutoML
from flaml.automl.task.factory import task_factory

automl = AutoML()
print("Built-in sklearn metrics:", sorted(automl.supported_metrics[0]))
print("classification estimators:", sorted(task_factory("classification").estimators.keys()))
```

## Classification

- **Metric**: for binary classification, `metric="roc_auc"` is common; for multiclass, `metric="log_loss"` is often robust.
- **Imbalanced data**:
  - pass `sample_weight` to `AutoML.fit()`;
  - consider setting class weights via `custom_hp` / `fit_kwargs_by_estimator` for specific estimators (see [FAQ](FAQ)).
- **Probability vs label metrics**: use `roc_auc` / `log_loss` when you care about calibrated probabilities.

## Regression

- **Default metric**: `metric="r2"` (minimizes `1 - r2`).
- If your target scale matters (e.g., dollar error), consider `mae`/`rmse`.

## Learning to rank

- Use `task="rank"` with group information (`groups` / `groups_val`) so metrics like `ndcg` and `ndcg@k` are meaningful.
- If you pass `metric="ndcg@10"`, also pass `groups` so FLAML can compute group-aware NDCG.

## Time series forecasting

- Use time-aware splitting. For holdout validation, set `eval_method="holdout"` and use a time-ordered dataset.
- Prefer supplying a DataFrame with a clear time column when possible.
- Optional time-series estimators depend on optional dependencies. To list what is available in your environment:

```python
from flaml.automl.task.factory import task_factory

estimators_all = task_factory("forecast").estimators.keys()
print(sorted(estimators_all))
```

## NLP (Transformers)

- Install the optional dependency: `pip install "flaml[hf]"`.
- When you provide a custom metric, ensure it returns `(metric_to_minimize, metrics_to_log)` with stable keys.

## Speed, stability, and tricky settings

- **Time budget vs convergence**: if you see warnings about not all estimators converging, increase `time_budget` or reduce `estimator_list`.
- **Memory pressure / OOM**:
  - set `free_mem_ratio` (e.g., `0.2`) to keep free memory above a threshold;
  - set `model_history=False` to reduce stored artifacts;
- **Reproducibility**: set `seed` and keep `n_jobs` fixed; expect some runtime variance.

## Persisting models (recommended)

Use `AutoML.pickle()` / `AutoML.load_pickle()` for persistence:

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=60)

automl.pickle("automl.pkl")
automl2 = AutoML.load_pickle("automl.pkl")
```

See also: [Task-Oriented AutoML](Use-Cases/Task-Oriented-AutoML) and [FAQ](FAQ).

````
