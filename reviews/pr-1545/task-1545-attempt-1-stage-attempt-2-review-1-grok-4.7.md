---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 1
timestamp: '2026-10-17T07:11:19Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 9c8f1b7f8ec196ed40d3d322b3065d83926d95752469a138d4e754f24fbc80b8
---

## Review Summary

- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-1-grok-4.7.md
- **Issues Found**: 8
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Inspected MLflow gating, `only_history`, autolog save/restore, and best-run indexing in `flaml/automl/automl.py` (2584–2588, 2766–2770, 3274–3337) and `flaml/fabric/mlflow.py` (268–288, 375–381, 764–844).
- [x] Inspected AutoFE fit scope and config parsing in `flaml/automl/ml.py` (`get_val_loss` / `train_estimator`), `flaml/fabric/autofe.py` (`fit` uses `TimeSeriesDataset.train_data`; line 166 rejects non-string configs), and `flaml/automl/automl.py` 2465–2472.
- [x] Inspected forecast smoothing init and the DataFrame branch in `flaml/automl/time_series/ts_model.py` (742–754, estimated `SimpleExpSmoothing` init) and latest-observation CV in `flaml/automl/time_series/ts_data.py` `cv_train_val_sets`.
- [x] Checked public defaults: `model_history` uses `is_fabric_runtime()` (`flaml/automl/automl.py` settings), Fabric detection does not import Spark (`flaml/fabric/__init__.py`), weighted MAPE passes `sample_weight` (`flaml/automl/ml.py`), and binary metrics are rejected for multiclass (`BINARY_ONLY_METRICS`).
- [x] Driver-reported pytest (261 passed) and the 2.7.0 build were not treated as proof. Those runs do not cover autolog-without-active-run retraining, failed-trial MLflow indexing, or regression visualization ranges.

## Issues

### Issue 1: Autolog without an active run disables final retraining

- **Severity**: High
- **File**: flaml/automl/automl.py
- **Line(s)**: 2584–2588, 2758–2770
- **Description**: If MLflow autolog is enabled and there is no active run, an integration is created with `only_history=False` and `parent_run_id is None`. Fit then forces `retrain_final=False`. Default `retrain_full=True` would otherwise retrain on the full training set for holdout, CV, and `max_iter==1`. The returned estimator stays the trial model, which was fit on a split or sample. Fabric history-only mode does not hit this because `only_history` is true. Logging policy is changing the trained model.
- **Risk**: Enabling autolog silently yields a different, weaker model than the same fit without autolog. The `max_iter==1` full-data fast path is also skipped.
- **Suggested Fix**: Do not couple `retrain_final` to MLflow parent-run presence. Log the retrained model only when a parent run exists, and always honor `retrain_full`.

### Issue 2: MLflow best-run index skips failed trials

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 844–850
- **Description**: `log_automl` uses `self.manual_run_ids[automl._best_iteration]`. Sequential search sets `_best_iteration` to `_track_iter` for every loop iteration, but `record_state` runs only when `analysis.trials[-1].last_result` is present (`flaml/automl/automl.py` 3274–3337). A failed trial consumes an iteration and appends nothing. After any earlier failure, the index is either out of range or points at a later run. `log_tune` (line 550) uses the same assumption against `analysis.best_iteration`.
- **Risk**: `fit()` raises `IndexError` after a successful search, or tags and copies the wrong child run as best, once an active run or autolog makes `manual_log` true.
- **Suggested Fix**: Store the MLflow run id on the search state when the trial is recorded, and look up the best trial by that id. Do not use the loop index as a dense run-list index.

### Issue 3: History-only integration still activates an MLflow experiment

- **Severity**: Medium
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 375–381
- **Description**: Fabric fit without an active run or autolog still constructs `MLflowIntegration` before `only_history` is set. With no active run, `__init__` calls `mlflow.set_experiment` (`flaml_default_experiment` or the notebook name) and `mlflow.get_experiment`. `only_history` is never read in this module, so it cannot skip that. `record_state` later returns before creating runs, but the process-global active experiment has already changed. In Fabric, construction also calls `get_mlflow_env_config` and notebook artifact lookup.
- **Risk**: A visualization-only fit creates or selects an experiment and sends later user `start_run()` calls into `flaml_default_experiment`. It can also contact Fabric services when the user did not ask to log.
- **Suggested Fix**: If neither an active run nor autolog is present, collect in-memory `infos` without constructing a client or calling `set_experiment`. Set that mode before any tracking-server call.

### Issue 4: Tune wrapper overwrites caller autolog and does not restore it

- **Severity**: Medium
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 268–288
- **Description**: `tune.run` wraps the evaluation function whenever Fabric, an active run, or autolog is detected (`flaml/tune/tune.py` 561–568). The wrapper calls `mlflow.autolog()` or `mlflow.autolog(disable=True)` on every trial and never restores the previous global or flavor settings. AutoML saves and restores those settings in `resume_mlflow`; tune does not call it.
- **Risk**: A Fabric or autolog `tune.run` leaves process-global autolog enabled or disabled, including `log_models`, after the call returns. That violates the requirement to preserve caller autolog state.
- **Suggested Fix**: Snapshot global and flavor autolog configs before wrapping, and restore them in a `finally` around `tune.run`, including on exceptions.

### Issue 5: SimpleForecaster DataFrame fit uses an unbound dataset

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 744–754
- **Description**: `joint_preprocess` sets `data` only in the `TimeSeriesDataset` branch. The DataFrame branch calls `_join` and then unconditionally reads `data.time_col` and `data.target_names`. AutoML passes a `TimeSeriesDataset`, but the else branch is the DataFrame API used by `SeasonalNaive`, `Naive`, `SeasonalAverage`, and `Average`.
- **Risk**: Direct `fit` on a DataFrame raises `UnboundLocalError` instead of training. Estimated smoothing initialization is not reached on that path.
- **Suggested Fix**: In the else branch, set `self.time_col` and `self.target_names` from the joined frame, and do not reference `data`.

### Issue 6: Visualization treats every validation loss as a unit-interval score

- **Severity**: Medium
- **File**: flaml/fabric/visualization.py
- **Line(s)**: 40–41, 547–549
- **Description**: `_get_aml_df` renames `validation_loss` to `score` and sets `score = 1 - loss`. Slice plots then fix the y-axis to `[0, 1]`. That inversion matches classification losses that are already `1 - metric`. It is wrong for RMSE, MAE, MSE, log loss, and MAPE, whose losses are unbounded. MAPE is the forecasting default.
- **Risk**: Regression and forecast plots clip or invert the metric. fANOVA importance is computed on this distorted score, so importance rankings can be misleading.
- **Suggested Fix**: Plot the stored loss, or convert with the metric's known direction. Do not hardcode `[0, 1]` unless the metric is bounded.

### Issue 7: Featurization validator accepts dicts that cannot be parsed

- **Severity**: Low
- **File**: flaml/automl/automl.py
- **Line(s)**: 2465–2467
- **Description**: Fit accepts a dict and tells the user that a dict is valid. `parse_autofe_config` only accepts `"off"`, `"auto"`, and `"force"`, then raises `ValueError` at `flaml/fabric/autofe.py` line 166. The README says custom dictionaries are not supported. The failure happens after data validation, while building `SearchState`.
- **Risk**: Callers who follow the validator message pass a dict and only fail later inside fit.
- **Suggested Fix**: Reject dicts in the validator with the same unsupported-config error, until custom configs are implemented.

### Issue 8: time_it executes arbitrary code strings

- **Severity**: Low
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 125
- **Description**: `time_it` calls `exec` on a non-callable argument. Library call sites use it as a decorator, but the function is importable and documented as accepting a code string. `test/fabric/test_mlflow_coverage.py` invokes `time_it("x = 1 + 1")`.
- **Risk**: Any future caller that passes user-controlled text gets code execution in the training process.
- **Suggested Fix**: Remove the string-execution mode and keep the decorator only.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 2

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 3

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 4

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 5

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 6

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 7

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 8

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

## Driver assessment after stage 1, attempt 2

The original response and verdict are preserved. Actual reviewer identity was
`grok-4.7`; the transport audit found 87 search and 62 view calls, no disallowed
tools or earlier/sibling feedback access, and no source drift. Stage 1 advances
**with unresolved findings** at High's two-attempt limit; it is not clean.

| Issue | Disposition                                                                | Evidence and resolution                                                                                                                                                                                                                                                                                                              |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Pre-existing, unresolved outside this diff                                 | Public base `17d293b01edc49734ccdb79ef04b68d0da72fab8`, `flaml/automl/automl.py:2635-2639`, already has the same no-parent guard forcing `retrain_final=False`. No unrelated training-policy change was made.                                                                                                                        |
| 2     | Pre-existing, unresolved outside this diff                                 | The public base indexes `manual_run_ids[analysis.best_iteration]` at `flaml/fabric/mlflow.py:499` and `manual_run_ids[automl._best_iteration]` at line 790. The dense-index assumption is inherited; the reviewer-described failed-trial sequence has not been independently runtime-reproduced in this review.                      |
| 3     | Fixed before stage 2                                                       | `only_history` is selected before construction can contact services. History-only integration initializes local state without a client, experiment selection/lookup, notebook lookup, or autolog mutation. Real AutoML and Tune regressions forbid those calls while preserving trial results/history.                               |
| 4     | New history-only exposure fixed; inherited active-logging concern retained | `_mlflow_wrapper` is AST-identical to public base lines 250-275. The new history-only path now returns the original evaluation function without invoking this wrapper or changing autolog. Broader active-run/autolog wrapper semantics remain an explicit inherited concern, not claimed fixed.                                     |
| 5     | Pre-existing, unresolved outside this diff                                 | `SimpleForecaster` is AST-identical to public base lines 684-736, as independently checked for the first attempt. The supported AutoML path supplies `TimeSeriesDataset`; no unrelated direct-DataFrame behavior was changed.                                                                                                        |
| 6     | Axis defect fixed; ranking claim not substantiated                         | Slice ranges are no longer `[0,1]`, and Tune is labeled with its actual objective. AutoML's explicitly labeled `1-loss` is an affine reflection, which does not itself distort variance-based fANOVA importance. The existing score representation is retained rather than treating the unproved ranking claim as a separate defect. |
| 7     | Fixed before stage 2                                                       | Unsupported dictionaries now raise a consistent error before task data validation. A regression forbids entering validation with this input.                                                                                                                                                                                         |
| 8     | Inherited intentional API, no introduced vulnerability established         | Public base `time_it` at `flaml/fabric/mlflow.py:80-117` already documents and implements code-string execution. An exact function diff contains only the added timing-log call. No new untrusted-input call path was identified; removing a documented public mode would be an unrelated compatibility break.                       |

The fixes passed the 282-test targeted suite and the wheel/source build recorded
in the sibling GPT artifact. Original findings, including low-priority and
pre-existing concerns, remain visible. These dispositions are driver judgments,
not changes to the reviewer's `ISSUES_FOUND` verdict.
