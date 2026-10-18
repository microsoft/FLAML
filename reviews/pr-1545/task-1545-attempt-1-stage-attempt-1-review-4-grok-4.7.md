---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 4
timestamp: '2026-10-17T13:15:50Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: c6946c2efe8143368b2e73c8f7b69fd0d0f2289c579914b7a431d4ee92931262
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-4-grok-4.7.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `flaml/automl/data.py` `add_time_idx_col` (about lines 266–530). DST uses wall-clock diffs, not UTC gaps, so a daily tz-aware series is not forced onto UTC. Not filed.
- [x] Read `flaml/automl/pipeline.py` `AutoMLPipeline.predict` (lines 123–148) against `generic_task.preprocess` list handling (lines 1132–1148). Flat lists are one row there, so the row-count check matches that contract. Not filed.
- [x] Read `flaml/automl/time_series/ts_data.py` `TimeSeriesDataset.__init__` (lines 44–58) and `DataTransformerTS.transform` (lines 502–541). Sort reorders the frame together; the transformer drops columns, not rows, so the `iloc` split in `time_series_task.py` lines 182–187 keeps row alignment.
- [x] Read `flaml/fabric/mlflow.py` `MLflowIntegration.__init__` (lines 322–370) and `record_state` (about lines 684–792). `only_history` returns before client setup; `autolog` and `manual_log` stay false, so history-only does not log models.
- [x] Read `flaml/automl/state.py` `sanitize` (line 386) and list `setdefault` (lines 163–167). `sanitize` copies before mutation. Not filed.
- [x] Read `flaml/automl/spark/metrics.py` lines 16–22 and 160–182, `flaml/automl/task/time_series_task.py` lines 285–291, `flaml/automl/time_series/ts_model.py` lines 149–161 and 737–754, `flaml/fabric/visualization.py` lines 405–419, and `benchmark/pmlb/runbenchmark.py` lines 121–125.
- [x] Driver-reported pytest (414 passed, 2 skips; Spark dtype contracts 12 passed) does not execute these paths. This review did not run tests. Pristine public base was not available, so regression-versus-carried-forward is not proven for every line.

## Issues

### Issue 1: Spark micro_f1 and macro_f1 use the wrong evaluator

- **Severity**: High
- **File**: flaml/automl/spark/metrics.py
- **Line(s)**: 168-182
- **Description**: `spark_metric_name_dict` advertises `micro_f1` and `macro_f1` for multiclass, and `AutoML.fit` recommends them when binary-only metrics are rejected. Both branches construct `MultilabelClassificationEvaluator` on the raw label and prediction columns. Those columns are scalar class ids after `to_spark()`, while that evaluator requires arrays of doubles. `f1` correctly uses `MulticlassClassificationEvaluator`. `macro_f1` also uses `f1MeasureByLabel`, which is one label's F1, not the unweighted mean across classes. Unlike `ndcg`, these branches never call `_process_df`, and wrapping a class id in a one-element array would still not be a multi-hot label.
- **Risk**: `use_spark=True` with `metric="micro_f1"` or `"macro_f1"` raises a Spark type error during evaluation, or, if a future Spark build accepts the columns, optimizes the wrong statistic. The exception is not `ValueError`, so `get_val_loss` will not turn it into an infinite trial loss.
- **Suggested Fix**: Score single-label micro-F1 with `MulticlassClassificationEvaluator` accuracy (micro-F1 equals accuracy for one label per row). Compute macro-F1 as the unweighted mean of per-class `fMeasureByLabel` on that same multiclass evaluator. Do not pass scalar columns to `MultilabelClassificationEvaluator`.

### Issue 2: Panel holdout binds frequency as time_idx

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 285-291
- **Description**: `TimeSeriesDataset.__init__` takes `(train_data, time_col, target_names, time_idx="time_idx", test_data=None)`. The `TS_FORECASTPANEL` holdout call passes `ts_data.frequency` in the `time_idx` position. Other constructors in this file pass `time_idx=` by keyword. Frequency is a pandas alias such as `"D"` or `"B"`, not the integer index column. The constructor then infers `.frequency` from the timestamps, so that field can still look right while `.time_idx` is wrong.
- **Risk**: Later `add_test_data`, AutoFE transform, and any `DataTransformerTS` built from this dataset use the frequency string as the index column name. The real `time_idx` column is no longer protected from categorical encoding or drop-as-id logic, and column lookup can raise `KeyError`. Group alignment for the panel split itself still uses the literal `"time_idx"` column, so the bug is delayed rather than immediate.
- **Suggested Fix**: Call `TimeSeriesDataset(..., time_idx=ts_data.time_idx, test_data=test_data)`. Do not pass frequency positionally.

### Issue 3: DataFrame fit of simple forecasters reads an unbound dataset

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 749-753
- **Description**: `SimpleForecaster.joint_preprocess` sets `data` only in the `TimeSeriesDataset` branch. The `else` branch builds `train_df` via `_join`, then both branches execute `self.time_col = data.time_col` and `self.target_names = data.target_names`. AutoML's normal forecast path passes a `TimeSeriesDataset`, so tests can miss this. Direct `fit` on a frame with a `ds` column, which `_join` accepts, always raises `NameError` before training.
- **Risk**: `Naive`, `SeasonalNaive`, and other `SimpleForecaster` subclasses cannot be fit from a dataframe even though the else branch exists for that input. Refit or custom call paths that do not wrap a `TimeSeriesDataset` fail before a model exists.
- **Suggested Fix**: In the else branch, set `self.time_col` and `self.target_names` from the joined frame (`TS_TIMESTAMP_COL` and `TS_VALUE_COL`) and do not read `data`.

### Issue 4: Spark grouped nDCG reorders counts by the wrong key

- **Severity**: Medium
- **File**: flaml/automl/spark/metrics.py
- **Line(s)**: 16-22, 203
- **Description**: For a NumPy array, `np.unique(..., return_index=True, return_counts=True)` aligns counts with sorted keys, and `c[np.argsort(i)]` restores first-appearance order. The pandas-on-Spark branch does not. `drop_duplicates().index.values` are index labels, while `value_counts().sort_index()` is ordered by group value. `argsort` of those labels does not restore first-appearance sizes. `ndcg@k` then slices `y_true[psum:psum+c]` with those sizes.
- **Risk**: Grouped Spark nDCG mixes rows from different groups whenever first-seen order differs from sorted group-key order, or when the series index is not a positional `RangeIndex`. The reported loss can rank the wrong model.
- **Suggested Fix**: Compute counts in first-appearance order, using positional indices (`reset_index(drop=True)` or `drop_duplicates` on a positional range) and do not combine value-sorted counts with label `argsort`.

### Issue 5: Forecast score swaps R² arguments

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 158-161
- **Description**: `sklearn.metrics.r2_score` is `(y_true, y_pred)`. The default branch calls `r2_score(y_pred, y_val)`. The branch immediately above correctly passes prediction then truth into `metric_loss_score`, whose signature is `(metric_name, y_predict, y_true)`. `AutoML.score` delegates to this method for forecast estimators.
- **Risk**: Public forecast `score()` returns R² of the true series against the forecast, not the reverse. Biased or scaled forecasts get a plausible but wrong number, so model comparison and tests that use `score` can accept a bad fit.
- **Suggested Fix**: Return `r2_score(y_val, y_pred)`.

### Issue 6: Multi-parameter contour plot drops trial markers

- **Severity**: Low
- **File**: flaml/fabric/visualization.py
- **Line(s)**: 417-419
- **Description**: `_get_contour_scatter` returns both a contour and a scatter of trials. The two-parameter branch adds both traces (line 405). The subplot branch unpacks `scatter` and adds only `contour`. Axis updates use the reverse combination, so titles land on the same cell as the matching contour; that part is consistent. The missing scatter is not.
- **Risk**: For three or more hyperparameters, contour cells show an interpolated surface with no evaluated points. Sparse trials look like a smooth response and hide failed or duplicate samples.
- **Suggested Fix**: `fig.add_trace(scatter, col=param_loc[combo[0]], row=param_loc[combo[1]])` immediately after the contour trace.

### Issue 7: H2O classification score is not a scalar accuracy

- **Severity**: Low
- **File**: benchmark/pmlb/runbenchmark.py
- **Line(s)**: 121-125
- **Description**: `runner_h2o` uses `perf.r2()` for `metric == "r2"` and otherwise `perf.accuracy()`. H2O's `accuracy()` returns threshold/accuracy pairs, not a float. `runner_flaml` stores `accuracy_score`, a float. The `__main__` filter currently runs regression only, so this branch is not taken by the default entry point. `flatten_results` still treats `score` as a measurement.
- **Risk**: Any classification call of `runner_h2o` writes a nested list into the score column and cannot be compared with FLAML accuracy. Removing the regression-only filter would silently publish invalid benchmark rows.
- **Suggested Fix**: Read the scalar accuracy from the H2O metric table (the accuracy at the maximizing threshold), and reject non-numeric scores before flattening.

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

## Driver assessment after stage 4, attempt 1

The original response and outcome remain intact. Actual model `grok-4.7`
used 91 view and 93 search calls; no disallowed tools, feedback access, or
source drift were found.

| Issue | Disposition                                        | Evidence and resolution                                                                                                                                                                                                                                              |
| ----- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Independently verified inherited concern, retained | `spark_metric_loss_score` is AST-identical to public base lines 50-230. The micro/macro evaluator behavior predates this PR; no new evaluator implementation is claimed.                                                                                             |
| 2     | Independently verified inherited concern, retained | `TimeSeriesTask.prepare_data`, including the positional panel constructor, is AST-identical to public base lines 222-322 as verified in the preceding stage.                                                                                                         |
| 3     | Independently verified inherited concern, retained | The same unchanged `SimpleForecaster` direct-frame branch is recorded in earlier assessments. The supported AutoML path uses a dataset; an unrelated direct-input redesign was not made.                                                                             |
| 4     | Independently verified inherited concern, retained | `ps_group_counts` is AST-identical to public base lines 16-22. Its first-appearance ordering defect remains an explicit existing issue.                                                                                                                              |
| 5     | Independently verified inherited concern, retained | The reversed R2 call was independently checked in stage 1 and is not changed by this PR.                                                                                                                                                                             |
| 6     | Fixed                                              | Every non-diagonal multi-parameter contour cell now includes its evaluated-trial scatter as well as the interpolated surface. A three-parameter figure has six matching contour/marker pairs.                                                                        |
| 7     | Fixed                                              | H2O classification extracts a scalar maximum accuracy from threshold/value pairs. Report flattening rejects successful records without a finite numeric scalar instead of publishing list-valued measurements. A successful mocked H2O call returns scalar accuracy. |

The 430-test suite, existing AutoFE integrations, lower-bound Optuna checks,
pandas 3 checks, build, and lint recorded in the sibling GPT artifact pass.
Driver dispositions do not change this reviewer's original dirty verdict.
