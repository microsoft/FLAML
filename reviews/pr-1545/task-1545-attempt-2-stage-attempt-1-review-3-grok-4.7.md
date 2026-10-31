---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 1
round: 3
timestamp: '2026-10-31T16:46:00Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: e03302315983879b8db4c228560ffa0beeb6643e2e98e9cbc073c94afe3e1668
---

## Review Summary

- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-1-review-3-grok-4.7.md
- **Issues Found**: 8
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read forecast CV/holdout in `flaml/automl/task/time_series_task.py` (lines 252–326, 401–458, 520–536) and `TimeSeriesDataset.__init__` in `flaml/automl/time_series/ts_data.py` (lines 46–58, 235–248).
- [x] Read budget wiring in `flaml/automl/state.py` (lines 328–344), `flaml/automl/automl.py` (lines 482, 1360–1390, 2443), and the guarded sibling in `flaml/automl/task/generic_task.py` (line 1227).
- [x] Read `add_time_idx_col` (`flaml/automl/data.py` lines 264–336), Spark conversion (lines 1185–1226), and `get_val_loss` (`flaml/automl/ml.py` lines 632–658).
- [x] Read statsmodels predict/fit in `flaml/automl/time_series/ts_model.py` (lines 340–430, 690–736) and TFT predict in `flaml/automl/time_series/tft.py` (lines 192–211).
- [ ] Tests were not executed here. Driver evidence (797 non-spark, 24 forecast-filter, 68 pandas) does not show unlimited-budget forecast CV, panel holdout, or a biased Spark timestamp sample.

## Issues

### Issue 1: Forecast CV crashes when the trial budget is None

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 430, 458
- **Description**: `evaluate_model_CV` always does `budget / n` and later `time.time() - start_time >= budget`. The default constructor budget is `-1` (`automl.py` line 482). That selects CV (`_decide_eval_method`, lines 1384–1388) and `state.py` then passes `train_time_limit`, which defaults to None. `generic_task.py` already uses `budget and budget / n`. The forecasting override does not.
- **Risk**: `AutoML().fit(..., task="ts_forecast")` with no time budget, or any forecast CV trial under an unlimited budget, raises `TypeError` on the first fold. A fix that only guards the division still fails at line 458.
- **Suggested Fix**: Treat None as unlimited: skip the per-fold division and the elapsed-budget break, matching the generic task. Add a regression with `time_budget=-1` and `eval_method="cv"`.

### Issue 2: Panel CV has no step size and splits the flat frame

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 323–326, 432
- **Description**: The univariate branch sets `state.kf.step_size`. The `TS_FORECASTPANEL` branch creates a `TimeSeriesSplit` and does not. `evaluate_model_CV` always calls `cv_train_val_sets(..., kf.step_size)`. That helper also slices the concatenated frame by row count, not within each `group_ids` series.
- **Risk**: With a numeric budget, panel CV raises `AttributeError` before any trial score. After a step size is added, folds can still mix series and leak future rows across groups.
- **Suggested Fix**: Set an explicit step size and split inside each series, keeping each series' latest observations in its validation slice. Do not read `step_size` unless it was assigned.

### Issue 3: Panel holdout stores frequency in time_idx and casts away nulls

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 261–294
- **Description**: `TimeSeriesDataset.__init__` takes `(train_data, time_col, target_names, time_idx="time_idx", test_data=None)`. The panel holdout call passes `ts_data.frequency` in the `time_idx` position. Other constructors in this file pass `time_idx=` by keyword. The same block rebases `time_idx` on a column subset and calls `.astype("int")`.
- **Risk**: The dataset is built with `time_idx` equal to an alias such as `"D"`. Later `enrich_dataset`, pipeline export, and `add_test_data` propagate that name and miss the integer column. Nullable `Int64` indexes from `add_time_idx_col` raise `IntCastingNaNError` on `pd.NA`. Without copy-on-write, the subtraction also mutates the stored training frame.
- **Suggested Fix**: Call `TimeSeriesDataset(..., time_idx=ts_data.time_idx, test_data=test_data)`. Copy before rebasing, and use a nullable integer cast that preserves `pd.NA`.

### Issue 4: Irregular time_idx assignment is label-based and can invent steps

- **Severity**: Medium
- **File**: flaml/automl/data.py
- **Line(s)**: 292, 333
- **Description**: Calendar branches build a `Series` aligned to `X.index`. The fewer-than-two-dates path and the irregular-gap path assign with `time_idx.loc[valid]`. On a non-unique index, boolean `.loc` is label-aligned and can raise or write one value onto every row that shares a label. `dates.get_indexer(...)` returns `-1` for an unmatched timestamp, and `steps[-1]` then stores the last step instead of failing.
- **Risk**: Repeated panel indexes or a dtype/timezone mismatch either abort preprocessing or silently attach the wrong step. Forecast cutoffs and TFT lags then use that index.
- **Suggested Fix**: Assign positionally, as the calendar branch does. If any indexer is negative, raise. Cover irregular dates with a duplicate index and a missing timestamp.

### Issue 5: Spark timestamp and double conversion can null values the sample never saw

- **Severity**: Medium
- **File**: flaml/automl/data.py
- **Line(s)**: 1148–1168, 1185–1208, 1220–1223
- **Description**: Integer inference is rechecked on the full column and raises on incompatible values. Timestamp and double decisions use only the sample (`convert_threshold`), then `try_cast` nulls every non-matching row. The pandas path validates datetime and numeric decisions on the full column before writing.
- **Risk**: A biased sample can classify a mostly non-date or mixed string column as timestamp or double and silently drop the values Spark did not sample. Forecasting and numeric features then train on nulls.
- **Suggested Fix**: Apply the same full-column threshold used by pandas before casting. If the full column fails, leave the original strings. Do not null rows that failed the chosen type.

### Issue 6: Holt-Winters searches seasonal period and Box-Cox but does not fit them

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 690–736
- **Description**: The search space includes `seasonal_periods` and `use_boxcox`. `fit` uses `seasonal_periods` only to disable seasonality on short series. The `ExponentialSmoothing` call passes trend, seasonality, and `initialization_method="estimated"`, but not `seasonal_periods` or `use_boxcox`. The training index is built with `to_datetime` and usually has no frequency.
- **Risk**: The default seasonal configuration can raise `ValueError` inside statsmodels. That `ValueError` is not the AutoFE exception swallowed in `state.py`, so one Holt-Winters trial can abort the search. If inference succeeds, the tuned period is not the fitted period.
- **Suggested Fix**: Pass `seasonal_periods` when `seasonal` is set and pass `use_boxcox`. If the index has no frequency and no period, fail that config with a trial-level error rather than an uncaught search error.

### Issue 7: Duplicate-row cleanup always asserts

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 530–536
- **Description**: After `drop_duplicates()`, the code asserts `X[[X.columns[0]]].duplicated() is None`. `duplicated()` returns a boolean Series, never None, so the assertion is false whenever this branch runs. It also checks column 0 rather than `time_col`.
- **Risk**: Any fully duplicate input row raises `AssertionError` after the cleanup warning, including rows whose timestamps are unique once exact duplicates are removed. Callers cannot recover the cleaned frame.
- **Suggested Fix**: Assert `X[time_col].duplicated().any()` is false, and raise a `ValueError` that names the timestamp column. Return the deduplicated frame when only exact copies were present.

### Issue 8: Validation-transform latency divides by an empty partition

- **Severity**: Medium
- **File**: flaml/automl/ml.py
- **Line(s)**: 632–658
- **Description**: After a preprocessor or AutoFE transform, `get_val_loss` uses `len(X_val.test_data)` for a `TimeSeriesDataset`, otherwise `X_val.shape[0]`, and divides by that count. It does not handle `test_data is None` or length 0. Predict paths in `ts_model.py` explicitly fall back when the test partition is empty. This division runs even when the metric is a callable that already returned a score.
- **Risk**: An empty holdout, an empty forecast test partition, or a custom splitter aborts the trial with `ZeroDivisionError` or `TypeError` after a successful fit, instead of recording the metric.
- **Suggested Fix**: Use the same empty-partition guard as predict. If the validation row count is 0, skip the latency adjustment or record transform time without dividing.

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

## Driver assessment

The original response is retained. Audit passed: `grok-4.7`, 68 search and
56 view calls, no prohibited tools or feedback access, and no source drift.

| Issue | Disposition and verification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Inherited: `evaluate_model_CV` is AST-identical to current public base lines 397-459. Unlimited-budget behavior remains separately documented.                                                                                                                                                                                                                                                                                                                                                                |
| 2     | Inherited: panel split construction and forecast CV are AST-identical to current public base. No new panel split algorithm is introduced here.                                                                                                                                                                                                                                                                                                                                                                |
| 3     | Inherited: `prepare_data` is AST-identical to current public base lines 222-322, including the positional panel constructor and rebasing behavior.                                                                                                                                                                                                                                                                                                                                                            |
| 4     | Duplicate-index corruption was not reproduced with an equal-index boolean mask. Positional assignment and an explicit negative-index guard were nevertheless added as closely related hardening, with irregular/NaT/duplicate-index regressions.                                                                                                                                                                                                                                                              |
| 5     | Fixed as a consistency improvement to the PR's dtype safeguards. All sampled numeric/timestamp candidates are checked against full-column support in one aggregate action. A biased sample cannot force a cast below the requested threshold. Accepted threshold-based coercion still deliberately maps nonmatching values to null, as documented; the API was not silently changed to require every value to parse. Real Spark regressions cover int/double/timestamp bias and preserved threshold behavior. |
| 6     | Inherited Holt-Winters parameter handling; see the broad-sweep assessment and current-base AST comparison.                                                                                                                                                                                                                                                                                                                                                                                                    |
| 7     | Inherited: `remove_ts_duplicates` is AST-identical to current public base lines 516-538. The unrelated cleanup-policy defect remains visible.                                                                                                                                                                                                                                                                                                                                                                 |
| 8     | Fixed. Forecast transformation timing measures only held-out rows; empty partitions contribute no per-row transform cost. Generic empty custom-metric validation skips the adjustment with a warning. Tests preserve the supplied metric and cover both dataframe and forecast-dataset empties.                                                                                                                                                                                                               |
