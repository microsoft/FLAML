---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 4
timestamp: '2026-10-17T14:49:10Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 877b86dd8c42a159d5708684c0a1d1afcf8ef59bd19b79bd682b27f246e74502
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-4-grok-4.7.md
- **Issues Found**: 8
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced default forecast CV from `flaml/automl/automl.py` (`time_budget` default -1 at line 473; `_decide_eval_method` returns `"cv"` when `time_budget < 0` at lines 1370-1376) through `flaml/automl/state.py` lines 320-326 (`budget = None`) and `flaml/automl/ml.py` lines 413-424 into `TimeSeriesTask.evaluate_model_CV`.
- [x] Compared that path with `flaml/automl/task/generic_task.py` line 1193 (`budget and budget / n`) and read `flaml/automl/task/time_series_task.py` lines 430 and 458.
- [x] Read `TimeSeriesTask.prepare_sample_train_data` (lines 511-516) against `TimeSeriesDataset.move_validation_boundary` (lines 235-246) and the tail slice in `AutoMLState.prepare_sample_train_data` (lines 254-256).
- [x] Read `remove_ts_duplicates` (lines 520-541), panel holdout construction (lines 288-294), and panel CV splitter setup (lines 323-326).
- [x] Read `HoltWinters.fit` (lines 706-724), `SeasonalNaive.predict` (lines 788-795), and `SearchState` starting-point mutation (`flaml/automl/state.py` lines 163-170).
- [x] Inspected forecast gap alignment in `StatsModelsEstimator.predict`, AutoFE fold fit in `get_val_loss`, and `hashFiles('.coverage')` at `.github/workflows/python-package.yml` line 250. Contour axis updates in `plot_contour` land on the transposed cell with matching titles, so that was not filed. An exact `hashFiles('.coverage')` pattern was not treated as a proved dotfile miss.
- [ ] Tests were not executed in this review. Driver-reported passes are not treated as proof; the default CV path below is still present in the current source.

## Issues

### Issue 1: Default time-series CV divides a null budget

- **Severity**: Critical
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 430, 458
- **Description**: `evaluate_model_CV` does `budget_per_train = budget / n` and later `time.time() - start_time >= budget`. For the default `time_budget=-1`, `_decide_eval_method` selects `"cv"`, and `_compute_with_config_base` passes `budget=None`. `None / n` raises `TypeError` before any fold runs. The generic task already uses `budget and budget / n`. Fixing only the division still leaves the elapsed comparison raising `TypeError` on `None`.
- **Risk**: `AutoML.fit(..., task="ts_forecast")` without a positive time budget, the documented default, crashes on the first CV trial instead of fitting. A positive budget hides it, which is why budgeted tests can pass.
- **Suggested Fix**: Match the generic guard: `budget_per_train = budget and budget / n`, and break only when `budget is not None and time.time() - start_time >= budget`. Add a forecast fit with `time_budget` unset and `eval_method="cv"`.

### Issue 2: Time-series subsampling keeps the oldest rows

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 511-516
- **Description**: The comment says the tail is kept. `shift = sample_size - len(train_data)` is negative, and `move_validation_boundary` assigns `train_data[:steps]`. For a negative step, pandas keeps the head and moves the recent rows into `test_data`. `AutoMLState.prepare_sample_train_data` already uses `iloc[-sample_size:]` for a `TimeSeriesDataset`, but the task override replaces that path. Subsampling is enabled for holdout when `n_rows > min_sample_size * 4` (default 40000).
- **Risk**: Large holdout forecasts are trained on the oldest window and scored on the recent holdout, so search selects configs for the wrong regime.
- **Suggested Fix**: Copy the dataset, set `train_data = train_data.iloc[-sample_size:]`, and keep the existing validation `test_data`. Do not use `move_validation_boundary` to subsample.

### Issue 3: Duplicate-timestamp cleanup always asserts

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 529-540
- **Description**: `X.duplicated()` matches full rows, not `time_col`. If any full-row duplicate exists, `drop_duplicates()` runs and then `assert X[[X.columns[0]]].duplicated() is None`. `duplicated()` returns a Series, so `is None` is always false and the assert always raises. It also checks column 0, not `time_col`. Conflicting timestamps with different other values never enter the branch and are kept.
- **Risk**: One exact duplicate row aborts every `ts_forecast` fit. Distinct rows that share a timestamp are silently accepted and can break frequency inference.
- **Suggested Fix**: Detect duplicates with `X[time_col].duplicated()`. After dropping full-row duplicates, assert `not X[time_col].duplicated().any()`. Provenance versus public base could not be checked from this snapshot; the defect is in the proposed tree.

### Issue 4: Panel CV splitter has no step size

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 323-326, 432
- **Description**: The univariate branch sets `state.kf.step_size`. The `TS_FORECASTPANEL` branch creates a `TimeSeriesSplit` and does not. `evaluate_model_CV` always calls `ts_data.cv_train_val_sets(..., kf.step_size)`. Once Issue 1 is fixed, or when `time_budget >= 0`, this raises `AttributeError`. The splitter is also not group-aware: it slices the flat frame by `period * n_groups` rows.
- **Risk**: Panel forecasting with `eval_method="cv"` crashes, or, if a step size is bolted on without grouping, mixes series inside a fold.
- **Suggested Fix**: Set an explicit step size and split within each `group_ids` series, preserving the latest observation in each series. Do not read `step_size` unless it was assigned.

### Issue 5: Holt-Winters drops tuned seasonal arguments

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 666-671, 716-722
- **Description**: The search space tunes `seasonal_periods` (init 7) and `use_boxcox`, and the short-series check reads `seasonal_periods`. `HWExponentialSmoothing(...)` receives only `damped_trend`, `seasonal`, and `trend`. The init point uses `seasonal="add"`. statsmodels requires `seasonal_periods` when seasonal is set and the index frequency does not imply one. `initialization_method="estimated"` is also not passed, unlike `SimpleExpSmoothing`; the supported extra is `statsmodels>=0.12.2`.
- **Risk**: Selecting `holt-winters` either raises on the default seasonal point or fits a period inferred from the index instead of the tuned value. `use_boxcox` never changes the model.
- **Suggested Fix**: Pass `seasonal_periods` when `seasonal` is set, pass `use_boxcox`, and pass `initialization_method="estimated"`. Assert those constructor kwargs in a test.

### Issue 6: Panel holdout stores frequency in time_idx

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 288-294
- **Description**: `TimeSeriesDataset.__init__` parameter 4 is `time_idx`, not frequency. This call passes `ts_data.frequency` positionally, so `time_idx` becomes an alias such as `"D"` or `"MS"`. Frequency is inferred again and is not a constructor argument. Other call sites pass `time_idx=` by keyword.
- **Risk**: Holdout panel datasets report the wrong index-column name. Pipeline reconstruction, `enrich_dataset`, and any `X[dataset.time_idx]` lookup miss the real `time_idx` column. The public panel test uses this holdout path.
- **Suggested Fix**: Call `TimeSeriesDataset(train_data, ts_data.time_col, ts_data.target_names, time_idx=ts_data.time_idx, test_data=test_data)`.

### Issue 7: Seasonal naive integer forecast ignores season and step

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 788-795
- **Description**: `predict(int)` loops over `i` but always appends `forecast(steps=self.season).iloc[0]`. `forecast` does not advance state, so every horizon step is the same first SES value. The model is `SimpleExpSmoothing` with `smoothing_level=1`, which has no seasonal component, and the dataframe path calls `super().predict`. The tuned `season` parameter does not affect predictions. Coverage tests only check `len(preds) == 5`.
- **Risk**: `snaive` returns a repeated last-level forecast rather than a seasonal cycle for both integer and timestamp inputs.
- **Suggested Fix**: Repeat the last `season` observations with `i % season`. Use that cycle for timestamp predictions too, and assert values, not only length.

### Issue 8: List starting points are mutated in place

- **Severity**: Low
- **File**: flaml/automl/state.py
- **Line(s)**: 163-170
- **Description**: For a list of starting points, missing `fe.*` keys are filled with `point.setdefault(...)`. Those dicts are the caller's objects, not copies.
- **Risk**: A caller's `starting_points` gains AutoFE keys as a side effect of constructing `SearchState`.
- **Suggested Fix**: Copy each point before `setdefault`, and assign the copies to `init_config`.

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

## Driver assessment after stage 4, attempt 2

Original findings and verdict remain unchanged. The runner confirmed
`grok-4.7`; the audit counted 77 view and 68 search calls with no forbidden
tools, feedback reads, or source drift. Stage 4 advances **with unresolved
findings** at the High two-attempt cap, not clean.

| Issue | Driver disposition and independently checked evidence                                                                                                                                                                                                    |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Retained critical baseline concern. `evaluate_model_CV` is AST-identical to public base lines 397-459, confirming the earlier null-budget finding. The existing default-budget CV defect is not claimed fixed by this consolidation review.              |
| 2     | Retained baseline concern. `prepare_sample_train_data` is AST-identical to public base lines 507-513.                                                                                                                                                    |
| 3     | Retained baseline concern. `remove_ts_duplicates` was independently verified AST-identical to public base lines 516-538 in the prior stage.                                                                                                              |
| 4     | Retained baseline concern. The panel splitter setup and CV consumer occur in the independently identical public `prepare_data` and `evaluate_model_CV` methods.                                                                                          |
| 5     | Retained baseline concern. `HoltWinters.fit` is AST-identical to public base lines 627-681. The prediction-boundary fix does not claim to repair these older constructor-argument omissions.                                                             |
| 6     | Retained baseline concern. The positional panel holdout constructor is in the unchanged public `prepare_data` method.                                                                                                                                    |
| 7     | Retained baseline algorithm concern. The only `SeasonalNaive.predict` change from base lines 742-750 is replacing `[0]` with `.iloc[0]` for index compatibility plus a comment. The repeated-first-value seasonal algorithm is inherited.                |
| 8     | Not substantiated. `normalize_starting_point` invokes `AutoMLState.sanitize`, whose `.copy()` creates a new dictionary before list completion uses `setdefault`. Current source and unchanged-input regressions contradict the asserted caller mutation. |

These dispositions preserve every original severity and finding above.
Independently verified baseline issues remain visible for follow-up rather
than being silently relabeled as fixed. Introduced findings from the companion
review were corrected and validated before stage 5.
