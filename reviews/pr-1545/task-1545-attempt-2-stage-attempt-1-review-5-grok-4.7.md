---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 1
round: 5
timestamp: '2026-10-31T16:41:14Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: e3e3662b48fa610de30e5f079024e09e967b70c49486415282a97dbc34392369
---

## Review Summary

- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-1-review-5-grok-4.7.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `flaml/automl/data.py` `add_time_idx_col` / `_nat_aware_int_series` (lines 250–345) against `test/automl/test_add_time_idx_col.py`, `test/automl/test_data_coverage.py` `TestAddTimeIdxCol`, and `test/automl/test_lossless_conversion.py`.
- [x] Read `flaml/automl/time_series/ts_model.py` `StatsModelsEstimator.predict` (lines 346–455) and `SeasonalAverage.fit` (lines 851–868) against `test/automl/test_forecast.py` and `test/automl/test_ts_data.py`.
- [x] Read AutoFE isolation in `flaml/automl/task/generic_task.py` (lines 895–910, 1024–1045) and `flaml/fabric/README.md` (lines 72–77); searched tests for `every class represented` (source only).
- [x] Read `flaml/automl/ml.py` `get_val_loss` (lines 619–659) and `flaml/fabric/autofe.py` `transform` (lines 529–543); the only latency assertion is `test/fabric/test_autofe_isolation.py` line 365 and uses DataFrames.
- [x] Driver pytest totals were treated as supporting evidence only. No tests were executed in this review. Forecast integer-horizon, export, DST, and singleton-class paths do have assertions and are not filed.

## Issues

### Issue 1: AutoFE missing-class rejection is untested

- **Severity**: Medium
- **File**: flaml/automl/task/generic_task.py
- **Line(s)**: 1028–1038
- **Description**: The holdout guard that raises `AutoFE requires disjoint partitions with every class represented` has no test. `test_autofe_rejects_singleton_classes_before_cross_validation` only hits the earlier singleton check (lines 899–903) under `eval_method="cv"`. The documented contract (`flaml/fabric/README.md` lines 76–77) says missing class coverage is rejected instead of copying rows. That copy is `_handle_missing_labels_fast`, the `elif` immediately after this guard. No test uses `featurization="auto"` with a non-stratified holdout that drops a class. Caller-supplied `X_val` never enters this check, and no test asserts that case either.
- **Risk**: A regression that falls through to the overlap repair would put the same observation in train and validation under AutoFE. CI would stay green. Explicit incomplete validation is also unlocked.
- **Suggested Fix**: Add a deterministic holdout test (`split_type="uniform"` or a stubbed split) that expects this `ValueError` and does not call the overlap handler. Add a second test for caller-supplied validation that omits a class, and reject that input if it is part of the documented contract.

### Issue 2: Forecast prediction-latency accounting is untested

- **Severity**: Medium
- **File**: flaml/automl/ml.py
- **Line(s)**: 626–634, 656–658
- **Description**: `get_val_loss` times preprocessor and AutoFE `transform`, then, for a `TimeSeriesDataset`, divides only by `len(X_val.test_data)`. Forecast CV passes the same fold dataset as `X_train` and `X_val` (`flaml/automl/task/time_series_task.py` line 439). `Featurization.transform` transforms both `train_data` and `test_data` (`flaml/fabric/autofe.py` lines 533–543). `test_prediction_latency_includes_only_serving_transform_cost` covers regression DataFrames only and never this denominator.
- **Risk**: Forecast `pred_time` can include training-partition transform cost, then spread it over validation rows only. `pred_time_limit` can reject or accept forecast trials for the wrong reason. An empty `test_data` would also divide by zero, and that path is untested.
- **Suggested Fix**: Add a forecast holdout and CV test with a timed transform. Assert per-row latency equals validation-only transform time divided by validation rows, and that fitting or training-partition transform time is excluded.

### Issue 3: Nullable month-start and year calendar indexes are untested

- **Severity**: Medium
- **File**: flaml/automl/data.py
- **Line(s)**: 250–262, 308–313
- **Description**: `freq == "MS"` and `freq == "Y"` are the only callers of `_nat_aware_int_series`. Complete MS series are checked in `test_regular_calendar_frequencies_follow_public_step_indexing`. No test passes `NaT`, a duplicate index, or a partial mask through those branches. Daily and month-end `NaT` tests use `get_indexer` or calendar-step indexing instead. `test_yearly_frequency` uses `YS` and expects steps `0..4`, so it does not hit `freq == "Y"`, which would store absolute years.
- **Risk**: A bad nullable cast on month-start data can write the int64 minimum or mis-assign duplicate indexes without failing CI. A pandas alias of `"Y"` can also violate the step-index contract with no failing test.
- **Suggested Fix**: Add an MS series with `NaT`, a repeated timestamp, and a non-unique index, and assert nullable `Int64` values. Add a direct `"Y"` case that locks the intended index, or remove that branch if `infer_freq` never returns it.

### Issue 4: Misaligned forecast timestamps are untested

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 396–403, 441–448
- **Description**: Both predict paths raise if requested timestamps are missing from the frequency grid, duplicated, or not increasing. Tests cover mixed train/future spans, exogenous gaps, and in-sample seasonal-naive subsets. Nothing matches `must be unique, increasing, and aligned`. Mock forecast tests also leave `_index_generated` false, so they do not lock the future-path `index=` argument at lines 411–415.
- **Risk**: Removing the check makes `get_indexer` `-1` become `iloc[-1]`, a silent wrong forecast rather than an error. Decreasing timestamps would be reordered the same way.
- **Suggested Fix**: Add future and in-sample cases for a duplicate, a decreasing pair, and an off-grid timestamp, each expecting that `ValueError` and no model call. Assert the generated-index future path passes `index=forecast_dates`.

### Issue 5: Seasonal-average lag cap is not asserted

- **Severity**: Low
- **File**: test/automl/test_forecast.py
- **Line(s)**: 288–295, 298–310
- **Description**: `SeasonalAverage.fit` caps `maxlag` at `(len(train_df) - 2) // 2` (`ts_model.py` lines 857–862). `test_seasonal_average_caps_lag_order_on_short_training_data` only asserts `estimator.season == 6`, which is set before the cap. `test_seasonal_average_leaves_residual_degrees_of_freedom` only asserts `df_resid > 0`, not that the selected order is at most 1.
- **Risk**: The cap can be dropped and both tests can still pass whenever the uncapped fit succeeds with positive residual degrees of freedom.
- **Suggested Fix**: Assert the fitted AR lag order is at most 4 on the 11-row case and at most 1 on the 5-row case.

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

## Driver assessment

Original findings are preserved. Audit passed with 118 search, 53 view, and
6 glob calls; runtime identity and independence were verified.

| Issue | Resolution and evidence                                                                                                                                                                                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Added deterministic generated-holdout missing-class/no-overlap-repair coverage. Caller-supplied validation deliberately remains as supplied, including class subsets; a separate test verifies no copying or augmentation, and the README now scopes rejection to automatically generated splits. |
| 2     | Added actual AutoML forecast holdout/CV timing tests. The shared partition transform now times only validation rows, not training history or fitting, and empty custom validation avoids division by zero.                                                                                        |
| 3     | Added MS and explicit legacy `"Y"` cases with NaT, repeated timestamps, and duplicate row labels. Both assert exact nullable `Int64` values.                                                                                                                                                      |
| 4     | Added duplicate/decreasing/off-grid cases for future and in-sample requests and an explicit generated-index future-date assertion. Invalid inputs are rejected before backend prediction calls when the fitted index is known.                                                                    |
| 5     | Strengthened tests with exact `ar_select_order(maxlag=...)` spies and fitted-lag bounds, not only successful fitting or positive residual degrees of freedom.                                                                                                                                     |

Post-fix targeted regressions pass. The broader contract suite reports
816 passes with 8 explicit skips and 20 subtests; actual Spark dtype
validation reports 23 passes.
