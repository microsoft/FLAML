---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 1
round: 4
timestamp: '2026-10-31T16:45:47Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: fc8a1ee9fd047b940f38f897e59fe8eba4d03ffaa393c29169c1b35e98e457f7
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-1-review-4-grok-4.7.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `TimeSeriesTask.prepare_sample_train_data` (`flaml/automl/task/time_series_task.py` 511-516), `move_validation_boundary` (`flaml/automl/time_series/ts_data.py` 235-247), and the caller `AutoMLState._compute_with_config_base` (`flaml/automl/state.py` 298-336) against `AutoMLState.prepare_sample_train_data` (`state.py` 251-256).
- [x] Read `StatsModelsEstimator.predict` (`flaml/automl/time_series/ts_model.py` 346-455), `create_forward_frame` (`ts_data.py` 558-576), and `test_forecast_cv_uses_fitted_business_day_frequency` (`test/automl/test_ts_data.py` 238-256).
- [x] Read `HoltWinters.fit` (`ts_model.py` 690-736), `SimpleForecaster.joint_preprocess` (`ts_model.py` 759-779), `SeasonalAverage.fit` (`ts_model.py` 852-867), and panel holdout construction (`time_series_task.py` 288-296).
- [x] Read `add_time_idx_col` and helpers (`flaml/automl/data.py` 250-393) against `test/automl/test_add_time_idx_col.py` and `test/automl/test_lossless_conversion.py`. Nullable timestamps, DST folds, and calendar-vs-step indexing match those tests.
- [x] Read AutoFE fold isolation in `get_val_loss` (`flaml/automl/ml.py` 618-643) and `Featurization.fit` (`flaml/fabric/autofe.py` 402-470). Training labels are fit-only; validation uses `transform`.
- [ ] Did not execute tests. Driver-reported 797/24 passes are supporting evidence only and do not cover the cases below.

## Issues

### Issue 1: Forecast subsampling keeps the oldest rows and moves the validation boundary

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 511-516, used with flaml/automl/time_series/ts_data.py 235-244
- **Description**: The comment says the tail is kept. `shift = sample_size - len(train_data)` is negative while subsampling, and `move_validation_boundary` then does `train_data[:steps]`, which is the oldest `sample_size` rows. It also appends the discarded recent rows to `test_data`. `AutoMLState.prepare_sample_train_data` already keeps `iloc[-sample_size:]` and leaves `test_data` alone, but search calls `task.prepare_sample_train_data` (`state.py` 311). Fit therefore sets `train_end_date` at the end of the stale window. Scoring still passes the original `state.X_val`, so prediction must bridge from that old origin to the real holdout.
- **Risk**: Holdout search on series longer than `MIN_SAMPLE_TRAIN * 4` (10000 * 4) trains early trials on the oldest history. The validation forecast is no longer the next `period` after a recent fit boundary. Misaligned gaps raise `ValueError` and abort the fit; aligned gaps produce a huge horizon and a loss that does not measure the intended forecast.
- **Suggested Fix**: Copy the dataset, set `train_data = train_data.iloc[-sample_size:]`, and leave `test_data` unchanged. Do not call `move_validation_boundary` for subsampling.

### Issue 2: Integer horizons are built from the dataset frequency, not the fitted frequency

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 350-352, 384-399
- **Description**: `predict(n)` always calls `create_forward_frame(self.frequency, n, train_end_date, ...)`. The later branch correctly prefers the fitted `model._index_freq` when it is a `DateOffset`, then `get_indexer`s the synthesized dates. `TimeSeriesDataset.cv_train_val_sets` copies the parent frequency, and `test_forecast_cv_uses_fitted_business_day_frequency` requires `fold.frequency != "B"` while the fitted index is business-day. `predict(fold)` can still succeed; `predict(n)` builds non-business dates and then rejects them as unaligned.
- **Risk**: After a CV fit, or any fit whose dataset frequency alias differs from the model offset, the public integer horizon raises or forecasts the wrong calendar. That breaks the contract that integer horizons start on the fitted training grid.
- **Suggested Fix**: For an int horizon, forecast `steps=n` from the fitted index, or build the frame with the same offset later used for `forecast_dates` (`_index_freq` when it is a `DateOffset`).

### Issue 3: Holt-Winters does not apply tuned seasonal_periods or use_boxcox

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 670-736
- **Description**: The search space tunes `seasonal_periods` (7, 12, 4, 52, 6) and `use_boxcox`. `seasonal_periods` is read only to disable seasonality on short series. `HWExponentialSmoothing` receives `damped_trend`, `seasonal`, `trend`, and `initialization_method="estimated"`, but neither `seasonal_periods` nor `use_boxcox`. `initialization_method` is correctly explicit. Daily indexes can still infer period 7, so a trial that selected 4 or 12 fits period 7, or raises when the index frequency does not imply a period.
- **Risk**: Logged and exported Holt-Winters configs do not describe the fitted seasonality or Box-Cox transform. Search comparisons among those choices are meaningless, and non-inferable frequencies fail at fit.
- **Suggested Fix**: Pass `seasonal_periods=self.params["seasonal_periods"]` when `seasonal` is set, and pass `use_boxcox=self.params["use_boxcox"]`. Assert those kwargs on a frequency that would not infer the tuned period.

### Issue 4: Panel holdout stores the frequency string as time_idx

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 288-294
- **Description**: `TimeSeriesDataset.__init__` takes `(train_data, time_col, target_names, time_idx="time_idx", test_data=None)`. The panel holdout call passes `ts_data.frequency` in the `time_idx` position. Frequency is computed inside `__init__` and is not a constructor argument. `self.time_idx` becomes an alias such as `"D"` or `"MS"` while the column remains `"time_idx"`.
- **Risk**: `add_test_data`, pipeline preprocessing, and AutoFE dataset reconstruction copy `X.time_idx` and then look for a column with the frequency name. Panel holdout search can drop or mis-handle the real step column and treat an unrelated column as the time index.
- **Suggested Fix**: Pass `time_idx=ts_data.time_idx` by keyword. Do not pass `frequency` positionally.

### Issue 5: DataFrame fit of simple forecasters references an undefined data

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 765-776
- **Description**: `joint_preprocess` sets `data` only in the `TimeSeriesDataset` branch. The `else` branch builds `train_df` via `_join`, then unconditionally executes `self.time_col = data.time_col` and `self.target_names = data.target_names`. `SeasonalNaive`, `Naive`, `SeasonalAverage`, and `Average` all call this helper. AutoML's dataset path is safe; a DataFrame/`y` fit is not.
- **Risk**: Direct or exported DataFrame fits raise `NameError` before training. The coded fallback is unreachable.
- **Suggested Fix**: Assign `time_col` and `target_names` inside the dataset branch. In the DataFrame branch, keep the values already set by `super().fit` / `_join`.

### Issue 6: SeasonalAverage lag selection can call statsmodels with a non-positive order

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 860-866
- **Description**: `maxlag = min(self.season, (len(train_df) - 2) // 2)` is 0 when `len(train_df) < 4`. A 2- or 3-point series can still be a valid `TimeSeriesDataset`. `ar_select_order(..., maxlag=0)` is not a legal order. The next line also passes `selection_res.ar_lags` through with no guard when selection returns no lags (`None`).
- **Risk**: Short or low-signal seasonal-average fits raise from statsmodels instead of a bounded FLAML error, including early CV folds that this cap was added to survive.
- **Suggested Fix**: If the residual-degree cap is below 1, fit a constant-only `AutoReg` (`lags=0`) or raise a FLAML `ValueError`. Map a `None`/`[]` selection result to `lags=0`.

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

## Driver assessment

Original findings and verdict remain unchanged. Audit passed with 60 search
and 52 view calls and no feedback access/source drift.

| Issue | Disposition                                                                                                                                                                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Independently verified inherited behavior: current public-base `prepare_sample_train_data` is AST-identical at lines 507-513. A broader sampling-policy fix is not included in this reconciliation.                                                        |
| 2     | Fixed. Integer and timestamp-based forecasts share the fitted calendar; the business-day CV test now exercises both entry points.                                                                                                                          |
| 3     | Inherited except for the PR's explicit estimated-initialization keyword; the existing ignored parameter choices are not newly introduced.                                                                                                                  |
| 4     | Inherited current-base panel holdout constructor; retained outside the merge fix scope.                                                                                                                                                                    |
| 5     | `joint_preprocess` is AST-identical to current public base lines 711-731. The exported task adapter supplies a `TimeSeriesDataset`, so the asserted exported-DataFrame failure is not substantiated. The raw helper fallback remains an inherited concern. |
| 6     | Runtime evidence contradicts the zero/None-lag assertion: two-/three-row fits produce constant-only models successfully. Added exact cap and fitted-value assertions. No unrelated statsmodels short-index prediction redesign is claimed.                 |
