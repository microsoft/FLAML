---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 2
round: 4
timestamp: '2026-10-31T18:12:46Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 602f31ba72f2104d91f8e074886a05ad91a53131c65809a5658262e52af89248
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-2-review-4-grok-4.7.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `flaml/automl/task/time_series_task.py` lines 509-540 and `flaml/automl/time_series/ts_data.py` `move_validation_boundary` (lines 236-246). Negative subsample shift keeps `train_data[:steps]`, the oldest rows, and appends the dropped tail to validation.
- [x] Read `flaml/automl/time_series/ts_model.py` HoltWinters search space (lines 682-697) and constructor (lines 743-749). `seasonal_periods` and `use_boxcox` are never passed. Integer-horizon path in `StatsModelsEstimator.predict` (lines 354-428) does start after `train_end_date`.
- [x] Read `remove_ts_duplicates` (lines 518-540), `AutoMLPipeline.predict` row check (`flaml/automl/pipeline.py` lines 152-167), and `_eval_estimator` latency division (`flaml/automl/ml.py` lines 724-730) versus the empty-partition guard at lines 664-671.
- [x] Checked forecast/time-index tests (`test/automl/test_forecast.py`, `test/automl/test_ts_data.py`, `test/automl/test_add_time_idx_col.py`) and driver-reported passes. No test asserts the forecast subsample window or that Holt-Winters receives the tuned period. Driver results were not re-executed.
- [ ] Did not run pytest, pre-commit, or git diff. Read-only inspection only.

## Issues

### Issue 1: Forecast subsampling keeps the oldest rows and enlarges validation

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 509-513, with `flaml/automl/time_series/ts_data.py` 241-246
- **Description**: The comment says the tail is kept. `shift = sample_size - len(train_data)` is negative whenever the trial is smaller than the training partition. `move_validation_boundary` then does `train_data[:steps]`, which for a negative slice is the head, and concatenates `train_data[steps:]` onto `test_data`. Example: 100 training rows, sample size 20, shift -80. Training becomes rows 0-19. Rows 20-99 move into validation. `AutoMLState.prepare_sample_train_data` already uses `iloc[-sample_size:]` and leaves the holdout unchanged, but the time-series override bypasses it. Default `sample=True` enables this once training rows exceed `min_sample_size * 4` (40000). A zero shift is a no-op; a positive shift correctly absorbs later holdout rows.
- **Risk**: Large-series search scores models fit on the distant past against a validation set that includes recent training rows plus the holdout. Sample-size steps are not comparable, and the config retrained on the real boundary can be the wrong one. `train_end_date` for those trials is also the end of the old head, so integer horizons start in the middle of history.
- **Suggested Fix**: If `sample_size < len(train_data)`, shallow-copy the dataset and set `train_data` to `train_data.iloc[-sample_size:].copy()` without moving those rows into `test_data`. Use `move_validation_boundary` only when `sample_size` exceeds the training length. Add a regression that a 20-row sample of a 100-row series with a holdout trains on the last 20 training timestamps and leaves the holdout length unchanged.

### Issue 2: Holt-Winters does not apply tuned seasonal_periods or use_boxcox

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 691-694, 731-749
- **Description**: The search space tunes `seasonal_periods` over `{7, 12, 4, 52, 6}` and `use_boxcox`, and the initial point is `seasonal="add"`. `seasonal_periods` is read only to force `seasonal=None` when `len(train) < 2 * seasonal_periods`. `HWExponentialSmoothing` receives `damped_trend`, `seasonal`, `trend`, and `initialization_method="estimated"`, but not `seasonal_periods` or `use_boxcox`. The training index is assigned with `to_datetime(...)`, so `.freq` is often missing. statsmodels then either raises because a seasonal model has no period, or infers one from frequency (`D` becomes 7, `B` becomes 5) and ignores the tuned value. `initialization_method="estimated"` is correctly explicit.
- **Risk**: Seasonal trials either abort the search with an uncaught `ValueError` or silently fit the frequency-implied period. Distinct `seasonal_periods` and `use_boxcox` choices produce the same model, so the logged best config does not describe the fitted smoother. Exported refits repeat the omission, so pipeline and native predictions can agree while both ignore the selected period.
- **Suggested Fix**: Pass `seasonal_periods=self.params["seasonal_periods"]` whenever `seasonal` is not None, and pass `use_boxcox=self.params["use_boxcox"]`. Monkeypatch `ExponentialSmoothing` and assert those kwargs on daily data with a period the frequency would not infer, such as 4.

### Issue 3: Duplicate-row cleanup asserts a condition that is always false

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 530-538
- **Description**: `validate_data` calls `remove_ts_duplicates` after timestamp coercion. `X.duplicated()` matches full rows, not `time_col`. If any exact duplicate row exists, `drop_duplicates()` runs and then `assert X[[X.columns[0]]].duplicated() is None`. `duplicated()` returns a Series, so `is None` is always false and the assert always raises. The check also uses column 0 rather than `time_col`, so a non-leading timestamp column would be the wrong key even after the comparison is fixed. Conflicting timestamps that differ in other columns are not dropped, which is reasonable, but they never reach a deliberate error either because the assert fires first whenever any full-row duplicate existed.
- **Risk**: Forecast fit crashes on identical repeated rows that the warning says were removed. The message claims conflicting timestamp values, which may not be the actual condition. Clean unique series are unaffected.
- **Suggested Fix**: After dropping exact duplicate rows, raise `ValueError` if `X[time_col].duplicated().any()`, and do not use `assert` for this input check. Cover both an exact duplicate row, which should be removed, and the same timestamp with different targets, which should fail cleanly.

### Issue 4: String-metric forecast evaluation still divides by an empty validation length

- **Severity**: Low
- **File**: flaml/automl/ml.py
- **Line(s)**: 724-730
- **Description**: `get_val_loss` skips `validation_transform_time / num_val_rows` when the validation partition is empty (lines 664-671). `_eval_estimator` then unconditionally does `(time.time() - pred_start) / num_val_rows`, and for a `TimeSeriesDataset` `num_val_rows` is `len(test_data)` with no empty check. Predict falls back to the training partition when test data is empty, so the numerator is a real prediction over training rows while the denominator is 0. The empty-validation regressions call a custom metric and therefore take the other branch; they do not exercise this division. `period=0` is an `int` and is not rejected before holdout setup.
- **Risk**: A string metric such as default `mape`, combined with an empty forecast test partition, raises `ZeroDivisionError` instead of the warning used for transform latency.
- **Suggested Fix**: If `num_val_rows == 0`, do not divide; warn and return a defined `pred_time`. Reject `period <= 0` in time-series split setup. Add a string-metric empty-test case beside the existing custom-metric test.

### Issue 5: Exported classifier row check miscounts 1D inputs

- **Severity**: Low
- **File**: flaml/automl/pipeline.py
- **Line(s)**: 159-167
- **Description**: After a successful `predict`, integral class decoding estimates the input row count as `X.shape[0]` whenever `shape` exists, and as 1 for a non-empty list whose first element is not a list. A one-row `Series` has `shape[0] == n_features`. A list of row tuples is not a list of lists, so it is counted as one row. Both can be valid estimator inputs whose prediction length is the real row count. Numpy rows and DataFrames are handled by `shape[0]`. The integral-value checks after this are sound.
- **Risk**: `predict` raises "one class index per input row" for valid one-row Series input or tuple rows, so an exported classifier cannot inverse-transform those calls.
- **Suggested Fix**: Count samples with a row helper that treats a 1D feature vector as one row and a sequence of sequences, including tuples, as `len(X)`. Compare that count with `len(codes)` only after squeezing a single-column prediction.

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

## Driver disposition

The original response and verdict are retained. The focused verification was
independent and valid: exact `grok-4.7` identity, 60 search and 55 view calls,
no prohibited tools/feedback access, and no source drift.

| Issue | Disposition                                                                                                                                                                                                                                                                                              |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Inherited current-public-base sampling behavior, independently verified by AST comparison; no new sampling implementation is introduced.                                                                                                                                                                 |
| 2     | Inherited current-public-base Holt-Winters parameter omission, apart from the intentional explicit estimated-initialization keyword. The issue remains documented for separate correction.                                                                                                               |
| 3     | Inherited current-public-base cleanup defect, previously verified; no new duplicate-cleanup policy is introduced.                                                                                                                                                                                        |
| 4     | The unchanged string-metric path requires nonempty validation. The PR's new transform-time adjustment now safely handles empty custom-metric validation; it does not claim to make the inherited `period=0`/string-metric path valid.                                                                    |
| 5     | Fixed. The label decoder checks explicit batch dimensions but does not guess how an estimator interprets 1D vectors. Nested tuple rows are counted as a batch. New tests cover Series/NumPy single-row inputs, tuple rows, and a univariate Series batch, while preserving malformed-class-index checks. |

The pipeline and append-log integration suites pass 79 tests after the final
driver fixes. These fixes are not presented as an additional clean reviewer
sign-off; all original reports remain available.
