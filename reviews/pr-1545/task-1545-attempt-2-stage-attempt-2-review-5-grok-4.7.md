---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 2
round: 5
timestamp: '2026-10-31T18:08:34Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 629102b68db941e744a97a09f5dd482dd2a66ec84159dee6bbd6690d40183f97
---

## Review Summary

- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-2-review-5-grok-4.7.md
- **Issues Found**: 4
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `HoltWinters.fit` (`flaml/automl/time_series/ts_model.py` 682–751) against `test_holt_winters_integer_horizon_starts_after_fitted_training_boundary` (`test/automl/test_ts_data.py` 217) and `TestHoltWinters` (`test/automl/test_ts_coverage.py` 726–777).
- [x] Read panel holdout reconstruction (`flaml/automl/task/time_series_task.py` 248–272) against `test_forecast_panel` (`test/automl/test_forecast.py` 871–972). No other `ts_forecast_panel` / `group_ids` test asserts the split.
- [x] Read `remove_ts_duplicates` (`flaml/automl/task/time_series_task.py` 518–540), called from `validate_data` at line 159. No test references it.
- [x] Compared append-log contract in `flaml/automl/automl.py` 2211–2218 and 2846 with writer tests in `test/automl/test_training_log_append.py` and the assertion-free `append_log=True` fit in `test/automl/test_multiclass.py` 573–636. No AutoFE+append fit test.
- [x] Spot-checked contracts that are locked: time-index calendar/DST/wide-int tests in `test/automl/test_add_time_idx_col.py` and `test/automl/test_lossless_conversion.py`; seasonal-naive lag and export agreement in `test/automl/test_forecast.py` and `test/fabric/test_pipeline_contracts.py`; Spark dtype contracts in `test/spark/test_dtype_contracts.py`; Fabric default/opt-out tests in `test/fabric/test_runtime.py`.
- [ ] Did not execute tests. Driver-reported pytest/pre-commit/build results were treated as supplied evidence only, not as proof of unasserted branches.

## Issues

### Issue 1: Holt-Winters seasonal search choices are not applied or asserted

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 691–697, 743–749
- **Description**: The search space tunes `seasonal_periods` (7, 12, 4, 52, 6; init 7) and `use_boxcox`, and the short-series guard reads `seasonal_periods`. `HWExponentialSmoothing(...)` receives only `damped_trend`, `seasonal`, `trend`, and `initialization_method="estimated"`. The new constructor test forces `seasonal=None`, so it never checks a seasonal fit. `TestHoltWinters` either disables seasonality or overrides multiplicative seasonality to additive on daily data, and never reads the fitted period or Box-Cox flag. `test_log_training_metric_ts_models` documents holt-winters but the list at `test/automl/test_forecast.py` 1058 omits it.
- **Risk**: A trial can select period 4 or 12, or `use_boxcox=True`, and still fit the default/inferred period with no transform. Daily indexes often have no frequency, so the default `seasonal="add"` point can also raise before search records a result. Existing tests stay green either way.
- **Suggested Fix**: Pass `seasonal_periods` when `seasonal` is set and pass `use_boxcox`. Monkeypatch `ExponentialSmoothing` on daily data with `seasonal_periods=4` and `use_boxcox=True`, and assert those kwargs. Include holt-winters in the log-training-metric estimator list or correct that docstring.

### Issue 2: Panel holdout reconstruction has no partition assertion

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 254–272
- **Description**: For `ts_forecast_panel` holdout, training rows are sorted by group, timestamp, and `time_idx`, then split at `time_idx.max() - period`. The only caller is `test_forecast_panel`, which sorts the fixture first, requires pytorch-forecasting, and asserts only length, finite values, and SMAPE better than an all-zero forecast (`test/automl/test_forecast.py` 972). It never inspects which rows became training versus validation.
- **Risk**: A sort-then-positional split, or a cutoff that uses row order instead of `time_idx`, can move later panel rows into training. The zero-forecast bar still passes. Univariate explicit-validation isolation tests do not exercise this branch.
- **Suggested Fix**: Add a forecast-panel test that does not need TFT: shuffled groups, unequal series lengths, and a repeated timestamp. Stop before search and assert every training `time_idx` is `<= cutoff`, every validation `time_idx` is `> cutoff`, and known future targets are absent from training.

### Issue 3: Conflicting timestamp duplicates are untested and the guard cannot fail

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 518–540
- **Description**: `remove_ts_duplicates` is on every dataframe forecast fit. It drops only full-row duplicates. The follow-up check is `X[[X.columns[0]]].duplicated() is None`. `duplicated()` returns a boolean Series, which is never `None`, so the assertion cannot fire. It also ignores `time_col`. No test calls this function or fits conflicting timestamps.
- **Risk**: Same timestamp with different targets or regressors is kept. The warning text claims those rows are rejected. Training can then use an arbitrary duplicate.
- **Suggested Fix**: After `drop_duplicates()`, raise if `X[time_col].duplicated().any()`. Test both an exact duplicate row (dropped, other rows unchanged) and a same-timestamp conflict (error, no silent keep).

### Issue 4: AutoML append_log is not asserted with AutoFE or as a non-resume

- **Severity**: Low
- **File**: flaml/automl/automl.py
- **Line(s)**: 2444, 2846
- **Description**: Writer tests lock continuing record IDs and current-run checkpoints. The only `AutoML.fit(..., append_log=True)` test (`test/automl/test_multiclass.py` 573–636) prints validation accuracy and does not read the log. Nothing fits twice with `featurization="auto"` and asserts that record IDs continue, the previous file is preserved, and `best_loss` is not taken from the prior run.
- **Risk**: Dropping `append_log` in `fit()`, or coupling append with search resume / AutoFE replay, would not fail the suite. The documented contract is only unit-tested below the fit path.
- **Suggested Fix**: Fit a one-trial AutoFE model to a temp log, fit again with `append_log=True` and a worse fixed config, and assert record IDs continue, the first record is unchanged, and the second model's `best_loss` is its own trial.

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

## Driver disposition

Original findings and verdict remain intact. Runtime/independence audit passed:
`grok-4.7`, 137 search and 43 view calls, no prohibited tools, feedback access,
or source drift.

| Issue | Disposition                                                                                                                                                                                                                                                                                     |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Repeated inherited current-public-base parameter omission. The inaccurate log-training test description was corrected to describe its actual selected estimators, without claiming Holt-Winters coverage that does not exist.                                                                   |
| 2     | Panel holdout split construction is unchanged from current public base. The earlier AST comparison and documented baseline limitation remain; this reconciliation does not introduce a new panel split.                                                                                         |
| 3     | Current-public-base duplicate cleanup is unchanged. The report's assertion wording is inverted: `is None` is false for a Series, so an executed assertion raises rather than being incapable of firing. The actual inherited defect remains visible.                                            |
| 4     | Added real consecutive `AutoML.fit` calls with AutoFE and append enabled. The test reads records and checkpoints, checks the original byte prefix, continuing IDs and `fe.*` settings, and verifies that a worse second run retains its own best loss rather than resuming the previous search. |

All 79 pipeline and append integration tests pass. The complete review consists
of the six initial Medium lenses and these two focused verification invocations;
no prior findings or verdicts have been overwritten.
