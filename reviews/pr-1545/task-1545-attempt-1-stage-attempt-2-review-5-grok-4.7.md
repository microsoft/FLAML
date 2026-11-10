---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 5
timestamp: '2026-10-17T18:44:59Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 24b799c4e84c40340bb97178bead65eb300b84dc65530025be47f75080332a82
---

## Review Summary

- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-5-grok-4.7.md
- **Issues Found**: 4
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read Holt-Winters search space and fit (`flaml/automl/time_series/ts_model.py` lines 655–723) against `test/automl/test_ts_coverage.py` `TestHoltWinters` and `test/automl/test_forecast.py` lines 857–897. No test spies on `ExponentialSmoothing` kwargs.
- [x] Read estimated-init coverage: `test/automl/test_forecast.py` `test_simple_forecaster_sets_initialization_method` and `test/automl/test_ts_data.py` `test_simple_forecasters_use_estimated_initialization_everywhere` cover only `SimpleExpSmoothing` (`Naive`/`SeasonalNaive`). `pyproject.toml` line 71 allows `statsmodels>=0.12.2`.
- [x] Read `SeasonalNaive.predict` (`ts_model.py` lines 788–797) and `test_seasonal_naive_fit_predict_int` (`test/automl/test_ts_coverage.py` lines 791–798), plus integer passthrough in `flaml/automl/task/time_series_task.py` lines 382 and 396 and `AutoML.predict` line 946.
- [x] Confirmed preserved contracts that are actually asserted: weighted MAPE (`test/automl/test_metric_sample_weight.py`), binary-only metrics (`test/automl/test_binary_metric_on_multiclass.py`), and latest CV observation (`test/automl/test_ts_data.py` `test_automl_cv_scores_latest_observation`).
- [x] Compared plot callers: `test/fabric/test_visualization.py` lines 60–76 discard figures; `test/fabric/test_visualization_contracts.py` does not call `plot_parallel_coordinate`, `plot_edf`, or `plot_timeline`. Multi-result EDF is `flaml/fabric/visualization.py` lines 469–477.
- [x] Read `test_forecast_panel` (`test/automl/test_forecast.py` lines 684–778): `ImportError` returns success; MAPE/sMAPE are printed only.
- [ ] Did not execute pytest. Driver-reported runs are evidence of execution, not of these assertion gaps. Pristine-base provenance was not available in this worktree.

## Issues

### Issue 1: Holt-Winters tuned seasonality and estimated init are untested and not applied

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 655–723
- **Description**: The search space tunes `seasonal_periods` (7, 12, 4, 52, 6) and `use_boxcox`, and the short-series guard reads `seasonal_periods`. `HWExponentialSmoothing(...)` receives only `damped_trend`, `seasonal`, and `trend`. `initialization_method="estimated"` is not passed. The comment at line 707 still refers to heuristic initialization. `SimpleExpSmoothing` does pass `"estimated"`, and those are the only constructor assertions (`test_simple_forecaster_sets_initialization_method`, `test_simple_forecasters_use_estimated_initialization_everywhere`). Holt-Winters tests fit the model and check length or in-place seasonal/trend overrides (`test_ts_coverage.py` `TestHoltWinters`); daily fixtures can let statsmodels infer period 7, so omitting `seasonal_periods=7` still passes. `test_log_training_metric_ts_models` documents holt-winters but the list at line 864 does not include it.
- **Risk**: Search choices for season length and Box-Cox do not change the fitted model. On `statsmodels>=0.12.2`, initialization can follow a version-dependent default instead of the required estimated init. Existing tests cannot fail for either omission.
- **Suggested Fix**: Pass `seasonal_periods` when `seasonal` is set, pass `use_boxcox`, and pass `initialization_method="estimated"`. Monkeypatch `ExponentialSmoothing` and assert those kwargs, using a period the index frequency would not infer (for example 4 on daily data).

### Issue 2: Seasonal-naive integer forecasts are checked only for length

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 788–797
- **Description**: `SeasonalNaive.predict` on an int forecasts `steps=self.season` and keeps `.iloc[0]` on every iteration. `i` is unused, so `predict(5)` with `season=3` repeats one value. The fitted model is non-seasonal `SimpleExpSmoothing`, so `season` never enters the fit. `test_seasonal_naive_fit_predict_int` only asserts `len(preds) == 5`. `AutoML.preprocess` returns an int unchanged, and `AutoML.predict` forwards it, so `predict(n)` reaches this path.
- **Risk**: A seasonal-naive integer horizon does not cycle the seasonal lag. The only test of that path stays green.
- **Suggested Fix**: Assert `predict(n)` repeats the training value from `season` steps back. Implement that lag cycle, or reject integer horizons if they are unsupported, and test both.

### Issue 3: Several public plots have no output assertions

- **Severity**: Low
- **File**: test/fabric/test_visualization.py
- **Line(s)**: 60–76
- **Description**: `test_plot_parallel_coordinate`, `test_plot_edf`, and `test_plot_timeline` call the plotters and discard the figures. Contract tests do not call these three. `plot_edf` also accepts a list or dict of tune results (`visualization.py` lines 469–477); nothing calls that branch.
- **Risk**: Wrong axes, empty traces, or a broken multi-study EDF can pass as long as the call does not raise.
- **Suggested Fix**: Assert trace count, axis titles, and score values for one AutoML result and one tune result. Add a two-study `plot_edf` case that checks both `tune_id` series.

### Issue 4: Panel forecast test can pass without running or checking predictions

- **Severity**: Low
- **File**: test/automl/test_forecast.py
- **Line(s)**: 684–778
- **Description**: `test_forecast_panel` catches `ImportError`, prints that pytorch-forecasting is missing, and returns. That is a pass, not a skip. When the import works, MAPE and sMAPE are printed and never asserted. The reported driver commands do not include `test_forecast.py`.
- **Risk**: Missing optional dependencies and arbitrary panel forecasts are both reported as success.
- **Suggested Fix**: Use `pytest.importorskip` (or `pytest.skip`). Assert prediction length, finite values, and a documented error ceiling.

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

## Driver assessment after stage 5, attempt 2

The original response and verdict remain unchanged. Actual model `grok-4.7`
used 41 view, 125 search, and 2 glob calls; source and independence audits
passed. Stage 5 advances **with unresolved findings** at the High cap.

| Issue | Disposition and evidence                                                                                                                                                                                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Explicit estimated initialization is now pinned in the Holt-Winters constructor and asserted, honoring the user's uniform-initialization requirement rather than relying on a dependency default. The separate seasonal-period/Box-Cox omissions were independently verified as inherited and are not claimed fixed.                               |
| 2     | The underlying seasonal-naive algorithm remains an independently verified baseline issue. The PR's existing `.iloc[0]` compatibility edit does not introduce that algorithm; it is retained explicitly for separate follow-up.                                                                                                                     |
| 3     | Parallel coordinates were already tested through the public dispatcher. New exact EDF/timeline and list/dictionary multi-study assertions now cover scores, probabilities, durations, offsets, trace names, and axis labels.                                                                                                                       |
| 4     | Missing panel dependencies now produce an explicit pytest skip rather than success. Predictions must match target length, be finite, and beat a zero-volume SMAPE baseline; zero denominators are handled explicitly. The dependency is absent from the main local environment, so the panel quality assertion is not claimed as executed locally. |

The 643-test suite, 53-test offline NLP suite, 26-test Ray reporting suite,
installed-Spark import guard, build, and source lint recorded in the companion
assessment pass. The panel dependency skip and native Spark persistence's
Linux-CI requirement remain visible validation limitations.
