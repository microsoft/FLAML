---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 1
round: 1
timestamp: '2026-10-31T16:45:58Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 1d60fdb184ad0565d95937642ca25ee118661ef0063de9897a4e251f156d8f66
---

## Review Summary

- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-1-review-1-grok-4.7.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `flaml/automl/time_series/ts_model.py` `StatsModelsEstimator.predict` (integer horizon at lines 351-352, model-calendar alignment at 385-394), `HoltWinters.fit` (search space and constructor at 678-736), and `SeasonalAverage.fit` (lines 862-866).
- [x] Read `test/automl/test_ts_data.py` `test_forecast_cv_uses_fitted_business_day_frequency` and `test_holt_winters_integer_horizon_starts_after_fitted_training_boundary`, plus `test/automl/test_forecast.py` seasonal-naive integer-horizon coverage. Those tests do not call `predict(n)` on a fold whose stored frequency differs from the fitted calendar.
- [x] Checked Fabric opt-in in `flaml/fabric/__init__.py` and `flaml/automl/automl.py` (`model_history` defaults to `is_fabric_runtime()`), history-only MLflow in `flaml/fabric/mlflow.py` (`only_history` returns before creating a client; `record_state` exits while `autolog`/`manual_log` stay false), and training-log append record-id continuation in `flaml/automl/training_log.py`.
- [x] Confirmed no merge conflict markers in `flaml/` or `test/` Python sources. `initialization_method="estimated"` is passed for Holt-Winters and simple exponential smoothing.
- [ ] Did not execute pytest. Driver-reported 797/24/9/68 passes are supporting evidence only and do not cover the untested paths below.

## Issues

### Issue 1: Holt-Winters search ignores seasonal period and Box-Cox

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 678-736
- **Description**: The search space tunes `seasonal_periods` (init 7; choices 7, 12, 4, 52, 6) and `use_boxcox`, and the short-series guard reads `seasonal_periods`. `HWExponentialSmoothing(...)` receives only `damped_trend`, `seasonal`, `trend`, and `initialization_method="estimated"`. The default point is `seasonal="add"`. statsmodels requires `seasonal_periods` when seasonal is set and the index frequency does not imply one (daily `"D"` does not). `use_boxcox` is never applied.
- **Risk**: Seasonal Holt-Winters trials raise, or fit a frequency-implied period instead of the tuned period. Box-Cox never changes the model. The default starting point fails on ordinary daily series, so the estimator cannot search the seasonality it advertises.
- **Suggested Fix**: Pass `seasonal_periods=self.params["seasonal_periods"]` when `seasonal` is set, and pass `use_boxcox=self.params["use_boxcox"]`. Assert those constructor kwargs on daily data with a period the frequency would not infer (for example 4).

### Issue 2: Integer horizons are built on the dataset frequency, then aligned to the fitted calendar

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 351-352, 385-394
- **Description**: `predict(n)` builds the requested dates with `create_forward_frame(self.frequency, ...)`. Later, future alignment prefers the fitted model's `_index_freq` when that is a `DateOffset`. `self.frequency` is the dataset attribute copied into CV folds, not re-inferred from the fold. `test_forecast_cv_uses_fitted_business_day_frequency` already requires `fold.frequency != "B"` while the fitted training dates are business-day regular. `predict(fold)` uses the model calendar; `predict(n)` does not. The same path serves exported pipelines. The training boundary (`train_end_date`) itself is correct.
- **Risk**: Documented ARIMA/SARIMAX integer forecasts raise `ValueError` ("aligned with the training frequency") whenever the stored frequency and fitted calendar differ, including pandas environments where business-day data is inferred as `"D"` or `"24h"`. If the model has no `DateOffset`, the n step-ahead values are labeled on the dataset grid instead of the fitted observation grid.
- **Suggested Fix**: For integer `X`, generate the forward frame from the fitted model frequency when it is a `DateOffset`, and fall back to `self.frequency` only when the model has no calendar. Add a `predict(n)` assertion on the existing business-day CV fold.

### Issue 3: SeasonalAverage fit crashes when lag selection is empty or the series is short

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 862-866
- **Description**: `maxlag = min(self.season, (len(train_df) - 2) // 2)` is 0 or negative for fewer than 4 rows, and `ar_select_order` requires a positive `maxlag`. When a valid selection returns no lags, `selection_res.ar_lags` is `None`, and `AutoReg(..., lags=None)` raises. There is no fallback to a constant model.
- **Risk**: Explicit `savg` fits and search trials fail on short folds and on series where the information criterion selects no autoregressive lags, instead of fitting the constant/average model the class is meant to provide.
- **Suggested Fix**: Clamp `maxlag` to at least 1 only when enough rows exist; otherwise fit `AutoReg(..., lags=0)`. If `ar_lags` is `None` or empty, fit `lags=0`.

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

## Driver Decision

{
"tier": "Medium",
"source": {
"tier": "primary estimate",
"models": "current live task-tool catalog and Medium complementary-family policy",
"stages": "review-code Medium default",
"safetyStops": "existing controls; no added execution or credit ceiling"
},
"changeSize": "Follow-up to a completed High review of the unchanged PR head. Public main has 16 new commits across 33 files, predominantly focused fixes and regression tests, with five bounded merge conflicts.",
"cyclomaticComplexity": "Moderate integration complexity in preprocessing, time-series semantics, training-log replay, dependency markers, and zero-shot defaults. The initial broad consolidation is unchanged; new merge interactions warrant six independent lenses.",
"rationale": "Use a fresh Medium six-lens review of merge readiness, with the full reconciled source diff available, rather than repeating the prior 24-invocation High cycle. Reassess introduced and tightly coupled concerns against current public main and preserve unrelated baseline evidence.",
"primaryModel": "gpt-6-astra",
"reviewers": \[
{
"model": "grok-4.7",
"reasoningEffort": "xhigh",
"eligibility": "Newest complementary Grok frontier in the current live catalog, at maximum supported reasoning."
}
\],
"overrideSource": "None for this invocation; previous task-specific model overrides are not reused.",
"contract": "direct",
"reviewAttempt": 2,
"maxPromptBytes": 1300000,
"inputBudgetRationale": "The complete source snapshot is 1231946 bytes before context; raise only the input allowance to retain all source and test evidence.",
"taskId": 1545,
"head": "54473f7da1b7afee8299810780f5dea080fba0dc",
"base": "386ff4005da683d8e8a9340b6d9641b89b34728b",
"repositoryPolicy": ".github/copilot-instructions.md; no repository-local review tier or model override found",
"humanIntent": "Review PR 1545 with review-code, make needed changes, push the existing upstream branch, and make the code ready for merging. Do not merge or bypass human approval."
}

## Driver assessment and scope

The original response and verdict above are preserved. Runtime model
`grok-4.7` used 119 search, 43 view, and 2 glob calls. Transport, exact model,
read-only boundary, feedback independence, and frozen-source checks passed.

| Issue | Disposition                                                                                                                                                                                                                                                                                                                                                                            |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Verified against current public base `386ff4005`: `HoltWinters.fit` is AST-identical after removing the PR's explicit estimated-initialization keyword. Ignored seasonal/Box-Cox choices remain an inherited concern, not a new merge regression. Actual statsmodels infers period 7 for daily data, so the blanket claim that daily input cannot infer a period is not substantiated. |
| 2     | Fixed. Integer frame generation and later alignment now share the same fitted-calendar resolver. The existing business-day CV regression additionally asserts `predict(5)` equals the fitted model forecast.                                                                                                                                                                           |
| 3     | Not substantiated as stated. Actual statsmodels accepts `ar_select_order(maxlag=0)` and `AutoReg(lags=None)`. Added two-/three-row fitted-value regressions and exact lag-cap spies. A separate two-row future-index limitation is not claimed fixed.                                                                                                                                  |

### Review progress

All six required Medium lenses completed with valid independent evidence.
Introduced and tightly coupled issues were reproduced and addressed; unchanged
public-base concerns are independently verified and retained, not treated as
fixed. Focused verification of the resulting correctness and coverage changes
will use new stage-attempt artifacts without showing prior feedback.

The reviewed merge preserves upstream seasonal definitions, append semantics,
nullable/DST-safe indexing, and public defaults. Validation after fixes:
816 targeted passes, 8 explicit skips, 20 subtests; 21 seasonal tests; 9
portfolio tests; and 23 actual Spark dtype tests. The raw reviewer verdict
remains `issues_found`, not a clean sign-off.

### Final driver summary

The Medium review completed all six independent lenses and two focused
verification invocations, all using runtime `grok-4.7` at `xhigh`. All eight
skill-owned reports are retained, together with the earlier review history.
Every invocation was valid and independently audited; their original
`issues_found` verdicts are not relabeled as clean.

Introduced and tightly coupled findings are addressed. The final focused
verification also led to an AutoFE append/non-resume integration test and a
label-decoder correction for ambiguous one-dimensional and tuple-row inputs.
Independent current-base comparisons distinguish the remaining inherited
forecast, overlap-policy, and MLflow lifecycle concerns from merge regressions;
they remain documented rather than silently expanding this PR's scope.

Final local regression result: **821 passed, 8 explicit skips, 20 subtests
passed**. Seasonal assertions, portfolio isolation, pandas 3 timestamp handling,
actual Spark conversion checks, and clean installed-wheel profiles were also
exercised. Hosted checks for the new commit and normal human review approval
remain separate publication gates; this process does not merge the PR.
