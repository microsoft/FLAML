---
producer: review-code
schema_version: 1
tier: Medium
contract: direct
execution_backend: copilot-cli
attempt: 2
stage_attempt: 1
round: 2
timestamp: '2026-10-31T16:42:47Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 773058b3a5f2247193b818e45e8375addc67d20c97d4d1be92583b85f091d516
---

## Review Summary

- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-2-stage-attempt-1-review-2-grok-4.7.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Inspected import layering: `flaml/__init__.py`, `flaml/automl/automl.py` (lines 49–73), `flaml/tune/__init__.py`, `flaml/tune/tune.py` (lines 26–41), and `flaml/fabric/mlflow.py` (lines 65–78).
- [x] Compared public AutoFE default in `flaml/automl/automl.py` (line 516) and `flaml/fabric/README.md` with `SearchState` (`flaml/automl/state.py` lines 62–97) and its export in `flaml/automl/__init__.py`.
- [x] Traced `allow_label_overlap` through constructor settings (`automl.py` lines 397, 489), `fit` (lines 1999, 2398–2400), and `retrain_from_log` (line 1337), plus `_setting_keys` (line 432).
- [x] Read `AutoMLPipeline.fit` (`flaml/automl/pipeline.py` lines 107–150) against `make_automl_pipeline` and the native `predict`/`preprocess` path (`automl.py` lines 902–1044).
- [x] Confirmed discarded `len(...)` calls in `flaml/automl/task/time_series_task.py` (lines 114 and 165) have no validation side effect; `TimeSeriesDataset.X_val` (`ts_data.py` lines 158–160) only returns a frame.
- [ ] Did not execute tests. Driver-supplied results (797 non-Spark passed, focused forecast 24 passed, pre-commit and build passed) are supporting evidence only and do not cover import-time warning filters or constructor-only `allow_label_overlap=False`.

## Issues

### Issue 1: Fabric MLflow import mutates public process defaults

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 65–78
- **Description**: Importing AutoML or Tune imports this module whenever MLflow is installed (`automl.py` line 72, `tune/tune.py` line 41; `import flaml` reaches both). The module body always sets `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR`, starts a process-global `ThreadPoolExecutor` (`cpu_count * 4`), and, unless `FLAML_IS_CLEAN_LOGS` is set away from its default `"1"`, raises MLflow/Synapse loggers to CRITICAL and installs process-wide `warnings.simplefilter` filters that ignore `FutureWarning` and `UserWarning`. None of this is gated by `is_fabric_runtime()` or `mlflow_logging`.
- **Risk**: A normal public import changes caller warning and logging behavior and starts worker threads even when Fabric defaults are off and MLflow logging is never enabled. That breaks the stated contract that Fabric behavior is explicit or locally detected and that a public install does not change process defaults (`flaml/fabric/README.md` lines 1–20, 30–48).
- **Suggested Fix**: Keep the MLflow import lazy or side-effect free. Create the executor on first log, and apply logger-level and warning filters only inside FLAML's own logging calls (local `catch_warnings`), behind an explicit enablement flag. Do not mutate `os.environ` or global warning filters at import.

### Issue 2: SearchState defaults AutoFE on while AutoML defaults it off

- **Severity**: Medium
- **File**: flaml/automl/state.py
- **Line(s)**: 62–97
- **Description**: `SearchState` is public (`flaml/automl/__init__.py` exports it) and defaults `featurization="auto"`. `AutoML` defaults to `"off"` unless `FLAML_FEATURIZATION` is set (`automl.py` line 516; README AutoFE row). `fit` passes the resolved value, but any direct `SearchState(...)` does not. With `flaml.fabric.autofe` importable, `"auto"` injects `fe.*` dimensions for tabular learners. If that import fails, the same default raises `ImportError` instead of behaving as `"off"`. `test/nlp/test_default.py` already constructs `SearchState` without this argument.
- **Risk**: The search-space layer opts into AutoFE, or fails closed, contrary to the public opt-in policy. A later caller that forgets the keyword silently changes the search.
- **Suggested Fix**: Default `SearchState.featurization` to `"off"` and require an explicit `"auto"` or `"force"`. Keep the ImportError only for an explicit non-off request when AutoFE is absent.

### Issue 3: allow_label_overlap is not one settings policy

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 432–476, 1337, 1999, 2398–2400
- **Description**: Constructor documents `allow_label_overlap` and stores it (line 489). `fit` is written to inherit that value only when the argument is `None`, but the signature default is `True`, so `AutoML(allow_label_overlap=False).fit(...)` stays on the overlapping holdout path. `preserve_checkpoint` has the same dead `None` check (signature default `True`). `retrain_from_log` does the opposite: it reads only the constructor setting and has no override. The flag is also absent from `_setting_keys`, so the MLflow configuration snapshot cannot record the policy that `fit` actually used. `learner_selector` is missing from that allowlist as well.
- **Risk**: Train/validation isolation depends on which entry point was used. A constructor-only opt-out does not affect `fit`, while replay uses a different source. Logged configurations cannot reproduce the split.
- **Suggested Fix**: Use one resolver: fit/retrain defaults of `None` mean "inherit constructor," and pass the resolved value through. Add `allow_label_overlap` and `learner_selector` to `_setting_keys`. Apply the same pattern to `preserve_checkpoint`.

### Issue 4: Resampler export refit reimplements Pipeline.fit

- **Severity**: Medium
- **File**: flaml/automl/pipeline.py
- **Line(s)**: 115–150
- **Description**: `make_automl_pipeline` stores the task resampler on `AutoMLPipeline`. The no-resampler path calls `super().fit`. The resampler path re-parses `step__param`, fits a nested `Pipeline`, assigns `self.steps` to that nested pipeline's steps plus the original estimator, resamples, and fits the estimator outside sklearn's `_fit`. `sklearn.base.clone` of an exported pipeline with a resampler therefore refits through a second fit implementation.
- **Risk**: Metadata routing, `set_output`, memory-cache step replacement, and future `Pipeline.fit` checks apply only to the non-resampler path. A failed `estimator.fit` leaves `self.steps` already replaced. The export contract is no longer "a sklearn Pipeline plus a label inverse."
- **Suggested Fix**: Keep resampling in a fit-only step, or a final-estimator wrapper, and always `return super().fit(...)`. Do not assign `self.steps` inside `fit`.

### Issue 5: Forecast validation still evaluates discarded lengths

- **Severity**: Low
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 114, 165
- **Description**: After the validation-boundary rewrite, `validate_data` still executes `len(pre_data.X_val)` and `len(val_df)` and discards both results. `X_val` is a plain column slice (`ts_data.py` lines 158–160); `to_dataframe` already checks row alignment. These calls do not establish the training boundary.
- **Risk**: The merge left a fake dependency in the forecast data-prep abstraction. A later edit can look like it preserved a length check that is not there.
- **Suggested Fix**: Delete both statements. If a boundary check is still required, assign and assert it explicitly.

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

Original findings and verdict are preserved. The exact-model, read-only,
independence, and source-integrity audit passed (44 search, 51 view calls).

| Issue | Disposition and evidence                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Independently verified inherited public-base import/lifecycle policy. Executor construction alone does not start worker threads. No new process-global policy is introduced by this reconciliation; the prior documented concern remains outside this change.                                                                                                                                                                                      |
| 2     | Fixed. Public `SearchState` defaults to `"off"`; explicit `"auto"`/`"force"` still opts in. Tests cover ordinary default construction and an unavailable AutoFE backend. AutoML's already-resolved explicit mode is unchanged.                                                                                                                                                                                                                     |
| 3     | Current public-base and PR signatures both have the same `allow_label_overlap=True` and `preserve_checkpoint=True` defaults. The constructor-policy inconsistency and related metadata gaps are inherited; no unrelated entry-point policy rewrite is claimed.                                                                                                                                                                                     |
| 4     | Retained design trade-off, not a reproduced regression. Ordinary sklearn transformers cannot resample `y`. The dedicated refit path uses sklearn preprocessing/cache behavior and explicitly restricts row-aligned metadata; standard `Pipeline.fit` itself can replace fitted steps before a final-estimator failure. Documentation clarifies the supported classic `step__parameter` interface rather than promising arbitrary metadata routing. |
| 5     | Fixed. Removed the two discarded length expressions; no validation behavior depended on them.                                                                                                                                                                                                                                                                                                                                                      |
