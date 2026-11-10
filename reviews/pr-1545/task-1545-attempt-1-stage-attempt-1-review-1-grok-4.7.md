---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 1
timestamp: '2026-10-17T06:22:27Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: b5db4218950dbe8e84f3b6fb558aa77bb77428cfa6b4e194acf5f0bc2394c41d
---

______________________________________________________________________

## producer: review-code schema_version: 1 tier: High contract: direct execution_backend: copilot-cli attempt: 1 stage_attempt: 1 round: 1 timestamp: 2026-03-22T18:45:00Z outcome: issues_found reviewer_model: grok-4.7 input_sha256: 318efb949c7e76b73babca908e69881f03898cd81768a88c3ceb99ce8dadb3f9

## Review Summary

- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-1-grok-4.7.md
- **Issues Found**: 8
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Fabric detection and public defaults in `flaml/fabric/__init__.py` and `flaml/automl/automl.py`: `FLAML_FABRIC_RUNTIME` is explicit and does not import Spark; `model_history` defaults to `is_fabric_runtime()`; `featurization` defaults to `"off"`; off-Fabric MLflow integration is created only with an active run or autolog.
- [x] AutoFE fit isolation in `flaml/automl/ml.py` (`get_val_loss`, `train_estimator`) and `flaml/fabric/autofe.py`: featurization is fit on the training partition before resampling; a no-op fit leaves `pipeline is None` while `_is_fitted` is True (`autofe.py` lines 382-384, 457-461).
- [x] Preserved forecast/metric contracts: `TimeSeriesDataset.cv_train_val_sets` includes the last observation; `sklearn_metric_loss_score` passes `sample_weight` into MAPE; `BINARY_ONLY_METRICS` rejects binary-only metrics on multiclass; `flaml/version.py` is `2.7.0`; `SimpleExpSmoothing` passes `initialization_method="estimated"`.
- [x] Defect paths read directly: Holt-Winters constructor (`ts_model.py` 651-720), TS CV budget (`time_series_task.py` 426, `state.py` 298-321), `AutoML.score` vs `predict` (`automl.py` 928 vs 975/1006), pipeline construction (`automl.py` 866-872, 3723-3728), and MLflow import/restore (`mlflow.py` 68-76, `automl.py` 2852-2896).
- [x] Driver result `python -m pytest test\automl\test_ts_data.py ...` (217 passed, 7 skips) and actionlint were not treated as proof. Those commands do not cover no-op AutoFE repr, Holt-Winters constructor args, or unlimited-budget forecast CV.
- [ ] Live Fabric, Docker, and a fresh full CI run were not available in this read-only invocation.

## Issues

### Issue 1: No-op AutoFE repr aborts a successful retrain

- **Severity**: High
- **File**: flaml/fabric/autofe.py
- **Line(s)**: 382-384, 457-461
- **Description**: `Featurization.fit` treats an empty transformer list as a successful no-op (`_is_fitted = True`, `pipeline` left `None`). `__repr__` and `_repr_mimebundle_` unconditionally call `self.pipeline.__repr__()`. That no-op is reachable: tree learners drop the numerical stage, categorical encoding is skipped when there are no category columns, and selection/extraction may be `"null"`. After retrain, `flaml/automl/automl.py:3592` logs `self._trained_estimator.autofe`, which raises `AttributeError` after the model has already been trained. The same crash happens for any later `str()`/`repr()` of the pipeline, including notebook display.
- **Risk**: `fit()` fails on a supported AutoFE success path and skips the MLflow retrain logging that follows line 3592. Because that exception leaves `fit()` before `resume_mlflow()` at line 2896, caller autolog also stays disabled.
- **Suggested Fix**: Make `__repr__` and `_repr_mimebundle_` tolerate `pipeline is None` (for example return `"Featurization(no-op)"`). Guard the retrain log so a display failure cannot abort fit or skip autolog restore.

### Issue 2: Holt-Winters ignores tuned seasonality and estimated initialization

- **Severity**: High
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 651-720
- **Description**: The search space tunes `seasonal_periods` (init 7) and `use_boxcox`, but `HWExponentialSmoothing(...)` receives only `damped_trend`, `seasonal`, and `trend`. `seasonal_periods` is used only to disable seasonality on short series. `initialization_method="estimated"` is not passed. `SimpleExpSmoothing` does pass it, and the forecast extra allows `statsmodels>=0.12.2`, whose default is still the legacy heuristic. The adjacent comment still refers to heuristic initialization.
- **Risk**: Seasonal Holt-Winters trials either raise (`seasonal_periods` required when the index frequency does not imply a period) or fit the wrong period. `use_boxcox` never changes the model. On 0.12.x this also violates the approved uniform `estimated` smoothing initialization.
- **Suggested Fix**: Pass `seasonal_periods` when `seasonal` is set, pass `use_boxcox`, and pass `initialization_method="estimated"`. Add a constructor assertion like the existing SimpleExpSmoothing test.

### Issue 3: Unlimited-budget time-series CV divides None

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 426
- **Description**: `evaluate_model_CV` always does `budget_per_train = budget / n`. For `time_budget < 0`, `AutoMLState._compute_with_config_base` sets `budget = None` (`state.py` 298-306) and forwards `train_time_limit` (`state.py` 321), which is also `None` by default. The generic task uses `budget and budget / n` and therefore survives an unlimited budget.
- **Risk**: `ts_forecast` with `eval_method="cv"` (or auto selecting CV) and `time_budget=-1` raises `TypeError` on every trial, so search cannot produce a model.
- **Suggested Fix**: Use the generic guard: `budget_per_train = None if not budget else budget / n`.

### Issue 4: Importing AutoML installs internal MLflow side effects

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 68-76
- **Description**: `flaml/__init__.py` imports AutoML, and `flaml/automl/automl.py` imports `flaml.fabric.mlflow` whenever MLflow imports succeed. That module always starts a process-wide `ThreadPoolExecutor` and, because `FLAML_IS_CLEAN_LOGS` defaults to `"1"`, calls `warnings.simplefilter("ignore")` for `FutureWarning` and `UserWarning`. It also forces `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR` and raises several MLflow loggers to `CRITICAL`.
- **Risk**: Any public process that has MLflow installed loses user and future warnings for the rest of the process, including warnings unrelated to FLAML, and gains a non-daemon-shutdown thread pool. That makes the internal runtime the default rather than an explicit Fabric/MLflow opt-in.
- **Suggested Fix**: Do not change global warning filters or start workers at import. Apply logger/warning changes only inside an explicit logging session, and default `FLAML_IS_CLEAN_LOGS` off outside Fabric.

### Issue 5: AutoML.score drops the forecast time column before AutoFE

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 928
- **Description**: `predict` and `predict_proba` call `autofe.transform(X, time_col)`. `score` calls `autofe.transform(X)` with no time column. Time-series `preprocess` returns a dataframe that still contains the timestamp (`DataTransformerTS` does not drop it). Without `time_col`, `Featurization.transform` treats that column as a feature.
- **Risk**: Forecasting plus AutoFE can fail in `score` (datetime fed to a scaler/extractor) or score a different matrix than `predict`, so model selection and reported scores disagree.
- **Suggested Fix**: Pass `getattr(estimator, "time_col", None)` from `score`, matching `predict`.

### Issue 6: Exported pipeline can contain a False preprocessor

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 866-872, 3723-3728
- **Description**: `feature_transformer` builds `Pipeline([("precessor", data_precessor), ("autofe", autofe)])` whenever `autofe` is set. `data_precessor` is `False` when `skip_transform` is set or the input is sparse (`generic_task.py` sets `_transformer = False`). Sparse data disables AutoFE search, but `_featurization` stays `"auto"`, so `automl_pipeline` does not return `None` and wraps that `False` step. sklearn `Pipeline` rejects a non-estimator step.
- **Risk**: `automl_pipeline`, `feature_transformer`, and `register_automl_pipeline` raise `TypeError` after an otherwise valid fit with `skip_transform=True` or sparse data and featurization left at `"auto"`.
- **Suggested Fix**: Omit a missing/false preprocessor. If AutoFE was excluded, return `None` from `automl_pipeline` unless the remaining steps are real estimators.

### Issue 7: Caller autolog is not restored if fit raises

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 2852-2855, 2895-2896
- **Description**: `MLflowIntegration.update_autolog_state` disables caller autolog and stores resume parameters. `resume_mlflow()` runs only at the end of `fit()`, after `_search()`. There is no `finally`. `__del__` is not a reliable restore point, and the integration object is retained on the `AutoML` instance.
- **Risk**: Any fit exception, including Issue 1, leaves the caller's MLflow autolog disabled for the rest of the process. That breaks the requirement to preserve caller autolog state.
- **Suggested Fix**: Call `resume_mlflow()` in a `finally` around the search/retrain body.

### Issue 8: SimpleForecaster dataframe path uses an undefined variable

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 737-754
- **Description**: `joint_preprocess` sets `data` only inside `if isinstance(X_train, TimeSeriesDataset)`. The `else` branch builds `train_df` from a dataframe, then unconditionally reads `data.time_col` and `data.target_names`. Naive, seasonal naive, average, and seasonal average all use this helper.
- **Risk**: A dataframe fit raises `UnboundLocalError` instead of training. The AutoML `TimeSeriesDataset` path avoids it, so the broken branch is untested.
- **Suggested Fix**: Set `self.time_col` and `self.target_names` in both branches before use.

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

## Driver assessment after stage 1, attempt 1

The original reviewer response above is retained. The first, script-generated
frontmatter block is authoritative; the second block was unsolicited reviewer
text and does not establish provenance. The effective effort decision is retained
once in the GPT stage-1 artifact.

Transport audit: actual model `grok-4.7`, 106 grep and 63 view calls, no
write/execute/delegation calls and no prior/sibling feedback access. Both
reviewers observed the frozen source tree.

| Issue | Driver disposition                     | Evidence and verification                                                                                                                                                                                                                                                                                        |
| ----- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Partially confirmed and fixed          | `None.__repr__()` is valid, so the claimed retraining crash through repr was not reproduced. Notebook MIME display on a successful no-op did fail. `Featurization._repr_mimebundle_` now handles no pipeline. `test_successful_noop_display_and_transform` verifies both interfaces and identity transformation. |
| 2     | Pre-existing; deferred outside this PR | AST comparison proves `HoltWinters` is identical to public base `17d293b01`, lines 605-681. No consolidation change introduced these constructor omissions. The initialization decision concerned simple forecasters, not a migration of unchanged Holt-Winters.                                                 |
| 3     | Pre-existing; deferred outside this PR | `TimeSeriesTask.evaluate_model_CV` is AST-identical to public base `17d293b01`, lines 397-459. Its unlimited-budget bug is retained here as an unresolved pre-existing concern, not silently counted as fixed.                                                                                                   |
| 4     | Pre-existing; deferred outside this PR | Public-base `flaml/fabric/mlflow.py` already has the same environment assignment (55), executor construction (58), clean-log default (61), and warning filters (67-68). Executor construction alone does not start workers. No unrelated import-policy rewrite was made.                                         |
| 5     | Confirmed and fixed with GPT issue 7   | Score supplies the time column and Featurization retains fitted timestamp context for exported transforms. Real forecast score/export and direct pipeline-transform regressions pass.                                                                                                                            |
| 6     | Confirmed and fixed with GPT issue 6   | False preprocessing sentinels become valid `None` passthroughs. The skip-transform export parity test passes.                                                                                                                                                                                                    |
| 7     | Confirmed, tightly coupled, and fixed  | A fit wrapper restores MLflow in `finally`; integration configuration is resolved before autolog is disabled. A forced search failure now restores exactly the captured settings.                                                                                                                                |
| 8     | Pre-existing; deferred outside this PR | `SimpleForecaster` is AST-identical to public base `17d293b01`, lines 684-736. The shared base fit already requires `TimeSeriesDataset`; the cited DataFrame path is not introduced by this PR.                                                                                                                  |

The combined driver regression command over pipeline contracts, lossless data
conversion, visualization contracts, runtime/MLflow, AutoFE isolation, log
reconstruction, and existing data coverage completed **215 passed** under
`pytest -n 2 --dist=loadfile`. Stage 1 has not advanced; both reviewers must run
attempt 2 on the updated snapshot.
