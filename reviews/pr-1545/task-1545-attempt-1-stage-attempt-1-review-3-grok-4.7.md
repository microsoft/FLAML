---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 3
timestamp: '2026-10-17T10:23:53Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 7dc9fe3bdf9e69dbb03f5cf384cc236636f57a94a7eab8142cdee38235c3d1b0
---

## Review Summary

- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-3-grok-4.7.md
- **Issues Found**: 11
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read import-time Fabric telemetry and logger guards: `flaml/__init__.py` lines 14–26, `flaml/fabric/telemetry.py` lines 8–34, `flaml/fabric/logger.py` lines 6–16, `flaml/fabric/__init__.py` lines 14–30.
- [x] Read Spark session mutation and integer conversion: `flaml/automl/spark/__init__.py` lines 5–115, `flaml/automl/data.py` lines 1045–1312, import chain from `flaml/automl/automl.py` line 30.
- [x] Read MLflow lifecycle, version checks, futures, and autolog restore: `flaml/fabric/mlflow.py` lines 66–88, 141–143, 270–305, 370–476, 679–710, 820–922, 958–1040; `flaml/automl/automl.py` lines 90–100, 1920–1921, 2526–2536, 2706–2718, 3098–3335, 3405–3608; `flaml/tune/tune.py` lines 548–565.
- [x] Read forecast and exported-pipeline failure paths: `flaml/automl/time_series/ts_model.py` lines 400–437, `flaml/automl/pipeline.py` lines 62–65 and 109–116.
- [x] Checked AutoFE fold fitting does not use the validation split: `flaml/automl/ml.py` lines 604–608 and `flaml/automl/task/generic_task.py` lines 1161–1248. No leakage defect found on that path.
- [ ] Did not execute tests or compare this tree to the pristine public base. Driver-reported pytest/build results are not treated as proof. Provenance of unchanged lines is uncertain where the supplied diff was truncated.

## Issues

### Issue 1: Import-time telemetry can fail `import flaml`

- **Severity**: High
- **File**: flaml/__init__.py
- **Line(s)**: 14–26
- **Description**: `import flaml` always imports `log_telemetry` and calls it. `log_telemetry` calls `report_usage_telemetry` with no try/except when Fabric detection is true and the synapse import succeeded (`flaml/fabric/telemetry.py` lines 8–15 and 26–31). Any exception from that report aborts the import. An invalid `FLAML_FABRIC_RUNTIME` value also raises `ValueError` from `is_fabric_runtime()` during that import. Only `ModuleNotFoundError` is swallowed for the synapse import; other import/runtime failures propagate. The same pattern is in `flaml/fabric/logger.py` lines 6–16, which `flaml/automl/automl.py` imports at module load.
- **Risk**: A telemetry or synapse failure makes the whole library unimportable, including NumPy-only public use when the Fabric marker or override is set. Import also performs a usage report, which is outside the detection contract that must not contact services.
- **Suggested Fix**: Keep detection local. Call telemetry lazily, wrap the report in try/except, and never let telemetry or an invalid override fail `import flaml`. Catch synapse import failures other than a missing optional package the same way.

### Issue 2: Importing AutoML disables Spark ANSI for the whole process

- **Severity**: High
- **File**: flaml/automl/spark/__init__.py
- **Line(s)**: 5, 75–115
- **Description**: `flaml/automl/automl.py` imports this module, so `import flaml` reaches `disable_spark_ansi_mode()` whenever AutoML imports. If a Spark session is already active, the module sets `spark.sql.ansi.enabled` to false and registers restore only at process exit. `spark.conf.get` is outside the try (line 75), so a read failure fails the import after ANSI may already have been changed. A missing config is treated as enabled (line 80), but restore skips when the saved value is `None` (line 102), so the disable is permanent. `os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"` runs even when no session exists.
- **Risk**: One import changes cast/overflow semantics for every later query on that shared session, including non-FLAML work. A failed import can leave ANSI off with no atexit restore. Concurrent notebook queries observe the mutated session.
- **Suggested Fix**: Do not change Spark or process environment at import. Scope disable/restore to the Spark call in try/finally. If the original key is missing, unset it on restore rather than leaving false. Keep the no-session path that does not start a JVM.

### Issue 3: String MLflow version checks drop run isolation and can delete other runs

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 82–85, 282, 958–974, 994–1010
- **Description**: Comparisons such as `mlflow.__version__ > "2.5.0"` and `<= "2.17.0"` are lexicographic. `"2.10.0" > "2.5.0"` is false because `'1' < '5'`, so MLflow 2.10–2.99 take the legacy branch. `adopt_children` then searches with an empty filter. It `delete_run`s runs that started after `self.start_time` and share params, deletes empty runs, and can reparent runs with no parent. Tune enables this path (`experiment_type == "tune"` and autolog, `flaml/tune/tune.py` lines 552–565; `update_autolog_state` lines 456–462). The same comparison skips `extra_tags` in `_mlflow_wrapper` (line 282), so the sid tag that should isolate jobs is not set. A sid containing a quote also breaks the filter string at line 974.
- **Risk**: On current 2.10+ servers, concurrent Tune/AutoML jobs in one experiment can delete or reparent each other's runs. The comment claims this only happens at `<= 2.5.0`, which the comparison does not implement.
- **Suggested Fix**: Compare with `packaging.version.parse`. Always filter by the job sid, escape filter values, and never delete a run that does not carry this job's sid.

### Issue 4: Best-run lookup uses the search counter, not the logged-run list

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 843
- **Description**: Sequential search sets `_best_iteration` to `_track_iter` (`flaml/automl/automl.py` lines 3098 and 3274). `record_state` appends to `manual_run_ids` only when a trial returns a result (lines 3286–3288). A failed trial takes the else branch at lines 3330–3335 and does not append, but the loop counter still advances. `log_automl` then does `self.manual_run_ids[automl._best_iteration]` with no length check. Parallel search enumerates only successful trials, so this is specific to the sequential path.
- **Risk**: Any failed trial before the best successful trial raises `IndexError` after a model has already been selected. `fit` fails, and Issue 6 can replace that error if autolog restore also fails.
- **Suggested Fix**: Key logged runs by `_track_iter`, or store the winning run id when the best trial is recorded. If the id is missing, log and continue instead of indexing off the end.

### Issue 5: Model logging races with final retrain and failures are only warnings

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 860–905
- **Description**: `log_automl` submits lambdas that read `automl.model` and `automl.pipeline_signature` when the worker runs, not when submitted. It sets `has_model = True` immediately. `_search` calls `log_automl` at `flaml/automl/automl.py` line 3406, then may replace `_trained_estimator` during retrain at line 3528, and only then waits at line 3591. The retrain logging path at lines 3561–3566 is skipped because `has_model` is already true. `wait_futures` (lines 3595–3608) logs future exceptions as warnings. Worker threads also call `mlflow.start_run` / `end_run` (`log_model` lines 579–615) on a process-wide pool.
- **Risk**: The logged artifact can be the pre-retrain model, a model being mutated by retrain, or nothing, while `fit` still returns the retrained in-memory model. Fluent MLflow calls from pool threads can also attach artifacts to the wrong run.
- **Suggested Fix**: Snapshot the model and signature before submit. Wait for that log, or log only after retrain. Set `has_model` only after success. Do not use the fluent API from worker threads; pass an explicit run id to the client.

### Issue 6: Autolog restore in `finally` can hide the fit error and leave autolog off

- **Severity**: High
- **File**: flaml/automl/automl.py
- **Line(s)**: 90–100, 1920–1921
- **Description**: `fit` is wrapped so `resume_mlflow()` runs in `finally`. `resume_mlflow` (`flaml/fabric/mlflow.py` lines 909–922) calls `_mlflow.autolog` and `_set_flavor_autolog` with no catch. An exception there replaces the fit exception. Flavor restore deletes each config before the next restore, so a mid-loop failure permanently drops the remaining caller flavors. `__del__` also calls `resume_mlflow` (line 1106); exceptions there are ignored by the interpreter. Disable happens earlier by mutating global `AUTOLOGGING_INTEGRATIONS` (lines 447–454).
- **Risk**: A training failure is reported as an autolog error, or a successful fit is reported as a failure, and the caller's autolog state stays disabled. Concurrent fits can restore each other's snapshots incorrectly.
- **Suggested Fix**: Catch and log restore errors inside `finally` so they cannot replace the fit exception. Restore from a copy, and only clear a flavor after its `autolog` call succeeds. Serialize global autolog updates.

### Issue 7: Autolog with no active run disables final retrain and creates an experiment

- **Severity**: High
- **File**: flaml/automl/automl.py
- **Line(s)**: 2529–2536, 2714–2718
- **Description**: Default `mlflow_logging` is true (line 508) and `retrain_full` defaults to true (line 488). If any MLflow flavor autolog is enabled and there is no active run, fit still constructs `MLflowIntegration(..., only_history=False)`. The constructor then calls `mlflow.set_experiment("flaml_default_experiment")` (`flaml/fabric/mlflow.py` lines 380–386) and `MlflowClient()`. After that, lines 2714–2718 force `retrain_final = False` with no warning. An empty active `run_name` also does `tags["mlflow.runName"]` (line 379) and raises `KeyError` if the tag is absent. `get_experiment` at line 386 is unguarded if `experiment_id` is missing.
- **Risk**: A global `mlflow.autolog()` from unrelated code silently skips the full retrain, so the returned model can be the subsampled trial model. Tracking-server errors abort fit before training. This overrides the public retrain default.
- **Suggested Fix**: Do not change `retrain_final` because logging has no parent run. Log the retrained model only when a parent run exists. Isolate experiment creation so a tracking failure cannot abort training, and use `.get` for `mlflow.runName`.

### Issue 8: Spark whole-number conversion silently nulls integers outside 32-bit range

- **Severity**: High
- **File**: flaml/automl/data.py
- **Line(s)**: 1225–1229, 1254–1312
- **Description**: String columns whose sampled values are finite whole numbers are inferred as `"int"` and applied with `try_cast(... as int)` plus `try_cast(round(double) as int)`. Spark `int` is 32-bit. A value such as `"3000000000"` parses as a finite double with a zero fractional part, so it is inferred as int, then both casts return NULL under ANSI `try_cast`. The comments treat out-of-range doubles as NULL on purpose, but that also drops integers that fit in `bigint` and in double's exact integer range. No warning is raised when a converted column becomes all null.
- **Risk**: IDs, epoch milliseconds, and other large whole-number features become NULL before training. The model fits on missing data instead of failing.
- **Suggested Fix**: Infer `bigint` when any finite whole value is outside int32, and cast to `long`. If a cast nulls a value that parsed as a finite number, raise or warn instead of continuing silently.

### Issue 9: Legacy forecast fallback swallows unrelated KeyError and drops exogenous values

- **Severity**: Medium
- **File**: flaml/automl/time_series/ts_model.py
- **Line(s)**: 411–432
- **Description**: When `train_end_date` is missing, every `KeyError` from `predict` is treated as an in-sample index miss. The fallback calls `get_forecast(steps=len(X))` or `forecast(steps=len(X))`. The `forecast` branch does not pass `exog`. A missing exog column, a bad kwarg, or an internal statsmodels key error takes the same path.
- **Risk**: Saved models without a training boundary return a forecast from the training end, possibly without exogenous inputs, instead of the original error. The result can be misaligned with the requested dates.
- **Suggested Fix**: Catch only the index-miss case, or test membership before predict. Re-raise other `KeyError`s. Pass `exog` on every fallback forecast.

### Issue 10: Exported pipeline predict truncates labels with `astype(int)`

- **Severity**: Medium
- **File**: flaml/automl/pipeline.py
- **Line(s)**: 109–114
- **Description**: If a label transformer is present, `predict` does `np.asarray(prediction).astype(int).reshape(-1)` before `inverse_transform`. Non-integral scores truncate toward zero. NaN becomes a garbage integer on NumPy 1.x and raises on NumPy 2. A 2-d prediction is flattened, so the inverse-transform length no longer matches the rows. `transform` on `_TaskPreprocessor` (line 62) also uses the unfitted task string via `getattr(self, "task_", self.task)` instead of raising `NotFittedError`.
- **Risk**: A reconstructed classification pipeline can map a row to the wrong class, or fail only on some NumPy versions, after a successful fit.
- **Suggested Fix**: Require finite integral class indices and raise a clear error otherwise. Inverse-transform a 1-d vector of the same length as `X`. Raise `NotFittedError` before preprocess if `task_` is missing.

### Issue 11: Importing MLflow support starts a process-wide pool that can hang shutdown

- **Severity**: Medium
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 66–76
- **Description**: If MLflow is installed, importing AutoML imports this module and constructs `ThreadPoolExecutor(max_workers=cpu_count * 4)` immediately. `atexit` calls `shutdown(wait=True)`. The module also sets MLflow loggers to `CRITICAL` and ignores `FutureWarning` and `UserWarning` globally. The pool is shared by every integration, and submitted work calls MLflow. Forked Spark or joblib workers inherit non-joinable pool threads.
- **Risk**: A stuck logging call blocks interpreter exit. Fork-after-import can deadlock. Public users who never enable logging still pay the thread and warning-filter side effects, and MLflow errors are hidden.
- **Suggested Fix**: Create the pool lazily on first log, and recreate it after fork. Shut down with a timeout and `wait=False` on interpreter exit. Do not change global warning filters at import.

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

### Issue 9

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 10

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 11

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

## Driver assessment after stage 3, attempt 1

The original findings and outcome remain intact. Actual model `grok-4.7`
used 56 view and 53 search calls. The audit found no disallowed tools,
feedback access, or source drift.

| Issue | Disposition                                                                | Evidence and resolution                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Reporting/factory failures fixed; deliberate configuration errors retained | Optional telemetry reporting and diagnostic-logger creation now warn instead of aborting core work. Tests inject real callback failures. Invalid `FLAML_FABRIC_RUNTIME` values still raise, as the approved explicit configuration contract and its tests require; unrelated broken imports are not indiscriminately hidden. Fabric-specific telemetry remains distinct from the local-only detector.                                                  |
| 2     | Verified inherited concern, retained                                       | Public-base comparison already establishes the import-time Arrow setting and active-session ANSI behavior. The PR's no-session guard avoids JVM startup; the broader existing Spark lifecycle is not claimed fixed.                                                                                                                                                                                                                                    |
| 3     | Verified serious inherited concern, retained                               | Comparing `adopt_children` with public base lines 879-1001 shows only two tag-namespace substitutions. Version ordering, filter construction, and delete/reparent behavior are inherited. `_mlflow_wrapper` is AST-identical to base lines 250-275. This remains an explicit separate run-isolation concern, not a claimed fix.                                                                                                                        |
| 4     | Verified inherited concern, retained                                       | The dense manual-run indexing assumption was independently verified in previous stages. New expected AutoFE failures produce recorded infinite-loss trials rather than silently dropping result entries; the broader inherited failed-result case remains unresolved.                                                                                                                                                                                  |
| 5     | Verified inherited concern, retained                                       | Comparing `log_automl` with base lines 757-843 shows tag changes, a trial-metadata wait, and a timing log only. Deferred model reads, early `has_model`, and best-artifact/retrain ordering already exist. The new snapshot builder does not establish a fix for when the older deferred callback chooses its input.                                                                                                                                   |
| 6     | New failure masking fixed; other claims qualified                          | A secondary restoration failure is logged without replacing an original training exception; restoration failure after success remains visible. Current flavor snapshots are deleted only after a successful restoration hook, contrary to the claimed premature deletion. General concurrent global-autolog ownership remains an inherited limitation.                                                                                                 |
| 7     | Verified inherited concern, retained                                       | Existing no-parent retrain coercion and experiment selection were independently verified in earlier stages. Tracking errors are not silently converted to successful logging.                                                                                                                                                                                                                                                                          |
| 8     | Fixed and verified on real Spark                                           | Decimal-backed integer conversion preserves 64-bit values and scientific notation beyond float64 exactness. Full-column validation widens an inferred int32 type when sampling misses a large value; incompatible finite values raise rather than become null. Out-of-range sampled integers remain strings with a warning. Twelve real Spark tests cover ANSI on/off, int32/int64 bounds, precision, sparse sampling, booleans, and malformed inputs. |
| 9     | Fixed                                                                      | Legacy step forecasting is selected only from explicit generated-index metadata when no saved boundary exists. Unrelated prediction `KeyError`s propagate, and every forecast fallback retains exogenous inputs and kwargs.                                                                                                                                                                                                                            |
| 10    | Fixed for the new pipeline contract                                        | Predictions must have one finite integral class index per input row; valid single-column classifier output is retained. Task preprocessing tracks fitted state explicitly and raises `NotFittedError` rather than using an unfitted task string.                                                                                                                                                                                                       |
| 11    | Verified inherited concern, retained                                       | Import-time executor and warning-filter behavior is present in public base. Construction alone starts no worker threads. Blocking submitted work and concurrent/forked use remain separate inherited lifecycle risks.                                                                                                                                                                                                                                  |

The 399-test suite, twelve real local Spark tests, existing logging save/load
test, lint checks, and wheel/source build recorded in the sibling GPT artifact
pass. These driver resolutions do not alter the original `ISSUES_FOUND`
verdict or erase inherited findings.
