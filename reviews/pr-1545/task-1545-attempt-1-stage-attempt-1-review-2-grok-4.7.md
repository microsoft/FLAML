---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 2
timestamp: '2026-10-17T08:00:01Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 518c0d52b87ba7193ab1c2bb32aafb0218c7d9526c338cf9131aedbf3f814374
---

## Review Summary

- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-2-grok-4.7.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced the public import path `flaml/__init__.py` (line 11) → `flaml/tune/__init__.py` (line 38) → `flaml/tune/tune.py` (line 40) and `flaml/automl/automl.py` (lines 63–70) into `flaml/fabric/mlflow.py` module-level side effects (lines 64–77) and `MLflowIntegration.__init__` (lines 369–390).
- [x] Read AutoML fit policy in `flaml/automl/automl.py` (lines 2464–2470, 2567–2585, 2757–2769, 3737–3740), `flaml/automl/pipeline.py`, `flaml/automl/__init__.py`, and the optional AutoFE guards in `flaml/automl/ml.py` and `flaml/automl/state.py`.
- [x] Read `flaml/visualization/__init__.py`, `flaml/fabric/visualization.py` imports and `_get_aml_df`, `flaml/fabric/__init__.py`, `flaml/fabric/README.md`, `pyproject.toml` extras (`synapse`, `fabric_python`) and ruff `ignore` (`F403`), plus `notebook/trident/generate_test.py` (line 17) and `notebook/trident/demo_1_flight_delays_automl.ipynb` (line 51).
- [x] Driver-reported pytest and build results were not treated as proof of design correctness. This review did not execute tests.
- [ ] Live Fabric delivery and Docker builds were not inspected; the assignment forbids executing commands and contacting services.

## Issues

### Issue 1: Importing public FLAML mutates process-global MLflow state

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 64–77
- **Description**: `import flaml` loads `flaml.tune`, which imports `flaml.fabric.mlflow` whenever MLflow is installed. That module, at import time, sets `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR`, starts a process-wide `ThreadPoolExecutor` (`cpu_count * 4` by default) with an `atexit` shutdown, and, because `FLAML_IS_CLEAN_LOGS` defaults to `"1"`, raises third-party loggers to `CRITICAL` and globally ignores `FutureWarning` and `UserWarning`. `AutoML` repeats the same import. None of this waits for `mlflow_logging`, an active run, or autolog intent.
- **Risk**: Installing optional MLflow makes the internal logging runtime the default for every public import. User and dependency warnings disappear, MLflow logging is silenced, and interpreter exit can block on the import-time pool. Explicit opt-out cannot undo work already done at import.
- **Suggested Fix**: Keep `fabric.mlflow` import lazy, and move the environment write, thread pool, logger levels, and warning filters into a context manager entered only when a logging integration is actually constructed. Restore previous filters and levels on exit. Do not start workers or register `atexit` at module import.

### Issue 2: Logging integration overrides training policy and the active experiment

- **Severity**: High
- **File**: flaml/automl/automl.py
- **Line(s)**: 2567–2585, 2757–2769
- **Description**: With MLflow installed and `mlflow_logging` left at its default, `fit()` treats any enabled autolog integration as logging intent and constructs `MLflowIntegration(..., only_history=False)` even when there is no active run. That constructor (`flaml/fabric/mlflow.py` lines 380–386) calls `mlflow.set_experiment("flaml_default_experiment")` when no experiment is active. `fit()` then forces `retrain_final = False` whenever `parent_run_id` is missing and `only_history` is false, after the normal `retrain_full` calculation and without a warning. The same constructor path is used by `tune.run()` (`flaml/tune/tune.py` lines 561–565).
- **Risk**: `mlflow.autolog()` without an active run silently changes the fitted model (no full retrain) and redirects later `start_run()` calls into FLAML's experiment. Logging policy is coupled to training policy, so an explicit `retrain_full=True` does not win.
- **Suggested Fix**: Keep autolog as a logging signal only. Do not change `retrain_final` from the integration. If there is no active run, record in-memory history or require the caller to pass an experiment; do not call `set_experiment` as a side effect of fit. If a combination is unsupported, raise rather than coerce.

### Issue 3: Public notebooks still install from an internal release feed

- **Severity**: High
- **File**: notebook/trident/generate_test.py
- **Line(s)**: 17
- **Description**: The consolidated public tree still tells users to install `flaml[synapse]` from `https://automlsaeastus.blob.core.windows.net/releases/FLAML-latest-py3-none-any.whl`. That URL is the install instruction in `notebook/trident/demo_1_flight_delays_automl.ipynb` line 51 and the other Trident demo notebooks, and `generate_test.py` writes it into `all_in_one_test.ipynb`. Committed outputs in `notebook/trident/FLAML Demo - Overview.ipynb` also record `synapsemldatascience.blob.core.windows.net`. Website docs do not use these feeds. This contradicts the approved boundary that GitHub is the only maintained library source and internal feeds are excluded.
- **Risk**: Readers of the public notebooks install an unmanaged `FLAML-latest` wheel that can diverge from the GitHub 2.7.0 package, including a private `synapse` extra payload. The public repo remains a client of the internal feed instead of the source the feed should consume.
- **Suggested Fix**: Point notebook install cells and `generate_test.py` at the GitHub or PyPI distribution and a public extra (`fabric_python` or `automl`). Remove internal blob URLs from source cells and regenerated outputs.

### Issue 4: Ensemble silently discards an explicit featurization choice

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 2464–2470
- **Description**: `fit()` validates `featurization` as `auto`, `off`, or `force`, then, if `ensemble` is set, assigns `featurization = "off"` with only a TODO comment. The incompatible `resampler` plus `ensemble` combination raises `ValueError` (`flaml/automl/automl.py` lines 605–612). The fit docstring says an explicit featurization value overrides `FLAML_FEATURIZATION`, but it does not say ensemble erases that value. The Fabric README documents the exclusion; the API does not enforce it.
- **Risk**: `AutoML(featurization="force", ensemble=True)` succeeds and returns a model with no AutoFE pipeline. Callers cannot distinguish an honored setting from a dropped one, and exported-pipeline composition never starts.
- **Suggested Fix**: Reject `ensemble` combined with `featurization` other than `"off"` using the same `ValueError` pattern as the resampler check, or warn and record the coerced value on the result. Do not drop an explicit setting silently.

### Issue 5: Public visualization facade is an unbounded star import

- **Severity**: Medium
- **File**: flaml/visualization/__init__.py
- **Line(s)**: 1
- **Description**: The public module is `from ..fabric.visualization import *`. `flaml/fabric/visualization.py` defines no `__all__` and imports Plotly, pandas, and NumPy at module level (lines 5–10). Star import therefore publishes those modules plus `AutoML`, tune types, and every public helper as `flaml.visualization` attributes. Ruff ignores `F403`, so this is not a lint miss; it is the public boundary. Plotly is only in the `synapse` and `fabric_python` extras, not core or `automl`.
- **Risk**: `import flaml.visualization` fails without an optional extra, and any new import or helper in the Fabric module becomes public API. The stable plot functions cannot be versioned separately from implementation imports.
- **Suggested Fix**: Give `flaml.fabric.visualization` an explicit `__all__` of plot and importance functions, re-export that list from `flaml.visualization`, and import Plotly inside the plot functions or behind a clear optional-dependency error.

### Issue 6: Optional Fabric features are wired as one import bundle

- **Severity**: Medium
- **File**: flaml/automl/__init__.py
- **Line(s)**: 1–9, 16–22
- **Description**: `flaml.automl` unconditionally imports `Featurization` from `flaml.fabric.autofe`, while `flaml/automl/ml.py` and `flaml/automl/state.py` still treat AutoFE as optional. Separately, `flaml/automl/automl.py` lines 63–84 and `flaml/tune/tune.py` lines 36–55 import Kusto logging, MLflow integration, and telemetry in one `try`. `flaml.fabric.mlflow` imports MLflow at module level, so a missing MLflow install disables telemetry and substitutes a local `KustoLogger` that does not match `flaml/fabric/logger.py`. The package-level fallbacks are dead when Fabric imports succeed, and divergent when they fail.
- **Risk**: An AutoFE import failure takes down all of `flaml.automl` and is reported as "install flaml[automl]". A missing MLflow install disables Fabric telemetry that does not need MLflow. Callers cannot depend on one logger or one optional-feature boundary.
- **Suggested Fix**: Lazy-import `Featurization` behind the existing `ml.py` guard, and split logger, telemetry, and MLflow imports so each missing dependency disables only that feature. Use one no-op logger from `flaml.fabric.logger`.

### Issue 7: Exported pipeline aliases live AutoML state

- **Severity**: Medium
- **File**: flaml/automl/pipeline.py
- **Line(s)**: 80–91
- **Description**: `make_automl_pipeline` says it builds an artifact from the supplied model, but it stores `automl._state.task` on `_TaskPreprocessor` and appends the estimator object itself as the last step. `AutoML.automl_pipeline` (`flaml/automl/automl.py` lines 3737–3740) passes `self.model` through on every access. No clone is taken. A later `pipeline.fit` therefore calls `fit` on the same estimator AutoML holds, and a later `AutoML.fit` mutates the task object the exported pipeline still references.
- **Risk**: Refit, registration, and final retraining are not independent compositions. Mutating or refitting the exported pipeline changes `automl.model`, and a second search changes pipelines already handed to MLflow or the caller.
- **Suggested Fix**: Clone the estimator and copy the task or preprocessing state needed for transform before building the pipeline. Document that `automl_pipeline` is a snapshot, and return `None` or a detached pipeline consistently when AutoFE is absent.

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

## Driver assessment after stage 2, attempt 1

Original findings and verdict are retained. Runtime identity was `grok-4.7`;
the audit found 52 search, 47 view, and 1 glob calls, no disallowed tools or
prior/sibling feedback access, and no source drift during the batch.

| Issue | Disposition                          | Evidence and resolution                                                                                                                                                                                                                                                                                                                                            |
| ----- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Verified inherited concern, retained | The public base already has the progress environment assignment, executor construction, clean-log default, and warning filters (`flaml/fabric/mlflow.py:55,58,61,67-68`), as independently checked in stage 1. Constructing `ThreadPoolExecutor` does not itself start workers. No unrelated process-global logging redesign was made.                             |
| 2     | Verified inherited concern, retained | Existing public experiment selection and the no-parent `retrain_final=False` guard were independently verified in stage 1. New history-only paths are isolated, including the newly covered Spark latency probe; active-autolog baseline policy is not claimed fixed.                                                                                              |
| 3     | Fixed                                | Nine demo install cells and the combined-notebook generator now install public GitHub main. Stale install outputs were cleared. The `synapse` extra is declared publicly and retained for demos requiring Spark; it is not a private package source. Source scanning and isolated generator/schema validation pass without executing workloads.                    |
| 4     | Fixed compatibly                     | The documented ensemble exclusion is preserved, but it now emits a warning under the current fit's logging configuration and records effective feature mode `off`. This avoids a new compatibility-breaking exception.                                                                                                                                             |
| 5     | Fixed                                | Both modules have an explicit, bounded export surface; the facade no longer exposes NumPy, AutoML, or incidental implementation imports. Missing Plotly raises an actionable optional-extra error. Export identity and dependency-error regressions pass.                                                                                                          |
| 6     | Fixed                                | Logger and telemetry initialization are independent of MLflow, using the existing shared logger implementation. Telemetry depends on the lightweight Tune logger formatter instead of importing the AutoML package. The package-level Featurization export is lazy; explicitly selecting unavailable AutoFE raises a clear error instead of silently disabling it. |
| 7     | Fixed                                | The shared builder deep-copies the fitted pipeline graph, including task, preprocessing, and learner. `sklearn.clone` alone would discard fitted state and was not used as a snapshot substitute. Refitting a returned pipeline leaves the original model unchanged; the additional memory cost is documented.                                                     |

The 332-test suite, installed core/AutoFE smoke checks, build, and subsequent
10-test metadata/generator run recorded in the sibling GPT artifact passed.
Missing-MLflow and missing-AutoFE subprocesses verify independent feature
boundaries. Earlier inherited concerns remain visible, and these driver
dispositions do not change the original `ISSUES_FOUND` outcome.
