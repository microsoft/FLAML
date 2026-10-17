---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 2
timestamp: '2026-10-17T09:20:18Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: b1bfed5139066b487406193dfb45fecf9b4733e97337ae42ceb89a5374215306
---

## Review Summary

- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-2-grok-4.7.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced the public import chain in `flaml/__init__.py` (lines 4–7, 14–27), `flaml/automl/automl.py` (lines 30, 71–76, 2528–2536), `flaml/tune/tune.py` (lines 35–43), and `flaml/fabric/visualization.py` (lines 18–25).
- [x] Inspected import-time process mutation in `flaml/fabric/mlflow.py` (lines 64–77) and `flaml/automl/spark/__init__.py` (lines 5, 114–115), against the opt-in contract in `flaml/fabric/README.md` and `flaml/fabric/__init__.py`.
- [x] Compared visualization objective handling in `flaml/fabric/visualization.py` (lines 54–62, 228–231, 318–319, 365–366) with `flaml/fabric/fanova/evaluator.py` (lines 48, 74, 189–198) and `test/fabric/test_visualization_contracts.py`.
- [x] Compared AutoFE composition in `flaml/automl/automl.py` `feature_transformer` (lines 828–836), `flaml/automl/pipeline.py` `make_automl_pipeline` (lines 114–127), and `flaml/automl/ml.py` `train_estimator` (lines 471–477).
- [ ] Pristine public-base comparison was not completed: reading the archive was denied without an interactive approval. Introduced-versus-unchanged status for the Spark helper was not inferred from that archive.
- [ ] Tests, builds, and actionlint were not executed by this review. Driver-reported pytest and build results were not treated as proof of these design contracts. Notebook, NOTICE, and benchmark hash summaries were not treated as behavioral evidence.

## Issues

### Issue 1: Optional MLflow presence activates Fabric runtime side effects at import

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 64–77
- **Description**: `import flaml` loads AutoML (`flaml/__init__.py` lines 4–7), which eagerly imports `flaml.fabric.mlflow` whenever MLflow is installed (`flaml/automl/automl.py` lines 71–76). `flaml/tune/tune.py` lines 35–43 and `flaml.visualization` (via `flaml/fabric/visualization.py` line 18) do the same. That module, at import and before any fit, active run, autolog intent, or Fabric check, starts a `ThreadPoolExecutor` sized to `cpu_count * 4`, registers an atexit shutdown, sets `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR`, raises several third-party loggers to CRITICAL, and globally ignores `FutureWarning` and `UserWarning`. `FLAML_IS_CLEAN_LOGS` defaults to `"1"`, so this is the default. This contradicts the lazy boundary used by `register_automl_pipeline` and the README statement that MLflow is imported when registration is called, not merely because AutoML is imported.
- **Risk**: A normal public import in any environment that happens to have MLflow hides unrelated warnings, changes MLflow behavior for the whole process, and reserves a large thread pool even when logging is off and the user is not on Fabric.
- **Suggested Fix**: Keep `flaml.fabric.mlflow` import inside the fit/tune/register call that has already decided logging is active. Move the thread pool, environment write, logger levels, and warning filters into that same opt-in path, and scope warning filters so they are restored. Do not run them at module import.

### Issue 2: AutoML plot types do not share one objective direction

- **Severity**: Medium
- **File**: flaml/fabric/visualization.py
- **Line(s)**: 61–62, 228–231, 318–319, 365–366
- **Description**: `_get_aml_df` rewrites `validation_loss` to `1 - loss` and `plot_optimization_history` treats the larger value as better (`comp = max`, axis title "Score (1 - validation loss)"). `plot_parallel_coordinate` and `plot_contour` hardcode AutoML as `better = "min"` and therefore select the reversed color scale. The fANOVA adapter separately creates its study with `direction="maximize"` (`flaml/fabric/fanova/evaluator.py` line 74). Tune plots consult `default_mode`; AutoML plots do not. `test_optimization_history_tracks_best_score` locks in the higher-is-better transform, but no contract test covers the color-scale direction.
- **Risk**: Parallel-coordinate and contour plots color the worse AutoML trial as better. Callers cannot interpret "score" consistently across the public `plot()` entry point.
- **Suggested Fix**: Define one AutoML objective direction next to the score transform and use it for history, color scales, and fANOVA. If the public score remains `1 - validation_loss`, AutoML color scales must use the max-is-better scale.

### Issue 3: Public visualization reads an integration the public default never creates

- **Severity**: Medium
- **File**: flaml/fabric/visualization.py
- **Line(s)**: 54–55
- **Description**: Every AutoML plot goes through `_get_aml_df`, which reads `aml.mlflow_integration.infos` with no guard. `fit()` initializes that attribute to `None` and constructs it only when MLflow is installed, logging is enabled, and either Fabric runtime or an active run/autolog is present (`flaml/automl/automl.py` lines 2528–2536). `model_history=True` does not create it. Search state keeps the best config, not the trial history these plots require. The public facade re-exports these functions, while contract tests inject a fake integration and never cover the default public object.
- **Risk**: `flaml.visualization.plot(automl, "optimization_history")` raises `AttributeError` after an ordinary off-Fabric fit. Enabling model history still does not produce a plottable history. Trial history is an optional logging side channel rather than part of the AutoML result.
- **Suggested Fix**: Store plot history on the AutoML result independently of MLflow, or have the public plot API raise a documented error when history was not collected. Do not require Fabric or an active MLflow run for the public facade to have a defined result.

### Issue 4: Importing AutoML mutates process and Spark state

- **Severity**: Medium
- **File**: flaml/automl/spark/__init__.py
- **Line(s)**: 5, 114–115
- **Description**: `flaml/automl/automl.py` line 30 imports the Spark helper on every AutoML load, including the NumPy-only public import and the no-PySpark path. That helper assigns `PYARROW_IGNORE_TIMEZONE` to `"1"` rather than using `setdefault`, then calls `disable_spark_ansi_mode()` and registers its restore function with atexit. The helper avoids starting a JVM when no session exists, but if a session is already active it changes `spark.sql.ansi.enabled` and pandas-on-Spark ANSI options for the rest of the process.
- **Risk**: Importing the public library overwrites a user PyArrow setting and can change SQL semantics of an unrelated active Spark session until exit. That is a broader side effect than the Fabric detector, which correctly avoids Spark.
- **Suggested Fix**: Set the PyArrow variable only when a Spark feature is requested, and only if the user has not set it. Apply ANSI changes inside the Spark training path and restore them before returning, not at import.

### Issue 5: AutoFE is exposed through inconsistent public compositions

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 828–836
- **Description**: Fitted AutoFE is a side attribute, `estimator.autofe`, which `predict`, `predict_proba`, and `score` each apply manually. `feature_transformer` then builds a live `Pipeline([("precessor", data_precessor), ("autofe", autofe)])` that aliases the fitted transformer and uses a misspelled public step name. `make_automl_pipeline` instead deep-copies a different schema, `feature_transformer` / `autofe` / `estimator` (`flaml/automl/pipeline.py` lines 114–127). `train_estimator` uses a third shape when `X_train` is None: an unfitted `Pipeline([("autofe", ...), ("estimator", ...)])` (`flaml/automl/ml.py` lines 474–477). The README's detached-snapshot guarantee applies only to the export path.
- **Risk**: Fitting or cloning `feature_transformer` refits the transformer still used by `AutoML.predict`. `named_steps["preprocessor"]` does not exist. `model.predict` skips AutoFE unless the caller remembers the side channel. The three public objects cannot be substituted for one another.
- **Suggested Fix**: Build one fitted, detached pipeline and use it for `feature_transformer`, export, and inference. Keep the step name consistent (`feature_transformer` or `preprocessor`), and stop sharing the live `autofe` object.

### Issue 6: fANOVA patches a private Optuna class for the whole process

- **Severity**: Medium
- **File**: flaml/fabric/fanova/evaluator.py
- **Line(s)**: 48, 189–198
- **Description**: Constructing `FanovaImportanceEvaluator` replaces `optuna.importance._fanova._tree._FanovaTree._get_node_value` on the imported class. Public `plot_param_importance` constructs that evaluator. The replacement is process-global, idempotent only by function name, and coupled to a private Optuna symbol. It is not limited to FLAML trials, and the README's "does not mutate" statement covers only the search space.
- **Risk**: Other Optuna fANOVA users in the process inherit FLAML's scalar coercion. An Optuna upgrade that moves or renames `_get_node_value` breaks public hyperparameter-importance plots, or the patch silently stops matching.
- **Suggested Fix**: Coerce node values at the FLAML boundary, or wrap a local tree type. Do not assign over Optuna's private class method.

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

## Driver assessment after stage 2, attempt 2

The original response and outcome are preserved. Actual model `grok-4.7`
used 96 view and 50 search calls; the audit found no disallowed tools,
feedback access, or source drift. One optional baseline-directory read was
denied for lack of interactive approval. It was not an independent baseline
comparison, and no such signoff is claimed.

Stage 2 advances **with unresolved findings**, not clean, under High's
two-attempt gate. Before stage 3:

| Issue | Disposition                                                                | Evidence and resolution                                                                                                                                                                                                                                                                                                                                                  |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Verified inherited concern, retained                                       | The environment write, executor construction, logger levels, and warning filters already exist in public base. Prior artifacts retain exact baseline locations. Construction does not itself start worker threads; this review does not claim a process-global logging redesign.                                                                                         |
| 2     | Fixed                                                                      | A shared AutoML score-direction constant drives optimization history and both color-scale choices. Actual parallel-coordinate and contour color sequences are asserted as higher-is-better.                                                                                                                                                                              |
| 3     | Fixed with explicit precondition                                           | History-dependent plotting raises a descriptive `ValueError` when trial history is unavailable. This preserves public memory/logging defaults instead of silently enabling history collection. A real public fit with retained models but no trial history exercises the error.                                                                                          |
| 4     | Verified inherited concern, retained                                       | Comparing `flaml/automl/spark/__init__.py` against public base shows the PyArrow assignment and import-time disable/restore calls are unchanged. The PR adds a no-active-session guard to prevent JVM startup; it introduces no new active-session mutation.                                                                                                             |
| 5     | New AutoFE composition defects fixed; intentional API differences retained | AutoFE-enabled `feature_transformer` now uses a detached, refittable task-aware pipeline and the same `feature_transformer`/`autofe` step names as export. It copies preprocessing only, not the learner. Raw `model`, feature-only transformer, and unfitted log recipe have different existing purposes; they are documented rather than made falsely interchangeable. |
| 6     | Fixed locally                                                              | Scalar node-value compatibility is implemented by local Optuna 2/3 subclasses used only by each FLAML evaluator. No Optuna class method is replaced. A fresh subprocess computes importance and verifies the original Optuna method identity is unchanged. The version-bounded private adapter dependency remains explicit.                                              |

The 373-test suite, two existing AutoFE integration tests, and wheel/source
build recorded in the sibling GPT artifact pass. Ray and imbalanced-learn
remain optional local skips. The original inherited concerns are not erased
by these fixes or by any later clean stage.
