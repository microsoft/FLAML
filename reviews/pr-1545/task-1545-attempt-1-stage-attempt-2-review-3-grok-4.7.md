---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 3
timestamp: '2026-10-17T12:21:02Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 2603a1b944513e0e5e761fc2d410d32319d15dfd96ce2d19af733afc62c1988e
---

## Review Summary

- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-3-grok-4.7.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `flaml/automl/task/time_series_task.py` (`validate_data`, `prepare_data`, `preprocess`, `remove_ts_duplicates`) and `flaml/automl/time_series/ts_data.py` (`DataTransformerTS`, `TimeSeriesDataset.X_all` / `cv_train_val_sets`). Row-preserving transform makes the `iloc` train/val split safe; duplicate-timestamp handling is not.
- [x] Read `flaml/fabric/autofe.py` fit/transform/`CardinalitySelector`, `flaml/automl/state.py` rejection handler, and `flaml/tune/tune.py` sequential evaluation. Only two AutoFE failures become infinite loss; other `ValueError`s escape `tune.run`.
- [x] Read `flaml/automl/automl.py` imports and `flaml/fabric/mlflow.py` module init. With MLflow installed, AutoML import applies process-wide warning filters and starts a thread pool.
- [x] Read `flaml/automl/spark/__init__.py`. Import always overwrites `PYARROW_IGNORE_TIMEZONE` and, if a session is already active, changes ANSI mode outside a restore-safe try.
- [x] Read `flaml/automl/pipeline.py`, `flaml/automl/ml.py` (`train_estimator`, `get_val_loss`), forecast `predict`/`enrich`, and `.github/workflows/python-package.yml`. Integer-horizon `np.integer` mismatch was not filed: estimators already accept only `int`. Pandas pin condition does not match its comment.
- [ ] Did not execute tests or claim driver pytest/build results as proof. Pristine public base was not available, so findings are defects in the proposed tree, not proved regressions.

## Issues

### Issue 1: Duplicate forecast rows crash cleanup and conflicting timestamps are kept

- **Severity**: High
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 517-537
- **Description**: `validate_data` calls `remove_ts_duplicates` after timestamp coercion. `duplicated()` ignores `time_col` and matches full rows only, so identical timestamp rows with different targets or features are kept. If any full-row duplicate exists, `drop_duplicates()` runs and then `assert X[[X.columns[0]]].duplicated() is None`. `duplicated()` returns a Series, and `is None` is always false, so the assert always raises. It also checks column 0, not `time_col`.
- **Risk**: A forecast frame with an exact duplicate row aborts `fit` with `AssertionError` even after the duplicate was removed. Conflicting duplicate timestamps are not rejected and can shift frequency, `time_idx`, and validation windows.
- **Suggested Fix**: Deduplicate and validate on `time_col`. After cleanup, raise `ValueError` only when `X[time_col].duplicated().any()`. Do not use `assert` or `is None`.

### Issue 2: AutoFE candidate failures abort the whole search

- **Severity**: High
- **File**: flaml/fabric/autofe.py
- **Line(s)**: 441-461, 565-586
- **Description**: The `auto`/`force` space includes cardinality selection and PCA. `CardinalitySelector` can set an all-false mask when every `categorical__*` column exceeds the threshold and no numeric column remains. `pipeline.fit` then raises `ValueError` (PCA/variance on zero columns), or fit succeeds and the learner later raises `ValueError` on zero features. The `except` only converts the all-constant `VarianceThreshold` case into `AutoFEIncompatibleDataError`. `AutoMLState._compute_with_config_base` re-raises every other `ValueError`, and sequential `tune.run` (`flaml/tune/tune.py` line 925) does not catch it.
- **Risk**: One sampled featurization candidate, especially ID-like categorical data, kills the AutoML run instead of being scored as infinite loss.
- **Suggested Fix**: After a successful featurization fit, raise `AutoFEIncompatibleDataError` if the pipeline would emit zero columns. Convert expected fit failures (`ValueError`, and numerical `LinAlgError`) from the featurization pipeline into that error before the learner is fit.

### Issue 3: Importing AutoML with MLflow installed changes process-global state

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 66-78
- **Description**: `flaml/automl/automl.py` line 72 imports this module whenever MLflow is installed, including a plain `import flaml` with AutoML extras. Import creates a process-wide `ThreadPoolExecutor` (`cpu_count * 4` workers by default) and, because `FLAML_IS_CLEAN_LOGS` defaults to `"1"`, calls `warnings.simplefilter("ignore")` for `FutureWarning` and `UserWarning`. `atexit` shutdown uses `wait=True`.
- **Risk**: Public users who happen to have MLflow installed lose warnings from pandas, sklearn, and their own code, and pay for a large thread pool before any logging is requested. A stuck logging task can block interpreter exit. This makes internal logging behavior the default whenever the optional package is present.
- **Suggested Fix**: Create the executor lazily on first log submission and shut it down without blocking forever. Scope warning filters to MLflow calls, or default `FLAML_IS_CLEAN_LOGS` off outside an explicit Fabric logging opt-in.

### Issue 4: Spark helper import overwrites environment and can disable ANSI mode

- **Severity**: Medium
- **File**: flaml/automl/spark/__init__.py
- **Line(s)**: 5, 53-115
- **Description**: `automl.py` line 30 imports this module on every AutoML load. Line 5 assigns `PYARROW_IGNORE_TIMEZONE="1"` rather than `setdefault`, even when PySpark is absent. If a session is already active, `disable_spark_ansi_mode()` runs at import. `spark.conf.get("spark.sql.ansi.enabled")` is outside the try, so a missing key fails the import. `None` is treated as enabled, but restore skips a saved `None`, and the only restore is an atexit handler captured at import.
- **Risk**: AutoML import clobbers a caller’s Arrow timezone setting. An already-active Spark session can have ANSI mode turned off for the rest of the process, or `import flaml` can raise before user code runs. Import-then-start-Spark, the common order, is not protected anyway.
- **Suggested Fix**: Use `setdefault` for the env var. Do not change Spark settings at import. If a Spark operation must disable ANSI, do it in a `try/finally` around that operation, and pass a default to `spark.conf.get`.

### Issue 5: Forecast period and cv_step_size are not bounded

- **Severity**: Medium
- **File**: flaml/automl/task/time_series_task.py
- **Line(s)**: 298-317, 347-352
- **Description**: `period` is checked only with `isinstance(..., int)`. Zero or negative values pass. Holdout uses `assert period < num_samples`, which is true for `period <= 0`, and `move_validation_boundary(0)` does not create a holdout. For CV, `cv_step_size` defaults to `period`. When `step_size == 0` and the size check is true, line 311 divides by zero. A negative step builds validation windows with the wrong length and direction.
- **Risk**: `fit(..., period=0)` or `cv_step_size=0` either crashes or evaluates empty/misaligned folds and can select a model on invalid scores.
- **Suggested Fix**: Reject `period <= 0` and `cv_step_size <= 0` with `ValueError` before split construction. Require both to be positive integers.

### Issue 6: Ubuntu pandas pin covers non-Spark jobs

- **Severity**: Low
- **File**: .github/workflows/python-package.yml
- **Line(s)**: 139-144
- **Description**: The step comment says pandas is capped where PySpark is installed, but the condition is every Ubuntu job except Python 3.10. It does not require `matrix.test-type == 'spark'`. Ubuntu 3.11+ `notspark` jobs, which never install PySpark, still run `pip install "pandas>=2.0,<3"`.
- **Risk**: Linux CI does not exercise pandas 3 for the main AutoML/forecast suite, so a pandas 3 regression can pass CI. Windows jobs are unpinned and do not close that gap.
- **Suggested Fix**: Apply the pin only to jobs that install PySpark. Let `notspark` jobs install the current pandas extra.

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

## Driver assessment after stage 3, attempt 2

The original response and outcome are preserved. Actual model `grok-4.7`
used 55 view and 53 search calls; no disallowed tools, feedback reads, or
source drift were found. Stage 3 advances **with unresolved findings** under
High's two-attempt limit; it is not clean.

| Issue | Disposition                                        | Evidence and resolution                                                                                                                                                                                                                                                                               |
| ----- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Independently verified inherited concern, retained | `remove_ts_duplicates` is AST-identical to public base `17d293b01edc49734ccdb79ef04b68d0da72fab8`, lines 516-538. The assertion and conflicting-timestamp behavior predate this PR; no unrelated duplicate-policy rewrite was made.                                                                   |
| 2     | Fixed                                              | Cardinality's all-feature-removal case raises the same specific rejected-candidate error before PCA or the learner runs. Regression cases cover PCA and null extraction. Arbitrary `ValueError`s or programming errors are not broadly suppressed.                                                    |
| 3     | Independently verified inherited concern, retained | Existing import-time MLflow executor/filter behavior is documented in prior assessments. Constructor invocation alone starts no workers. No process-global lifecycle redesign is claimed.                                                                                                             |
| 4     | Independently verified inherited concern, retained | Existing Arrow assignment and active-session ANSI behavior were compared with public base in prior stages; the PR's no-session guard remains intact.                                                                                                                                                  |
| 5     | Independently verified inherited concern, retained | `prepare_data` is AST-identical to public base lines 222-322, and `decide_split_type` to lines 325-358. The nonpositive period/CV-step handling is an existing input-validation issue.                                                                                                                |
| 6     | Fixed across all affected setup steps              | The old workflow also installed PySpark in Ubuntu non-Spark jobs, contrary to the report's premise. Both the pandas cap and all three version-specific PySpark installers now require the Spark test type. The dependency-check script only reports installed packages and requires no policy change. |

The 414-test suite and build recorded in the sibling GPT artifact pass.
Workflow assertions and actionlint verify the complete Spark-only setup gate.
Non-Spark pandas resolution still follows the remaining dependency constraints;
this is not a claim that every supported Python version can install pandas 3.
