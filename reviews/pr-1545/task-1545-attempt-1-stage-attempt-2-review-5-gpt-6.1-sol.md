---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 5
timestamp: '2026-10-17T18:49:38Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: ca0a6537aa3504f720aee239a4c8353b63844b24967b3cf46b00541e0fd1b686
---

## Review Summary

- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-5-gpt-6.1-sol.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced feature-only fitting through `flaml\automl\pipeline.py:26-82`, target normalization in `flaml\automl\data.py:697-706`, and reconstruction/refit assertions in `test\fabric\test_autofe_isolation.py` and `test\fabric\test_pipeline_contracts.py`.
- [x] Compared visualization implementations with assertions in `test\fabric\test_visualization.py` and `test\fabric\test_visualization_contracts.py`, including objective naming, running-best values, categorical handling, and native Ray reporting.
- [x] Examined the Spark import regression guard, Spark session setup tests, and dependency/marker selection in `.github\workflows\python-package.yml`.
- [x] Compared the supplied classification-head test replacement with the maintained offline NLP selector; confirmed the selected tests exercise utilities and direct model execution rather than `AutoML.fit()`.
- [x] Considered the supplied driver-reported validation results, including 634 passes, the 51-test offline NLP subset, Ray reporting, and actual Spark dtype execution, without treating those results as coverage of the cases below.
- [ ] No commands or tests were executed by this reviewer. Findings are based on the supplied old/new diff and read-only source/test inspection.

## Issues

### Issue 1: Unsupervised feature-only refits still try to encode an absent target

- **Severity**: Medium
- **File**: `flaml\automl\pipeline.py`
- **Line(s)**: 79-80
- **Description**: `_TaskPreprocessor.fit_transform(X, y=None)` forwards the absent target to `DataTransformer.fit_transform()`, whose classification branch calls `LabelEncoder.fit_transform(None)` at `data.py:697-706`. Consequently, an otherwise unsupervised reconstructed recipe fails, for example `Featurization(config=[{"stage": "numerical", "method": "scaler_standard", "columns": ["a"]}], task="classification").fit(frame)`. An exported feature-only pipeline containing no supervised stage has the same problem when refitted with just `X`. Existing reconstruction/refit tests always supply `y`, so they miss this path.
- **Risk**: The advertised standalone reconstruction and feature-only transformer interfaces unnecessarily require labels even for scaling, PCA, or no-op preprocessing.
- **Suggested Fix**: Let feature-only preprocessing fit without target normalization when `y` is absent. Add reconstruction and cloned feature-pipeline tests using only `X` for unsupervised configurations; retain an explicit target requirement for supervised stages such as LDA.

### Issue 2: An initial NaN trial hides every subsequent running-best value

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 237-238
- **Description**: Running-best values use Python `min()`/`max()` over prefixes without handling NaN. For Tune scores `[NaN, 0.5, 0.2]` in minimization mode, every prefix starts with NaN, so the entire best-score line remains NaN despite two successful observations. AutoML histories have the equivalent problem after conversion to `1 - validation_loss`. `test_optimization_history_tracks_best_score` covers only finite observations.
- **Risk**: A failed or invalid first trial makes the optimization-progress line disappear and conceals later improvements.
- **Suggested Fix**: Calculate the running best while excluding invalid observations, leaving an undefined value only until a valid observation exists. Add assertions for an initial and intermediate NaN in both optimization modes and for AutoML history conversion.

### Issue 3: Parallel-coordinate plotting overwrites a hyperparameter named `score`

- **Severity**: Low
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 324-334
- **Description**: `hp_df["score"] = score` replaces an existing hyperparameter column with the objective. A valid Tune configuration containing a parameter named `score` therefore loses that parameter’s values in the plot, including when explicitly requested through `params=["score"]`. The auxiliary-metric collision test exercises `_get_tune_df()` but does not cover this separate parameter-column collision.
- **Risk**: The plot silently presents objective values under a hyperparameter’s name, producing misleading diagnostics.
- **Suggested Fix**: Use a collision-free objective/color column without modifying parameter columns. Add a real Tune regression case asserting both the requested `score` parameter’s dimension values and the independent objective/color values.

### Issue 4: The no-JVM regression guard is vacuous in the maintained CI matrix

- **Severity**: Medium
- **File**: `test\test_misc_coverage.py`
- **Line(s)**: 52-71
- **Description**: This unmarked test runs in `notspark` jobs, where the workflow no longer installs PySpark. Its subprocess then bypasses the Spark-context check through `if 'pyspark' in sys.modules`. Conversely, jobs that install PySpark select only `-m "spark"`, so they deselect this test. The active-session Spark tests do not exercise importing FLAML with PySpark installed and no JVM running.
- **Risk**: This is a coverage gap, not a demonstrated current JVM-start defect: removing the new no-active-session safeguard can escape its dedicated regression test.
- **Suggested Fix**: Run an installed-PySpark version of this fresh-process guard in the Spark jobs, and assert that PySpark is actually available before checking that FLAML import leaves the Spark context inactive. Keep the separate no-optional-dependencies import checks.

### Issue 5: Offline NLP coverage removes the end-to-end AutoML training guard

- **Severity**: Medium
- **File**: `.github\workflows\python-package.yml`
- **Line(s)**: 208-250
- **Description**: Both main test selectors now ignore `test\nlp`. The replacement offline step selects utility tests and the rewritten classification-head test, which calls `load_model()`, performs a direct forward pass, and backpropagates without invoking `AutoML.fit()` or `TrainerForAuto`. The supplied diff shows that the replaced head-switch tests previously exercised `AutoML.fit()`. Other NLP fit/predict/CV tests remain excluded, while the GPU helper is not pytest-collected.
- **Risk**: The 51 reported offline passes do not protect FLAML’s NLP training integration against regressions in shared AutoML orchestration, training arguments, trainer construction, or prediction. This is missing integration coverage, not proof that those paths currently fail.
- **Suggested Fix**: Reuse the tiny local checkpoint/tokenizer fixtures for a bounded offline `AutoML.fit()` and prediction smoke test, covering classification and regression head adaptation. Collect it in the maintained offline step and assert trained-model availability, output shape, and finite results.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Feature-only preprocessing reaches target encoding when no target was supplied.
- **How verified**: Source call-chain inspection; reproduction not executed.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Prefix reduction does not exclude an initial NaN.
- **How verified**: Inspected the reduction and existing finite-only assertions; tests not executed.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Objective insertion overwrites a valid parameter column.
- **How verified**: Inspected DataFrame construction and collision coverage; tests not executed.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Dependency installation and marker selection prevent the guard from testing installed PySpark.
- **How verified**: Cross-checked workflow selectors, test marking, and the conditional subprocess assertion.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: The maintained offline subset does not replace the removed end-to-end training coverage.
- **How verified**: Compared the supplied test replacement, current selected test bodies, and CI commands; tests not executed.

## Driver assessment after stage 5, attempt 2

Original findings and verdict are retained. Actual model `gpt-6.1-sol`
used 45 view and 18 search calls; evidence, independence, and source-integrity
audits passed. Stage 5 advances **with unresolved findings** under High's
two-attempt cap, not clean.

| Issue | Resolution before stage 6                                                                                                                                                                             | Verification                                                                                                                                                                                                                                      |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Feature-only task preprocessing skips target normalization when `y` is absent. LDA retains an explicit target-label requirement.                                                                      | Unsupervised reconstructed scaling and cloned feature-only PCA fit with `X` alone; a supervised LDA case rejects missing labels clearly.                                                                                                          |
| 2     | Running-best calculation uses a linear cumulative reduction over finite observations with forward filling after the first valid score.                                                                | Initial/intermediate NaNs are covered for both Tune directions and AutoML's transformed score history.                                                                                                                                            |
| 3     | Parallel coordinates use a collision-free objective column rather than replacing a parameter named `score`.                                                                                           | A real Tune result preserves the score-named parameter's values and independent objective colors.                                                                                                                                                 |
| 4     | Added a Spark-marked fresh-process import guard that requires PySpark to be installed, then asserts no active Spark context before and after importing FLAML. Existing dependency-free guards remain. | The installed-PySpark case passes in the real Spark validation environment and is selected by the maintained Spark CI invocation.                                                                                                                 |
| 5     | The maintained offline head fixture now also runs bounded `AutoML.fit()` and prediction for classification/regression adaptation, with a tiny local model/tokenizer.                                  | The new test exposed a real modern-Transformers `TrainerForAuto.log` signature mismatch, now fixed by forwarding supported positional/keyword arguments. All 53 offline NLP tests pass, including actual AutoML/Trainer execution and prediction. |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ml_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\automl\test_split.py::test_group_split_with_sample_weight_alignment test\automl\test_split.py::test_group_split_with_sample_weight_series_alignment test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py test\test_misc_coverage.py -m "not spark" -n 2 --dist=loadfile -q --tb=short --disable-warnings
643 passed, 2 optional skips

python -m pytest test\nlp\test_hf_utils_coverage.py test\nlp\test_autohf_classificationhead.py -q --tb=short --disable-warnings
53 passed with offline flags and local fixtures

python -m pytest test\fabric\test_ray_reporting.py test\fabric\test_visualization_contracts.py -q --tb=short --disable-warnings
Ray 2.4.0 environment: 26 passed

python -m pytest test\test_misc_coverage.py::TestFlamlInit::test_installed_spark_import_does_not_start_jvm -q --tb=short --disable-warnings
Installed PySpark environment: 1 passed

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

Some initial plot assertions required conversion of Plotly's numeric
object-dtype arrays before NumPy comparison; their numeric expectations were
not weakened. The offline NLP test was corrected to inspect FLAML's documented
model/tokenizer bundle rather than assume a bare estimator object.
