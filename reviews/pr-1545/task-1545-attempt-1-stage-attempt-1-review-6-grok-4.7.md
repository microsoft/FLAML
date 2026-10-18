---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 6
timestamp: '2026-10-17T19:44:12Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 8f84cbfe9c6c7643757c786327ba2935a84c43ff279ff8ce617b6c513a869ab7
---

## Review Summary

- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-6-grok-4.7.md
- **Issues Found**: 10
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `.coveragerc` omit list (line 16) against the offline NLP `coverage run` step in `.github/workflows/python-package.yml` (lines 236–249) and the `codecov.yml` ignore list.
- [x] Read fixed pickle paths in `test/spark/test_0sparkml.py` (171–172), `test/spark/test_automl.py` (60–61), `test/automl/test_regression.py` (176–177), `test/nlp/test_autohf.py` (28), and `test/nlp/test_autohf_custom_metric.py` (80), against `test/_artifact_paths.py`.
- [x] Read `auto_convert_dtypes_pandas` docstring (`flaml/automl/data.py` 1462–1468) against the full-column integer/numeric scans (1688–1695, 1796–1816) and `test/automl/test_data_coverage.py` (736–737).
- [x] Read `notebook/trident/automl_plot.ipynb` source line 545, `flaml/fabric/fanova/README.md`, and `benchmark/pmlb/analyze.ipynb` source lines 1214–1227.
- [x] Read `flaml/fabric/visualization.py` line 196, `flaml/fabric/mlflow.py` signature logs (179–204), public AutoML docstring typos (`flaml/automl/automl.py` 304–327 and 2177–2199), `flaml/fabric/autofe.py` line 603, and `benchmark/pmlb/runbenchmark.py` line 18.
- [ ] Tests, actionlint, and package builds were not executed by this reviewer. Driver-reported results were not treated as proof.
- [ ] No pristine-base archive was available, so introduced-versus-existing provenance is limited to the current tree and the supplied manifest.

## Issues

### Issue 1: NLP coverage collection is discarded by the omit list

- **Severity**: Medium
- **File**: `.coveragerc`
- **Line(s)**: 16
- **Description**: `.coveragerc` omits `*/flaml/automl/nlp/*`. The workflow still runs `test/nlp/test_hf_utils_coverage.py` and `test/nlp/test_autohf_classificationhead.py` under `coverage run` with `COVERAGE_PROCESS_START` (`.github/workflows/python-package.yml` 241–249). Those executions cannot appear in the combined Codecov report. `codecov.yml` does not ignore `flaml/automl/nlp`.
- **Risk**: The new offline NLP job looks like coverage evidence, but NLP source coverage stays invisible. Regressions in the public NLP path will not show up in the coverage comment or patch report.
- **Suggested Fix**: Remove `*/flaml/automl/nlp/*` from the omit list if that job is meant to measure coverage. If exclusion is intentional, drop `COVERAGE_PROCESS_START` from that step and state in the workflow comment that NLP is executed but excluded from coverage.

### Issue 2: Pickle tests still use shared working-directory paths

- **Severity**: Medium
- **File**: `test/spark/test_automl.py`
- **Line(s)**: 60–61
- **Description**: Spark and non-Spark tests pickle to fixed CWD names and do not use `artifact_path` or `tmp_path`. `test/spark/test_automl.py` and `test/automl/test_regression.py` both use `automl_xgboost_spark.pkl`. `test/spark/test_0sparkml.py` uses `automl_spark.pkl`. Both NLP tests use `automl.pkl`. No cleanup is present in those Spark tests. The notspark job runs pytest with `-n 2 --dist=loadfile`; Spark and NLP are separate jobs, so the published matrix does not collide those files, but a local full run with xdist can.
- **Risk**: Parallel workers can overwrite each other's pickles and fail or assert against the wrong model. Leftover pickles and `seqclass.log` remain in the worktree after a run.
- **Suggested Fix**: Route every pickle and log through `artifact_path` or `tmp_path`, and give each module its own filename. Keep the Spark job serial for the JVM, but do not rely on job isolation for filesystem safety.

### Issue 3: `sample_ratio` docs overstate what pandas numeric inference samples

- **Severity**: Medium
- **File**: `flaml/automl/data.py`
- **Line(s)**: 1462–1468, 1688–1695, 1796–1816
- **Description**: The public docstring says `sample_ratio` is the fraction of rows used for type inference and that `1.0` forces full-frame inference. The integer regex and `pd.to_numeric` paths always scan `full_cleaned`. Sampling applies to later timedelta, datetime, and category heuristics. `test_sample_ratio_caps_inference_but_preserves_full_data` repeats the cost claim while its integer column takes the full scan. The full scan is an intentional all-or-nothing safety check, not a sampling bug.
- **Risk**: Callers who set `sample_ratio=0.1` to cut inference cost still pay a full string and numeric pass on object columns, which is often the expensive path. The docstring also implies a sample can decide int versus double, which the code correctly refuses to do.
- **Suggested Fix**: Document that integer and numeric decisions always use the full column, and that `sample_ratio` bounds only the later heuristics. Update the test docstring to match. Do not sample the integer path.

### Issue 4: Plot notebook points at the wrong f-ANOVA package

- **Severity**: Medium
- **File**: `notebook/trident/automl_plot.ipynb`
- **Line(s)**: 545
- **Description**: The hyperparameter-importance markdown links to `https://github.com/optuna/optuna-fast-fanova/` and says "This plot use". `flaml/fabric/fanova/README.md` says importance uses Optuna's pure-Python `FanovaImportanceEvaluator` and needs no Cython `build_ext`. The same cell also says hyperparameters "avaliable".
- **Risk**: Users can install or build the wrong package and conclude Fabric plotting requires a native extension.
- **Suggested Fix**: Link to Optuna's `FanovaImportanceEvaluator` or `flaml/fabric/fanova/README.md`. Fix the grammar and spelling in that cell.

### Issue 5: Fabric diagnostic logs stringify caller-controlled values

- **Severity**: Low
- **File**: `flaml/fabric/visualization.py`
- **Line(s)**: 196
- **Description**: `plot()` logs `kwargs={kwargs}`. Signature inference logs `error: {e}` to the Kusto logger (`flaml/fabric/mlflow.py` 185–186 and 203–204). Tune's startup log was reduced to counts and flags (`flaml/fabric/README.md` 142–144). The Kusto logger is a no-op outside Fabric, and Fabric scrubbers exist, but these calls still interpolate caller values before scrubbing.
- **Risk**: A large or data-bearing kwarg, or an MLflow exception that embeds sample values, can be sent to Fabric diagnostics. Off Fabric this is only a wasted format.
- **Suggested Fix**: Log `fig_type` and kwarg keys, or the known `learner`/`params` fields. Log exception type only, and keep the full exception on the local debug logger.

### Issue 6: Public AutoML docstrings ship spelling errors

- **Severity**: Low
- **File**: `flaml/automl/automl.py`
- **Line(s)**: 304, 317, 324, 327, 2177, 2190, 2196, 2199
- **Description**: Both fit docstrings say "performes", "forcely", and "Whetehr". `website/docs/Examples/Integrate - Spark.md` lines 104–105 repeat "performes" and "forcely".
- **Risk**: These are the user-facing parameter descriptions for parallel tuning, Spark cancellation, featurization, and log append. The misspellings make the new public API look unfinished.
- **Suggested Fix**: Use "performs", "forcibly", and "Whether" in both docstrings and the Spark example page.

### Issue 7: AutoFE registry is named `avaliable_methods`

- **Severity**: Low
- **File**: `flaml/fabric/autofe.py`
- **Line(s)**: 55, 58, 603
- **Description**: The stage/method registry and its docstring use "avaliable". The name is module-level and referenced by `get_transformer` and config validation. It is not listed in `flaml/fabric/__init__.py`.
- **Risk**: Public consolidation freezes a misspelled name. A later rename without an alias can break internal importers.
- **Suggested Fix**: Add `available_methods` as the canonical dict, keep `avaliable_methods` as an alias, and fix the docstring spelling.

### Issue 8: Benchmark config constant is misspelled

- **Severity**: Low
- **File**: `benchmark/pmlb/runbenchmark.py`
- **Line(s)**: 18
- **Description**: The script-wide settings dict is `GLOABL_CONF`, and every later lookup uses that spelling.
- **Risk**: Naming-only. It is not a library API, but it is the setting users edit to rerun the public benchmark.
- **Suggested Fix**: Rename to `GLOBAL_CONF` at the definition and all call sites in this file.

### Issue 9: FLAML loss-curve cell plots the Databricks function

- **Severity**: Low
- **File**: `benchmark/pmlb/analyze.ipynb`
- **Line(s)**: 1214–1227
- **Description**: The cell defines `plot_flaml_runs_loss_curve` with the title "Loss Curve for Databricks AutoML Experiment", then calls `plot_dbx_runs_loss_curve(dbx_runs_example)`. The FLAML function is never called. `plot_dbx_runs_loss_curve` is defined earlier, so the cell does not raise; it repeats the Databricks plot.
- **Risk**: Anyone rerunning the benchmark notebook never sees the FLAML loss curve from this cell.
- **Suggested Fix**: Call `plot_flaml_runs_loss_curve` on the FLAML runs frame and set the title to FLAML.

### Issue 10: Coverage-comment step documents a file the workflow does not create

- **Severity**: Low
- **File**: `.github/workflows/python-package.yml`
- **Line(s)**: 357
- **Description**: The PR comment step says coverage text was tee'd into `pytest-coverage.txt`. The combine step runs `coverage report -m` and writes `coverage.xml`; it does not create `pytest-coverage.txt`. The action itself uses `pytest-xml-coverage-path: ./coverage.xml`, so the comment path is valid.
- **Risk**: Operators debugging a missing PR comment will look for a file and a pytest summary line that this workflow never produces.
- **Suggested Fix**: Delete the `pytest-coverage.txt` sentence and state that the comment is built from `coverage.xml` only.

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

## Driver assessment after stage 6, attempt 1

The original findings and verdict are preserved. Actual model `grok-4.7`
used 35 view and 48 search calls; source and independence audits passed.

| Issue | Resolution and verification                                                                                                                                                                                                                                                                                                                                                  |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Removed the NLP omission and the invalid negated source entry. A real offline coverage run passes 53 tests; its combined JSON records NLP execution, including 180 utility lines, 32 trainer lines, 31 training-argument lines, 8 collator lines, and 70 general NLP utility lines.                                                                                          |
| 2     | Extended scoped artifact paths to 22 additional references across Spark, NLP, and regression tests. Shared NLP settings now allocate distinct per-call logs, and consumers read the returned filename. The published CI suites were already separated; this strengthens filesystem ownership without claiming that full parallel Spark/NLP execution is otherwise supported. |
| 3     | Public docs and test descriptions now distinguish full-column integer/numeric decisions from sampled later heuristics and mandatory full-column conversion checks. Numeric safety scans were not weakened.                                                                                                                                                                   |
| 4     | The plot notebook links to Optuna's pure-Python evaluator, explicitly states no native extension is required, and fixes the associated wording.                                                                                                                                                                                                                              |
| 5     | Central plot diagnostics retain only known plot type/counts; signature/artifact errors retain exception type rather than caller values or exception text. Local detailed logging remains. Secret-marker regressions confirm centralized calls do not contain supplied payloads. The existing single-message logging interface is retained.                                   |
| 6     | Corrected the public AutoML parameter docstrings and Spark example spelling.                                                                                                                                                                                                                                                                                                 |
| 7     | `available_methods` is canonical, with `avaliable_methods` retained as the identical compatibility alias. An identity assertion covers the alias.                                                                                                                                                                                                                            |
| 8     | `GLOBAL_CONF` is canonical throughout the benchmark runner; the old name remains an alias for existing importers.                                                                                                                                                                                                                                                            |
| 9     | The FLAML loss-curve cell invokes the FLAML function/data, labels FLAML correctly, and computes elapsed seconds without relying on pandas datetime storage units. A source-cell regression asserts exact values and title.                                                                                                                                                   |
| 10    | The workflow comment now describes only its actual Cobertura XML input; the nonexistent pytest text file is no longer referenced.                                                                                                                                                                                                                                            |

The companion assessment records 661 passing targeted tests, the real
53-test offline NLP coverage run, package builds, source lint, and actionlint.
The notebook/API source-cell checks and centralized-diagnostic regressions are
included in that validation. Original review outcomes are not rewritten.
