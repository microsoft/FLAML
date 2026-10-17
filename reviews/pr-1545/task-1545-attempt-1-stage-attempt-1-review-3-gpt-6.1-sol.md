---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 3
timestamp: '2026-10-17T10:21:38Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 3ac8f0d675b92d57a9f157c212f2a55c665970b8cec236fdb912b91d465c619f
---

## Review Summary

- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-3-gpt-6.1-sol.md
- **Issues Found**: 9
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced the new timestamp sorting in `flaml\automl\time_series\ts_data.py:54-55` through explicit validation-partition reconstruction in `flaml\automl\task\time_series_task.py:175-187`.
- [x] Inspected AutoFE initialization, parameter changes, fitting, reconstruction, and fitted-state checks in `flaml\fabric\autofe.py`, together with `test\fabric\test_autofe_isolation.py` and `test\fabric\test_pipeline_contracts.py`.
- [x] Followed AutoFE exceptions through `flaml\automl\ml.py:get_val_loss`, cross-validation evaluation, and `flaml\tune\tune.py:run`.
- [x] Examined target-label handling and early validation failures in `flaml\automl\pipeline.py:75-125`, and null-value handling in `flaml\fabric\visualization.py:523-573`.
- [x] Compared the new xdist invocation with shared artifact writes in `test\automl\test_python_log.py`, `test\automl\test_forecast.py`, and `test\automl\test_score.py`.
- [x] Inspected both new benchmark scripts and their failure-result and repeat-execution paths.
- [x] Considered the supplied driver-reported targeted test, build, and installation evidence. Introduced-versus-existing comparisons use the supplied diff; no external baseline archive or other review artifacts were accessed.
- [ ] Runtime reproductions and fresh CI were not executed by this reviewer; this was a read-only source and test review.

## Issues

### Issue 1: Timestamp sorting can move explicit validation rows into training

- **Severity**: High
- **File**: `flaml\automl\time_series\ts_data.py`
- **Line(s)**: 54-55
- **Description**: Sorting every constructor's `train_data` interacts incorrectly with `TimeSeriesTask.validate_data`. That method concatenates transformed training and explicit validation rows, constructs another `TimeSeriesDataset`, then restores the split using only `move_validation_boundary(-val_len)` (`time_series_task.py:175-187`). The new constructor sorts this combined frame before the boundary is restored. For example, training dates January 1, 3, 5, and 7 with explicit validation dates January 2 and 4 become training dates January 1-4 and validation dates January 5 and 7. The caller's validation rows and labels have silently entered training.
- **Risk**: Final learners and AutoFE can fit on explicitly held-out observations, violating the isolation requirement and invalidating evaluation results.
- **Suggested Fix**: Preserve partition membership while preprocessing and sorting: reconstruct training and validation frames separately rather than splitting the sorted union by length. If non-future validation dates are unsupported, reject them explicitly before fitting instead of reassigning their rows. Add an explicit-validation regression with interleaved timestamps and identifiable held-out labels.

### Issue 2: A data-dependent AutoFE candidate failure aborts the entire search

- **Severity**: High
- **File**: `flaml\automl\ml.py`
- **Line(s)**: 604-608
- **Description**: AutoFE fitting executes without a trial-level boundary for expected transformation incompatibilities. A feature that varies globally but is constant within one CV training fold is valid input for a tree learner, yet selecting `fe.selection="variance"` makes `VarianceThreshold.fit` raise because the fold has no surviving features. The exception propagates through `evaluate_model_CV` and the unguarded evaluation call at `flaml\tune\tune.py:925`, terminating `AutoML.fit` rather than rejecting that candidate. A one-feature frame containing one constant block followed by another, evaluated with non-shuffled two-fold CV, demonstrates this boundary.
- **Risk**: Enabling automatic feature search can make otherwise trainable datasets fail unpredictably when an incompatible candidate is sampled, even after successful earlier trials.
- **Suggested Fix**: At the AutoML trial-evaluation boundary, classify and log expected AutoFE/data incompatibilities as failed or infinite-loss trials and continue searching. Do not silently substitute a different pipeline or suppress arbitrary programming errors. Cover a failing variance-selection candidate followed by a viable no-selection candidate.

### Issue 3: `set_params` leaves AutoFE falsely fitted and silently passing data through

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 234-245, 251-260, 531
- **Description**: `set_params` reinitializes the existing instance with `self.__init__(**current)`. The constructor clears `pipeline` and column metadata but never resets `_is_fitted`, so a previously fitted instance retains `_is_fitted=True`. Consequently, fitting a standard scaler, changing its parameters to a min-max scaler, and calling `transform` without refitting passes `check_is_fitted` and returns unchanged input because `pipeline` is now `None`. Learned `time_col_` also survives this reinitialization.
- **Risk**: Consumers receive untransformed features while the object reports itself fitted under its new configuration.
- **Suggested Fix**: Initialize or explicitly clear all fitted-state fields during reinitialization, including `_is_fitted` and timestamp context. Add a test asserting that parameter changes invalidate a fitted featurizer and that `transform` raises `NotFittedError` until another successful fit.

### Issue 4: Reconstructed forecasting AutoFE loses timestamp context for DataFrame fits

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 389-398
- **Description**: `time_col_` is captured before reconstructed task preprocessing runs. With `Featurization(config=..., task="ts_forecast").fit(X_dataframe, y)`, it is set to `None`, although `_TaskPreprocessor.fit_transform` subsequently creates a `TimeSeriesDataset`. Fitting then excludes the timestamp from `_input_columns`, but later `transform(future_dataframe)` takes the non-time-series branch. Task preprocessing preserves the timestamp, and `static_preprocess` rejects it as an unexpected column. Thus this documented reconstruction route can fit successfully but cannot transform; `fit_transform` fails immediately during its transform phase.
- **Risk**: Forecast configuration reconstruction works for the covered embedded-dataset case but fails for ordinary DataFrame/target inputs.
- **Suggested Fix**: Resolve and retain timestamp context after task preprocessing has normalized the input. Ensure prediction applies task preprocessing and timestamp separation consistently. Add reconstruction coverage for DataFrame-plus-target fitting and future DataFrame transformation.

### Issue 5: Failed pipeline refits can silently replace the old model's label mapping

- **Severity**: Medium
- **File**: `flaml\automl\pipeline.py`
- **Line(s)**: 75-93, 109-114
- **Description**: `AutoMLPipeline.fit` assigns and fits `label_transformer_` before validating fit parameters or resampler compatibility. For an already fitted exported classifier, refitting with different target labels and a forbidden `estimator__sample_weight` raises at `_validate_resampler`, before touching the old feature pipeline or learner. Those components remain fitted, but subsequent `predict` decodes their old class codes using the newly assigned labels. For example, a failed refit from `approve/deny` to `cat/dog` leaves the old learner returning predictions labeled `cat/dog`.
- **Risk**: After an early validation failure, a still-usable pipeline can produce silently mislabeled predictions and report incorrect `classes_`.
- **Suggested Fix**: Validate parameters before mutating learned label state, and commit a newly fitted label transformer only after successful fitting. Alternatively, invalidate the complete pipeline on failure. Test a failed refit with changed labels and require either unchanged predictions or `NotFittedError`, never relabeled old predictions.

### Issue 6: Slice plots raise on valid null-valued categorical parameters

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 541-544
- **Description**: `hp.value_counts()` excludes missing values by default, but the next line indexes those counts for every row. A valid tuning parameter such as `class_weight=tune.choice([None, "balanced"])` therefore produces `KeyError` for the `None` row. Conditional parameters missing from some trial configurations have the same failure.
- **Risk**: Slice visualization fails on common valid tuning histories instead of plotting available observations.
- **Suggested Fix**: Define explicit handling for null and absent parameter values. Filter missing parameter rows together with their scores, or normalize/count a displayed null category without unchecked indexing. Cover both a categorical `None` choice and a conditionally absent parameter.

### Issue 7: New xdist execution exposes shared model-file races

- **Severity**: Medium
- **File**: `.github\workflows\python-package.yml`
- **Line(s)**: 210-218
- **Description**: The non-Spark suite now runs files concurrently, but several included files still write the checkout-relative `automl.pkl`. In particular, `test\automl\test_python_log.py:112-118` writes that file, computes predictions, then reloads it and checks prediction equality. Meanwhile, `test\automl\test_forecast.py` repeatedly truncates and writes the same file, and `test\automl\test_score.py` also writes and reloads it. `--dist=loadfile` serializes tests within a file, not across these files.
- **Risk**: A worker can load another test's model or a partially written pickle, causing intermittent deserialization failures, feature mismatches, or false assertion failures.
- **Suggested Fix**: Move shared writable artifacts to per-test `tmp_path` locations before parallelizing their files, or explicitly serialize the remaining artifact-sharing tests. Check other generated files for the same cross-file ownership problem.

### Issue 8: Benchmark download failures return a shape the result writer cannot handle

- **Severity**: Low
- **File**: `benchmark\pmlb\runbenchmark.py`
- **Line(s)**: 32-34, 81-83, 126-127
- **Description**: Both runners catch dataset-fetch failures and return only `score` and `duration`. Result flattening later unconditionally reads `metrics["metric"]`. A single failed download therefore reaches a supposedly handled path but ultimately raises `KeyError("metric")` after the remaining benchmark work.
- **Risk**: Transient external-data failures prevent the final report from being generated; the zero-score fallback also disguises failed evaluation as measured performance.
- **Suggested Fix**: Use a consistent result shape with metric and explicit failure status, and have report generation distinguish failed evaluations from scores. Add a fetch-failure test that exercises final result flattening.

### Issue 9: Dataset extraction can delete CSV outputs on repeat execution

- **Severity**: Low
- **File**: `benchmark\pmlb\extract_dataset.py`
- **Line(s)**: 13-17
- **Description**: The conversion loop processes every directory entry, including `.csv` outputs left by an earlier invocation. For an ordinary existing CSV filename, replacing `"tsv"` with `"csv"` leaves its path unchanged: the script writes to that same file and then deletes it with `os.remove`. If directory enumeration visits a newly converted TSV before its corresponding CSV entry, it deletes the newly generated result as well.
- **Risk**: Repeat execution is not reliably idempotent and can remove benchmark datasets depending on directory contents and enumeration order.
- **Suggested Fix**: Process only the intended `.tsv` inputs, replace the filename suffix rather than arbitrary substrings, and delete only those TSV inputs after successful conversion. Cover a directory already containing generated CSV files.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Explicit validation membership must survive timestamp normalization.
- **How verified**: Static trace through dataset construction and boundary restoration; runtime reproduction not executed.

### Issue 2

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Expected candidate incompatibility must not terminate the whole automatic search.
- **How verified**: Inspected AutoFE fitting, CV propagation, and the sequential evaluation call; runtime reproduction not executed.

### Issue 3

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Reinitialization clears the pipeline but preserves its successful-fit flag.
- **How verified**: Traced constructor assignments, `set_params`, fitted-state checking, and the no-pipeline transform branch.

### Issue 4

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Timestamp context is captured before reconstructed input normalization.
- **How verified**: Traced DataFrame reconstruction through `_TaskPreprocessor`, feature-column recording, and transformation.

### Issue 5

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Label-state mutation precedes validation that can leave the old learner intact.
- **How verified**: Inspected label-encoder assignment, resampler validation, and prediction decoding; runtime reproduction not executed.

### Issue 6

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Count construction excludes values that the following lookup still visits.
- **How verified**: Inspected hyperparameter-frame construction and slice-plot count lookup.

### Issue 7

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Parallel test files retain shared writable artifact paths.
- **How verified**: Compared the workflow's xdist scheduling with concrete pickle write/read sites; concurrent tests not executed.

### Issue 8

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Failure results violate the downstream result-shape assumption.
- **How verified**: Compared both exception returns with the final flattening loop.

### Issue 9

- **Status**: Open
- **What changed**: None; read-only review.
- **Why**: Existing outputs are included in an input-conversion-and-delete loop.
- **How verified**: Inspected filename transformation and deletion targets; script not executed.

## Driver assessment after stage 3, attempt 1

Original findings and verdict are retained. The audit confirmed actual model
`gpt-6.1-sol`, 32 view and 5 search calls, read-only tools, no earlier/sibling
feedback access, and no source drift.

| Issue | What changed and why                                                                                                                                                                                                                                                                             | Verification                                                                                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Time-series preprocessing reconstructs training and explicit validation frames separately. Sorting no longer splits a reordered union using only its length.                                                                                                                                     | Identifiable interleaved validation labels remain entirely in the validation partition. The fixture uses a learner compatible with the small input, stopping before search to inspect partition membership.              |
| 2     | Known variance-selection and LDA partition incompatibilities raise a specific AutoFE error. The AutoML trial boundary logs only that error as an infinite-loss rejected trial, then continues. Missing estimators are handled explicitly, and rejected trials do not log model artifacts.        | A constant-fold variance candidate is rejected before a valid no-selection candidate succeeds; arbitrary implementation `ValueError`s still propagate. Active MLflow history retains both trials and the correct winner. |
| 3     | Featurizer construction/reinitialization resets fitted state and timestamp metadata.                                                                                                                                                                                                             | Changing a fitted scaler's parameters raises `NotFittedError` until refit, after which the requested scaling is used.                                                                                                    |
| 4     | Timestamp context is captured after reconstructed task preprocessing; task metadata inference also handles missing attributes on a new forecast task.                                                                                                                                            | DataFrame-plus-target `fit_transform` and future-frame transformation preserve the timestamp column.                                                                                                                     |
| 5     | New label encoders are committed only after successful fitting, not before parameter validation.                                                                                                                                                                                                 | An early rejected refit with different labels retains the original predictions and `classes_`.                                                                                                                           |
| 6     | Slice plots normalize missing values to a distinct displayed category and map explicit counts, including null values.                                                                                                                                                                            | Both categorical `None` and conditionally absent values retain aligned points and finite counts.                                                                                                                         |
| 7     | A process- and module-owned temporary-artifact helper replaces 38 shared filename references in 12 test files. The inventory also covered shared iris, resume, California, CO2, and sparse-classification logs, not only pickles. Existing within-file behavior and CI parallelism are retained. | Helper ownership/containment tests pass, the bounded codemod is idempotent, and the existing logging save/load test passes. Source formatting and syntax checks cover all changed test files.                            |
| 8     | Both benchmark runners return complete result shapes with explicit download-failure status, error detail, and no measured score. Report flattening preserves statuses. H2O is initialized only after a successful fetch.                                                                         | Fetch failures for both runners reach report generation without a `KeyError` or fabricated zero score; no real benchmark service is started.                                                                             |
| 9     | Extraction uses context-managed streams, converts only `.tsv` inputs, replaces suffixes rather than arbitrary substrings, and removes only successfully converted TSVs.                                                                                                                          | Repeated extraction preserves existing CSVs and filenames containing `tsv` in the stem.                                                                                                                                  |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
399 passed, 2 optional skips (Ray and imbalanced-learn)

python -m pytest test\automl\test_python_log.py -q --tb=short --disable-warnings
1 passed

python -m pytest test\spark\test_dtype_contracts.py -q --tb=short --disable-warnings
12 passed on isolated PySpark 4.0.4 / Python 3.13.5 / Java 17

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

The local Spark launcher initially stalled because its UUID command loaded a
user PowerShell profile. The owned process tree was stopped; a session-only
launcher shim adds `-NoProfile`, without changing user/global configuration.
The successful runs use a real local Spark session, not a mocked SQL backend.
A cleanup diagnostic referenced a PID that was subsequently confirmed no
longer running.

Stage 3 has not advanced. Both required reviewers will receive attempt 2;
earlier inherited concerns remain retained separately.
