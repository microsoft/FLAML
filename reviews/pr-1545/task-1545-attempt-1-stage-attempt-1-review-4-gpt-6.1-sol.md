---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 4
timestamp: '2026-10-17T13:10:23Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: ce7d291baa65b27f00c0bd6c347d276b381ef0001f8a2e1999062da434460fec
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-4-gpt-6.1-sol.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Reviewed the supplied old/new diff against public base `17d293b01edc49734ccdb79ef04b68d0da72fab8` and inspected current source without accessing external baseline archives or other review artifacts.
- [x] Traced AutoFE training through `flaml\automl\task\generic_task.py`, `flaml\automl\data.py`, `flaml\automl\ml.py`, and `flaml\fabric\autofe.py`; inspected isolation and pipeline-contract tests.
- [x] Checked forecasting boundaries, integer horizons, timestamp ordering, and CV slicing in `flaml\automl\time_series\ts_data.py` and `ts_model.py` against the supplied forecasting tests.
- [x] Inspected pandas/Spark conversion logic and the coverage in `test\automl\test_lossless_conversion.py`, `test\automl\test_data_coverage.py`, and `test\spark\test_dtype_contracts.py`.
- [x] Compared fANOVA distribution handling with `OptunaSearch.convert_search_space()`, declared Optuna version bounds, and `test\fabric\test_fanova.py`.
- [x] Read the benchmark notebook’s successful-fetch path and `test\test_benchmark_contracts.py`; the latter exercises download failures but not successful training.
- [x] Considered the supplied driver evidence, including 414 passing targeted tests, 12 passing Spark dtype tests, and package-build/install smoke results, as reported evidence rather than independent execution.
- [ ] No commands or tests were executed by this reviewer, as required by the read-only assignment. Reproductions below are derived from source inspection.

## Issues

### Issue 1: Fold-local AutoFE still learns from globally fitted preprocessing

- **Severity**: High
- **File**: `flaml\automl\ml.py`
- **Line(s)**: 604-608
- **Description**: The new fold-local `autofe.fit_transform()` receives data that has already undergone a full-data preprocessing fit. `GenericTask.validate_data()` fits `DataTransformer` on all input rows at line 208, and `DataTransformer` fits its median imputer before CV partitioning. For example, a CV training partition containing `[0, NaN, 2]` and validation values `[50, 100, 150]` reaches AutoFE as `[0, 50, 2]`, rather than the training-only imputation `[0, 1, 2]`. Changing only validation values therefore changes the PCA/scaler parameters learned for that training fold. The isolation tests call `compute_estimator()` directly with complete data, bypassing this earlier fit.
- **Risk**: The new AutoFE path does not satisfy the approved held-out-row isolation requirement. Validation rows influence candidate transformations and can bias model selection, despite AutoFE’s own `fit()` being invoked separately per fold.
- **Suggested Fix**: In the opt-in AutoFE evaluation path, fit all data-dependent preprocessing on each training partition and apply that fitted preprocessing to validation rows before AutoFE. Keep the ordinary `featurization="off"` behavior separate if necessary. Add an end-to-end `AutoML.fit()` CV test with missing training values that changes only held-out rows and verifies unchanged training-fold transformations.

### Issue 2: Categorical stringification silently merges distinct feature values

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 348-351
- **Description**: `static_preprocess()` unconditionally converts categorical values to strings and rebuilds the categorical column. Distinct valid categories such as integer `1` and string `"1"` consequently become identical. With `pd.Categorical([1, "1", 1, "1"])`, an ordinal featurizer learns only one category and emits one code for both original values. This conversion also runs when all stages are absent or null, so a successful no-op featurizer can alter and merge input values.
- **Risk**: AutoFE can irreversibly discard predictive information without an error or warning. Exported and reconstructed transformations cannot recover the original category distinction.
- **Suggested Fix**: Preserve categorical identity using a fitted, collision-free encoding rather than unconditional `str()` conversion. If mixed category types cannot be supported, reject them explicitly instead of merging them. Add tests for mixed-type category values and for categorical no-op round trips.

### Issue 3: Integer forecast shortcut skips the saved prediction boundary

- **Severity**: Medium
- **File**: `flaml\automl\time_series\ts_model.py`
- **Line(s)**: 350-353
- **Description**: The new shortcut calls `forecast(steps=X)` directly for models such as Holt-Winters that lack `get_forecast`. When the fitted `TimeSeriesDataset` contains a validation partition, `self.end_date` is that partition’s last timestamp, while the underlying model ends at `self.train_end_date`. For training dates January 1-20 and validation dates January 21-25, `predict(3)` now forecasts January 21-23. Predicting the equivalent frame created from `self.end_date` follows the existing gap-aware path and forecasts January 26-28. Previously integer input passed through `enrich()` and that date-aware path.
- **Risk**: Retained holdout models, including AutoML runs with final retraining disabled, silently return forecasts for the wrong time window. Integer and timestamp-based requests no longer have equivalent semantics.
- **Suggested Fix**: Route integer horizons through the existing date-aware logic, or restrict the shortcut to cases with equivalent boundaries and prediction indexes. Add Holt-Winters coverage comparing integer and equivalent timestamp requests when validation rows remain attached to the fitted dataset.

### Issue 4: Timestamp coercion writes into an incompatible pandas string column

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 319
- **Description**: After successfully converting a string timestamp Series, the code assigns it using `X.loc[:, TS_TIMESTAMP_COL] = ts`. This is an in-place assignment into the existing column dtype, not a guaranteed dtype replacement. On pandas 3, a `StringDtype` column cannot accept Timestamp values and raises `TypeError`. Thus a valid input such as a `ds` Series of three ISO dates with `dtype="string"` fails in the newly added coercion branch. Object-backed columns can instead retain object dtype, contradicting the intended datetime consistency.
- **Risk**: The new string-input support fails on a supported pandas version, and downstream consumers can receive a non-datetime `ds` column even when `time_idx` was computed from datetime values.
- **Suggested Fix**: Replace the column with the converted Series using column replacement or `isetitem`, with explicit copying where required for view handling. Add object- and StringDtype-input tests asserting both datetime `ds` dtype and correct `time_idx` values on pandas 3.

### Issue 5: Legacy Optuna log and quantized distributions bypass replay expansion

- **Severity**: Medium
- **File**: `flaml\fabric\fanova\evaluator.py`
- **Line(s)**: 14-17, 185-194
- **Description**: When modern `FloatDistribution`/`IntDistribution` are unavailable, the type helpers recognize only legacy `UniformDistribution` and `IntUniformDistribution`. Supported Optuna 2.x versions also have separate `LogUniformDistribution`, `IntLogUniformDistribution`, and `DiscreteUniformDistribution` classes, which the tuning converter actually returns. These distributions fall through `_expand_distribution()` unchanged. For example, replaying observations `[1e-4, 0.1]` against `LogUniformDistribution(1e-3, 1.0)` then fails `create_trial()` validation instead of widening the replay bounds. Existing legacy tests cover only ordinary integer-uniform distributions.
- **Risk**: Hyperparameter-importance analysis fails on supported Optuna installations when warm starts, early stopping, or recorded trials produce out-of-space log/quantized values.
- **Suggested Fix**: Recognize and normalize every supported legacy numeric distribution, preserving logarithmic semantics and compatible quantization grids. Add lower-bound Optuna coverage for log-float, log-integer, and discrete-float replay expansion without mutating the original distributions.

### Issue 6: Successful benchmark downloads forward report metadata into learner fitting

- **Severity**: Medium
- **File**: `benchmark\pmlb\build_benchmark.ipynb`
- **Line(s)**: 1476-1477, 1490
- **Description**: The notebook’s `runner_flaml()` puts `"status": "ok"` and `"error": None` inside `automl_settings` and expands that dictionary into `AutoML.fit()`. Neither is an AutoML fit option, so both enter `fit_kwargs` and are forwarded to learner fitting. The initial LightGBM fit consequently receives an unsupported `status` keyword and raises `TypeError`. These fields are already correctly included in the returned benchmark result; they do not belong in training settings. The added contract test exits through the download-failure branch and misses this path.
- **Risk**: The benchmark-building notebook stops on successful dataset downloads instead of producing measurements and selecting benchmark tasks.
- **Suggested Fix**: Remove `status` and `error` from `automl_settings`, retaining them only in result records. Add a successful-fetch regression test using a small local dataset that reaches learner fitting.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Complete training-partition isolation is required for the opt-in AutoFE path.
- **How verified**: Static trace through validation, median imputation, CV slicing, and AutoFE fitting; reproduction not executed.

### Issue 2

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Categorical normalization must not merge distinct source values or alter a no-op result.
- **How verified**: Inspected unconditional string conversion and downstream ordinal encoding; reproduction not executed.

### Issue 3

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Integer forecasts must retain existing date-boundary and delayed-gap semantics.
- **How verified**: Compared the supplied old/new prediction flow with `TimeSeriesDataset.end_date` and estimator boundary assignment; reproduction not executed.

### Issue 4

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Timestamp coercion must replace an incompatible column dtype.
- **How verified**: Inspected the StringDtype coercion and `.loc` assignment path; pandas reproduction not executed.

### Issue 5

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Replay expansion must cover the declared Optuna 2.x compatibility range.
- **How verified**: Compared distribution recognition, replay construction, native conversion branches, and existing legacy tests; lower-version reproduction not executed.

### Issue 6

- **Status**: Open
- **What changed**: Pending driver fix.
- **Why**: Benchmark report metadata must remain separate from learner fit parameters.
- **How verified**: Traced the notebook’s successful-fetch settings through AutoML fit kwargs to learner fitting; notebook workload not executed.

## Driver assessment after stage 4, attempt 1

Original findings and verdict are preserved. Runtime model `gpt-6.1-sol`
used 35 view and 8 search calls; transport and source-integrity audits passed
with no prohibited tools or feedback access.

| Issue | What changed and why                                                                                                                                                                                                                                                                                                                                                                                                                               | Verification                                                                                                                                                                                                                                                                            |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Opt-in AutoFE no longer consumes globally fitted feature statistics. Task validation records raw features and target mapping; each estimator owns a preprocessor fitted inside its training fold/holdout. Inference, feature export, full pipeline export, final training, and log retraining use that estimator's fitted preprocessor. Constant fields remain available for explicit AutoFE selection. Ordinary feature-off behavior is retained. | End-to-end AutoML CV changes only held-out values and still imputes `[0, NaN, 2]` as `[0, 1, 2]` for the training fold. A forecasting regression excludes explicit validation values from imputation. Raw pandas/NumPy prediction, export/refit, replay, and feature-off behavior pass. |
| 2     | Removed unconditional categorical stringification. No-op transforms preserve categorical values; unsupported mixed-type ordinal categories raise an explicit candidate incompatibility rather than merge values.                                                                                                                                                                                                                                   | Mixed integer `1` and string `"1"` remain distinct in no-op output and are explicitly rejected by ordinal feature encoding.                                                                                                                                                             |
| 3     | The integer shortcut is limited to equivalent saved/model boundaries; attached validation boundaries use the existing date-aware gap logic.                                                                                                                                                                                                                                                                                                        | A correctly initialized Holt-Winters model fitted with attached validation data produces identical integer- and timestamp-request forecasts. The direct estimator fixture supplies the required target argument and smoothing configuration.                                            |
| 4     | String timestamp coercion copies the input and replaces the column dtype rather than assigning timestamps into a string array.                                                                                                                                                                                                                                                                                                                     | Object/StringDtype cases assert datetime output and exact indexes; a separate pandas 3.0.6 environment passes the same assertions.                                                                                                                                                      |
| 5     | Replay expansion recognizes legacy log-float, log-integer, and discrete-float distributions, preserving supported log scales and compatible step grids.                                                                                                                                                                                                                                                                                            | New legacy-class tests pass with modern aliases disabled; actual Optuna 2.8.0 executes the fANOVA suite with 16 passes and 17 modern-class-only skips.                                                                                                                                  |
| 6     | Removed report-only status/error fields specifically from the notebook training-settings dictionary, retaining them in result records.                                                                                                                                                                                                                                                                                                             | A successful local fetch reaches real AutoML/LightGBM fitting. This fixes the driver-introduced broad replacement error rather than just extending the failure-path test.                                                                                                               |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
430 passed, 2 optional skips

python -m pytest test\fabric\test_autofe.py -k "test_numpy_autofe or test_reconstruct" -q --tb=short --disable-warnings
2 passed

python -m pytest test\fabric\test_fanova.py -q --tb=short --disable-warnings
Optuna 2.8.0 environment: 16 passed, 17 modern-distribution-only skips

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

Pandas 3.0.6 additionally passes raw-input imputation, exported prediction
parity, pipeline refitting, and timestamp dtype replacement. Source lint
passes. Stage 4 remains in progress pending both attempt-2 reviews.
