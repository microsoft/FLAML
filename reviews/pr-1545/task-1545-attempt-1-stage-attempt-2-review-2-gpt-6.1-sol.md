---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 2
timestamp: '2026-10-17T09:20:58Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 514eeb37bb795365224f8c089d6e8f5e104cf5591ca654dbf44dbee543e92689
---

## Review Summary

- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-2-gpt-6.1-sol.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read `.github\copilot-instructions.md`, the supplied diff, and `pyproject.toml`; inspected optional-dependency and public API boundaries.
- [x] Traced time-series construction, positional splitting, and task validation through `flaml\automl\time_series\ts_data.py` and `flaml\automl\task\time_series_task.py`; inspected `test\automl\test_ts_data.py`.
- [x] Inspected AutoFE reconstruction, configuration export, and task preprocessing in `flaml\fabric\autofe.py`, `flaml\automl\data.py`, and `flaml\automl\pipeline.py`, alongside the isolation and pipeline-contract tests.
- [x] Compared `flaml\fabric\visualization.py:90-116` with the existing `OptunaSearch.convert_search_space()` implementation in `flaml\tune\searcher\suggestion.py:904-973`.
- [x] Inspected the suite-wide MLflow persistence patch in `test\conftest.py:23-105` and the MLflow integration tests.
- [x] Considered the supplied driver validation results as reported evidence only. No commands or tests were executed by this reviewer.
- [ ] Direct reads of the pristine public-base archive were permission-denied. Base comparisons therefore used the supplied diff; no finding below depends on an independently unverified pre-existing defect.

## Issues

### Issue 1: Frequency normalization accepts shuffled data without normalizing the actual time axis

- **Severity**: High
- **File**: `flaml\automl\time_series\ts_data.py`
- **Line(s)**: 54-68, 232-255
- **Description**: The constructor now sorts a detached timestamp series for frequency inference, but stores `train_data` in its original order. Holdout and CV boundaries still slice that original order positionally. For example, observations ordered January `[3, 1, 4, 2]` are accepted with daily frequency, but a two-row holdout trains on `[3, 1]` and validates on `[4, 2]`. Training therefore includes an observation later than a validation observation. `test_unsorted_input()` checks only that a frequency exists, not that chronological ordering is preserved.
- **Risk**: Previously rejected shuffled inputs can now produce temporal leakage, invalid validation scores, and incorrect training/forecast boundaries.
- **Suggested Fix**: Establish one canonical row order before storing and splitting time-series data, preserving feature/target alignment and panel identities. Alternatively, reject unordered input explicitly. Add tests asserting that training timestamps precede validation timestamps for shuffled input and that forecast boundaries follow the normalized data.

### Issue 2: Reconstructed AutoFE uses a different, incompatible task-preprocessing abstraction

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 275-300, 388-397
- **Description**: `Featurization(config=...)` always installs the generic `DataTransformer`, then invokes it before unpacking a `TimeSeriesDataset`. This differs from both parameter-based AutoFE and `_TaskPreprocessor`, which handles forecasting with `DataTransformerTS`. A transformer fitted successfully with `Featurization(params={"fe.numerical": "scaler_standard"}).fit(dataset)` cannot be reconstructed using `Featurization(config=original.config, task="ts_forecast").fit(dataset)`: the generic transformer receives the dataset and `y=None`, then attempts label encoding instead of extracting the embedded training targets.
- **Risk**: Configuration reconstruction does not compose with a supported AutoFE input type, despite fitted time-series transformation and pipeline export supporting that type.
- **Suggested Fix**: Share task-aware preprocessing between reconstruction and pipeline export. Extract dataset targets and timestamp context before applying learned feature stages, and use the forecasting preprocessor where appropriate. Add a time-series configuration round-trip test using embedded targets and numeric regressors.

### Issue 3: AutoFE configuration export is not a faithful serialization boundary

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 435-445
- **Description**: When selection is absent or `"null"`, `keep_cols` remains a pandas `Index` and is placed directly into the extraction configuration. Consequently, fitting PCA and calling `json.dumps(featurizer.config)` raises `TypeError`, breaking the JSON persistence workflow demonstrated in `notebook\trident\featurization.ipynb:1723-1726`. The same block also applies a selector's support mask to the original `X.columns`, although an earlier `ColumnTransformer` can reorder features. For standalone input ordered `[numeric, categorical]`, ordinal encoding followed by cardinality selection can therefore report the numeric column as dropped when the categorical column was actually removed.
- **Risk**: Valid fitted configurations cannot always be persisted, and exported transformation metadata can identify the wrong features.
- **Suggested Fix**: Track feature names through each pipeline stage, interpreting support masks against the selector's actual input names. Materialize exported names as JSON-compatible lists. Add JSON serialization/reconstruction coverage with extraction enabled and selection disabled, plus mixed-order categorical/numeric provenance coverage.

### Issue 4: Visualization duplicates search-space conversion and drops supported Ray domains

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 90-116
- **Description**: `optuna_space()` recognizes only `flaml.tune.sample.Domain` subclasses. With supported Ray installed, public numeric constructors such as `flaml.tune.randint()` and `uniform()` are imported from Ray by `flaml\tune\__init__.py:5-21`. Those domains fail this guard and disappear from visualization, including for sequential `tune.run(use_ray=False)` results. An all-numeric search consequently produces an empty importance space. This duplicates the existing converter in `OptunaSearch.convert_search_space()`, which already supports both domain families, sampler types, quantization, and integer upper-bound conventions. The duplicate also converts an exclusive integer upper bound into an inclusive Optuna bound.
- **Risk**: Installing an optional supported backend changes whether analysis works, and the analysis representation can disagree with the space actually searched.
- **Suggested Fix**: Reuse or extract the established search-space converter rather than maintaining a second conversion policy. Add visualization-level tests for Ray domains, logarithmic/quantized domains, and exclusive integer bounds; checking only that importance values sum to one does not establish that every parameter survived conversion.

### Issue 5: Global test setup changes the persistence failure contract being validated

- **Severity**: Medium
- **File**: `test\conftest.py`
- **Line(s)**: 58, 61-105
- **Description**: Every pytest process globally replaces MLflow's private `FileStore._read_yaml`. After retries, persistent YAML corruption is translated into `MissingConfigException` so the affected run can be skipped. This is not just worker isolation: all MLflow integration tests execute against corruption-recovery behavior absent from the shipped library. A permanent metadata-read failure can therefore become an incomplete history during testing rather than exposing the original persistence failure.
- **Risk**: The suite can mask run-history corruption and provide misleading evidence about the production logging lifecycle. This identifies a test-contract mismatch; it does not establish that corruption occurred during the reported driver runs.
- **Suggested Fix**: Remove suite-wide corruption suppression. Use isolated tracking stores or a transactional tracking backend for concurrent tests, and assert complete expected histories. If corruption recovery is intended product behavior, implement and test that explicitly rather than injecting it only into the test environment.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Frequency inference and positional splitting need a consistent chronological invariant.
- **How verified**: Static constructor-to-split trace; regression execution pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Reconstruction must use task-compatible preprocessing.
- **How verified**: Static comparison of parameter-based, reconstructed, and exported-pipeline paths; round-trip execution pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Exported configuration must be serializable and preserve actual stage feature identities.
- **How verified**: Static tracing of `keep_cols`, selector masks, and the documented JSON consumer; regression execution pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Visualization should follow the established domain-conversion contract.
- **How verified**: Compared public constructor imports and both converters; backend-specific regression execution pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: Integration validation should not silently substitute an unshipped persistence contract.
- **How verified**: Inspected unconditional patch installation and exception translation; no corruption experiment executed.

## Driver assessment after stage 2, attempt 2

The original response and verdict remain unchanged. The runner confirmed
`gpt-6.1-sol`; the audit counted 42 view, 12 search, and 2 glob calls with no
disallowed tools or prior/sibling feedback access. Two optional external
baseline reads were denied because interactive directory approval was
unavailable. This limitation is disclosed above; current source and the
pristine archive were unchanged during review.

Stage 2 advances **with unresolved findings** at the High two-attempt gate,
not clean. No third architecture attempt is run. Before stage 3, the driver
addressed these findings:

| Issue | Resolution                                                                                                                                                                                                                                        | Verification                                                                                                                                                                                        |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Training rows are stably ordered by timestamp before storage and positional splitting. Feature/target/group rows and original index labels stay aligned; explicit forecast-request order remains unchanged.                                       | Shuffled holdout/CV cases assert every training timestamp precedes validation. A time column also used as an index name works, caller data is unchanged, and explicit prediction order is retained. |
| 2     | Configuration reconstruction reuses the task-aware preprocessing adapter. It obtains embedded targets, timestamp names, and custom time-index metadata from forecasting datasets instead of applying the generic transformer to a dataset object. | A JSON configuration round trip with embedded forecasting targets, numeric regressors, custom time-index metadata, and future-frame transformation passes.                                          |
| 3     | Exported columns are materialized as JSON-compatible lists. Selection masks are interpreted against the actual preceding stage's feature order, and only generated prefixes are removed.                                                          | PCA with no selection serializes and reconstructs; categorical-first transformed output correctly reports the categorical source column as dropped even when input order was numeric-first.         |
| 4     | Visualization delegates domain conversion to `OptunaSearch.convert_search_space`, sharing sampler, quantization, log, and integer-bound policies with tuning.                                                                                     | Native integer/log/quantized distributions retain their bounds/semantics. A real-Ray regression is present but skipped locally because the optional supported Ray backend is not installed.         |
| 5     | Removed suite-wide `FileStore._read_yaml` replacement and its corruption-to-missing translation. Per-worker stores remain isolated; run cleanup no longer silently swallows persistence failures.                                                 | Malformed YAML raises the actual YAML error. A real local FileStore retains all expected child trials. The wider integration suite passes with no test-only corruption recovery.                    |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
373 passed, 2 optional skips (Ray and imbalanced-learn)

python -m pytest test\fabric\test_autofe.py -k "test_numpy_autofe or test_reconstruct" -q --tb=short --disable-warnings
2 passed

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

The color-scale regressions were adjusted only to allow machine-rounding in
uniform stop positions; every actual color remains compared exactly in order.
Driver fixes do not relabel this dirty review as clean. Subsequent reviewers
are told not to attempt the unapproved external-directory reads.
