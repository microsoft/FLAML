---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 4
timestamp: '2026-10-17T14:33:21Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 82bc70ce2900142a7865c510dbf1f6f91eee54e5804d76293f0fc76775720bf0
---

## Review Summary

- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-4-gpt-6.1-sol.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Reviewed the supplied public-base diff and traced AutoFE data flow through `flaml\automl\task\generic_task.py`, `ml.py`, `state.py`, and `pipeline.py`, including validation splitting, fold-local fitting, final retraining, and export.
- [x] Inspected `test\fabric\test_autofe_isolation.py` and `test_pipeline_contracts.py`; checked whether their inputs exercise rare-class augmentation and mixed column labels through the complete AutoML path.
- [x] Reviewed runtime detection, MLflow logging/restoration, forecasting boundaries, dtype conversion, and their associated contract tests.
- [x] Followed the telemetry import graph through `flaml\__init__.py`, `flaml\tune\__init__.py`, and `flaml\tune\tune.py`, comparing it with `test\test_misc_coverage.py`.
- [x] Traced H2O response typing, model selection, and scoring in `benchmark\pmlb\runbenchmark.py`; inspected the mocked classification test in `test\test_benchmark_contracts.py`.
- [x] Compared the Ray return path in `flaml\tune\tune.py` with the analysis-type guards in `flaml\fabric\visualization.py`.
- [x] Considered the supplied driver-reported validation results as supporting evidence only. No commands or tests were executed by this reviewer, and no other review artifacts were accessed.

## Issues

### Issue 1: Pre-split augmentation defeats AutoFE's held-out-row isolation

- **Severity**: High
- **File**: `flaml\automl\task\generic_task.py`
- **Line(s)**: 197-214, 855-885, 1004-1018
- **Description**: The new fold-local AutoFE path still receives data after `prepare_data()` duplicates rare-class observations **before** holdout/CV splitting. A concrete case is 20 majority observations and one minority observation, using default `auto_augment=True`, stratified two-fold CV, and an LDA AutoFE candidate. Augmentation creates 20 copies of that minority observation; both folds necessarily contain copies of the same original row and label. Consequently, the supervised featurizer learns from an observation also present in validation. The missing-label overlap handler provides another route to the same violation. This is an interaction between the new AutoFE path and existing split preparation, not a claim that augmentation itself is newly introduced.
- **Risk**: AutoFE validation losses become optimistically biased, violating the explicit requirement that feature engineering must not learn from held-out rows or labels.
- **Suggested Fix**: For AutoFE-enabled fits, establish disjoint partitions before augmentation and augment only each training partition. Do not repair label coverage by copying observations across that boundary; reject impossible singleton-class splits explicitly. Add isolation coverage using original-row identifiers and default stratified splitting, rather than only custom splitter objects.

### Issue 2: Mixed column-label normalization happens after sklearn has already rejected the input

- **Severity**: Medium
- **File**: `flaml\automl\pipeline.py`
- **Line(s)**: 80-81
- **Description**: `_TaskPreprocessor.fit_transform()` invokes `DataTransformer.fit_transform()` before `Featurization.static_preprocess()` applies its collision-safe column mapping. For a numeric DataFrame with columns `["0", 0]`, the former constructs a `ColumnTransformer` using the mixed-type labels at `data.py:645-668`. sklearn rejects mixed string/integer feature names or selectors, so AutoML fails before reaching the new mapping. The added mixed-label test at `test_pipeline_contracts.py:374-387` calls `Featurization` directly and therefore bypasses the failing preprocessing step.
- **Risk**: The documented mixed-label support works in isolation but not through AutoML training or reconstructed preprocessing.
- **Suggested Fix**: Normalize and retain column identity before the first sklearn preprocessing operation, or use safe positional selectors with a persisted schema mapping. Extend coverage to `AutoML.fit()`, reordered-input prediction, and exported-pipeline refitting with mixed numeric column labels.

### Issue 3: The new telemetry fallback test deterministically fails during package import

- **Severity**: Medium
- **File**: `test\test_misc_coverage.py`
- **Line(s)**: 36-47
- **Description**: This test sets `sys.modules["flaml.fabric.telemetry"] = None` and expects `import flaml` to succeed. However, `flaml\tune\tune.py:28` imports `log_telemetry` unconditionally, and `flaml\tune\__init__.py:38` imports that module. AutoML's initial import may catch the resulting `ImportError`, but the subsequent top-level import of `flaml.tune.searcher` retries Tune outside that handler and raises again. The subprocess cannot print `OK`. AutoML also has an unconditional telemetry import at `automl.py:51`.
- **Risk**: The merged non-Spark workflow collects this test, so the full suite contains a deterministic failure not covered by the reported targeted command.
- **Suggested Fix**: Make the import contract consistent. Either support unavailable telemetry throughout Tune and AutoML, or change this test to simulate an unavailable optional Synapse reporter rather than an unconditionally required internal module. Preserve a meaningful assertion that optional telemetry failure does not disable ordinary package use.

### Issue 4: H2O classification targets remain numeric, so AutoML trains regressors

- **Severity**: Medium
- **File**: `benchmark\pmlb\runbenchmark.py`
- **Line(s)**: 112-116, 128-135
- **Description**: The classification branch casts `target` to integer and then constructs an `H2OFrame`, but never converts the H2O response to a factor. H2O determines classification versus regression from the response type; integer-valued numeric responses still select regression. The subsequent `perf.accuracy()` call therefore does not produce the intended classification measurement. The new test supplies a generic mock with an `accuracy()` method, so it cannot detect this mismatch.
- **Risk**: Real classification benchmarks fail or train the wrong task instead of producing valid classification comparisons.
- **Suggested Fix**: Convert `hf["target"]` to a factor before splitting for classification, while retaining numeric responses for regression. Extract accuracy using behavior that supports both binary and multiclass models, and verify response typing with a type-aware test or small real H2O case.

### Issue 5: The H2O benchmark uses test labels for model and threshold selection

- **Severity**: Medium
- **File**: `benchmark\pmlb\runbenchmark.py`
- **Line(s)**: 124-135
- **Description**: `leaderboard_frame=test` ranks H2O candidates on the same partition subsequently passed to `aml.leader.model_performance(test)`. Thus, the reported test score has already influenced which model is evaluated. This affects the default regression benchmark loop as well as classification. Additionally, `max(perf.accuracy())` selects a classification threshold using those test labels. By contrast, the FLAML runner leaves its test partition outside model selection.
- **Risk**: The resulting comparisons systematically favor H2O through test-set selection and do not measure unseen-data performance.
- **Suggested Fix**: Keep the final test partition out of the H2O leaderboard and threshold selection. Use training CV or a separate validation partition for selection, then score final predictions once on an untouched test set. Update the classification contract test so it does not require maximizing test-set threshold accuracy.

### Issue 6: Visualization rejects results returned by the Ray tuning backend

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 80-84, 106-115
- **Description**: These guards accept only FLAML's `ExperimentAnalysis` subclass. The `use_ray=True` branch in `flaml\tune\tune.py:695-714` returns the native object from `ray.tune.run()` directly, which is an instance of the base Ray analysis class, not FLAML's subclass. Therefore, a successful public `tune.run(..., use_ray=True)` result is rejected by the new visualization APIs. The Ray-domain conversion test does not exercise this return-type mismatch.
- **Risk**: Users of the supported Ray backend cannot plot or analyze their tuning results through the advertised facade.
- **Suggested Fix**: Normalize Ray results into a compatible analysis object, including search-space metadata, or explicitly support native Ray analysis objects in the visualization adapter. Add coverage connecting an actual Ray-backend return object to history plotting and parameter-importance extraction.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: AutoFE isolation must account for duplication before partitioning.
- **How verified**: Static trace of augmentation, stratified splitting, and fold-local feature fitting; regression execution pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Column normalization must precede sklearn's feature-name validation.
- **How verified**: Inspected preprocessing order and the direct-only mixed-label test; end-to-end regression pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: The test expectation conflicts with unconditional imports.
- **How verified**: Static import-graph analysis; no subprocess was executed by this reviewer.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Numeric H2O responses do not select classification.
- **How verified**: Inspected response construction and the mocked accuracy test; real H2O validation pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: Final evaluation labels must not participate in model or threshold selection.
- **How verified**: Traced the same `test` partition into leaderboard ranking and final scoring.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: Visualization must accept the result type returned by the selected backend.
- **How verified**: Compared the native Ray return path with FLAML-subclass-only guards; backend integration coverage pending.

## Driver assessment after stage 4, attempt 2

The original response and verdict are retained. Actual model `gpt-6.1-sol`
used 52 view and 10 search calls. Independence, allowed-tool, artifact, and
source-integrity checks passed. Stage 4 advances **with unresolved findings**
under High's two-attempt limit; there is no third correctness attempt.

| Issue | Resolution before stage 5                                                                                                                                                                                                                                                                                                          | Verification                                                                                                                                                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | AutoFE-enabled classification forms disjoint partitions before augmentation. Existing rare-class augmentation is reused through a cloneable training-only sampler after feature fitting. Singleton splits and missing class coverage raise instead of copying rows across the boundary. Ordinary feature-off behavior is retained. | Default stratified CV asserts disjoint original row IDs, unchanged validation size, pre-augmentation AutoFE fit sizes, and augmented training-class counts. A singleton regression raises before CV, while feature-off singleton augmentation still works. |
| 2     | One collision-safe column mapping is shared by task preprocessing and AutoFE. It runs before sklearn sees mixed names and is retained for reordered inference and refits.                                                                                                                                                          | Real AutoML fits, reordered predictions, full exports, and cloned refits cover string/integer mixtures, nonconsecutive integers, and generated-name collisions.                                                                                            |
| 3     | The coverage test now removes the optional Synapse reporter, not an internal module that is part of the installed package. It asserts that reporting is unavailable but ordinary AutoML import remains usable.                                                                                                                     | The corrected subprocess and the entire miscellaneous-coverage module pass as part of the expanded suite.                                                                                                                                                  |
| 4     | H2O classification responses are converted to factors before splitting. Numeric responses remain numeric for regression.                                                                                                                                                                                                           | Type-aware frame tests cover binary, multiclass, and regression response behavior.                                                                                                                                                                         |
| 5     | H2O leader selection uses training CV, not the final test frame. Classification accuracy is measured from final predictions on untouched test labels, without selecting a threshold on those labels.                                                                                                                               | Type-aware tests reject test-frame leaderboard/validation use and verify final prediction accuracy; regression retains independent final scoring. Public H2O source documents factor responses and CV-ranked default leaderboards.                         |
| 6     | Native Ray analysis objects are retained, given search-space metadata, and accepted through the common analysis base type.                                                                                                                                                                                                         | A real Ray 2.4.0 backend on Python 3.10.19 completes tuning, then feeds history plotting and importance extraction. Its reporting suite passes 19 tests, including native Ray domains.                                                                     |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py test\test_misc_coverage.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
557 passed, 2 optional skips

python -m pytest test\fabric\test_ray_reporting.py test\fabric\test_visualization_contracts.py -q --tb=short --disable-warnings
Ray 2.4.0 / Python 3.10.19 environment: 19 passed

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

The Ray test uses one CPU, zero GPUs, bounded local object-store resources,
and explicit cleanup. The first initialization attempt encountered Windows'
unavailable GPU-discovery command; explicitly selecting zero GPUs removed that
unneeded dependency. No user/global environment was modified, and no owned Ray
services remained afterward.

H2O service execution was not performed. Its response/partition contract is
validated by type-aware test doubles and authoritative public H2O Python source,
not claimed as a live H2O benchmark.
