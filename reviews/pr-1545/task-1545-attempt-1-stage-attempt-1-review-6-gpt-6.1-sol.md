---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 6
timestamp: '2026-10-17T19:45:27Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 41842395deb0dc502046943a755f9143087cde01d2f6f4e328e236d326a00670
---

## Review Summary

- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-6-gpt-6.1-sol.md
- **Issues Found**: 4
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Compared the supplied dtype-conversion diff with `flaml\automl\data.py` and the sampling/timedelta cases in `test\automl\test_data_coverage.py`.
- [x] Traced validation preprocessing and prediction timing through `flaml\automl\ml.py`, `flaml\automl\automl.py`, and the AutoFE isolation/pipeline contract tests.
- [x] Inspected AutoFE warning gates in `flaml\fabric\autofe.py` and their unconditional invocation from `flaml\automl\state.py`.
- [x] Checked the standalone-featurization notebook example against the fitted-column contract and `flaml\fabric\README.md`.
- [ ] Runtime reproductions were not performed: this assignment permits read-only inspection only. The supplied driver validation was considered; no tests or commands were run by this reviewer.

## Issues

### Issue 1: An all-null inference sample bypasses full-column timedelta detection

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 1977–1978, 2020–2021
- **Description**: Full-column timedelta revalidation remains inside `if len(non_na) > 0`. Consequently, an all-null sample skips it even when the full column readily satisfies the conversion threshold. A deterministic source-level reproduction is a 1,000-row column initially containing `"PT1H"`, with `None` assigned at `df.sample(n=100, random_state=0).index`. Default sampling then sees only nulls and falls through to categorical conversion, whereas `sample_ratio=1.0` recognizes the 900 valid durations as timedeltas.
- **Risk**: The new sampling default can silently change a valid duration column into categorical/string data based on which rows are sampled, contradicting the full-column revalidation guarantee and changing downstream feature handling.
- **Suggested Fix**: Allow full-column timedelta validation whenever sampling is active and `n_full_nonnull > 0`, independently of whether the sample contains usable values. Add sampled-versus-full regression coverage for an all-null inference sample, asserting dtype, values, index, and missing-value preservation.

### Issue 2: Prediction-latency constraints exclude the tuned AutoFE stage

- **Severity**: Medium
- **File**: `flaml\automl\ml.py`
- **Line(s)**: 616–625, 631–646, 687–705
- **Description**: Validation preprocessing and `autofe.transform(X_val)` finish before `_eval_estimator()` starts its prediction timer. Returned `pred_time` therefore measures only the downstream learner, although `AutoML.predict()` applies those feature transforms on every prediction call. Expensive feature-engineering candidates receive no corresponding inference-latency cost.
- **Risk**: AutoML can accept a pipeline under `pred_time_limit` whose actual prediction path exceeds that limit. Reported latency also understates the serving cost of the selected AutoFE pipeline.
- **Suggested Fix**: Include validation-only inference transformation time in the returned and logged per-row prediction latency, excluding feature fitting and training-only resampling. Preserve the feature-off behavior and add holdout/CV coverage with a controlled slow transform and fast learner to verify the actual constraint.

### Issue 3: Disabled AutoFE emits unsupported-feature warnings for ordinary fits

- **Severity**: Low
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 114–124
- **Description**: Sparse-data and NLP checks emit warnings before checking `raw_config == "off"`. `SearchState` invokes this function for every learner even when featurization is disabled, so ordinary default sparse/NLP fits warn that AutoFE was turned off despite the user never requesting it.
- **Risk**: Repeated, misleading warnings make supported public workflows appear partially unsupported and obscure actionable diagnostics.
- **Suggested Fix**: Return immediately for `"off"` before checking unsupported AutoFE inputs. Cover disabled sparse/NLP fits with no AutoFE warnings, while retaining warnings for explicit `"auto"` or `"force"` requests.

### Issue 4: The standalone-featurization demo passes raw labeled data to a feature-only transformer

- **Severity**: Low
- **File**: `notebook\trident\featurization.ipynb`
- **Line(s)**: 1087–1089
- **Description**: The notebook adds `target` before splitting `train` and `test`, then calls `automl.model.autofe.transform(test)` directly. This fitted featurizer expects the task-preprocessed feature columns, not the raw frame containing `target`. Its column-equality guard raises `ValueError("AutoFE input columns must match the fitted columns.")`.
- **Risk**: Following the published walkthrough stops at the standalone-transform example, before reconstruction and registration. The retained successful display output does not represent the current API contract.
- **Suggested Fix**: Pass feature-only data through `automl.preprocess()` before invoking the lower-level featurizer, or demonstrate `automl.feature_transformer` for raw inputs and explain the distinction. Clear the affected stale output and add a small local-fixture check for this example’s input contract.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Empty inference samples must not suppress valid full-column duration detection.
- **How verified**: Source-path inspection; runtime regression verification pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Latency constraints must account for the newly tuned inference transformations.
- **How verified**: Traced validation and prediction paths; controlled timing regression verification pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Disabled optional functionality should not generate unsupported-feature warnings.
- **How verified**: Inspected warning order and `SearchState` invocation; warning-capture regression verification pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: The walkthrough must use the current raw-input versus processed-feature interfaces correctly.
- **How verified**: Compared notebook source with fitted-column validation; local-fixture verification pending.

## Driver assessment after stage 6, attempt 1

Original findings and verdict remain intact. Actual model `gpt-6.1-sol`
used 26 view and 9 search calls; independence, allowed-tool, artifact, and
source-integrity checks passed.

| Issue | What changed and why                                                                                                                                                                                                                       | Verification                                                                                                                                                                                                                                                                        |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Sampled timedelta detection now permits full-column validation even when the selected sample has no nonmissing values.                                                                                                                     | A deterministic all-null 100-row sample from a 1,000-row duration column produces exactly the full-inference dtype, values, original index, and missing-value mask.                                                                                                                 |
| 2     | Validation-only task preprocessing and AutoFE transformation durations are added to returned/logged per-row prediction latency. Feature fitting and training-only resampling remain outside that timer; feature-off behavior is unchanged. | Controlled holdout/CV cases combine 0.2s preprocessing and 0.4s feature transformation over four validation rows, yielding about 0.15s per row despite much larger simulated fitting costs. The reported value crosses a 0.1s constraint boundary only when transforms are enabled. |
| 3     | `"off"` returns before unsupported-input warnings.                                                                                                                                                                                         | Sparse/NLP cases are silent when disabled and still warn for explicit auto/force requests.                                                                                                                                                                                          |
| 4     | Both standalone and reconstructed notebook examples drop target labels and use the correct raw-feature versus task-preprocessed interface. Affected stale outputs are cleared.                                                             | Exact notebook source cells execute against a local fitted fixture and produce unlabeled transformed frames of the expected length.                                                                                                                                                 |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ml_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\automl\test_split.py::test_group_split_with_sample_weight_alignment test\automl\test_split.py::test_group_split_with_sample_weight_series_alignment test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py test\test_misc_coverage.py -m "not spark" -n 2 --dist=loadfile -q --tb=short --disable-warnings
661 passed, 2 optional skips

python -m coverage run -m pytest test\nlp\test_hf_utils_coverage.py test\nlp\test_autohf_classificationhead.py -q --tb=short --disable-warnings
53 passed; combined coverage JSON contains executed NLP source

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

Source lint and actionlint pass. Public `main` remains the reviewed
`17d293b01edc49734ccdb79ef04b68d0da72fab8`; the published PR head has not
changed. Stage 6 remains in progress pending its second reviewer attempt.
