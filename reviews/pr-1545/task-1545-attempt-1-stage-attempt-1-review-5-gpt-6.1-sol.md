---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 5
timestamp: '2026-10-17T16:49:53Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 586a0ed9df89f300bfa8235b7743e53f4bca887fdb973e28cdaec76a60901a1e
---

## Review Summary

- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-5-gpt-6.1-sol.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Compared the supplied old/new diff with current source and tests, including `flaml\automl\pipeline.py`, `flaml\fabric\autofe.py`, and their isolation/pipeline contract tests. Baseline comparisons use the supplied diff, not an external pristine archive.
- [x] Inspected `.github\workflows\python-package.yml`, the workflow test invocations, and `pyproject.toml` to assess test collection and platform coverage.
- [x] Examined forecasting accuracy assertions and Spark serialization coverage in `test\automl\test_forecast.py`, `test\spark\test_0sparkml.py`, and `AutoML.load_pickle()`.
- [x] Compared MLflow lifecycle implementation with mocked and real integration coverage in `test\fabric\test_runtime.py` and the scenarios in `test\automl\check_mlflow_behaviour.py`.
- [x] Statically traced uncovered date-format and feature-name cases through `auto_convert_dtypes_pandas()` and `CardinalitySelector.fit()`.
- [x] Considered the supplied driver-reported validation, including 557 targeted passes, 19 Ray-environment passes, and 12 actual Spark SQL passes. Those results do not establish coverage of the cases identified below.
- [ ] Reviewer-executed tests: not performed; this invocation permits read-only inspection only. Fresh full CI and live Fabric validation were not independently verified.

## Issues

### Issue 1: The workflow removes the entire NLP suite without a replacement

- **Severity**: Medium
- **File**: `.github\workflows\python-package.yml`
- **Line(s)**: 208-232
- **Description**: Both test variants now pass `--ignore=test/nlp`. No other inspected workflow runs that directory; the OpenAI workflow runs `test/autogen`, which is different. Consequently, neither the existing NLP integration tests nor the newly added `test_hf_utils_coverage.py` and modified classification-head tests execute in maintained CI. The supplied base invocation did not exclude NLP.
- **Risk**: Changes to shared fitting, prediction, preprocessing, and MLflow pipeline code can break public NLP behavior while every maintained test job remains green.
- **Suggested Fix**: Add a maintained NLP test job, or route an appropriate NLP subset into an existing job. Keep lightweight utility tests collected and use explicit dependency/platform skips for unsupported integrations rather than excluding the whole directory.

### Issue 2: The forecasting regression test no longer checks forecast quality

- **Severity**: Medium
- **File**: `test\automl\test_forecast.py`
- **Line(s)**: 67-71
- **Description**: The existing MAPE assertion has been commented out rather than replaced with an assertion appropriate to the intended forecasting behavior. `test_forecast_automl()` now only prints its measured error. For example, an all-zero prediction produces MAPE 1 on this positive CO2 series but would no longer fail this check. The new timestamp/index tests do not replace an end-to-end accuracy check.
- **Risk**: A substantial forecasting-quality regression can pass the suite despite the requirement to preserve public forecasting behavior.
- **Suggested Fix**: Restore a meaningful, justified MAPE bound. If time-budget variability makes the original threshold unreliable, use a fixed learner/configuration and deterministic iteration budget for the quality regression test instead of removing the assertion.

### Issue 3: Spark persistence coverage now bypasses model restoration

- **Severity**: Medium
- **File**: `test\spark\test_0sparkml.py`
- **Line(s)**: 170-177
- **Description**: The Spark-model round trip now calls `load_pickle(..., load_spark_models=False)` and comments out both restored-model prediction calls. `AutoML.load_pickle()` explicitly documents that this mode leaves Spark estimators without their underlying models. The remaining assertions and visualization comparison validate metadata only. The other round-trip tests found train ordinary XGBoost or sklearn models, even where Spark distributes their trials.
- **Risk**: Broken Spark artifact paths, model-class loading, or restored predictions are no longer detected by this test.
- **Suggested Fix**: Keep the metadata-only visualization case separately, and retain an actual Spark-model round-trip test using the default restoration mode. A small synthetic `rf_spark` case should assert that the restored model exists and that its predictions match the original.

### Issue 4: Real global-autolog behavior is missing from the MLflow test matrix

- **Severity**: Medium
- **File**: `test\fabric\test_runtime.py`
- **Line(s)**: 528-567, 653-690
- **Description**: The global-plus-flavor restoration case replaces both `AUTOLOGGING_INTEGRATIONS` and MLflow’s autolog hooks with mocks, so it cannot validate their real registry and patching side effects. The real subprocess round trip explicitly removes the global `"mlflow"` configuration and covers only flavor autologging. The global-autolog scenarios in `test\automl\check_mlflow_behaviour.py` are not collected by pytest’s default filename patterns; the Spark integration file containing similar scenarios is explicitly ignored by CI. This is a coverage gap, not evidence that restoration currently fails.
- **Risk**: The global configuration branch can mishandle caller state, logging intent, or model-artifact opt-outs without the existing tests detecting it.
- **Suggested Fix**: Extend the isolated real-MLflow subprocess test to cover global-only and global-plus-explicit-flavor configurations, with and without an active run. Assert actual registry restoration after successful and failed fits, expected run relationships, and absence of artifacts when `log_models=False`.

### Issue 5: Date-format coverage misses a newly rejected valid timestamp format

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 1418, 2079-2108
- **Description**: The new datetime hint rejects every value in a column such as `["01/02/24", "01/03/24", "01/04/24"]`: these contain neither four consecutive digits, a colon, nor three letters. Static tracing therefore skips `pd.to_datetime()` and returns a string schema, whereas the supplied old implementation attempted datetime parsing and recognizes these dates. Existing positive date-conversion tests use four-digit years and do not cover this regression.
- **Risk**: Common two-digit-year date columns silently lose timestamp semantics and downstream temporal feature handling.
- **Suggested Fix**: Recognize complete numeric date shapes with two-digit years without reopening the two-component bucket-label false positives. Add exact schema/value assertions for sampled and unsampled date columns, alongside the existing `"31-40"` and `"60+"` negative cases.

### Issue 6: Cardinality selection mistakes an ordinary numeric feature name for encoder metadata

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 576-581
- **Description**: `CardinalitySelector` identifies categorical features solely by a `categorical__` name prefix. With no column-transformer stage, `Featurization(params={"fe.selection": "cardinality"})` on numeric columns `{"categorical__signal": [0.0, 1.0, 2.0], "other": [3.0, 4.0, 5.0]}` drops the first column despite both being numeric. Adding a numerical scaling stage changes the generated prefix and prevents that drop. The double-underscore regression test covers `"a__b"`, but not names resembling generated prefixes.
- **Risk**: Feature selection depends on user column spelling and unrelated stage configuration, silently removing useful numeric features or rejecting an otherwise valid candidate.
- **Suggested Fix**: Determine categorical provenance from explicit feature metadata rather than raw input-name prefixes. Extend the existing prefix test with numeric `categorical__*` columns, with and without a numerical transformer, and assert identical retention decisions.

### Issue 7: The invalid-metric test accepts unrelated implementation failures

- **Severity**: Low
- **File**: `test\automl\test_ml_coverage.py`
- **Line(s)**: 180-185
- **Description**: `pytest.raises((ValueError, Exception))` is equivalent to accepting any `Exception`. It therefore passes for `NameError`, `AttributeError`, or other failures unrelated to deliberate invalid-metric handling. With `datasets` available, the current unknown-name lookup can raise `KeyError`, which this assertion also accepts despite the test describing an ImportError/ValueError path.
- **Risk**: The new coverage test can remain green when the intended error-handling behavior is broken.
- **Suggested Fix**: Exercise the intended dependency/error branch deterministically and assert its specific exception type and message. Handle any different public invalid-metric behavior explicitly rather than masking it with a catch-all assertion.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: NLP execution coverage has no maintained replacement.
- **How verified**: Static inspection of workflow selectors and pytest configuration; replacement execution pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: The end-to-end forecast error is calculated but no longer asserted.
- **How verified**: Supplied old/new diff and current test inspection; revised quality check pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Metadata-only loading bypasses the Spark restoration contract.
- **How verified**: Inspected the changed test, other round-trip callers, and `load_pickle()` implementation; real restored-model parity check pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Mocked global-autolog hooks do not establish real MLflow lifecycle behavior.
- **How verified**: Inspected collected tests, subprocess setup, scenario filenames, and workflow exclusions; real global-autolog matrix pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: The new date prefilter excludes complete two-digit-year dates.
- **How verified**: Static branch tracing against the supplied old implementation; targeted date-format regression test pending.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: User feature names are being treated as generated categorical provenance.
- **How verified**: Static tracing of selector construction, prefix generation, and support-mask calculation; targeted retention test pending.

### Issue 7

- **Status**: Open
- **What changed**: Pending.
- **Why**: The catch-all exception assertion cannot distinguish intended rejection from implementation failure.
- **How verified**: Inspected the assertion and invalid-metric lookup path; precise error-contract test pending.

## Driver assessment after stage 5, attempt 1

Original findings and outcome are retained. Actual model `gpt-6.1-sol`
used 50 view, 24 search, and 2 glob calls; the artifact, independence,
allowed-tool, and source-integrity audits passed.

| Issue | Resolution                                                                                                                                                                                                                                                 | Verification and limits                                                                                                                                                                                                                                                                                                                                            |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Added a maintained Ubuntu/Python 3.10 non-Spark CI step for NLP utilities and classification-head compatibility. Tokenization uses a local vocabulary; head-switch cases use tiny locally created BERT checkpoints and assert logits, loss, and gradients. | 51 tests pass with Hub/model downloads disabled. Larger network-model integrations are not claimed as part of this offline subset.                                                                                                                                                                                                                                 |
| 2     | Restored the established CO2 MAPE ceiling, with finite-value and horizon checks.                                                                                                                                                                           | The end-to-end forecast-quality test passes with the original 0.0051 ceiling.                                                                                                                                                                                                                                                                                      |
| 3     | Kept metadata-only visualization loading explicitly separate and added a native `rf_spark` default-load round trip that asserts restored models and prediction parity.                                                                                     | Native RF training ran locally, but Windows Spark ML file writing was blocked by missing Hadoop `winutils`; the success-path test is collected in maintained Linux Spark CI. Its local platform skip is explicit, not a claimed persistence pass. WSL was also inspected and has no Java runtime; no host OS packages or arbitrary native binaries were installed. |
| 4     | Added isolated real-MLflow global-only/global-plus-flavor cases, with/without active parent and with successful/failed fitting.                                                                                                                            | Actual registry equality, run relationships, and model-artifact opt-outs pass across the eight configurations. No mocked registry/hook is used in these cases.                                                                                                                                                                                                     |
| 5     | The datetime prefilter now recognizes complete three-component numeric dates with two-digit years while retaining bucket-label exclusions.                                                                                                                 | Sampled and full-frame cases assert the existing `timestamp` schema and exact datetime values.                                                                                                                                                                                                                                                                     |
| 6     | Cardinality selection receives explicit categorical provenance from the constructed feature stages; direct use falls back to categorical dtype rather than a user-controlled name prefix.                                                                  | Numeric `categorical__signal` is retained with and without scaling; existing categorical removal and rejected-candidate tests still pass.                                                                                                                                                                                                                          |
| 7     | Replaced the catch-all assertion with deterministic optional-HF absence and installed-HF unknown-name cases, each checking its specific exception and message.                                                                                             | Both cases pass. Existing differing invalid-name behavior is explicit rather than hidden behind `Exception`.                                                                                                                                                                                                                                                       |

Validation includes 634 targeted passes with 2 optional skips; 51 offline NLP
passes; 18 actual Spark guard passes with 1 explicit native-persistence
platform skip; and 9 strengthened AutoFE/replay/forecast/error-contract passes.
The exact core command is retained in the companion driver assessment.
Wheel/source builds, source lint, and actionlint pass.

Stage 5 remains in progress pending both second-attempt reviewers. Native
Spark persistence still requires hosted Linux CI evidence before a green-CI
claim; the local environment limitation is not converted into a success.
