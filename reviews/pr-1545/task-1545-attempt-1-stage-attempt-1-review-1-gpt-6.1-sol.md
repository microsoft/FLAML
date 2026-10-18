---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 1
timestamp: '2026-10-17T06:27:27Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 3c7b51fb169894da7205db7f8324ae423a787fd713003b2e01757767972b8e90
---

______________________________________________________________________

## producer: review-code schema_version: 1 tier: High contract: direct execution_backend: copilot-cli attempt: 1 stage_attempt: 1 round: 1 timestamp: "2026-10-17T06:09:38.912Z" outcome: issues_found reviewer_model: gpt-6.1-sol input_sha256: 318efb949c7e76b73babca908e69881f03898cd81768a88c3ceb99ce8dadb3f9

## Review Summary

- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-1-gpt-6.1-sol.md
- **Issues Found**: 13
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Reviewed the supplied consolidation diff against public base `17d293b01edc49734ccdb79ef04b68d0da72fab8`, including the stated current-main merge.
- [x] Traced AutoFE evaluation, per-fold fitting, final training, prediction, and export through `flaml\automl\ml.py`, `flaml\automl\state.py`, `flaml\automl\automl.py`, and `flaml\fabric\autofe.py`.
- [x] Inspected MLflow policy and artifact construction in `flaml\fabric\mlflow.py`, alongside `test\fabric\test_runtime.py` and `test\fabric\test_mlflow_coverage.py`.
- [x] Inspected numeric conversion and time-index arithmetic in `flaml\automl\data.py`, forecasting interfaces in `flaml\automl\time_series\ts_model.py`, and the associated data, forecasting, and isolation tests.
- [x] Inspected visualization implementation, package metadata, supplied workflow changes, and relevant notebook source cells. The findings below identify concrete source-derived failure cases, not merely missing coverage.
- [x] Considered the supplied validation report of **217 passed and 7 optional-library skips**, plus the reported Actionlint result; these results were not independently executed and do not cover all cases below.
- [ ] No commands, tests, builds, external-service calls, or live Fabric/Docker validation were performed in this read-only review.

## Issues

### Issue 1: Disabled autologging can discard an explicit model-artifact opt-out

- **Severity**: High
- **File**: `flaml\fabric\mlflow.py`
- **Line(s)**: 429–434
- **Description**: `_do_log_model` now ignores every configuration having `disable=True`. Consequently, `mlflow.autolog(disable=True, log_models=False)` followed by an active run and `AutoML.fit()` can produce `_do_log_model=True`: the explicit global setting is filtered out, and `all()` over the remaining enabled configurations defaults to true. The active run still enables FLAML's manual logging. The previous implementation read the global `log_models` setting without dropping it.
- **Risk**: FLAML automatically uploads model artifacts despite an explicit `log_models=False` setting. This violates the logging-opt-out requirement and can persist sensitive models in a tracking store where the caller intended to log only metrics.
- **Suggested Fix**: Preserve the explicit global model-logging restriction independently of whether autologging is enabled, while also respecting applicable flavor restrictions. Add active-run coverage for `disable=True, log_models=False`, verifying that automatic model/pipeline artifacts remain absent and explicit registration still works.

### Issue 2: Exported and explicitly registered classification pipelines lose original labels

- **Severity**: High
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 3723–3729
- **Description**: `automl_pipeline` includes feature preprocessing and `self.model`, but not the fitted label decoder. `DataTransformer.fit_transform()` label-encodes classification targets (`data.py:701–710`); only `AutoML.predict()` reverses that encoding (`automl.py:980–981`). Therefore, fitting with labels such as `"approve"`/`"deny"` or integers `10`/`20` produces an exported pipeline returning `0`/`1`, unlike the fitted AutoML object. `register_automl_pipeline()` registers this same pipeline.
- **Risk**: Deployment silently changes prediction values, output types, and class interpretation. The added registration round-trip test uses already canonical `0`/`1` labels, so it cannot detect this mismatch.
- **Suggested Fix**: Preserve label decoding and original class metadata in the exported prediction interface. Verify prediction equivalence between AutoML, its exported pipeline, and a reloaded registered model using string labels and noncanonical numeric labels.

### Issue 3: Trial artifacts can combine one model with another model's AutoFE

- **Severity**: Medium
- **File**: `flaml\fabric\mlflow.py`
- **Line(s)**: 692–703
- **Description**: `pickle_and_log_automl_artifacts()` receives the model being logged, but obtains preprocessing from `automl.feature_transformer`. That property now includes the AutoFE of the **globally selected** estimator (`automl.py:866–874`). When it returns a `Pipeline`, this branch appends the supplied model without using that model's own `autofe`. For example, with `retrain_full=False`, `model_history=True`, and `log_type="all"`, a nonwinning learner can be logged behind the winning learner's LDA/PCA transformation.
- **Risk**: Saved trial pipelines can fail with feature-count errors or silently make different predictions from the model represented by the trial. Background logging also makes reliance on mutable global selection state unsafe.
- **Suggested Fix**: Construct each artifact from task-level preprocessing and the explicitly supplied model's fitted AutoFE, not from the current global winner. Add a reload test covering two retained learners with different feature transformations and output dimensions.

### Issue 4: Valid scientific-notation integers are silently replaced with missing values

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 1867–1889, 1929–1934
- **Description**: For a column such as `["1e16", "2e16", "3e16"]`, the strict integer-string fast path does not match, but numeric parsing succeeds and identifies whole numbers. The precision-preserving path then rejects every source string: `int(s)` fails on exponent notation, and the float fallback accepts only magnitudes below `2**53`. It consequently constructs an all-`pd.NA` `Int64` column and reports successful integer conversion without rechecking the conversion threshold.
- **Risk**: A completely valid numeric column is silently erased during dtype conversion.
- **Suggested Fix**: Parse integral decimal/exponent forms exactly, for example using `Decimal` with integrality and range checks, or preserve the source representation when lossless conversion is unavailable. Revalidate successful conversion counts after precision recovery and add the scientific-notation case above.

### Issue 5: Featurization does not satisfy scikit-learn's cloning contract

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 212–254
- **Description**: Constructor parameters `params` and `config` are exposed as computed properties rather than stored constructor arguments. In particular, an unfitted instance created with `params=...` exposes `config=[]`. Scikit-learn cloning passes that derived list back to the constructor, selecting the reconstruction branch instead of the original parameter branch. `clone(Featurization(params={"fe.numerical": "scaler_standard"}))` therefore encounters the missing-task reconstruction error; providing a task still leaves constructor-parameter identity/configuration mismatches.
- **Risk**: The new scikit-learn pipelines cannot reliably participate in cloning, cross-validation, cached pipeline fitting, or grid search. This also affects AutoFE pipelines reconstructed from training logs.
- **Suggested Fix**: Implement a proper constructor/get-parameters contract that preserves initialization arguments unchanged and keeps learned/exported configuration separate. Cover cloning both unfitted and fitted instances, plus cross-validation of a log-reconstructed pipeline.

### Issue 6: Exported preprocessing contains an invalid boolean step when skip_transform is enabled

- **Severity**: Medium
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 866–874
- **Description**: `GenericTask.validate_data()` represents disabled task preprocessing with `_transformer=False` (`generic_task.py:197–199`). With AutoFE enabled, `feature_transformer` inserts that boolean directly into a scikit-learn pipeline as `("precessor", False)`. Direct `AutoML.predict()` handles the false sentinel, but `feature_transformer.transform()` and `automl_pipeline.predict()` do not: `False` is neither a transformer nor a supported passthrough step.
- **Risk**: A valid numeric AutoML fit using `skip_transform=True` becomes unusable through exported preprocessing, pipeline prediction, or model registration.
- **Suggested Fix**: Normalize the disabled-preprocessor sentinel to `None`/`"passthrough"` or omit that step when building pipelines. Add export and prediction-parity coverage with both `skip_transform=True` and AutoFE enabled.

### Issue 7: Time-series AutoFE loses its time-column context in score and exported pipelines

- **Severity**: Medium
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 866–874, 925–928
- **Description**: AutoFE fits time-series features after removing the timestamp (`autofe.py:350–352`). `AutoML.predict()` correctly supplies the estimator's `time_col` to `autofe.transform()`, but `score()` and the exported feature-transformer pipeline call it without that context. A valid PCA-only feature pipeline fitted on an exogenous column `"x"` then receives a prediction frame containing both `"ds"` and `"x"`; PCA rejects the unexpected timestamp/feature schema.
- **Risk**: Forecasting prediction can work through AutoML while scoring and exported/registered prediction fail on the same input.
- **Suggested Fix**: Retain time-column context in the fitted Featurization object, or consistently bind it on every scoring/export path. Verify scoring and exported prediction parity for a time-series model with a plain DataFrame input and a selected extraction transformation.

### Issue 8: Transform silently becomes an identity operation after a failed refit

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 338–342, 406–412, 428–456
- **Description**: `fit()` clears `pipeline` and sets `_is_fitted=False` before fitting. If fitting fails, `_transform()` subsequently sees `pipeline is None` and returns the input as though a successful no-op transformation had been fitted. The same happens on a fresh, unfitted instance. The new `test_failed_refit_is_not_fitted` checks `check_is_fitted()`, but does not exercise `transform()`.
- **Risk**: Failed preprocessing can silently feed untransformed data to an already fitted downstream estimator, producing incorrect predictions instead of reporting the invalid fitted state.
- **Suggested Fix**: Enforce the existing fitted-state hook at the public transform entry point. Keep successful no-op fits valid via `_is_fitted=True`, and test that both fresh and failed-refit instances raise `NotFittedError` on transformation.

### Issue 9: Selecting the largest tied timestamp delta merges distinct observations

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 509–510, 566–568
- **Description**: The largest-mode rule is incompatible with subsequent floor division. For timestamps `2024-01-01`, `2024-01-02`, and `2024-01-04`, inferred frequency is absent and the equally frequent deltas are one and two days. The code chooses two days and emits `time_idx=[0, 0, 1]`, merging the first two distinct timestamps. Longer alternating-gap series have the same problem.
- **Risk**: Panel forecasting receives duplicate time indices within a series, corrupting temporal identity or failing downstream dataset construction. `tft.py` uses this index for training boundaries and decoder construction.
- **Suggested Fix**: Choose a compatible base interval and validate alignment/injectivity, or reject unsupported spacing explicitly rather than flooring distinct observations into the same slot. Add an assertion for the example above, preserving the missing-day gap.

### Issue 10: Forecast hyperparameter importance invokes the wrong search-space interface

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 71–78
- **Description**: `optuna_space()` reconstructs every AutoML learner's space using `search_space(data_size=..., task=...)`. Forecast learners such as ARIMA and SARIMAX inherit `TimeSeriesEstimator.search_space(data, task, pred_horizon)` (`ts_model.py:86`), which does not accept `data_size`. Consequently, hyperparameter-importance plotting for a fitted forecast AutoML result raises `TypeError` before evaluation.
- **Risk**: The public visualization interface does not support forecasting results despite accepting AutoML results generally.
- **Suggested Fix**: Reuse the actual resolved learner search space, or dispatch with the same task-specific arguments already used by `SearchState` (`state.py:87–90`). Add forecast-result coverage for `get_param_importance()`/`plot_param_importance()`.

### Issue 11: Optimization history plots the worst score as the best score

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 40–41, 210–220
- **Description**: `_get_aml_df()` converts validation loss to `score = 1 - loss`, but the AutoML plotting branch calculates the cumulative **minimum** of those scores. For improving losses `[0.5, 0.1]`, the score points are `[0.5, 0.9]` while the displayed best-score line remains `[0.5, 0.5]`. The axis is additionally labeled as loss despite plotting scores.
- **Risk**: The optimization chart misrepresents progress and the winning trial, potentially showing degradation as improvement.
- **Suggested Fix**: Use one consistent representation: cumulative maximum for `1-loss` scores, or cumulative minimum for unmodified losses, with matching labels and appropriate axis bounds. Assert the actual plotted trace values rather than only constructing a figure.

### Issue 12: New OpenAI demos import a namespace absent from the proposed package

- **Severity**: Low
- **File**: `notebook\trident\demo_5_code_integrate_openai.ipynb`
- **Line(s)**: 729; also `notebook\trident\demo_6_math_integrate_chatgpt.ipynb:308`
- **Description**: Both new demos use `from flaml import oai, tune`. The proposed `flaml\__init__.py` does not export `oai`, and there is no `flaml.oai` module/package in the source tree. The legacy implementation is under `flaml.autogen.oai`, whose parent package explicitly directs users to the separate AutoGen repository.
- **Risk**: The notebooks fail at import when used with the public package this consolidation is intended to maintain.
- **Suggested Fix**: Update the examples to a supported client/API in the separate maintained project, or exclude obsolete demos. Do not reintroduce a removed public namespace solely to accommodate these samples.

### Issue 13: The math demo references an endpoint list that is never initialized

- **Severity**: Low
- **File**: `notebook\trident\demo_6_math_integrate_chatgpt.ipynb`
- **Line(s)**: 366, 1451
- **Description**: Independently of the import failure above, the only executable `config_list` initializer is commented out. The apparent assignment at line 376 belongs to a Markdown example, not a code cell. The later request nevertheless passes `config_list=config_list`, causing `NameError` in a fresh kernel after the preceding tuning work.
- **Risk**: The documented end-to-end example cannot complete and can fail only after paid tuning requests have already occurred.
- **Suggested Fix**: Initialize the endpoint list in an executable setup cell, or omit the argument consistently when relying on Fabric endpoint configuration. Validate notebook cell ordering with a mocked client.

## Resolution Log

_Updated by the driving agent as findings are addressed. No changes were made in this read-only review._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Explicit model-artifact opt-outs must remain effective.
- **How verified**: Static policy trace; disabled-autolog regression coverage pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Exported predictions must retain original target-label semantics.
- **How verified**: Static label-encoding/export trace; noncanonical-label round-trip coverage pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Each saved estimator must use its own fitted feature transformation.
- **How verified**: Static artifact-construction and caller trace; multi-model reload coverage pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Valid numeric values must not become missing during precision recovery.
- **How verified**: Static arithmetic/parser trace for scientific notation; regression coverage pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: AutoFE pipelines must support the advertised scikit-learn composition paths.
- **How verified**: Static constructor/property analysis; cloning and cross-validation coverage pending.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: Disabled preprocessing must be represented by a valid pipeline passthrough.
- **How verified**: Static sentinel-to-pipeline trace; export-parity coverage pending.

### Issue 7

- **Status**: Open
- **What changed**: Pending.
- **Why**: Forecast scoring and export must use the same timestamp handling as prediction.
- **How verified**: Static comparison of fit, predict, score, and export paths; regression coverage pending.

### Issue 8

- **Status**: Open
- **What changed**: Pending.
- **Why**: Unfitted and failed-refit transformations must fail explicitly.
- **How verified**: Static fitted-state trace and inspection of the existing failed-refit test; transform coverage pending.

### Issue 9

- **Status**: Open
- **What changed**: Pending.
- **Why**: Distinct observations must not collapse onto one time index.
- **How verified**: Static calculation of the three-timestamp example; regression coverage pending.

### Issue 10

- **Status**: Open
- **What changed**: Pending.
- **Why**: Forecast learners require their task-specific search-space interface.
- **How verified**: Static signature/call-site comparison; forecast visualization coverage pending.

### Issue 11

- **Status**: Open
- **What changed**: Pending.
- **Why**: Best-score visualization must agree with optimization direction.
- **How verified**: Static calculation of plotted scores and cumulative values; trace assertions pending.

### Issue 12

- **Status**: Open
- **What changed**: Pending.
- **Why**: Published examples must import APIs present in their supported package.
- **How verified**: Inspected notebook imports, package exports, and namespace locations; no notebook execution performed.

### Issue 13

- **Status**: Open
- **What changed**: Pending.
- **Why**: Notebook requests must not depend on undefined setup variables.
- **How verified**: Inspected executable cells versus Markdown content; mocked cell-order validation pending.

## Driver Decision

{
"contract": "direct",
"tier": "High",
"source": {
"tier": "Primary-agent estimate",
"reviewers": "Explicit driving-human override: high_gpt_grok_override",
"stageBehavior": "review-code High defaults",
"safetyStopPolicy": "Existing session controls; no new timeout, credit, or cycle bound"
},
"changeSize": "Large public/internal library consolidation: 142 changed files, with runtime and test changes spanning AutoML, tuning, forecasting, Fabric integrations, packaging, and CI. Notebook outputs, benchmarks, and notices dominate raw lines but do not account for the runtime breadth.",
"cyclomaticComplexity": "High qualitative complexity from environment-dependent defaults, asynchronous MLflow state and artifacts, per-fold feature engineering and resampling, temporal boundaries, compatibility branches, and optional-dependency failures.",
"rationale": "High is appropriate for interacting public contracts and concurrency across modules. There is no repository-local tier/model override. The default High Opus slot is unavailable in the live task catalog, so the human explicitly approved two available frontier families without reducing the six-stage coverage or per-stage retry gate.",
"primaryModel": "gpt-6-astra",
"reviewers": \[
{"slot": 1, "model": "gpt-6.1-sol", "reasoning": "max"},
{"slot": 2, "model": "grok-4.7", "reasoning": "xhigh"}
\],
"overrideSource": "User selected High with GPT-6.1-Sol and Grok-4.7 at maximum reasoning; explicitly omit unavailable Opus.",
"stageGate": "Six ordered stages. Run both required reviewers independently and concurrently. Advance when both are clean; otherwise fix, build/test, and run attempt 2. High may then advance with unresolved findings explicitly retained, but dispatch/evidence failures never count as clean or a valid dirty advancement.",
"scope": {
"repository": "microsoft/FLAML",
"pullRequest": 1545,
"branch": "lijiang/open-source-internal-merge",
"startingHead": "0e3c5d26f100710006c15a30782f8d80e84d5769",
"publicBase": "17d293b01edc49734ccdb79ef04b68d0da72fab8"
}
}

## Driver assessment after stage 1, attempt 1

The original response and verdict above are retained. Only the outer
script-generated frontmatter is authoritative. The duplicate frontmatter inside
the response is reviewer-generated text, not provenance.

Transport audit confirmed actual model `gpt-6.1-sol`, 40 view, 14 rg and 2 glob
calls, no disallowed tools or prior/sibling feedback access, and no source drift.
The initial reproduction batch had **19 failures and 1 pass**.

| Issue | What changed and why                                                                                                                                                                                                    | Verification                                                                                                                                                                                                  |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | The explicit global `log_models=False` restriction is retained independently of `disable`.                                                                                                                              | Disabled-global-autolog/active-run regression now passes; explicit registration remains covered.                                                                                                              |
| 2     | `AutoMLPipeline` preserves the fitted label decoder and original `classes_`; automatic and explicit artifact paths use it.                                                                                              | String and 10/20 labels match AutoML through export, pickle, and real MLflow registration/reload.                                                                                                             |
| 3     | A shared pipeline builder uses task preprocessing plus the specifically supplied trial model's AutoFE, never the current winner's composed transformer.                                                                 | A three-feature retained model logs/predicts correctly while the winner uses one-component LDA; the winner's steps are unchanged.                                                                             |
| 4     | Integral decimal/exponent strings are parsed with `Decimal` and checked for exactness/Int64 range; unsuccessful lossless recovery preserves strings and logs the reason instead of rounding or creating missing values. | Exact 1e16 and 2\*\*53+1 forms, negative values, and out-of-range exponent strings pass alongside the existing conversion suite.                                                                              |
| 5     | Constructor arguments are retained separately from learned export configuration; `get_params`/`set_params` preserve the sklearn clone contract.                                                                         | Fresh and fitted cloning, parameter replacement, and cross-validation of an AutoFE pipeline pass.                                                                                                             |
| 6     | Missing/false task and label preprocessors are normalized to valid passthroughs in exported pipelines.                                                                                                                  | `skip_transform=True` parity covers transformed features and exported predictions.                                                                                                                            |
| 7     | Fitted Featurization retains its timestamp column, and score passes the estimator time context explicitly.                                                                                                              | A real forecast model's exported predictions and score match its direct preprocessed estimator path. An unrelated reversed-R2 argument order in unchanged public `TimeSeriesEstimator.score` was not altered. |
| 8     | `transform` validates fitted state; successful no-op fits still set the fitted flag.                                                                                                                                    | Fresh/failed-refit transformations raise `NotFittedError`, while successful no-op transforms remain identity.                                                                                                 |
| 9     | The modal time step is used only when all observed intervals align; otherwise an exact common divisor preserves distinct temporal indices and gaps.                                                                     | Three timestamps at day offsets 0/1/3 now produce 0/1/3, not 0/0/1; other irregular and existing timezone/overflow cases pass.                                                                                |
| 10    | Importance plotting consumes the learner's resolved search state rather than calling every task with the tabular search-space signature.                                                                                | Forecast-style resolved-space test passes; the old constructor path is forbidden by its mock.                                                                                                                 |
| 11    | AutoML's `1-loss` scores use cumulative maximum, a score label and unconstrained plot range; Tune retains its own objective direction.                                                                                  | Actual trace values are asserted for improving and worsening trials.                                                                                                                                          |
| 12    | Removed the two obsolete OpenAI demos instead of restoring a removed API or introducing a new client migration.                                                                                                         | Confirmed both were PR-only additions, both import missing `flaml.oai`, and no notebook/test/site references name these files. Existing public examples remain untouched.                                     |
| 13    | Removed with the obsolete math demo above; no potentially paid execution can reach its undefined endpoint list.                                                                                                         | Verified the only assignment was commented code/Markdown and the later call used the undefined name.                                                                                                          |

Validation after fixes:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
215 passed
```

No stage has advanced yet. The driver will build the updated tree and submit
both required reviewers at stage 1, attempt 2 with a new immutable snapshot.

## Driver stage history

This section records advancement separately from immutable reviewer outcomes.
Both required runtime identities remain `gpt-6.1-sol` and `grok-4.7` under the
human-approved override above.

| Stage                         | Attempts                  | Outcome                                    | Carry-over                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------- | ------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 - Broad sweep               | 2, both valid and audited | Advanced-with-unresolved under High's gate | All seven GPT attempt-2 findings and Grok's new history/validation/axis defects were addressed before stage 2. Inherited no-parent final-retrain policy, dense run-index concerns, active-logging Tune wrapper semantics, and the SimpleForecaster direct-DataFrame concern remain explicitly retained. See both attempt-2 driver assessments for evidence and limits. |
| 2 - Architecture and patterns | 2, both valid and audited | Advanced-with-unresolved under High's gate | Introduced retry findings were addressed and validated before stage 3. Inherited import/logging/retrain/Spark concerns and the baseline RF timing error remain retained. Optional external-baseline reads were denied and are not claimed as reviewer evidence.                                                                                                        |
| 3 - Edge cases and robustness | 2, both valid and audited | Advanced-with-unresolved under High's gate | Introduced retry findings were fixed before stage 4 with 414 targeted passes. Inherited MLflow lifecycle/run-isolation, duplicate-timestamp cleanup, and forecast-input bound concerns remain explicitly retained.                                                                                                                                                     |
| 4 - Detailed correctness      | 2, both valid and audited | Advanced-with-unresolved under High's gate | Introduced retry findings fixed before stage 5 with 557 targeted passes and real Ray validation. Inherited forecast CV/subsampling/seasonal/panel defects and other previously recorded concerns remain retained; no clean-stage claim.                                                                                                                                |
| 5 - Testing and coverage      | 2, both valid and audited | Advanced-with-unresolved under High's gate | Introduced retry findings fixed before stage 6: 643 non-Spark passes, 53 offline NLP passes including AutoML training, real Ray and installed-Spark guards. Native Spark persistence and optional panel quality remain explicitly platform/dependency-limited locally.                                                                                                 |
| 6 - Polish and hardening      | 2, both valid and audited | Advanced-with-unresolved under High's gate | Final introduced findings were driver-resolved and validated after the cap: privacy, LDA partition handling, actual timing, fitted feature labels, regenerated notebook, accurate docs, and coverage failure gates. No third attempt or reviewer-clean claim.                                                                                                          |

### Final review outcome

All six ordered stages completed with both human-approved runtime reviewers:
`gpt-6.1-sol` at `max` and `grok-4.7` at `xhigh`; the unavailable Opus slot
was omitted only under the recorded human override. All 24 skill-owned
Markdown reports are retained. Every stage reached High's two-attempt cap
with valid dirty results, so the overall outcome is
**advanced-with-unresolved**, not clean.

Introduced findings were driver-resolved with regression coverage. Independently
verified inherited concerns remain visible rather than silently expanding
this consolidation into unrelated fixes: MLflow/Spark process-global lifecycle
and run ownership, default time-series CV budget/panel/subsampling behavior,
legacy seasonal and scoring semantics, and Spark metric/group ordering.
Local native Spark filesystem restoration and optional panel-quality execution
remain environment-limited; hosted Linux/runtime results must be reported
separately. Final post-gate fixes are validated driver changes, not a new
independent review attempt.

### Hosted CI follow-up after publication

The reviewed source and all 24 reports were published in
`44cd4cd2e06636784baf36ce7c350668dfbb9449`. Public installed-package jobs,
formatting, and documentation checks passed. Build run `39073896919`
exposed a dependency-resolution failure before test collection in the
Python 3.11/3.12 NumPy-1.x profiles: PyArrow 26 requires NumPy 2 at runtime,
while the established test profile intentionally selects NumPy 1.26.4.

The follow-up caps PyArrow below 26 only where the test extra selects NumPy
1.x. It does not add Arrow to the core/AutoML profiles or change the
Python 3.13/Windows ARM64 dependency paths. An isolated Python 3.12
environment imported NumPy 1.26.4, PyArrow 25.0.1, and the same hosted
Datasets 5.1.0 release and created a dataset successfully. Metadata marker
tests passed across the affected Linux/Windows Python profiles.

The corrected run `39075717540` passed the Linux test variants and public
installation jobs. Native Spark persistence was actually exercised in the
Python 3.12 Spark job (`test_persistence_contracts.py .`), whose summary was
88 passed and 2 skipped; this closes the prior local-only validation gap.

One remaining Windows ARM64/Python 3.11 job source-built XGBoost 3.2 with
Visual Studio's default x64 target and then failed to load its DLL
(`WinError 193`). Newer ARM64 wheels require Python 3.12, so the job now
preinstalls a CPU-native source build with both `CMAKE_GENERATOR` and
`CMAKE_GENERATOR_PLATFORM=ARM64`, without reusing cached mis-targeted wheels.
The variables are scoped to that one build step so other packages and newer
ARM64 jobs are unchanged. An immediate native fit/predict smoke check fails
the step if the resulting library is unusable.

Run `39081234773` showed that the environment-only target selection was
insufficient: the built wheel still failed its immediate native import.
The next correction configures the public XGBoost release directly with
`cmake -A ARM64`, uses a CPU-only/no-OpenMP build, and checks the PE machine
type of both the built and installed DLL before the fit/predict smoke.
The Python package then consumes that explicitly built local library; no
third-party source is patched and no architecture or matrix entry is skipped.

Stage 2 will use a new source snapshot after the 282-test suite, expanded
28-test pipeline checks, and successful wheel/source builds. Later clean stages
cannot erase the inherited unresolved findings or relabel stage 1 as clean.

### Driver input-size adjustment for stage 3 onward

The complete stage-3 source snapshot is 1,039,076 bytes before adding the
context and lens prompt. The driver sets the runner's prompt ceiling to
1,200,000 bytes to retain the complete code/test input rather than omit
reviewable changes. This is an input-size allowance only: reviewer identities,
maximum reasoning, six ordered stages, two-attempt High gates, and existing
execution safety controls are unchanged.

For stage 6, attempt 2, the complete source snapshot reached 1,208,482 bytes
before context. The same input-only allowance is raised to 1,300,000 bytes;
no source is omitted and no execution or stage limit is changed.
