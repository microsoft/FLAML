---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 1
round: 2
timestamp: '2026-10-17T07:57:54Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: c23adaebbfb70a61deddb6d64e68f2c74704fe925f2a1b4236bc7f7f60c773da
---

## Review Summary

- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-1-review-2-gpt-6.1-sol.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Reviewed the supplied diff and traced training, search initialization, prediction, and export through `flaml\automl\automl.py`, `state.py`, `ml.py`, and `pipeline.py`.
- [x] Inspected runtime detection and MLflow lifecycle handling in `flaml\fabric\__init__.py` and `mlflow.py`, including the history-only tests in `test\fabric\test_runtime.py`.
- [x] Compared AutoFE implementation and pipeline composition against `test\fabric\test_autofe_isolation.py`, `test_pipeline_contracts.py`, and the resampling contract in `test\automl\test_resampler.py`.
- [x] Inspected dependency declarations and formatting conventions in `pyproject.toml`, plus public-install coverage in `test\check_public_install.py`. No `AGENTS.md` was found at the repository root or under the inspected source/test trees.
- [x] Considered the supplied driver evidence: 282 targeted tests passed, the separate 28-test pipeline run passed, and wheel/source-distribution builds succeeded. These are reported results; this reviewer executed no commands.
- [ ] Live Fabric/Spark behavior and fresh full CI were not independently validated under this read-only assignment.

## Issues

### Issue 1: History-only integration still performs MLflow writes during Spark search

- **Severity**: High
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 2582–2587, 3006–3010
- **Description**: Fabric now creates an integration with `only_history=True` when neither a run nor autologging is active. However, `_search_parallel()` treats any non-null integration as permission to call `get_mlflow_log_latency()`. For a NumPy/pandas fit with `use_spark=True`, `max_iter=2`, and no logging intent, that helper starts and deletes an MLflow run; with the Fabric-default model history it also logs model artifacts. The existing history-only isolation test uses the single-trial path, which bypasses parallel search.
- **Risk**: Selecting Spark parallelism changes an otherwise in-memory history operation into tracking-service access and artifact writes. Training can fail when tracking is unavailable, despite history-only mode.
- **Suggested Fix**: Skip latency probing for history-only integrations. Also ensure any probe artifact respects `_do_log_model`. Add a mocked parallel-search regression that rejects MLflow client/run/artifact operations without starting a JVM.

### Issue 2: List-valued starting points silently bypass forced AutoFE

- **Severity**: Medium
- **File**: `flaml\automl\state.py`
- **Line(s)**: 93–95, 129–146
- **Description**: AutoFE domains are added to the search space, but initialization defaults are populated only when `starting_point` is not a list. The list is then assigned directly to `init_config`. Consequently, a valid call using `max_iter=1`, `featurization="force"`, and `starting_points={"rf": [{"n_estimators": 4}]}` trains without AutoFE: `_search()` consumes that point directly, and `train_estimator()` sees no `fe.*` parameters. The zero-shot `"data"` path also produces list-valued starting points.
- **Risk**: An explicit feature-engineering selection is silently ignored, including in ordinary single-learner, no-budget fits. The resulting model can also lack the requested exportable AutoFE pipeline.
- **Suggested Fix**: Normalize dictionary and list starting points consistently, completing missing AutoFE choices before any direct-training shortcut. Cover both explicit lists and zero-shot single-learner fits.

### Issue 3: Exported pipeline refitting omits the configured resampler

- **Severity**: Medium
- **File**: `flaml\automl\pipeline.py`
- **Line(s)**: 54–58, 79–92
- **Description**: Normal training invokes `_resample_training_data()` after AutoFE in `ml.py:472` and `567`. The exported pipeline contains preprocessing, AutoFE, and the bare learner; `AutoMLPipeline.fit()` delegates directly to sklearn's pipeline fitting, and neither preprocessing nor the learner invokes that resampling helper. Thus, after fitting AutoML with the existing `DuplicateMinorityResampler` test stub, refitting `clone(automl.automl_pipeline)` makes no resampler calls.
- **Risk**: Pipeline refits and cross-validation silently train on a different class distribution from the configuration selected by AutoML. A registered pipeline's refit behavior therefore does not preserve the original training contract.
- **Suggested Fix**: Preserve resampling as a cloneable, fit-only operation after AutoFE and before learner fitting, reusing the shared helper. Keep it out of prediction and enforce the existing weight/group restrictions. Add exported-pipeline coverage using the counter-based resampler stub.

### Issue 4: AutoFE strips genuine feature-name prefixes and creates duplicate columns

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 453–472
- **Description**: `_transform()` removes everything before the first `__` from every output name, assuming a `ColumnTransformer` generated the prefix. A selection-only pipeline has no such prefix. For example, `Featurization(params={"fe.selection": "cardinality"})` fitted on numeric columns `["a__b", "b"]` retains both features but returns columns `["b", "b"]`.
- **Risk**: Valid input schemas lose feature identity. Downstream learners can reject duplicate names, and exported configurations or feature-importance output can associate values with the wrong feature.
- **Suggested Fix**: Remove only prefixes known to have been introduced by the fitted column transformer, preserving original labels otherwise. Maintain an explicit output-name mapping and test existing `__` names with and without a column-transformer stage.

### Issue 5: The AutoFE installation extra omits its mandatory pandas dependency

- **Severity**: Medium
- **File**: `pyproject.toml`
- **Line(s)**: 186–188
- **Description**: The new `autofe` extra declares only scikit-learn. Pandas is not a required scikit-learn dependency, but `flaml\fabric\autofe.py:6` imports it unconditionally, as does the new pipeline adapter. A clean installation requesting only `flaml[autofe]` therefore does not supply the dependencies needed to import the advertised feature.
- **Risk**: The new opt-in installation surface is incomplete and fails in clean environments, while the existing core/AutoML smoke tests do not exercise this extra.
- **Suggested Fix**: Declare the feature's required dependencies, at least pandas, in the AutoFE extra without expanding NumPy-only core requirements. Add an isolated `[autofe]` install/import/transform smoke test.

### Issue 6: Disabling AutoFE breaks warm-starting from an AutoFE-generated configuration

- **Severity**: Medium
- **File**: `flaml\automl\state.py`
- **Line(s)**: 55–59, 93–105
- **Description**: `best_config_per_estimator` retains the newly generated `fe.*` keys. When those configurations are passed to a subsequent fit with `featurization="off"`—also the public default—the AutoFE domains are absent. With more than one iteration, `valid_starting_point()` indexes `search_space[name]` for those keys and raises `KeyError` rather than handling the disabled feature. This breaks the documented warm-start pattern after an AutoFE-enabled run.
- **Risk**: Users cannot reuse learned model hyperparameters while opting out of feature engineering; the same stored configuration works or crashes depending on the iteration budget.
- **Suggested Fix**: Reconcile starting-point keys with the selected AutoFE mode before validation. When off, retain compatible learner parameters and remove obsolete `fe.*` choices with an appropriate warning; never re-enable AutoFE implicitly. Cover dictionary and list warm starts across mode changes.

### Issue 7: MLflow run configuration metadata ignores fit-time overrides

- **Severity**: Low
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 426, 2403–2408
- **Description**: `_automl_user_configurations` captures constructor arguments, and fit arguments are captured only when that dictionary is empty. For example, constructing with `task="classification"` and fitting with `task="regression"` leaves classification in the metadata consumed by `MLflowIntegration.record_state()`. A constructor without settings captures only the first fit; subsequent fits keep that earlier configuration.
- **Risk**: Fabric display configuration and MLflow configuration artifacts can misreport the task, budget, or featurization settings for the run being inspected.
- **Suggested Fix**: Build a fresh per-fit configuration from constructor settings and explicit fit overrides using the same precedence as training. Avoid the separate, one-time introspection-based snapshot. Test constructor overrides and repeated fits.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: History-only integrations must remain isolated from tracking writes.
- **How verified**: Fix validation pending; source trace identifies the unconditional parallel latency probe.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Starting-point representation must not change explicit AutoFE behavior.
- **How verified**: Explicit-list and zero-shot regression coverage pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Exported refits must preserve the selected training operations.
- **How verified**: Counter-based exported-pipeline resampling regression pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Generated-name cleanup must preserve genuine input feature identity.
- **How verified**: Selection-only `["a__b", "b"]` schema regression pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: An installation extra must provide its directly required dependencies.
- **How verified**: Clean `[autofe]` installation smoke test pending.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: Warm-start normalization must respect explicit AutoFE opt-out.
- **How verified**: Multi-iteration warm-start mode-transition tests pending.

### Issue 7

- **Status**: Open
- **What changed**: Pending.
- **Why**: Run metadata must reflect that run's effective user configuration.
- **How verified**: Constructor/fit precedence and repeated-fit metadata assertions pending.

## Driver assessment after stage 2, attempt 1

Original findings and verdict are retained. Runtime identity was
`gpt-6.1-sol`; the transport audit found 34 view, 9 search, and 1 glob calls,
read-only tools, no previous/sibling feedback access, and no source drift.

| Issue | What changed and why                                                                                                                                                                                                                                    | Verification                                                                                                                                                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Parallel search skips latency probing in history-only mode. Logging-enabled probes receive the actual artifact opt-out as well as the history setting.                                                                                                  | Three mocked parallel-runner cases cover history-only, active-run/model opt-out, and active-run/model logging. The Spark serialization boundary is mocked; no JVM is started.           |
| 2     | List-valued initial configurations are completed with valid AutoFE initial choices, matching the dictionary path.                                                                                                                                       | Real explicit-list and zero-shot `"data"` single-trial fits both create non-null forced AutoFE pipelines.                                                                               |
| 3     | Exported pipelines retain a cloneable, fit-only resampler, use the shared resampling helper after preprocessing/AutoFE, and reuse the shared weight/group validator.                                                                                    | The counter-based resampler grows 60 rows to 108 on refit, is not called by prediction, and sees only 30 training rows per CV fold. Incompatible weights/groups raise.                  |
| 4     | Output cleanup strips only prefixes introduced by the fitted column transformer, not arbitrary substrings before `__`.                                                                                                                                  | Selection-only and numerical-transform cases preserve distinct `a__b` and `b` columns.                                                                                                  |
| 5     | `[autofe]` includes the ordinary AutoML import dependencies plus its stronger sklearn minimum. Core requirements remain NumPy-only. CI now creates separate environments for core, AutoML, and AutoFE, so extras cannot mask each other's dependencies. | An isolated installed `[autofe]` wheel imports and transforms successfully; an isolated NumPy-only wheel imports and tunes successfully. Both pass `pip check` with no MLflow or Spark. |
| 6     | Unsupported `fe.*` choices are removed before starting-point validation with an explicit warning; off mode cannot re-enable them through a list shortcut.                                                                                               | Dictionary/list warm starts with one and three iterations remain feature-off and preserve the caller's configuration.                                                                   |
| 7     | Per-fit metadata is rebuilt from that call's settings and overrides rather than retaining the constructor/first-fit snapshot. Effective feature exclusions are recorded.                                                                                | Real regression/classification fits cover constructor overrides and repeated changes to task, budget, and feature mode.                                                                 |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
332 passed, 1 optional imbalanced-learn skip

python -m pytest test\test_package_metadata.py -q --tb=short --disable-warnings
10 passed, including the subsequently added isolated notebook-generation test

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

The initial metadata fixture also exposed a positive-budget RF retrain
`_t1` error. This is independently verified baseline behavior:
`LGBMEstimator.fit` is AST-identical to public base lines 1581-1683, and
the public caller already passes `is_retrain=True`. It is not claimed fixed.
The metadata regression uses a real LightGBM fit to isolate the requested
configuration contract. The first parallel fixture likewise required mocking
the Spark trainable-serialization boundary before it could test the probe in
an environment without Spark.

Stage 2 has not advanced. Both required reviewers will receive a new
attempt-2 snapshot. A complete pristine public-base archive is also provided
as neutral reference source, enabling independent introduced-versus-existing
comparisons without exposing prior feedback.
