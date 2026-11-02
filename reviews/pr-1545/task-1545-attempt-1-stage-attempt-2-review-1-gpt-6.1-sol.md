---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 1
timestamp: '2026-10-17T07:12:59Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 34f13a2ada7f508cb6126a725f4fcffcdce4c7669f74595bf8a28322eba90333
---

## Review Summary

- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-1-gpt-6.1-sol.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced AutoFE configuration, fold fitting, and final retraining through `flaml\fabric\autofe.py`, `flaml\automl\state.py`, and `flaml\automl\ml.py`; inspected `test\fabric\test_autofe_isolation.py`.
- [x] Inspected pipeline export and log-retraining paths in `flaml\automl\pipeline.py`, `flaml\automl\automl.py`, and `flaml\fabric\mlflow.py`; compared coverage in `test\fabric\test_pipeline_contracts.py` and `test\automl\test_training_log_reconstruction.py`.
- [x] Reviewed runtime detection, public dependency metadata, logging opt-outs, and autolog restoration against `flaml\fabric\README.md`, `pyproject.toml`, and `test\fabric\test_runtime.py`.
- [x] Inspected timestamp indexing and forecasting behavior in `flaml\automl\data.py`, `flaml\automl\time_series\ts_data.py`, and `flaml\automl\time_series\ts_model.py`, including weighted MAPE, binary-only metric validation, and the latest-observation CV boundary.
- [x] Reviewed visualization implementation and assertions in `flaml\fabric\visualization.py` and `test\fabric\test_visualization_contracts.py`, plus the supplied notebook source changes.
- [x] Considered the supplied driver evidence: 261 targeted tests passed, and `python -m build --no-isolation` built both distributions. These are driver-reported results, not reviewer executions.
- [ ] No commands or tests were executed during this read-only review; the reproductions below are derived from source inspection.

## Issues

### Issue 1: Removing timezone offsets merges distinct observations during DST fallback

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 416–430, 462–470, 488–492
- **Description**: Both frequency inference and index arithmetic discard timezone offsets with `tz_localize(None)`. For `pd.date_range("2024-11-03 00:00", periods=5, freq="h", tz="America/New_York")`, the two distinct 01:00 observations become identical wall-clock timestamps. The resulting indices are `[0, 1, 1, 2, 3]`, merging observations. With only the two repeated-hour observations, the derived step is zero and the modulo operation raises `ZeroDivisionError`.
- **Risk**: Valid timezone-aware hourly data acquires duplicate time indices or fails conversion, breaking panel-forecast ordering and uniqueness requirements.
- **Suggested Fix**: Preserve offset/UTC information for intraday elapsed-time indexing. Apply wall-clock indexing only where calendar cadence requires it, and explicitly guard against nonpositive steps. Add fallback-DST hourly cases alongside the existing timezone tests.

### Issue 2: Exported pipelines use legacy task transformers without adapting their contracts

- **Severity**: Medium
- **File**: `flaml\automl\pipeline.py`
- **Line(s)**: 13–17, 38–47
- **Description**: `make_automl_pipeline()` installs `_transformer` directly as a sklearn pipeline step. For ordinary AutoML, this is `DataTransformer`, whose `fit_transform(X, y, task)` requires a third argument and returns `(X, y)`. Consequently, calling the new exported pipeline’s `fit(X, y)` fails rather than supporting refitting or cross-validation. Time-series prediction also bypasses `TimeSeriesTask.preprocess()`: `pipeline.predict(5)` reaches `DataTransformerTS.transform()` and fails on `X.drop`, while DataFrame prediction drops columns from the caller’s frame in place. If a fitted transformer drops a constant column, predicting twice on that same frame fails on the second call. The export tests currently exercise fitted prediction on string-named DataFrames, not these contracts.
- **Risk**: Exported/registered pipelines do not compose with supported forecasting inputs or normal sklearn refitting, and time-series prediction can mutate caller data.
- **Suggested Fix**: Introduce a sklearn-compatible adapter around task preprocessing. Its fitting interface should return features only, and transformation should preserve AutoML’s input normalization and copying behavior. Cover exported-pipeline refitting, integer forecast horizons, and repeated prediction on an unchanged DataFrame.

### Issue 3: Pipeline export fails after training through `retrain_from_log()`

- **Severity**: Medium
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 3738–3741
- **Description**: `automl_pipeline` reads `self._featurization`, but that attribute is assigned only by `fit()`. A fresh `AutoML` successfully trained with `retrain_from_log()` on a record containing `fe.*` settings therefore raises `AttributeError` when exporting its pipeline. Reusing an instance can instead leave the flag stale from a previous fit and incorrectly return `None`. Default registration also depends on another fit-only attribute, `_mlflow_exp_name`, in `flaml\fabric\mlflow.py`.
- **Risk**: A successfully reconstructed, fitted model cannot be exported or registered, contrary to the requirement that log reconstruction and pipeline export compose.
- **Suggested Fix**: Establish export state on every training path, preferably using the actual fitted featurizer/logged configuration rather than a fit-only flag. Make the registration experiment-name fallback safe for log-retrained instances. Add a fresh-instance `retrain_from_log()` → export → registration regression case.

### Issue 4: A single-trial fit silently ignores forced AutoFE

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 77–81
- **Description**: AutoFE search entries contain domains but no initial values. `SearchState` therefore omits them from `init_config`. For `AutoML(estimator_list=["rf"], max_iter=1, featurization="force", mlflow_logging=False)`, `_search()` takes its direct-retraining path and trains `state.init_config[0]` without completing those domains. No `fe.*` parameters reach `train_estimator()`, so `model.autofe` is `None` despite the explicit `"force"` selection.
- **Risk**: A valid public option silently has no effect in single-trial/refit workflows; forced transformations are not applied.
- **Suggested Fix**: Supply valid AutoFE initial choices or complete the configuration before the direct-training path. Verify that `"force"` materializes non-null stage choices with `max_iter=1` without manually supplying AutoFE starting points.

### Issue 5: AutoFE crashes on supported DataFrames with integer column labels

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 313–317, 517
- **Description**: Column labels are converted to strings only when the input is not already a DataFrame. Thus `Featurization(params={"fe.selection": "cardinality"}).fit(pd.DataFrame({0: [0., 1., 2.]}))` reaches `col.startswith(...)` with an integer and raises `AttributeError`. This also affects numeric DataFrames inside AutoML when cardinality selection is chosen without a preceding column transformer. Separately, numerical transformers receive integer labels as sklearn positional selectors, so nonconsecutive labels such as `10` and `20` can select nonexistent positions.
- **Risk**: Enabling AutoFE makes otherwise supported numeric DataFrame inputs fail during training.
- **Suggested Fix**: Use a consistent, collision-safe column-name mapping or resolve selectors explicitly by position throughout fitting and transformation. Avoid string operations on unchecked labels. Add coverage for both RangeIndex and nonconsecutive integer column labels.

### Issue 6: Slice plots hide valid objectives outside the classification-score interval

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 546–549
- **Description**: Every slice subplot fixes its y-axis to `[0, 1]`, although Tune scores are the original arbitrary objective values and AutoML’s `1 - validation_loss` can be negative. A Tune objective returning losses between 10 and 20 produces traces entirely outside the visible range. The first axis is also labeled `Score(1-loss)` for Tune, where no such conversion occurs.
- **Risk**: Valid regression/tuning results appear empty or omit important trials, misleading optimization analysis.
- **Suggested Fix**: Autorange the y-axes from the actual scores and label Tune plots with `default_metric`. Add contract assertions for objectives above one and negative AutoML scores.

### Issue 7: The AutoFE tutorial documents the opposite Fabric default

- **Severity**: Low
- **File**: `notebook\trident\featurization.ipynb`
- **Line(s)**: 615
- **Description**: The tutorial states, “On Fabric, it's set to `auto` by default.” The implementation and approved requirements keep AutoFE off in every environment unless explicitly selected or configured through `FLAML_FEATURIZATION`.
- **Risk**: Users following the tutorial can mistakenly assume feature engineering is enabled and misinterpret their results.
- **Suggested Fix**: Document `"off"` as the default in Fabric too, with explicit `"auto"`/`"force"` or environment configuration required to enable it.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Distinct DST observations must retain distinct time indices.
- **How verified**: Source-derived timestamp arithmetic; runtime regression verification pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: Export must adapt legacy task-transformer fitting and prediction contracts.
- **How verified**: Inspected transformer signatures, in-place mutation, and pipeline construction; execution pending.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: Log-retrained models need valid export and registration state.
- **How verified**: Traced constructor, `fit()`, `retrain_from_log()`, and export-property assignments; execution pending.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Explicit forced featurization must survive the direct-training path.
- **How verified**: Traced AutoFE domains through `SearchState.init_config` and `_search()`’s single-trial branch; execution pending.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: Integer-labeled DataFrames must remain usable when AutoFE is enabled.
- **How verified**: Inspected label normalization, selector construction, and cardinality-prefix checks; execution pending.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: Visualization must display the supported objective range accurately.
- **How verified**: Compared score construction with explicit slice-axis limits; rendering verification pending.

### Issue 7

- **Status**: Open
- **What changed**: Pending.
- **Why**: Tutorial defaults must match the approved opt-in behavior.
- **How verified**: Compared notebook source with constructor defaults and runtime-policy assertions.

## Driver assessment after stage 1, attempt 2

The original reviewer response, open-at-review statuses, and verdict above are
preserved. The transport audit confirmed actual model `gpt-6.1-sol`, 35 view and
12 search calls, read-only tools, no earlier/sibling feedback access, and no
source drift during the concurrent batch.

High's two-attempt gate advances stage 1 **with unresolved findings**, not clean.
There is no third broad-sweep attempt. The following carry-over fixes were
implemented and verified before stage 2:

| Issue | Driver resolution                                                                                                                                                                         | Verification                                                                                                                                                                                                                                                    |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Calendar cadence retains wall-clock indexing; intraday cadence uses elapsed UTC time. Nonpositive steps raise explicitly.                                                                 | Both a complete DST fallback and just its repeated-hour pair produce unique consecutive indices; existing daily/monthly/yearly DST and overflow tests remain passing.                                                                                           |
| 2     | Added a sklearn task-preprocessor adapter. Refit returns feature-only tabular data or the required forecasting dataset; prediction delegates to task normalization/copying.               | Exported classification supports refit and cross-validation with original labels. RF and naive forecasting exports support refit; repeated DataFrame prediction is nonmutating and supported integer horizons retain parity.                                    |
| 3     | Export eligibility uses the actual fitted AutoFE rather than a fit-only flag. Default registration safely falls back when a log-retrained instance lacks an experiment name.              | Fresh and previously fitted instances replay a real two-trial log, export, and predict correctly; a fresh replay is registered and loaded from MLflow. The fixture uses two trials because the pre-existing direct single-trial path writes no training record. |
| 4     | AutoFE domains now provide valid initial values; forced domains choose non-null methods in the direct single-trial path.                                                                  | A real `max_iter=1, featurization="force"` fit with no supplied AutoFE starting points builds and uses its pipeline.                                                                                                                                            |
| 5     | A fitted collision-safe mapping normalizes non-string feature labels, validates input identity, and preserves fitted column order; selectors no longer apply unchecked string operations. | RangeIndex, 10/20, mixed string/integer, and genuinely colliding generated names pass with cardinality and standard scaling, reordered input, and unchanged source frames.                                                                                      |
| 6     | Slice axes autorange. AutoML explicitly labels its affine score, while Tune uses the real objective name.                                                                                 | Negative and above-one scores remain present with no fixed axis range; Tune's cost label and exact trace values are asserted.                                                                                                                                   |
| 7     | The notebook now documents `off` in Fabric and explicit `auto`/`force` or environment selection.                                                                                          | Source Markdown matches the runtime policy and existing environment/constructor precedence assertions.                                                                                                                                                          |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\test_package_metadata.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
282 passed

python -m pytest test\fabric\test_pipeline_contracts.py -q --tb=short --disable-warnings
28 passed, including the additional RF-refit and true-name-collision assertions

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

These driver fixes do not rewrite the dirty reviewer outcome or establish
independent re-review signoff. The surviving inherited concerns are retained in
the sibling Grok artifact and the driver stage history.
