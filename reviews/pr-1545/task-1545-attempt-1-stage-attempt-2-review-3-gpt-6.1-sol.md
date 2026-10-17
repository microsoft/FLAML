---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 3
timestamp: '2026-10-17T12:21:06Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 299c4bbe37bf256aa24a71ede97d4e14661cf74eedee0950a870c82bf349bd8b
---

## Review Summary

- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-3-gpt-6.1-sol.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Traced AutoFE candidate failures through `flaml\fabric\autofe.py:439-455,565-576` and `flaml\automl\state.py:359-368`; inspected the variance-selection regression in `test\fabric\test_autofe_isolation.py:125-149`.
- [x] Inspected the new precision-preserving conversion branch in `flaml\automl\data.py:1755-1900` and the supplied lossless-conversion tests.
- [x] Inspected `flaml\fabric\fanova\evaluator.py:54-95` and `flaml\fabric\visualization.py:43-146`, including null handling, nested configurations, and objective-column normalization.
- [x] Confirmed existing nested-search support in `test\tune\test_tune.py:252-283`; compared it with visualization’s configuration extraction and `OptunaSearch.convert_search_space()`.
- [x] Read both the download-failure producer and downstream consumer in `benchmark\pmlb\build_benchmark.ipynb:1459-1514,1671-1678`.
- [x] Reviewed the driver-reported validation evidence, including 399 passing targeted tests and 12 passing Spark SQL contracts. Findings below concern additional boundary cases established by source tracing.
- [ ] Reproducer cases were not executed: this invocation permits read-only inspection only. Baseline comparisons use the supplied diff; no external baseline or other review artifacts were accessed.

## Issues

### Issue 1: Cardinality selection can still abort the entire AutoFE search

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 439-455, 565-576
- **Description**: `CardinalitySelector` can remove every feature, but only the corresponding `VarianceThreshold` failure is translated into `AutoFEIncompatibleDataError`. A concrete case is one categorical feature containing 20 distinct values repeated twice, evaluated with contiguous two-fold CV. Each training fold contains 20 unique categories, so `ordinal → cardinality → PCA` produces zero features and PCA raises an ordinary `ValueError`. `AutoMLState._compute_with_config_base()` rethrows that error instead of rejecting the candidate. With null extraction, the zero-feature frame reaches the learner and fails there instead.
- **Risk**: One ordinary, data-dependent candidate can terminate an otherwise valid AutoML search before a subsequent viable configuration is evaluated.
- **Suggested Fix**: Detect zero surviving features after cardinality selection and raise `AutoFEIncompatibleDataError`, including when no extractor follows. Add a CV regression where this candidate is followed by a viable candidate, while retaining propagation of unrelated implementation errors.

### Issue 2: Large integers cause valid boolean cells to become missing

- **Severity**: Medium
- **File**: `flaml\automl\data.py`
- **Line(s)**: 1837-1842, 1876-1884
- **Description**: The new precision-preserving branch converts Python booleans to `pd.NA`, then accepts the result whenever the remaining values meet `convert_threshold`. For an object column containing `[9007199254740993, 9007199254740994, True]`, the large integers select this branch, the boolean is discarded, and two successful conversions out of three exceed the default 0.6 threshold. In contrast, `[1, 2, True]` follows the ordinary numeric branch and preserves the boolean as `1`.
- **Risk**: An unrelated large value changes whether a valid, nonmissing cell survives conversion. The function returns an integer schema without reporting this data loss.
- **Suggested Fix**: Preserve boolean values consistently with the ordinary numeric/boolean conversion paths. If mixed booleans are intentionally unsupported, preserve the column or explicitly reject conversion rather than accepting newly introduced missing values. Add mixed large-integer/boolean regressions.

### Issue 3: fANOVA discards legitimate null categorical choices

- **Severity**: Medium
- **File**: `flaml\fabric\fanova\evaluator.py`
- **Line(s)**: 72-74
- **Description**: The validity mask treats every null parameter value as an absent observation, although `None` is a valid Optuna categorical choice. For `CategoricalDistribution([None, "balanced"])`, a frame containing those two choices with finite scores `[0.0, 1.0]` loses the `None` trial and returns `{}` before invoking Optuna. With more trials, importance is computed only from the non-null-choice subset.
- **Risk**: Common configurations such as `class_weight=None` are systematically excluded, producing empty or biased importance results despite valid completed trials.
- **Suggested Fix**: Make parameter validation distribution-aware and preserve the distinction between an absent conditional parameter and an explicitly supplied `None` categorical choice. Add importance tests for both valid null choices and genuinely absent parameters.

### Issue 4: Nested tuning configurations and importance distributions use different names

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 85-108
- **Description**: `get_hp_df()` expands configuration dictionaries only one level, whereas `OptunaSearch.convert_search_space()` recursively flattens distributions using `/`. For a supported tuning space such as `{"model": {"x": tune.uniform(0, 1)}}`, the hyperparameter frame contains a dictionary-valued `model` column, but the converted space requires `model/x`. Consequently, `get_param_importance()` raises the evaluator’s missing-column `ValueError`. Explicitly selecting `params=["model/x"]` fails earlier during frame selection.
- **Risk**: Importance analysis fails on valid nested Tune results. Existing nested-search support does not compose with the newly added reporting API.
- **Suggested Fix**: Normalize trial configurations and search spaces with the same flattening convention, and apply parameter selection after normalization. Add a visualization regression using a completed nested-space search.

### Issue 5: An auxiliary metric named score breaks objective extraction

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 43-50
- **Description**: Renaming the selected metric to `score` creates duplicate columns when trial results already contain an auxiliary metric named `score`. For example, a valid objective returning `{"loss": ..., "score": ...}` with `metric="loss"` produces two `score` columns. Subsequent `df["score"]` access returns a DataFrame instead of the expected Series. Importance evaluation then flattens both columns and raises the row-count mismatch error; history plots also receive the wrong output shape.
- **Risk**: Ordinary multi-metric Tune results break reporting or mix the objective with an unrelated metric.
- **Suggested Fix**: Extract the selected objective explicitly into one unambiguous Series or collision-safe internal column rather than renaming onto an existing column. Add a regression returning both the selected objective and an auxiliary `score`.

### Issue 6: Notebook download recovery produces records its consumer cannot read

- **Severity**: Medium
- **File**: `benchmark\pmlb\build_benchmark.ipynb`
- **Line(s)**: 1461-1466, 1512-1513, 1671-1676
- **Description**: The notebook’s runner catches dataset-fetch exceptions and persists `{"score": 0, "duration": 0}`, omitting `metric`. The later selection cell unconditionally reads `result[task]["metric"]`, so any caught download failure becomes a subsequent `KeyError`. This remains separate from the corrected failure-record handling in `runbenchmark.py`.
- **Risk**: A single transient download failure can invalidate a long benchmark run at its selection stage, while its checkpoint misleadingly records a measured zero score.
- **Suggested Fix**: Use a complete failure record with metric, explicit status, error, and no measured score; skip failed records before numeric comparisons. Reuse the script’s failure-record convention and cover the notebook producer/consumer together.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Cardinality’s zero-feature outcome is not classified as a recoverable candidate failure.
- **How verified**: Source trace only; regression execution pending.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: The precision branch can accept conversion after discarding valid boolean cells.
- **How verified**: Source trace of branch selection and threshold calculation; reproducer not executed.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: The missing-value mask excludes valid categorical `None` choices.
- **How verified**: Source trace of the validity mask and early return; reproducer not executed.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Configuration extraction and distribution conversion normalize nested names differently.
- **How verified**: Compared both implementations with existing nested-search coverage; reporting regression not executed.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: Objective renaming permits duplicate `score` columns and violates the expected Series shape.
- **How verified**: Source trace through objective extraction and evaluator shape validation; reproducer not executed.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: The notebook’s failure-record shape is incompatible with its downstream reader.
- **How verified**: Direct inspection of both source cells; notebook workload not executed.

## Driver assessment after stage 3, attempt 2

The original response and verdict remain intact. The audit confirmed
`gpt-6.1-sol`, 36 view and 8 search calls, read-only tools, no prior/sibling
feedback access, and no source drift. Stage 3 advances **with unresolved
findings** at High's two-attempt gate, not clean; there is no third robustness
attempt.

| Issue | Resolution before stage 4                                                                                                                                                                   | Verification                                                                                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Cardinality selection now raises the specific candidate-incompatibility error whenever no feature survives, before either extraction or learner fitting.                                    | Real contiguous-fold categorical cases reject the bad candidate and reach a viable candidate with both PCA and null extraction. Unrelated errors still propagate.                                |
| 2     | Precision-preserving integer conversion handles Python and NumPy booleans as exact 1/0 values, consistent with ordinary numeric conversion.                                                 | Large integers mixed with true/false values preserve every nonmissing cell.                                                                                                                      |
| 3     | Parameter validity is distribution-aware. Explicit Python `None` remains a categorical observation, while an absent parameter has a separate missing marker.                                | Real fANOVA evaluates None/balanced choices; a boundary test asserts that only the two present observations reach Optuna. Observed None also participates in categorical distribution expansion. |
| 4     | Configurations and distributions use the same existing `/` flattening convention. Parameter selection happens after normalization.                                                          | A completed nested Tune search reports `model/x` and computes importance for that same name.                                                                                                     |
| 5     | Objective extraction assigns the selected objective into a unique internal score column rather than renaming onto a colliding auxiliary column.                                             | A real multi-metric run retains the selected loss Series and valid importance output despite another metric named `score`.                                                                       |
| 6     | The benchmark notebook reuses the script's complete download-failure record and skips failed/incomplete legacy records before score comparisons. Modified cells' stale outputs are cleared. | Both notebook producer and selection consumer execute with a failed fetch and a successful record; only the successful record is selected, without running a benchmark workload.                 |

Validation:

```text
python -m pytest test\fabric\test_pipeline_contracts.py test\automl\test_lossless_conversion.py test\fabric\test_visualization_contracts.py test\fabric\test_runtime.py test\fabric\test_mlflow_coverage.py test\fabric\test_autofe_isolation.py test\fabric\test_fanova.py test\automl\test_training_log_reconstruction.py test\automl\test_data_coverage.py test\automl\test_ts_data.py test\automl\test_resampler.py test\test_package_metadata.py test\test_benchmark_contracts.py test\test_artifact_paths.py -n 2 --dist=loadfile -q --tb=short --disable-warnings
414 passed, 2 optional skips (Ray and imbalanced-learn)

python -m build --no-isolation
Built flaml-2.7.0.tar.gz and flaml-2.7.0-py3-none-any.whl
```

Actionlint also passes. These driver fixes do not rewrite the dirty review
outcome; inherited concerns remain retained in the companion assessment and
stage history.
