---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 6
timestamp: '2026-10-17T20:53:43Z'
outcome: issues_found
reviewer_model: gpt-6.1-sol
input_sha256: 88989bdbfb6010e003798247cd2c12106ba01e564f1badff817df18d35e1970b
---

## Review Summary

- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-6-gpt-6.1-sol.md
- **Issues Found**: 6
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Reviewed the supplied public-base diff and traced centralized diagnostics through `flaml\automl\automl.py`, `flaml\fabric\logger.py`, and `test\fabric\test_telemetry.py`.
- [x] Inspected AutoFE fitting and candidate rejection in `flaml\fabric\autofe.py:434–473` and `flaml\automl\state.py:333–376`, alongside the isolation and pipeline contract tests.
- [x] Traced visualization timing and feature-name inputs through `flaml\fabric\visualization.py`, `flaml\automl\model.py:219–239`, `flaml\automl\automl.py:866–872`, and `flaml\tune\trial.py:95–100,135–140`.
- [x] Compared the committed combined notebook’s installation cell with `notebook\trident\generate_test.py` and the generation assertions in `test\test_package_metadata.py`.
- [x] Cross-checked the runtime README’s import claims against the actual module-level MLflow imports and reviewed the dependency declarations in `pyproject.toml`.
- [x] Considered the supplied driver validation results as reported evidence only. No commands or tests were executed during this review.
- [ ] Fresh full CI, live Fabric execution, and notebook workloads were not independently verified; these remain outside this read-only review.

## Issues

### Issue 1: AutoML centralized diagnostics serialize caller-controlled objects

- **Severity**: High
- **File**: `flaml\automl\automl.py`
- **Line(s)**: 469–471, 2834–2838
- **Description**: The new Kusto messages interpolate the metric object, split object, custom learner names, and unrestricted `entrypoint` value directly. A valid callable metric can be a `functools.partial` whose representation contains bound arguments, or an object whose representation includes private configuration. These values consequently enter centralized diagnostics. Tune’s startup logging deliberately avoids this, but AutoML does not apply the same boundary.
- **Risk**: Fabric diagnostics can receive caller data or sensitive configuration that was supplied for computation rather than telemetry. This path also runs independently of `mlflow_logging=False`.
- **Suggested Fix**: Restrict centralized fields to allowlisted built-in identifiers, counts, and boolean flags. Represent custom metrics/splitters without formatting their objects, and validate the entrypoint against its documented choices. Add an AutoML counterpart to the existing private-payload logging test.

### Issue 2: Zero-within-class-variance LDA candidates can abort AutoFE searches

- **Severity**: Medium
- **File**: `flaml\fabric\autofe.py`
- **Line(s)**: 457–473
- **Description**: The LDA compatibility guard checks only class and sample counts. Valid data such as `X={"x": [0.0, 0.0, 1.0, 1.0]}` with `y=[0, 0, 1, 1]` passes that guard but has zero within-class rank. Default SVD LDA fails on this input, typically while indexing an empty singular-value array. This failure is not converted to `AutoFEIncompatibleDataError`; `AutoMLState` therefore cannot reject the candidate and continue.
- **Risk**: Selecting a data-incompatible LDA recipe can terminate an otherwise viable search, including after other candidates have trained successfully.
- **Suggested Fix**: Detect the incompatible LDA training-partition condition using the actual extraction-stage input and report it through `AutoFEIncompatibleDataError`, without broadly suppressing implementation errors. Add a search regression with this LDA candidate followed by a successful no-op candidate.

### Issue 3: Tune timelines fabricate sequential execution for parallel trials

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 517–522
- **Description**: Tune bar positions are calculated from the cumulative sum of trial durations rather than recorded start times. Two five-second trials starting together are plotted at bases zero and five, showing ten seconds of serial work instead of five seconds of overlapping execution. Trial start/update timestamps already exist on the analysis trials.
- **Risk**: The timeline misrepresents Ray/Spark concurrency, elapsed time, and scheduling gaps, undermining the new reporting surface’s usefulness for performance diagnosis.
- **Suggested Fix**: Position bars using actual trial timing metadata normalized to an experiment origin. Preserve those timings when constructing the reporting frame. Add assertions for overlapping trials and gaps between trials; retain cumulative-duration behavior only where it is explicitly identified as an approximation.

### Issue 4: Feature-importance plots can pair transformed importances with raw feature names

- **Severity**: Medium
- **File**: `flaml\fabric\visualization.py`
- **Line(s)**: 290–292
- **Description**: The plot assumes `result.feature_names_in_` describes the learner’s transformed inputs. When the learner does not expose a name attribute recognized by `BaseEstimator.feature_names_in_`, AutoML falls back to the original schema. CatBoost exposes `feature_names_`, which the adapter does not check; name-less custom learners have the same problem. With three raw features reduced to one LDA component, this supplies three labels for one importance value and makes `px.bar` fail. Equal-length transformations can instead produce misleading labels.
- **Risk**: Feature-importance reporting fails or mislabels features for supported AutoFE/learner combinations.
- **Suggested Fix**: Resolve names from the actual fitted model or retained AutoFE output schema, including CatBoost’s naming API, and verify alignment before plotting. Add coverage for dimensionality reduction with CatBoost or a custom learner that trains on NumPy arrays.

### Issue 5: The committed combined notebook does not install the public FLAML source

- **Severity**: Low
- **File**: `notebook\trident\all_in_one_test.ipynb`
- **Line(s)**: 9–10
- **Description**: The committed bootstrap installs auxiliary packages but no FLAML package. Its generator now explicitly includes the public GitHub FLAML requirement, so the checked-in notebook is stale relative to that change. The generation test checks a temporary regenerated notebook, not the committed artifact.
- **Risk**: Running the committed notebook uses whatever FLAML version is preinstalled—or fails when none is installed—instead of exercising the intended public consolidation.
- **Suggested Fix**: Regenerate and persist the combined notebook with the public installation requirement. Extend the contract test to inspect the committed bootstrap cell as well as regenerated output.

### Issue 6: The README incorrectly promises lazy MLflow imports

- **Severity**: Low
- **File**: `flaml\fabric\README.md`
- **Line(s)**: 112–114
- **Description**: The README says MLflow is imported when registration is called, “not merely when AutoML is imported.” However, `flaml\automl\automl.py:66–75` and `flaml\tune\tune.py` import MLflow and its integration at module load when installed. The registration wrapper is lazy, but the package-wide MLflow import is not.
- **Risk**: Users receive inaccurate expectations about startup cost and optional integration initialization.
- **Suggested Fix**: Describe the actual guarantee: MLflow is optional, and registration requires it when invoked. Remove the package-wide lazy-import claim unless import behavior is intentionally changed and separately verified.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: Pending.
- **Why**: Centralized diagnostics need a consistent caller-data boundary.
- **How verified**: Static source tracing only; the proposed AutoML payload regression was not executed.

### Issue 2

- **Status**: Open
- **What changed**: Pending.
- **Why**: A training-partition incompatibility should reject the candidate rather than abort the search.
- **How verified**: Inspected the LDA guards and exception path; the proposed singular-data regression was not executed.

### Issue 3

- **Status**: Open
- **What changed**: Pending.
- **Why**: A timeline must preserve actual overlap and scheduling gaps.
- **How verified**: Compared the cumulative-duration calculation with available trial timestamps; no plotting tests were executed.

### Issue 4

- **Status**: Open
- **What changed**: Pending.
- **Why**: Importance values must be labeled using the fitted learner’s feature space.
- **How verified**: Traced feature-name fallback and plotting inputs; no AutoFE/CatBoost reproduction was executed.

### Issue 5

- **Status**: Open
- **What changed**: Pending.
- **Why**: The persistent notebook should match its public-source generator.
- **How verified**: Directly inspected the committed installation cell, generator, and generation test.

### Issue 6

- **Status**: Open
- **What changed**: Pending.
- **Why**: Documentation should distinguish optional dependencies from deferred imports.
- **How verified**: Compared the README statement with module-level import paths.

## Final driver disposition after the High stage gate

Original findings and verdict are preserved. Actual model `gpt-6.1-sol`
used 18 view and 10 search calls. The artifact, identity, independence,
allowed-tool, and source-integrity audits passed. This was the second valid
attempt of the sixth stage; the approved High review is complete
**advanced-with-unresolved**, not reviewer-clean. No third attempt was run.

| Issue | Driver resolution and validation                                                                                                                                                                                                                                                                               |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | AutoML diagnostics now contain only fixed event names, counts, and boolean classifications. Entrypoint values are validated against the existing code-first/low-code choices before logging. A callable metric, custom splitter, and learner carrying private markers do not appear in centralized logs.       |
| 2     | A narrowly scoped LDA subclass checks its actual post-selection input for zero within-class variation and raises the existing candidate-incompatibility error. A cardinality-selection/LDA failure followed by a valid candidate now completes the search. Arbitrary implementation errors are not suppressed. |
| 3     | Reporting preserves trial start times and plots true overlap/gaps. Missing timing metadata is explicitly labeled as an approximate sequence. Synthetic overlapping/gapped histories and native Ray reporting tests pass.                                                                                       |
| 4     | CatBoost's `feature_names_` is recognized, fitted AutoFE output names are available through the standard accessor, and plotting checks alignment before rendering. CatBoost and a name-less custom NumPy learner both plot the correct one-dimensional LDA feature space.                                      |
| 5     | The committed combined notebook was regenerated from its source notebooks. Its bootstrap explicitly installs the public GitHub source, package ordering is deterministic, and execution counts/outputs are cleared. The contract checks the committed artifact as well as regeneration.                        |
| 6     | The README now describes optional MLflow availability and the lazy registration wrapper without promising package-wide deferred imports.                                                                                                                                                                       |

Final driver validation at this point: **668 targeted passes, 8 skips**
(two optional dependencies and six Linux-shell-only tests), plus **30 Ray
environment passes and one optional CatBoost skip**. All six missing-coverage
shell cases were separately executed in native Linux through WSL. Original
reviewer findings remain above; these post-gate driver fixes were validated
but are not presented as an additional independent reviewer sign-off.

The previously identified inherited MLflow/Spark lifecycle, forecast, and
metric issues remain explicitly retained in the earlier assessments. Local
native Spark persistence and optional panel-quality limitations also remain
visible pending the appropriate hosted/runtime environments.
