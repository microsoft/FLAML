---
producer: review-code
schema_version: 1
tier: High
contract: direct
execution_backend: copilot-cli
attempt: 1
stage_attempt: 2
round: 6
timestamp: '2026-10-17T20:54:12Z'
outcome: issues_found
reviewer_model: grok-4.7
input_sha256: 7331a16fd8f7164be8d8ece1c778ba3ed518ecb3bfb153743bb54a190fee41f6
---

## Review Summary

- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\\Users\\lijiang1.copilot\\session-state\\b0e4942d-f489-4b42-8a90-097ede4e88f7\\files\\pr-1545\\reviews\\pr-1545\\task-1545-attempt-1-stage-attempt-2-review-6-grok-4.7.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist

- [x] Read import-time side effects in `flaml/fabric/mlflow.py` lines 64–75 and callers `flaml/__init__.py` line 5, `flaml/automl/automl.py` line 72, and `flaml/tune/tune.py` line 41.
- [x] Read `UserWarning` sites that those filters hide: `flaml/automl/data.py` lines 806–812 and `flaml/fabric/visualization.py` lines 125–148.
- [x] Compared `mlflow_logging` docs at `flaml/automl/automl.py` lines 409 and 2288–2290 with the gate at lines 2543–2548 and the accurate `tune.run` text at `flaml/tune/tune.py` lines 484–487.
- [x] Compared `flaml/fabric/telemetry.py` lines 18–23 with `propagate = False` in `flaml/automl/logger.py` lines 45–62.
- [x] Checked `website/docs/FAQ.md` line 167 against `skip_transform` default `False` in `flaml/automl/automl.py` and the earlier FAQ example that sets `skip_transform=True`.
- [x] Checked coverage success-on-empty at `.github/workflows/python-package.yml` lines 261 and 329, plus `fail_ci_if_error: false` at line 349.
- [x] Read `flaml/fabric/README.md` line 113 and the MLflow/Fabric behavior table against the eager import. Intentional names `flaml.meric` and `avaliable_methods` were not treated as new typos.
- [ ] Tests and builds were not executed in this review. Driver-reported pytest, Spark, Ray, and wheel results were not treated as proof of these polish issues.

## Issues

### Issue 1: Importing FLAML applies Fabric logging policy process-wide

- **Severity**: High
- **File**: flaml/fabric/mlflow.py
- **Line(s)**: 64–75
- **Description**: If MLflow is installed, `import flaml` loads AutoML (`flaml/__init__.py` line 5), which imports `flaml.fabric.mlflow` (`flaml/automl/automl.py` line 72). `flaml.tune.tune` does the same at line 41. At import, before any fit, active run, autolog intent, or Fabric check, the module sets `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR`, creates a `ThreadPoolExecutor` of `cpu_count * 4` workers, and registers `atexit` shutdown with `wait=True`. Because `FLAML_IS_CLEAN_LOGS` defaults to `"1"`, it also raises several third-party loggers to `CRITICAL` and calls `warnings.simplefilter("ignore")` for `FutureWarning` and `UserWarning`. That hides later `UserWarning`s, including unseen-category prediction warnings in `flaml/automl/data.py` lines 806–812 and f-ANOVA warnings in `flaml/fabric/visualization.py` lines 125–148. `FLAML_IS_CLEAN_LOGS` is not documented. `flaml/fabric/README.md` line 113 says MLflow is imported when registration is called, not merely because AutoML is imported; registration is lazy, but this import is not.
- **Risk**: Public processes that happen to have MLflow installed lose warning and MLflow diagnostics, and shutdown can block on a stuck logging future. This makes internal notebook logging policy the default for ordinary users.
- **Suggested Fix**: Do not change global warnings, third-party logger levels, or the process environment at import. Apply any suppression only inside an explicit Fabric logging session, and default `FLAML_IS_CLEAN_LOGS` off. Create the executor on first submission and shut it down without waiting forever. Document any remaining opt-in.

### Issue 2: AutoML docs omit the MLflow logging gate

- **Severity**: Medium
- **File**: flaml/automl/automl.py
- **Line(s)**: 409, 2288–2290, 2543–2548
- **Description**: The constructor documents `mlflow_logging` as `default=True` and "Whether to log the training results to mlflow." `fit()` only points back to that setting. The implementation logs only when MLflow is installed and (`is_fabric_runtime()` or an active run or autologging). `tune.run` documents that gate (`flaml/tune/tune.py` lines 484–487). The AutoML docstrings do not.
- **Risk**: `help(AutoML)` tells public users that logging is on by default. They may expect tracking-server activity, or miss that an active run/autolog is required off Fabric.
- **Suggested Fix**: State that `True` only enables the integration, that off-Fabric logging still requires an active run or autologging, and that `False` disables it, including Fabric history collection.

### Issue 3: Telemetry logger can duplicate console output

- **Severity**: Low
- **File**: flaml/fabric/telemetry.py
- **Line(s)**: 18–23
- **Description**: Import adds a stdout handler to `flaml.fabric.telemetry` and leaves `propagate` at its default `True`. `flaml/automl/logger.py` and `flaml/tune/logger.py` set `propagate = False` specifically to avoid duplicate console lines. `import flaml` always loads this module and calls `log_telemetry`.
- **Risk**: A telemetry warning is printed by this handler and again by any root or ancestor handler. Every public import also attaches a logging handler as a side effect.
- **Suggested Fix**: Set `propagate = False` after adding the handler, or emit telemetry warnings through the existing FLAML logger instead of installing a second stdout handler at import.

### Issue 4: OOM FAQ inverts skip_transform

- **Severity**: Low
- **File**: website/docs/FAQ.md
- **Line(s)**: 167
- **Description**: The OOM section says that if data are already preprocessed, set `skip_transform` False. The parameter default is `False`, which keeps preprocessing on. The same FAQ earlier shows `skip_transform=True` to skip automatic preprocessing. Memory is saved by skipping FLAML's transform, not by leaving it enabled.
- **Risk**: Users following the OOM advice keep the heavier path and can make the memory problem worse.
- **Suggested Fix**: Tell users to set `skip_transform=True` when they have already preprocessed the data, matching the earlier example.

### Issue 5: Coverage check passes when no coverage exists

- **Severity**: Low
- **File**: .github/workflows/python-package.yml
- **Line(s)**: 261, 329, 349
- **Description**: The per-job combine step and the aggregate coverage job both `exit 0` when no coverage files exist. The aggregate job also runs `if: always()`. Codecov upload sets `fail_ci_if_error: false` and is skipped when `coverage.xml` is absent. A green "Combine coverage and upload to Codecov" check therefore does not prove that coverage was collected.
- **Risk**: A coverage-instrumentation regression can ship with a green coverage check and no report, which is a weak production observability gate.
- **Suggested Fix**: Fail the aggregate job when Ubuntu build jobs succeeded but uploaded no coverage data. Reserve `exit 0` for jobs that intentionally do not collect coverage.

## Resolution Log

_Updated by the driving agent as findings are addressed._

### Issue 1

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 2

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 3

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 4

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 5

- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

## Final driver disposition after the High stage gate

The original response and verdict remain intact. Actual model `grok-4.7`
used 49 view, 76 search, and 2 glob calls. All evidence/independence checks
passed. The second valid attempt completes stage 6 under the approved High
gate **advanced-with-unresolved**, not clean; there is no third attempt.

| Issue | Disposition and evidence                                                                                                                                                                                                                                                                                                                                             |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Retained, independently verified baseline concern. Earlier public-base comparisons establish the existing eager integration import, warning/logger settings, and executor lifecycle. Executor construction alone starts no workers. Documentation is corrected; a global lifecycle redesign is not claimed.                                                          |
| 2     | Fixed. Both public AutoML parameter descriptions explain optional availability, the off-Fabric active-run/autolog gate, and complete disabling of MLflow/Fabric history through `False`.                                                                                                                                                                             |
| 3     | Fixed. Telemetry uses normal logging propagation without installing its own console handler. A fresh-process regression asserts no extra handler and exactly one captured warning.                                                                                                                                                                                   |
| 4     | Fixed. The OOM FAQ now recommends `skip_transform=True` for already preprocessed input.                                                                                                                                                                                                                                                                              |
| 5     | Fixed. A successful Ubuntu test job or successful build matrix now fails on missing coverage; unsuccessful upstream work may explicitly skip empty aggregation. Uploading a nonexistent coverage file is an error. Native Linux shell tests additionally exposed and fixed the old literal `.coverage` array entry that made the missing-data condition unreachable. |

The companion final assessment records the post-gate regression and native
Ray results. All six coverage-shell combinations (per-job/aggregate against
success/failure/cancelled) were executed under Linux and return the intended
status. Codecov's external upload remains deliberately best-effort; successful
test instrumentation is no longer silently optional.

These driver resolutions do not erase the original findings or convert the
approved High outcome into a clean sign-off.
