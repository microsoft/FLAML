import copy
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from flaml import AutoML

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def tracking(monkeypatch, tmp_path, request):
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    uri = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
        if getattr(request, "param", None) == "sqlite"
        else (tmp_path / "tracking").as_uri()
    )
    mlflow.set_tracking_uri(uri)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            try:
                yield mlflow.tracking.MlflowClient()
            finally:
                while mlflow.active_run() is not None:
                    mlflow.end_run()
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def explicit_requirements(monkeypatch):
    original = mlflow.sklearn.log_model

    def log_model(*args, **kwargs):
        kwargs["pip_requirements"] = ["flaml", "numpy", "pandas", "scikit-learn", "optuna"]
        return original(*args, **kwargs)

    monkeypatch.setattr(mlflow.sklearn, "log_model", log_model)


@pytest.mark.parametrize("ensemble", [False, True])
def test_parent_artifact_matches_final_model_and_does_not_end_caller_run(tracking, explicit_requirements, ensemble):
    from sklearn.ensemble import StackingClassifier

    X = pd.DataFrame(np.random.RandomState(7).normal(size=(80, 3)), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 0).astype(int)
    automl = AutoML(
        estimator_list=["rf", "extra_tree"] if ensemble else ["rf"],
        max_iter=4 if ensemble else 2,
        n_jobs=1,
        verbose=0,
        eval_method="holdout",
        retrain_full=False,
        ensemble={"n_jobs": 1} if ensemble else False,
        model_history=True,
        log_type="all",
    )
    with mlflow.start_run() as parent:
        automl.fit(X, y)
        assert mlflow.active_run().info.run_id == parent.info.run_id
        assert tracking.get_run(parent.info.run_id).info.status == "RUNNING"
        restored = mlflow.sklearn.load_model(f"runs:/{parent.info.run_id}/model")
        np.testing.assert_array_equal(restored.predict(X), automl.predict(X))
        if ensemble:
            assert isinstance(restored.steps[-1][1], StackingClassifier)
        mlflow.log_metric("after_fit", 1.0)
    assert tracking.get_run(parent.info.run_id).data.metrics["after_fit"] == 1.0


def test_each_trial_artifact_retains_its_own_model_and_pruned_requirements(
    tracking, explicit_requirements, monkeypatch
):
    from packaging.requirements import Requirement

    from flaml.fabric.mlflow import MLflowIntegration

    X = pd.DataFrame({"value": np.arange(60.0), "squared": np.arange(60.0) ** 2})
    y = np.tile([0, 1], 30)
    expected = []
    original = MLflowIntegration.record_state

    def record(self, automl, state, estimator, *args, **kwargs):
        model = getattr(state, "trial_model", None) or state.trained_estimator
        features = model.autofe.transform(model._flaml_preprocessor.transform(X))
        expected.append((model.predict(features), np.asarray(features)))
        return original(self, automl, state, estimator, *args, **kwargs)

    monkeypatch.setattr(MLflowIntegration, "record_state", record)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        model_history=True,
        log_type="all",
        retrain_full=False,
        eval_method="holdout",
        featurization="auto",
        metric=lambda X, y, estimator, *args, **kwargs: (0.1 if estimator.params["n_estimators"] == 4 else 0.9, {}),
        starting_points={
            "rf": [
                {"n_estimators": 4, "fe.extraction": "PCA"},
                {"n_estimators": 8, "fe.extraction": "LDA"},
            ]
        },
    )
    with mlflow.start_run():
        automl.fit(X, y)
        run_ids = automl.mlflow_integration.manual_run_ids
        assert len(run_ids) == len(expected) == 2
        assert expected[0][1].shape[1] != expected[1][1].shape[1]
        for run_id, (predictions, features), trees in zip(run_ids, expected, (4, 8)):
            restored = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            np.testing.assert_array_equal(restored.predict(X), predictions)
            restored_features = restored.named_steps["autofe"].transform(
                restored.named_steps["feature_transformer"].transform(X)
            )
            np.testing.assert_allclose(restored_features, features)
            assert restored.steps[-1][1].estimator.n_estimators == trees
            requirements = tracking.download_artifacts(run_id, "model/requirements.txt")
            with open(requirements, encoding="utf-8") as file:
                names = {Requirement(line).name for line in file if line.strip()}
            assert "optuna" not in names
            assert "scikit-learn" in names


@pytest.mark.parametrize("flavor_only", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_serialized_remote_objective_restores_actual_worker_configuration(tracking, monkeypatch, flavor_only, fail):
    import cloudpickle
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml.fabric.mlflow import _mlflow_wrapper

    driver = mlflow.set_experiment("driver")
    if flavor_only:
        mlflow.sklearn.autolog(log_models=False, silent=True)
    else:
        mlflow.autolog(log_models=False, silent=True)

    def objective(config):
        import mlflow
        from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

        assert AUTOLOGGING_INTEGRATIONS["sklearn"]["log_models"] is False
        assert mlflow.tracking.fluent._active_experiment_id == driver.experiment_id
        if fail:
            raise ValueError("expected objective failure")
        return {"loss": 1.0}

    wrapped = cloudpickle.loads(cloudpickle.dumps(_mlflow_wrapper(objective, driver.experiment_id, autolog=True)))
    worker = mlflow.set_experiment("worker")
    mlflow.sklearn.autolog(log_models=True, silent=False)
    before = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
    experiment_environment = os.environ.get("MLFLOW_EXPERIMENT_ID")
    monkeypatch.setattr("socket.gethostname", lambda: "different-worker")
    if fail:
        with pytest.raises(ValueError, match="expected objective failure"):
            wrapped({})
    else:
        assert wrapped({}) == {"loss": 1.0}
    assert AUTOLOGGING_INTEGRATIONS == before
    assert mlflow.tracking.fluent._active_experiment_id == worker.experiment_id
    assert os.environ.get("MLFLOW_EXPERIMENT_ID") == experiment_environment
    assert mlflow.active_run() is None


@pytest.mark.parametrize("global_autolog", [False, True])
def test_remote_autolog_preserves_disabled_unavailable_flavor(tracking, global_autolog):
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
    from sklearn.linear_model import LinearRegression

    from flaml.fabric.mlflow import _apply_autolog_configurations

    disabled = {"disable": True, "log_models": False}
    AUTOLOGGING_INTEGRATIONS["unavailable_for_test"] = disabled.copy()
    configurations = {"unavailable_for_test": disabled.copy()}
    if global_autolog:
        configurations["mlflow"] = {"log_models": False, "silent": True}
    with mlflow.start_run() as run:
        _apply_autolog_configurations(configurations)
        LinearRegression().fit([[0], [1], [2]], [0, 2, 4])
    assert ("fit_intercept" in tracking.get_run(run.info.run_id).data.params) is global_autolog
    assert AUTOLOGGING_INTEGRATIONS["unavailable_for_test"] == disabled
    assert configurations["unavailable_for_test"] == disabled


def test_global_autolog_partial_configuration_is_not_retried_as_explicit(tracking, monkeypatch):
    from importlib import import_module

    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, autologging_integration

    from flaml.fabric.mlflow import _apply_autolog_configurations

    @autologging_integration("sklearn")
    def unavailable_autolog(log_models=True, disable=False, silent=False):
        raise ImportError("optional training backend is unavailable")

    monkeypatch.setattr(import_module("mlflow.sklearn"), "autolog", unavailable_autolog)
    mlflow.autolog(log_models=False, silent=True)
    configurations = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
    assert configurations["sklearn"]["disable"] is False
    assert not configurations["sklearn"].get("globally_configured", False)
    _apply_autolog_configurations(configurations)
    assert AUTOLOGGING_INTEGRATIONS["sklearn"] == configurations["sklearn"]


def test_summary_copy_preserves_parent_trial_identity(tracking):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run(tags={"flaml.owner_id": "outer", "flaml.trial_id": "outer-trial"}) as parent:
        integration = MLflowIntegration("tune")
        with mlflow.start_run(nested=True, tags={"flaml.owner_id": "inner", "flaml.trial_id": "inner-trial"}) as child:
            mlflow.log_metric("score", 0.5)
        integration.copy_mlflow_run(child.info.run_id, parent.info.run_id)
        stored = tracking.get_run(parent.info.run_id)
        assert stored.data.tags["flaml.owner_id"] == "outer"
        assert stored.data.tags["flaml.trial_id"] == "outer-trial"
        assert stored.data.metrics["score"] == 0.5


@pytest.mark.parametrize("drain_fails", [False, True])
def test_failed_fit_drains_artifact_producers_before_restoring_autolog(drain_fails):
    import importlib

    from flaml.automl.automl import _restore_mlflow_after_fit

    order = []
    integration = SimpleNamespace(resume_mlflow=lambda: order.append("restore"))

    def drain():
        order.append("drain")
        if drain_fails:
            raise RuntimeError("artifact drain failed")

    instance = SimpleNamespace(mlflow_integration=integration, wait_futures=drain)

    def fail(self):
        raise ValueError("original training error")

    with patch.object(importlib.import_module("flaml.automl.automl").logger, "exception") as logged, pytest.raises(
        ValueError, match="original training error"
    ):
        _restore_mlflow_after_fit(fail)(instance)
    assert order == ["drain", "restore"]
    if drain_fails:
        logged.assert_called_once()
        assert "draining" in logged.call_args.args[0]
    else:
        logged.assert_not_called()


def test_parallel_tune_adoption_joins_trial_identity_not_start_order(monkeypatch):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune", only_history=True)
    integration.autolog = True
    integration.experiment_id = "experiment"
    integration.best_iteration = 0
    integration.mlflow_client = Mock()
    for index in range(2):
        integration.record_trial(
            {"loss": index + 0.1}, SimpleNamespace(config={"x": index}, trial_id=str(index)), "loss"
        )
    runs = [
        SimpleNamespace(
            info=SimpleNamespace(
                run_id=f"run-{index}",
                run_name=f"trial-{index}",
                start_time=(integration.start_time + index + 1) * 1000,
                experiment_id="experiment",
                lifecycle_stage="active",
                status="FINISHED",
            ),
            data=SimpleNamespace(
                params={"x": str(index)}, metrics={}, tags={**integration.extra_tag, "flaml.trial_id": str(index)}
            ),
        )
        for index in range(2)
    ]
    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: runs)
    integration.mlflow_client.get_run.side_effect = lambda run_id: next(
        run for run in runs if run.info.run_id == run_id
    )
    written = {}
    monkeypatch.setattr(
        integration,
        "_log_info_to_run",
        lambda info, run_id, **kwargs: written.update({run_id: info["metrics"]["loss"]}),
    )
    result = SimpleNamespace()
    integration.adopt_children(result)
    assert written == {"run-0": 0.1, "run-1": 1.1}
    assert result.best_run_id == "run-0"


@pytest.mark.parametrize("objective_fails", [False, True])
def test_remote_restore_failure_does_not_replace_an_objective_error(tracking, monkeypatch, objective_fails):
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    import flaml.fabric.mlflow as module

    def objective(config):
        if objective_fails:
            raise ValueError("original objective error")
        return 1.0

    wrapped = module._mlflow_wrapper(objective, None)
    before = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
    monkeypatch.setattr("socket.gethostname", lambda: "different-worker")
    apply = Mock(side_effect=[None, RuntimeError("restore failed")])
    monkeypatch.setattr(module, "_apply_autolog_configurations", apply)
    expected = ValueError if objective_fails else RuntimeError
    with pytest.raises(expected, match="original objective error" if objective_fails else "restore failed"):
        wrapped({})
    assert apply.call_count == 2
    assert AUTOLOGGING_INTEGRATIONS == before


def test_new_winning_learner_does_not_clean_retained_model_history():
    from flaml.automl.model import ExtraTreesEstimator

    X = np.arange(120).reshape(60, 2)
    y = np.tile([0, 1], 30)

    def metric(X, y, estimator, *args, **kwargs):
        return (0.1 if isinstance(estimator, ExtraTreesEstimator) else 0.9), {}

    automl = AutoML(
        estimator_list=["rf", "extra_tree"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        metric=metric,
        eval_method="holdout",
        retrain_full=False,
        model_history=True,
        mlflow_logging=False,
        learner_selector="roundrobin",
    )
    automl.fit(X, y)
    assert automl.best_estimator == "extra_tree"
    for learner in ("rf", "extra_tree"):
        model = automl.best_model_for_estimator(learner)
        assert model is not None
        assert len(model.predict(X)) == len(y)


@pytest.mark.parametrize("clock_offset_ms", [0, -60000])
def test_owned_trial_adoption_uses_identity_not_worker_clock(tracking, clock_offset_ms):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run() as parent:
        integration = MLflowIntegration("tune")
        integration.autolog, integration.manual_log = True, False
        trial = SimpleNamespace(trial_id="trial", config={"x": 1})
        integration.record_trial({"loss": 0.1}, trial, "loss")
        integration.set_best_iter(SimpleNamespace(best_iteration=0, best_trial=trial))
        child = tracking.create_run(
            integration.experiment_id,
            start_time=int(integration.start_time * 1000) + clock_offset_ms,
            tags={**integration.extra_tag, "flaml.trial_id": trial.trial_id},
        )
        result = SimpleNamespace()
        integration.adopt_children(result)
        assert result.best_run_id == child.info.run_id
        stored = tracking.get_run(child.info.run_id)
        assert stored.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert stored.data.metrics["loss"] == 0.1


def test_adoption_retains_unrecorded_owned_empty_runs(tracking):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog, integration.manual_log = True, False
    integration.record_trial({"loss": 0.1}, SimpleNamespace(trial_id="recorded", config={}), "loss")
    created = {}
    for name, tags in {
        "recorded": {**integration.extra_tag, "flaml.trial_id": "recorded"},
        "stopped": {**integration.extra_tag, "flaml.trial_id": "stopped"},
        "failed": {**integration.extra_tag, "flaml.trial_id": "failed"},
        "artifact": {**integration.extra_tag, "flaml.trial_id": "artifact"},
        "annotated": {**integration.extra_tag, "flaml.trial_id": "annotated"},
        "retagged": {**integration.extra_tag, "flaml.trial_id": "retagged"},
        "unrelated": {"flaml.owner_id": "other", "flaml.trial_id": "stopped"},
    }.items():
        tags = {**tags, "mlflow.runName": f"initial-{name}"}
        created[name] = tracking.create_run(
            integration.experiment_id, start_time=int((integration.start_time + 1) * 1000), tags=tags
        ).info.run_id
        tracking.set_terminated(created[name], "FINISHED")
    tracking.set_terminated(created["failed"], "FAILED")
    tracking.log_text(created["artifact"], "retain this diagnostic", "trace.txt")
    tracking.set_tag(created["annotated"], "mlflow.note.content", "retain this explanation")
    tracking.set_tag(created["retagged"], "extra_tag.sid", "caller-authored value")
    integration.adopt_children()
    assert tracking.get_run(created["recorded"]).info.lifecycle_stage == "active"
    assert tracking.get_run(created["stopped"]).info.lifecycle_stage == "active"
    assert tracking.get_run(created["unrelated"]).info.lifecycle_stage == "active"
    for name in ("failed", "artifact", "annotated", "retagged"):
        assert tracking.get_run(created[name]).info.lifecycle_stage == "active"
    assert tracking.list_artifacts(created["artifact"])[0].path == "trace.txt"


def test_adoption_does_not_delete_an_unreported_owned_attempt(tracking, monkeypatch):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog, integration.manual_log = True, False
    run_ids = []

    def objective(config):
        run_ids.append(mlflow.active_run().info.run_id)

    integration.wrap_evaluation_function(objective)({}, _flaml_trial_id="unreported")
    delete = Mock(wraps=integration.mlflow_client.delete_run)
    monkeypatch.setattr(integration.mlflow_client, "delete_run", delete)
    integration.adopt_children()
    delete.assert_not_called()
    stored = tracking.get_run(run_ids[0])
    assert stored.info.lifecycle_stage == "active"
    assert stored.info.status == "FINISHED"


@pytest.mark.parametrize("autolog", [False, True])
@pytest.mark.parametrize("delivery", ["report", "return", "both", "caught", "caught_empty"])
def test_local_scheduler_stop_preserves_results_and_tracking(tracking, autolog, delivery):
    from flaml import tune
    from flaml.tune.scheduler import TrialScheduler

    class StopFirst(TrialScheduler):
        def set_search_properties(self, **kwargs):
            return True

        def on_trial_complete(self, *args):
            pass

        def on_trial_result(self, runner, trial, result):
            return self.STOP if trial.config["x"] == 1 else self.CONTINUE

    def objective(config):
        if delivery != "return":
            try:
                tune.report(loss=float(config["x"]))
            except StopIteration:
                if delivery == "caught_empty":
                    return {}
                if delivery != "caught":
                    raise
        if delivery != "report":
            return {"loss": float(config["x"]) + 10}

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(
            objective,
            config={"x": tune.choice([1, 2])},
            points_to_evaluate=[{"x": 1}, {"x": 2}],
            metric="loss",
            mode="min",
            num_samples=2,
            scheduler=StopFirst(),
            verbose=0,
        )
        expected = [11.0, 12.0] if delivery == "return" else [1.0, 2.0 if delivery == "report" else 12.0]
        assert [trial.last_result["loss"] for trial in result.trials] == expected
        assert result.trials[0].status == ("ERROR" if delivery == "caught_empty" else "TERMINATED")
        assert result.trials[1].status == "TERMINATED"
        assert result.best_config["x"] == (2 if delivery == "caught_empty" else 1)
        winner = tracking.get_run(result.best_run_id)
        assert winner.info.status == "FINISHED"
        assert winner.data.metrics == {"loss": result.best_result["loss"]}
        assert mlflow.active_run().info.run_id == parent.info.run_id
        if autolog:
            runs = tracking.search_runs(
                [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
            )
            first = next(run for run in runs if run.data.tags["flaml.trial_id"] == result.trials[0].trial_id)
            assert first.info.status == ("FAILED" if delivery == "caught_empty" else "FINISHED")


@pytest.mark.parametrize("autolog", [False, True])
@pytest.mark.parametrize("user_lagrange", [False, True])
def test_reported_metrics_are_captured_before_searcher_mutation(tracking, autolog, user_lagrange):
    from flaml import tune

    metrics = {"loss": 2.0, "memory": 2.0}
    if user_lagrange:
        metrics["loss_lagrange"] = 17.0

    def objective(config):
        tune.report(**metrics)

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run():
        result = tune.run(
            objective,
            config={"x": 1},
            metric="loss",
            mode="min",
            metric_constraints=[("memory", "<=", 1.0)],
            num_samples=1,
            verbose=0,
        )
        assert tracking.get_run(result.best_run_id).data.metrics == metrics
        assert {key: result.best_result[key] for key in metrics} == metrics
        if not user_lagrange:
            assert "loss_lagrange" not in result.best_result
        assert result.best_trial._last_callback_result["loss_lagrange"] > metrics["loss"]


@pytest.mark.parametrize("mutation", ["caller", "callback"])
def test_nested_report_metrics_are_snapshotted_before_mutation(tracking, monkeypatch, mutation):
    from flaml import tune
    from flaml.tune.searcher.blendsearch import CFO

    diagnostics = {"score": 2.0}
    searcher = CFO(space={"x": 1}, metric="loss", mode="min")
    callback = searcher.on_trial_result

    def on_result(trial_id, result):
        callback(trial_id, result)
        if mutation == "callback":
            result["diagnostics"]["score"] = 77.0

    monkeypatch.setattr(searcher, "on_trial_result", on_result)

    def objective(config):
        tune.report(loss=1.0, diagnostics=diagnostics)
        if mutation == "caller":
            diagnostics["score"] = 99.0

    with mlflow.start_run():
        result = tune.run(objective, search_alg=searcher, num_samples=1, verbose=0)
        assert result.best_result["diagnostics"] == {"score": 2.0}
        assert result.best_trial.metric_analysis["diagnostics/score"]["last"] == 2.0
        assert tracking.get_run(result.best_run_id).data.metrics["diagnostics.score"] == 2.0


def test_report_snapshot_does_not_clone_opaque_artifact_payloads():
    from flaml import tune

    class Artifact:
        def __deepcopy__(self, memo):
            raise AssertionError("An opaque payload must not be deep-copied.")

    artifact = Artifact()
    result = tune.run(
        lambda config: {"loss": 1.0, "artifacts": [artifact]},
        config={"x": 1},
        metric="loss",
        mode="min",
        num_samples=1,
        mlflow_logging=False,
        verbose=0,
    )
    assert result.best_result["artifacts"][0] is artifact
    assert result.best_trial._last_callback_result["artifacts"][0] is artifact


@pytest.mark.parametrize("autolog", [False, True])
def test_generated_timing_objective_is_selectable_and_logged(tracking, autolog):
    from flaml import tune

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run():
        result = tune.run(
            lambda config: {"auxiliary": 1.0},
            config={"x": 1},
            metric="time_total_s",
            mode="min",
            num_samples=1,
            verbose=0,
        )
        assert result.best_trial is result.trials[0]
        elapsed = result.best_result["time_total_s"]
        assert np.isfinite(elapsed) and elapsed >= 0
        assert tracking.get_run(result.best_run_id).data.metrics["time_total_s"] == elapsed


def test_successful_autolog_trial_inside_a_handled_exception_finishes(tracking):
    from flaml import tune

    mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run():
        try:
            raise ValueError("handled caller error")
        except ValueError:
            result = tune.run(
                lambda config: {"loss": 1.0}, config={"x": 1}, metric="loss", mode="min", num_samples=1, verbose=0
            )
            assert result.best_run_id is not None
            assert tracking.get_run(result.best_run_id).info.status == "FINISHED"


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/mlruns",
        "mlruns",
        r"C:\mlruns",
        r"\\server\share\mlruns",
        "sqlite:///:memory:",
        "sqlite://",
        "sqlite+pysqlite:///:memory:",
        "sqlite:///file::memory:?cache=shared&uri=true",
        "sqlite:///file:%3Amemory%3A?uri=true",
        "sqlite:///file:flaml?mode=memory&uri=true",
        "sqlite:///file:?uri=true",
        "sqlite:///file://?uri=true",
        "sqlite:///file://localhost?uri=true",
        "sqlite:///file:/flaml?vfs=memdb&uri=true",
    ],
)
def test_native_ray_rejects_nontransactional_tracking_paths(uri):
    from flaml.fabric.mlflow import _validate_ray_tracking_uri

    with pytest.raises(ValueError, match="transactional tracking"):
        _validate_ray_tracking_uri(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:///tracking.db",
        "sqlite:///file:tracking.db?mode=rwc&uri=true",
        "sqlite:///%3Amemory%3A",
        "https://tracking.example",
        "databricks",
        "databricks://profile",
    ],
)
def test_native_ray_accepts_transactional_or_service_tracking(uri):
    from flaml.fabric.mlflow import _validate_ray_tracking_uri

    _validate_ray_tracking_uri(uri)


@pytest.mark.parametrize(
    "uri", ["sqlite:///tracking.db", "sqlite:///file:tracking%20%23.db?mode=rwc&uri=true", "absolute-file"]
)
def test_sqlite_worker_tracking_uses_the_driver_database(tracking, tmp_path, monkeypatch, uri):
    import json
    import subprocess
    import sys
    from pathlib import Path

    import cloudpickle

    from flaml.fabric.mlflow import _mlflow_wrapper, _worker_tracking_uri

    driver, worker = tmp_path / "driver", tmp_path / "worker"
    driver.mkdir()
    worker.mkdir()
    monkeypatch.chdir(driver)
    if uri == "absolute-file":
        uri = f"sqlite:///{(driver / 'tracking.db').as_uri()}?mode=rwc&uri=true"
    resolved = _worker_tracking_uri(uri)
    assert _worker_tracking_uri(resolved) == resolved
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment = client.create_experiment("driver-store", artifact_location=(driver / "artifacts").as_uri())

    def objective(config):
        mlflow.log_metric("loss", 2.0)
        return mlflow.active_run().info.run_id

    with mlflow.start_run(experiment_id=experiment) as parent:
        wrapped = _mlflow_wrapper(objective, experiment, autolog=True)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import cloudpickle,json,sys; "
                "objective=cloudpickle.loads(sys.stdin.buffer.read()); "
                "print(json.dumps({'run_id':objective({},_flaml_trial_id='worker')}))",
            ],
            input=cloudpickle.dumps(wrapped),
            cwd=worker,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
            capture_output=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stderr.decode(errors="replace")
        run_id = json.loads(process.stdout.decode().splitlines()[-1])["run_id"]
        run = client.get_run(run_id)
        assert run.info.experiment_id == experiment
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert run.data.metrics == {"loss": 2.0}
        assert mlflow.get_tracking_uri() == uri
        assert mlflow.active_run().info.run_id == parent.info.run_id


def test_file_store_ray_logging_is_rejected_before_integration_or_evaluation(tracking, monkeypatch):
    import flaml.tune.tune as tuning

    integration, objective = Mock(), Mock()
    monkeypatch.setattr(tuning, "MLflowIntegration", integration)
    with mlflow.start_run() as parent:
        with pytest.raises(ValueError, match="transactional tracking"):
            tuning.run(objective, metric="loss", mode="min", use_ray=True, num_samples=1, verbose=0)
        assert mlflow.active_run().info.run_id == parent.info.run_id
    integration.assert_not_called()
    objective.assert_not_called()


@pytest.mark.parametrize("outcome", ["ERROR", "TERMINATED"])
@pytest.mark.parametrize("existing_status", ["RUNNING", "FAILED", "FINISHED"])
def test_native_driver_finishes_only_unfinished_owned_attempts(tracking, outcome, existing_status):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog = True
    run = tracking.create_run(integration.experiment_id, tags={**integration.extra_tag, "flaml.trial_id": "trial"})
    if existing_status != "RUNNING":
        tracking.set_terminated(run.info.run_id, existing_status)
    trial = SimpleNamespace(trial_id="trial", status=outcome)
    integration._finalize_native_trial(trial, run.info.run_id)
    expected = ("FAILED" if outcome == "ERROR" else "FINISHED") if existing_status == "RUNNING" else existing_status
    assert tracking.get_run(run.info.run_id).info.status == expected


@pytest.mark.parametrize("field", ["flaml.owner_id", "flaml.trial_id", "experiment", "deleted"])
def test_native_driver_does_not_finalize_an_unrelated_attempt(tracking, field):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog = True
    tags = {**integration.extra_tag, "flaml.trial_id": "trial"}
    if field.startswith("flaml."):
        tags[field] = "unrelated"
    experiment = tracking.create_experiment("unrelated") if field == "experiment" else integration.experiment_id
    run = tracking.create_run(experiment, tags=tags)
    if field == "deleted":
        tracking.delete_run(run.info.run_id)
    integration._finalize_native_trial(SimpleNamespace(trial_id="trial", status="TERMINATED"), run.info.run_id)
    assert tracking.get_run(run.info.run_id).info.status == "RUNNING"


@pytest.mark.parametrize("status", ["ERROR", "TERMINATED"])
@pytest.mark.parametrize("missing", [False, True])
def test_native_finalization_skips_invalid_or_unavailable_run_ids(tracking, monkeypatch, status, missing):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog = True
    lookup = Mock(wraps=integration.mlflow_client.get_run)
    monkeypatch.setattr(integration.mlflow_client, "get_run", lookup)
    integration._finalize_native_trial(SimpleNamespace(trial_id="trial", status=status), "f" * 32 if missing else 123)
    if not missing:
        lookup.assert_not_called()
    assert integration.infos == []


@pytest.mark.parametrize("accepted", ["unique", "ambiguous", "explicit", "mixed-terminal"])
def test_native_adoption_finalizes_only_the_unambiguous_accepted_attempt(tracking, accepted):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog, integration.manual_log, integration._native_ray = True, False, True
    run_ids = [
        tracking.create_run(
            integration.experiment_id, tags={**integration.extra_tag, "flaml.trial_id": "trial"}
        ).info.run_id
        for _ in range(1 if accepted == "unique" else 2)
    ]
    if accepted == "mixed-terminal":
        tracking.set_terminated(run_ids[-1], "FAILED")
    initial_statuses = [tracking.get_run(run_id).info.status for run_id in run_ids]
    trial = SimpleNamespace(trial_id="trial", config={"x": 1}, status="TERMINATED", last_result={"loss": 1.0})
    integration.record_trial(trial.last_result, trial, "loss", run_id=run_ids[-1] if accepted == "explicit" else None)
    analysis = SimpleNamespace(
        trials=[trial],
        best_iteration=0,
        best_trial=trial,
        best_config=trial.config,
        best_result=trial.last_result,
        best_run_id=None,
        best_run_name=None,
    )
    integration.log_tune(analysis, "loss")
    if accepted in ("ambiguous", "mixed-terminal"):
        assert analysis.best_run_id is None
        assert [tracking.get_run(run_id).info.status for run_id in run_ids] == initial_statuses
        assert all(not tracking.get_run(run_id).data.metrics for run_id in run_ids)
    else:
        assert analysis.best_run_id == run_ids[-1]
        assert tracking.get_run(run_ids[-1]).info.status == "FINISHED"
        assert all(tracking.get_run(run_id).info.status == "RUNNING" for run_id in run_ids[:-1])


@pytest.mark.parametrize("field", ["flaml.owner_id", "flaml.trial_id"])
def test_native_adoption_skips_all_writes_after_a_fresh_ownership_mismatch(tracking, monkeypatch, field):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run():
        integration = MLflowIntegration("tune")
        integration.autolog, integration.manual_log, integration._native_ray = True, False, True
        run_id = tracking.create_run(
            integration.experiment_id, tags={**integration.extra_tag, "flaml.trial_id": "trial"}
        ).info.run_id
        trial = SimpleNamespace(trial_id="trial", config={"x": 1}, status="TERMINATED")
        integration.record_trial({"loss": 1.0}, trial, "loss", run_id=run_id)
        integration.set_best_iter(SimpleNamespace(best_iteration=0, best_trial=trial))
        lookup = integration._runs_for_adoption

        def change_after_lookup(owner):
            runs = lookup(owner)
            tracking.set_tag(run_id, field, "another-owner-or-trial")
            return runs

        monkeypatch.setattr(integration, "_runs_for_adoption", change_after_lookup)
        result = SimpleNamespace(best_run_id=None)
        integration.adopt_children(result)
        stored = tracking.get_run(run_id)
        assert not stored.data.metrics and not stored.data.params
        assert "mlflow.parentRunId" not in stored.data.tags
        assert "flaml.child_counter" not in stored.data.tags
        assert "flaml.best_run" not in stored.data.tags
        assert stored.info.status == "RUNNING"
        assert result.best_run_id is None and integration.best_run_id is None


@pytest.mark.parametrize("reported", [False, True])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_native_failed_trials_reconcile_without_successful_metric_records(tracking, reported, ambiguous):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog, integration.manual_log, integration._native_ray = True, False, True
    failed_runs = [
        tracking.create_run(
            integration.experiment_id, tags={**integration.extra_tag, "flaml.trial_id": "failed"}
        ).info.run_id
        for _ in range(2 if ambiguous else 1)
    ]
    failed = SimpleNamespace(
        trial_id="failed", config={}, status="ERROR", last_result={"loss": -10.0} if reported else {}
    )
    integration._finalize_native_trial(failed, None)
    integration.record_trial(failed.last_result, failed, "loss")
    winner = SimpleNamespace(trial_id="winner", config={}, status="TERMINATED", last_result={"loss": 1.0})
    winner_run = tracking.create_run(
        integration.experiment_id, tags={**integration.extra_tag, "flaml.trial_id": "winner"}
    ).info.run_id
    integration.record_trial(winner.last_result, winner, "loss", run_id=winner_run)
    analysis = SimpleNamespace(
        trials=[failed, winner],
        best_trial=winner,
        best_iteration=1,
        best_config=winner.config,
        best_result=winner.last_result,
        best_run_id=None,
        best_run_name=None,
    )
    integration.log_tune(analysis, "loss")
    assert analysis.best_run_id == winner_run
    assert tracking.get_run(winner_run).data.metrics == {"loss": 1.0}
    for run_id in failed_runs:
        stored = tracking.get_run(run_id)
        assert stored.info.status == ("RUNNING" if ambiguous else "FAILED")
        assert not stored.data.metrics and "flaml.child_counter" not in stored.data.tags


@pytest.mark.parametrize("operation", ["lookup", "finalize"])
@pytest.mark.parametrize("missing", [False, True])
def test_native_run_lookup_preserves_nonmissing_store_errors(tracking, monkeypatch, operation, missing):
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_DOES_NOT_EXIST

    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog = True
    run_id = "f" * 32
    integration.infos = [{"trial_id": "trial", "run_id": run_id}]
    error = MlflowException("lookup failed", error_code=RESOURCE_DOES_NOT_EXIST if missing else PERMISSION_DENIED)
    monkeypatch.setattr(integration.mlflow_client, "get_run", Mock(side_effect=error))

    def lookup():
        if operation == "lookup":
            return integration._runs_for_adoption(integration.extra_tag["flaml.owner_id"])
        return integration._finalize_native_trial(SimpleNamespace(trial_id="trial", status="TERMINATED"), run_id)

    if missing:
        assert not lookup()
    else:
        with pytest.raises(MlflowException) as caught:
            lookup()
        assert caught.value is error


@pytest.mark.parametrize("field", ["flaml.owner_id", "flaml.trial_id", "experiment", "deleted", "run_id"])
def test_direct_adoption_lookup_never_mutates_a_mismatched_run(tracking, monkeypatch, field):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    integration.autolog, integration.manual_log = True, False
    tags = {**integration.extra_tag, "flaml.trial_id": "trial", "diagnostic": "retain"}
    if field.startswith("flaml."):
        tags[field] = "unrelated"
    experiment = tracking.create_experiment("unrelated") if field == "experiment" else integration.experiment_id
    run = tracking.create_run(experiment, tags=tags)
    if field == "deleted":
        tracking.delete_run(run.info.run_id)
    expected_id = "f" * 32 if field == "run_id" else run.info.run_id
    integration.record_trial(
        {"loss": 1.0}, SimpleNamespace(trial_id="trial", config={}, status="TERMINATED"), "loss", run_id=expected_id
    )
    if field == "run_id":
        monkeypatch.setattr(integration.mlflow_client, "get_run", Mock(return_value=tracking.get_run(run.info.run_id)))
    before = tracking.get_run(run.info.run_id)
    integration.adopt_children()
    after = tracking.get_run(run.info.run_id)
    assert after.data.tags == before.data.tags
    assert after.data.metrics == before.data.metrics
    assert after.info.status == before.info.status


@pytest.mark.parametrize("token", [123, 0, [], "repeat", b"repeat"])
def test_owned_run_pagination_rejects_invalid_or_repeated_tokens(tracking, monkeypatch, token):
    import flaml.fabric.mlflow as module

    integration = module.MLflowIntegration("tune")
    monkeypatch.setattr(module, "SEARCH_MAX_RESULTS", 1)
    monkeypatch.setattr(mlflow, "search_runs", Mock(return_value=[object()]))

    class Page(list):
        pass

    page = Page([object()])
    page.token = token
    lookup = Mock(return_value=page)
    monkeypatch.setattr(integration.mlflow_client, "search_runs", lookup)
    with pytest.raises(RuntimeError, match="invalid or repeated page token"):
        integration._search_owned_runs(integration.extra_tag["flaml.owner_id"])
    assert lookup.call_count == (2 if isinstance(token, (str, bytes)) else 1)


@pytest.mark.parametrize("backend", ["standalone", "spark", "ray"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_trial_exit_status_requires_an_explicit_ray_context(tracking, monkeypatch, backend, exit_code):
    import flaml.tune.tune as tuning
    from flaml.fabric.mlflow import _mlflow_wrapper

    created = []

    def objective(config):
        created.append(mlflow.active_run().info.run_id)
        raise SystemExit(exit_code)

    monkeypatch.setattr(tuning, "_use_ray", True)
    wrapped = _mlflow_wrapper(objective, "0", autolog=True)
    with pytest.raises(SystemExit) as caught:
        if backend == "spark":
            tuning._evaluate_spark_trial(wrapped, {}, "trial", {"_flaml_trial_id": "trial"})
        elif backend == "ray":
            with tuning._capture_trial_run("trial") as context:
                context["backend"] = "ray"
                wrapped({}, _flaml_trial_id="trial")
        else:
            wrapped({})
    assert caught.value.code == exit_code
    assert tracking.get_run(created[0]).info.status == ("FINISHED" if backend == "ray" and exit_code == 0 else "FAILED")


def test_unexpected_local_stop_iteration_remains_a_failed_objective(tracking):
    from flaml import tune

    def objective(config):
        raise StopIteration("unexpected user stop")

    mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        with pytest.raises(StopIteration, match="unexpected user stop"):
            tune.run(
                objective,
                config={"x": 1},
                metric="loss",
                mode="min",
                num_samples=1,
                verbose=0,
            )
        children = tracking.search_runs(["0"], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'")
        assert len(children) == 1 and children[0].info.status == "FAILED"
        assert mlflow.active_run().info.run_id == parent.info.run_id


def test_generator_exit_before_a_first_yield_is_not_a_success(tracking):
    from flaml.fabric.mlflow import _mlflow_wrapper

    created = []

    def objective(config):
        created.append(mlflow.active_run().info.run_id)
        raise GeneratorExit("no accepted report")
        yield {"loss": 1.0}

    with pytest.raises(GeneratorExit):
        list(_mlflow_wrapper(objective, "0", autolog=True)({}))
    assert tracking.get_run(created[0]).info.status == "FAILED"


@pytest.mark.parametrize("callable_object", [False, True])
def test_generator_returning_callables_fail_inside_the_owned_context(tracking, callable_object):
    from flaml.fabric.mlflow import _mlflow_wrapper

    def generator(config):
        yield {"loss": 1.0}

    class CallableGenerator:
        def __call__(self, config):
            yield {"loss": 1.0}

    def forward(config):
        return generator(config)

    with mlflow.start_run() as parent:
        wrapped = _mlflow_wrapper(CallableGenerator() if callable_object else forward, "0", autolog=True)
        with pytest.raises(TypeError, match="generator function"):
            wrapped({})
        children = tracking.search_runs(["0"], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'")
        assert len(children) == 1 and children[0].info.status == "FAILED"


def test_manual_tune_winner_uses_trial_identity_after_resultless_trial(tracking):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run():
        integration = MLflowIntegration("tune")
        winner = SimpleNamespace(trial_id="winner", config={"x": 1})
        integration.record_trial({"loss": 0.1}, winner, "loss")
        result = SimpleNamespace(
            best_iteration=1, best_trial=winner, best_config=winner.config, best_result={"loss": 0.1}
        )
        integration.log_tune(result, "loss")
        assert result.best_run_id == integration.manual_run_ids[0]


@pytest.mark.parametrize("autolog", [False, True])
def test_tune_report_results_are_logged_without_returning_a_value(tracking, autolog):
    from flaml import tune

    def objective(config):
        tune.report(loss=config["x"], converged=True, note="not a metric")

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(
            objective,
            config={"x": tune.choice([1, 2]), "enabled": True},
            metric="loss",
            mode="min",
            num_samples=2,
            use_ray=False,
            verbose=0,
        )
        assert len(result.trials) == 2
        assert result.best_run_id
        best = tracking.get_run(result.best_run_id)
        assert best.data.tags["flaml.trial_id"] == result.best_trial.trial_id
        assert best.data.metrics == {"loss": result.best_result["loss"]}
        assert best.data.params["x"] == str(result.best_config["x"])
        assert best.data.params["enabled"] == "True"
        assert mlflow.active_run().info.run_id == parent.info.run_id


def test_tune_adopts_successful_metrics_once(tracking):
    from flaml import tune

    mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run():
        result = tune.run(
            lambda config: {"loss": config["x"]},
            config={"x": tune.choice([1, 2])},
            metric="loss",
            mode="min",
            num_samples=2,
            use_ray=False,
            verbose=0,
        )
        assert result.best_run_id
        assert len(tracking.get_metric_history(result.best_run_id, "loss")) == 1


def test_tune_cleanup_failure_preserves_original_objective_error(tracking, monkeypatch):
    from flaml import tune
    from flaml.fabric.mlflow import MLflowIntegration

    def fail(config):
        raise ValueError("original objective failure")

    monkeypatch.setattr(MLflowIntegration, "adopt_children", Mock(side_effect=RuntimeError("tracking unavailable")))
    with mlflow.start_run(), pytest.raises(ValueError, match="original objective failure"):
        tune.run(fail, config={"x": 1}, num_samples=1, use_ray=False, verbose=0)


def test_remote_trial_lineage_does_not_use_unrelated_active_worker_run(tracking, monkeypatch):
    from flaml.fabric.mlflow import _mlflow_wrapper

    with mlflow.start_run() as driver:

        def objective(config):
            return mlflow.active_run().info.run_id

        wrapped = _mlflow_wrapper(objective, driver.info.experiment_id, autolog=True)
    with mlflow.start_run() as worker:
        monkeypatch.setattr("socket.gethostname", lambda: "remote-worker")
        child_id = wrapped({}, _flaml_trial_id="trial")
        assert mlflow.active_run().info.run_id == worker.info.run_id
        assert tracking.get_run(worker.info.run_id).info.status == "RUNNING"
    assert tracking.get_run(child_id).data.tags["mlflow.parentRunId"] == driver.info.run_id


def test_repeated_pipeline_registration_keeps_immutable_version_sources(tracking, explicit_requirements):
    from uuid import uuid4

    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    from flaml.fabric.mlflow import register_automl_pipeline

    X, y = np.arange(12).reshape(6, 2), np.array([0, 1] * 3)
    name = "registered-" + uuid4().hex
    versions = []
    with mlflow.start_run() as parent:
        for value in (0, 1):
            pipeline = Pipeline([("model", DummyClassifier(strategy="constant", constant=value))]).fit(X, y)
            versions.append(register_automl_pipeline(SimpleNamespace(automl_pipeline=pipeline), model_name=name))
            assert mlflow.active_run().info.run_id == parent.info.run_id
        for value, version in enumerate(versions):
            restored = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
            np.testing.assert_array_equal(restored.predict(X), np.full(len(y), value))
    assert versions[0].source != versions[1].source


def test_final_model_mismatch_is_not_marked_as_published(tracking):
    from flaml.fabric.mlflow import MLflowIntegration

    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    with mlflow.start_run():
        integration = MLflowIntegration("automl")
        with mlflow.start_run(nested=True), pytest.raises(RuntimeError, match="run"):
            integration.log_final_model(automl)
        assert not integration.has_model


def test_completed_model_futures_are_observed_once_and_released(tracking):
    import importlib
    from concurrent.futures import Future

    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("automl")
    future = Future()
    future.set_exception(ValueError("trial artifact failure"))
    integration.futures_log_model[future] = "artifact"
    with patch.object(importlib.import_module("flaml.fabric.mlflow").logger, "warning") as logged:
        integration.wait_model_futures()
        integration.wait_model_futures()
    assert not integration.futures_log_model
    logged.assert_called_once()
    assert str(logged.call_args.args[-1]) == "trial artifact failure"


@pytest.mark.parametrize("autolog", [False, True])
@pytest.mark.parametrize("context_api", ["tune", "train", "air", "legacy"])
@pytest.mark.parametrize("tracking", ["sqlite"], indirect=True)
def test_native_ray_analysis_retains_trial_association(tracking, monkeypatch, autolog, context_api):
    import importlib
    import inspect
    import sys
    from types import ModuleType

    from flaml import tune

    module = importlib.import_module("flaml.tune.tune")
    current_trial = None
    trials = [
        SimpleNamespace(trial_id="empty", config={"x": 0}, last_result=None),
        SimpleNamespace(trial_id="winner", config={"x": 1}, last_result=None),
    ]
    returned = SimpleNamespace(trials=trials, best_trial=trials[1], best_config={"x": 1}, best_result={"loss": 1.0})

    def ray_run(objective, **kwargs):
        nonlocal current_trial
        for trial in trials:
            current_trial = trial.trial_id
            arguments = (
                {"checkpoint_dir": "ray-checkpoint"}
                if "checkpoint_dir" in inspect.signature(objective).parameters
                else {}
            )
            result = objective(trial.config, **arguments)
            trial.last_result = (
                {
                    **result,
                    "done": True,
                    "config": trial.config,
                    "config/x": trial.config["x"],
                    "timestamp": 1234,
                    "warmup_time": 0.25,
                }
                if result is not None
                else None
            )
        return returned

    ray = ModuleType("ray")
    ray.tune = SimpleNamespace(run=ray_run)
    ray.train = SimpleNamespace()
    air = ModuleType("ray.air")
    air.session = SimpleNamespace()

    def get_context():
        return SimpleNamespace(get_trial_id=lambda: current_trial)

    if context_api == "tune":
        ray.tune.get_context = get_context
    elif context_api == "train":
        ray.train.get_context = get_context
    elif context_api == "air":
        air.session.get_trial_id = lambda: current_trial
    else:
        ray.tune.get_trial_id = lambda: current_trial
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.tune", ray.tune)
    monkeypatch.setitem(sys.modules, "ray.train", ray.train)
    monkeypatch.setitem(sys.modules, "ray.air", air)
    monkeypatch.setitem(sys.modules, "ray.air.session", air.session)
    if context_api in {"tune", "train", "legacy"}:
        monkeypatch.setitem(sys.modules, "ray.air", None)
        monkeypatch.setitem(sys.modules, "ray.air.session", None)
    monkeypatch.setattr(module, "ray_available", False)

    def objective(config, checkpoint_dir=None):
        assert checkpoint_dir == "ray-checkpoint"
        return {"loss": 1.0} if config["x"] else None

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(
            objective,
            config={"x": tune.choice([0, 1])},
            metric="loss",
            mode="min",
            use_ray=True,
            num_samples=2,
            verbose=0,
        )
        assert result is returned
        best = tracking.get_run(result.best_run_id)
        assert best.data.tags["flaml.trial_id"] == "winner"
        assert best.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert best.data.metrics == {"loss": 1.0}


@pytest.mark.parametrize("api", ["air", "tune", "train"])
@pytest.mark.parametrize("anonymous", [None, 1.25])
def test_ray_report_does_not_inject_empty_metrics_or_disappear(monkeypatch, api, anonymous):
    import importlib
    import sys
    from types import ModuleType

    from flaml.tune.result import DEFAULT_METRIC

    module = importlib.import_module("flaml.tune.tune")
    received = []

    def report(*, metrics):
        received.append(metrics)

    ray = ModuleType("ray")
    ray.__version__ = "2.4.0" if api == "air" else "2.49.2"
    ray.tune, ray.train = SimpleNamespace(), SimpleNamespace()
    air = ModuleType("ray.air")
    air.session = SimpleNamespace()
    if api == "air":
        air.session = SimpleNamespace(report=report)
    elif api == "tune":
        ray.tune.report = report
    else:
        ray.train.report = report
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.tune", ray.tune)
    monkeypatch.setitem(sys.modules, "ray.train", ray.train)
    monkeypatch.setitem(sys.modules, "ray.air", air)
    monkeypatch.setitem(sys.modules, "ray.air.session", air.session)
    if api != "air":
        monkeypatch.setitem(sys.modules, "ray.air", None)
        monkeypatch.setitem(sys.modules, "ray.air.session", None)
    monkeypatch.setattr(module, "_use_ray", True)
    module.report(anonymous, loss=0.5)
    expected = {"loss": 0.5}
    if anonymous is not None:
        expected[DEFAULT_METRIC] = anonymous
    assert received == [expected]


@pytest.mark.parametrize("objective_fails", [False, True])
def test_remote_trial_uses_driver_tracking_store_and_restores_worker_store(
    tracking, monkeypatch, tmp_path, objective_fails
):
    from flaml.fabric.mlflow import _mlflow_wrapper

    driver_uri = mlflow.get_tracking_uri()
    with mlflow.start_run() as driver:

        def objective(config):
            assert mlflow.get_tracking_uri() == driver_uri
            mlflow.log_metric("loss", 1.0)
            if objective_fails:
                raise ValueError("objective failed in the driver store")
            return mlflow.active_run().info.run_id

        wrapped = _mlflow_wrapper(objective, driver.info.experiment_id, autolog=True)
    worker_uri = (tmp_path / "worker-tracking").as_uri()
    mlflow.set_tracking_uri(worker_uri)
    worker_client = mlflow.tracking.MlflowClient()
    try:
        with mlflow.start_run(experiment_id="0") as worker:
            monkeypatch.setattr("socket.gethostname", lambda: "remote-worker")
            if objective_fails:
                with pytest.raises(ValueError, match="objective failed in the driver store"):
                    wrapped({}, _flaml_trial_id="remote")
            else:
                child_id = wrapped({}, _flaml_trial_id="remote")
                assert tracking.get_run(child_id).data.metrics["loss"] == 1.0
            assert mlflow.get_tracking_uri() == worker_uri
            assert mlflow.active_run().info.run_id == worker.info.run_id
            assert worker_client.get_run(worker.info.run_id).info.status == "RUNNING"
        children = tracking.search_runs(
            [driver.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{driver.info.run_id}'"
        )
        assert len(children) == 1
        assert children[0].info.status == ("FAILED" if objective_fails else "FINISHED")
        assert children[0].data.metrics["loss"] == 1.0
    finally:
        mlflow.set_tracking_uri(driver_uri)


@pytest.mark.parametrize("objective_fails", [False, True])
def test_objective_error_survives_fluent_run_termination_failure(tracking, monkeypatch, objective_fails):
    from flaml.fabric.mlflow import _mlflow_wrapper

    def objective(config):
        if objective_fails:
            raise ValueError("primary objective error")
        return 1.0

    end_run = mlflow.tracking.fluent.end_run

    def failing_end(*args, **kwargs):
        end_run(*args, **kwargs)
        raise RuntimeError("secondary termination error")

    with mlflow.start_run() as parent:
        wrapped = _mlflow_wrapper(objective, parent.info.experiment_id, autolog=True)
        with monkeypatch.context() as scoped:
            scoped.setattr(mlflow.tracking.fluent, "end_run", failing_end)
            error = ValueError if objective_fails else RuntimeError
            message = "primary objective error" if objective_fails else "secondary termination error"
            with pytest.raises(error, match=message):
                wrapped({})
        assert mlflow.active_run().info.run_id == parent.info.run_id


@pytest.mark.parametrize(
    "backend,identity_matches", [(None, True), ("spark", True), ("ray", True), ("local", False), ("local", True)]
)
def test_stopiteration_requires_the_matching_local_trial(tracking, monkeypatch, backend, identity_matches):
    from contextlib import nullcontext

    import flaml.tune.tune as tuning
    from flaml.fabric.mlflow import _mlflow_wrapper

    created = []
    trial_id = "current" if identity_matches else "different"
    monkeypatch.setattr(
        tuning, "_runner", SimpleNamespace(running_trial=SimpleNamespace(trial_id="current", status="TERMINATED"))
    )

    def objective(config):
        created.append(mlflow.active_run().info.run_id)
        raise StopIteration("objective stopped")

    wrapped = _mlflow_wrapper(objective, "0", autolog=True)
    context = tuning._capture_trial_run(trial_id, backend=backend) if backend is not None else nullcontext()
    with context, pytest.raises(StopIteration, match="objective stopped"):
        wrapped({}, _flaml_trial_id=trial_id)
    expected = "FINISHED" if backend == "local" and identity_matches else "FAILED"
    assert tracking.get_run(created[0]).info.status == expected


@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize("ended_status", [None, "FINISHED", "FAILED"])
def test_clean_stop_fallback_does_not_force_a_successful_owned_run_failed(tracking, monkeypatch, backend, ended_status):
    import flaml.tune.tune as tuning
    from flaml.fabric.mlflow import _mlflow_wrapper

    created = []
    monkeypatch.setattr(
        tuning, "_runner", SimpleNamespace(running_trial=SimpleNamespace(trial_id="trial", status="TERMINATED"))
    )

    def objective(config):
        created.append(mlflow.active_run().info.run_id)
        if ended_status:
            mlflow.end_run(status=ended_status)
            mlflow.start_run(nested=True)
        if backend == "ray":
            raise SystemExit(0)
        raise StopIteration("scheduler stop")

    wrapped = _mlflow_wrapper(objective, "0", autolog=True)
    monkeypatch.setattr(mlflow.tracking.fluent, "end_run", Mock(side_effect=RuntimeError("fluent termination failed")))
    with tuning._capture_trial_run("trial", backend=backend), pytest.raises(
        SystemExit if backend == "ray" else StopIteration
    ):
        wrapped({}, _flaml_trial_id="trial")
    assert tracking.get_run(created[0]).info.status == (ended_status or "FINISHED")


def test_failed_parent_summary_still_adopts_completed_trial_metrics(tracking, monkeypatch):
    from flaml import tune

    log_metric = mlflow.tracking.MlflowClient.log_metric

    def fail_summary(self, run_id, key, value, *args, **kwargs):
        if key == "num_child_runs":
            raise RuntimeError("parent summary unavailable")
        return log_metric(self, run_id, key, value, *args, **kwargs)

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_metric", fail_summary)
    mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        with pytest.raises(RuntimeError, match="parent summary unavailable"):
            tune.run(lambda config: {"loss": 1.0}, config={"x": 1}, metric="loss", num_samples=1, verbose=0)
        children = tracking.search_runs(
            [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
        )
        assert len(children) == 1
        assert children[0].data.metrics["loss"] == 1.0
        assert len(tracking.get_metric_history(children[0].info.run_id, "loss")) == 1
        assert mlflow.active_run().info.run_id == parent.info.run_id


@pytest.mark.parametrize("autolog", [False, True])
def test_resultless_tune_does_not_require_a_best_trial(tracking, autolog):
    from flaml import tune

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(lambda config: None, config={"x": 1}, metric="loss", num_samples=1, verbose=0)
        assert result.best_run_id is None
        assert mlflow.active_run().info.run_id == parent.info.run_id


def test_reported_then_errored_trial_is_not_recorded_as_an_empty_success(tracking):
    from flaml import tune

    def objective(config):
        tune.report(loss=0.5)
        return {}

    with mlflow.start_run() as parent:
        result = tune.run(objective, config={"x": 1}, metric="loss", num_samples=1, verbose=0)
        assert result.trials[0].status == "ERROR"
        children = tracking.search_runs(
            [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
        )
        assert not children
        assert result.best_run_id is None


@pytest.mark.parametrize("termination_fails", [False, True])
def test_failed_fluent_activation_terminates_its_owned_run_without_masking_the_error(
    tracking, monkeypatch, termination_fails
):
    from flaml.fabric.mlflow import _mlflow_wrapper

    created = []
    create_run = mlflow.MlflowClient.create_run

    def remember_run(self, *args, **kwargs):
        run = create_run(self, *args, **kwargs)
        created.append(run.info.run_id)
        return run

    objective = Mock()
    wrapped = _mlflow_wrapper(objective, "0", autolog=True)
    monkeypatch.setattr(mlflow.MlflowClient, "create_run", remember_run)
    monkeypatch.setattr(mlflow, "start_run", Mock(side_effect=ValueError("activation failed")))
    if termination_fails:
        monkeypatch.setattr(mlflow.MlflowClient, "set_terminated", Mock(side_effect=RuntimeError("termination failed")))
    with pytest.raises(ValueError, match="activation failed"):
        wrapped({})
    objective.assert_not_called()
    assert len(created) == 1
    assert tracking.get_run(created[0]).info.status == ("RUNNING" if termination_fails else "FAILED")


@pytest.mark.parametrize("behavior", ["end", "nested"])
@pytest.mark.parametrize("outcome", ["success", "exception", "empty"])
def test_local_autolog_cleanup_does_not_end_the_caller_run(tracking, behavior, outcome):
    from flaml.fabric.mlflow import _mlflow_wrapper

    with mlflow.start_run() as parent:

        def objective(config):
            if behavior == "end":
                mlflow.end_run()
            else:
                mlflow.start_run(nested=True)
                mlflow.log_metric("diagnostic", 1.0)
            if outcome == "exception":
                raise ValueError("local objective failure")
            return {} if outcome == "empty" else 1.0

        wrapped = _mlflow_wrapper(objective, parent.info.experiment_id, autolog=True)
        if outcome == "exception":
            with pytest.raises(ValueError, match="local objective failure"):
                wrapped({})
        else:
            assert wrapped({}) == ({} if outcome == "empty" else 1.0)
        assert mlflow.active_run().info.run_id == parent.info.run_id
        assert tracking.get_run(parent.info.run_id).info.status == "RUNNING"
        children = [run for run in tracking.search_runs(["0"]) if run.info.run_id != parent.info.run_id]
        expected_status = "FINISHED" if outcome == "success" else "FAILED"
        assert children and all(run.info.status == expected_status for run in children)


@pytest.mark.parametrize("behavior", ["manual_log", "manual_run", "autolog_end", "autolog_nested"])
def test_remote_objective_owns_an_isolated_fluent_stack(tracking, monkeypatch, tmp_path, behavior):
    from flaml.fabric.mlflow import _mlflow_wrapper

    driver_uri = mlflow.get_tracking_uri()
    with mlflow.start_run() as driver:

        def objective(config):
            assert mlflow.get_tracking_uri() == driver_uri
            if behavior == "manual_run":
                assert mlflow.active_run() is None
                with mlflow.start_run():
                    mlflow.log_metric("user_metric", 1.0)
            elif behavior == "manual_log":
                assert mlflow.active_run() is None
                mlflow.log_metric("user_metric", 1.0)
            elif behavior == "autolog_end":
                mlflow.end_run()
                assert mlflow.active_run() is None
            else:
                mlflow.start_run(nested=True)
                mlflow.log_metric("user_metric", 1.0)
            return 1.0

        wrapped = _mlflow_wrapper(objective, "0", autolog=behavior.startswith("autolog"))
    worker_uri = (tmp_path / "worker-store").as_uri()
    mlflow.set_tracking_uri(worker_uri)
    client = mlflow.MlflowClient()
    try:
        with mlflow.start_run() as worker:
            before = list(mlflow.tracking.fluent._active_run_stack.get())
            monkeypatch.setattr("socket.gethostname", lambda: "remote-stack-worker")
            assert wrapped({}) == 1.0
            assert mlflow.tracking.fluent._active_run_stack.get() == before
            assert mlflow.active_run().info.run_id == worker.info.run_id
            assert mlflow.get_tracking_uri() == worker_uri
            assert client.get_run(worker.info.run_id).info.status == "RUNNING"
        created = [run for run in tracking.search_runs(["0"]) if run.info.run_id != driver.info.run_id]
        assert created and all(run.info.status == "FINISHED" for run in created)
    finally:
        mlflow.set_tracking_uri(driver_uri)


def test_remote_tracking_restores_implicit_uri_and_environment(tracking, monkeypatch, tmp_path):
    from mlflow.tracking._tracking_service import utils as tracking_utils

    from flaml.fabric.mlflow import _mlflow_wrapper

    driver_uri = mlflow.get_tracking_uri()

    def objective(config):
        assert mlflow.get_tracking_uri() == driver_uri
        os.environ["MLFLOW_TRACKING_URI"] = (tmp_path / "changed-by-objective").as_uri()
        return 1.0

    wrapped = _mlflow_wrapper(objective, "0")
    worker_uri = (tmp_path / "implicit-worker").as_uri()
    mlflow.set_tracking_uri(None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", worker_uri)
    monkeypatch.setattr("socket.gethostname", lambda: "implicit-uri-worker")
    try:
        assert tracking_utils._tracking_uri is None
        assert wrapped({}) == 1.0
        assert tracking_utils._tracking_uri is None
        assert os.environ["MLFLOW_TRACKING_URI"] == worker_uri
        subsequent = (tmp_path / "subsequent").as_uri()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", subsequent)
        assert mlflow.get_tracking_uri() == subsequent
    finally:
        mlflow.set_tracking_uri(driver_uri)


@pytest.mark.parametrize("autolog", [False, True])
def test_completed_trial_wins_over_a_better_report_from_an_errored_trial(tracking, autolog):
    from flaml import tune

    def objective(config):
        if config["x"] == 0:
            tune.report(loss=0.0)
            return {}
        return {"loss": 1.0}

    if autolog:
        mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(
            objective,
            config={"x": tune.choice([0, 1])},
            points_to_evaluate=[{"x": 0}, {"x": 1}],
            metric="loss",
            mode="min",
            num_samples=2,
            use_ray=False,
            verbose=0,
        )
        assert [trial.status for trial in result.trials] == ["ERROR", "TERMINATED"]
        assert result.best_config == {"x": 1}
        assert result.best_trial is result.trials[1]
        assert tracking.get_run(result.best_run_id).data.metrics["loss"] == 1.0
        assert tracking.get_run(parent.info.run_id).data.params["x"] == "1"


@pytest.mark.parametrize("diagnostic", [False, True])
def test_empty_dict_failure_remains_a_failed_owned_autolog_run(tracking, diagnostic):
    from flaml import tune

    def objective(config):
        if diagnostic:
            mlflow.log_metric("diagnostic", 1.0)
        return {}

    mlflow.autolog(log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = tune.run(objective, config={"x": 1}, metric="loss", num_samples=1, use_ray=False, verbose=0)
        assert result.trials[0].status == "ERROR"
        runs = tracking.search_runs(["0"], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'")
        assert len(runs) == 1
        assert runs[0].info.status == "FAILED"
        assert runs[0].data.metrics == ({"diagnostic": 1.0} if diagnostic else {})


@pytest.mark.parametrize("content", ["input", "rename", "system_tag", "nested", "artifact_unavailable"])
def test_unrecorded_owned_runs_retain_user_lineage_and_names(tracking, monkeypatch, content):
    import flaml.fabric.mlflow as module

    with mlflow.start_run() as parent:
        integration = module.MLflowIntegration("tune")
        integration.autolog, integration.manual_log = True, False

        def objective(config):
            if content == "input":
                dataset = mlflow.data.from_pandas(pd.DataFrame({"feature": [1, 2]}))
                mlflow.log_input(dataset, context="training")
            elif content == "rename":
                mlflow.set_tag("mlflow.runName", "user-owned-name")
            elif content == "system_tag":
                mlflow.set_tag("mlflow.user", "edited owner")
            elif content == "nested":
                with mlflow.start_run(nested=True):
                    mlflow.log_metric("diagnostic", 1.0)
            return mlflow.active_run().info.run_id

        run_id = integration.wrap_evaluation_function(objective)({}, _flaml_trial_id="unreported")
        before = tracking.get_run(run_id)
        if content == "artifact_unavailable":
            monkeypatch.setattr(integration.mlflow_client, "list_artifacts", Mock(side_effect=RuntimeError("offline")))
            sibling = integration.wrap_evaluation_function(lambda config: mlflow.active_run().info.run_id)(
                {}, _flaml_trial_id="recorded"
            )
            integration.record_trial({"loss": 1.0}, SimpleNamespace(config={}, trial_id="recorded"), "loss")
        monkeypatch.setattr(module, "IS_RENAME_CHILD_RUN", True)
        integration.adopt_children()
        after = tracking.get_run(run_id)
        assert after.info.lifecycle_stage == "active"
        assert after.data.tags == before.data.tags
        assert "flaml.child_counter" not in after.data.tags
        assert mlflow.active_run().info.run_id == parent.info.run_id
        if content == "input":
            assert len(after.inputs.dataset_inputs) == 1
        elif content == "nested":
            children = tracking.search_runs(["0"], filter_string=f"tags.`mlflow.parentRunId` = '{run_id}'")
            assert len(children) == 1 and children[0].data.metrics == {"diagnostic": 1.0}
        elif content == "artifact_unavailable":
            assert tracking.get_run(sibling).data.metrics["loss"] == 1.0
            integration.mlflow_client.list_artifacts.assert_not_called()


@pytest.mark.parametrize("reported", [False, True])
@pytest.mark.parametrize("metric", ["time_total_s", "warmup_time"])
def test_explicit_time_objective_is_not_removed_as_framework_metadata(tracking, reported, metric):
    from flaml import tune

    def objective(config):
        if reported:
            tune.report(**{metric: 2.5})
        else:
            return {metric: 2.5}

    with mlflow.start_run():
        result = tune.run(objective, config={"x": 1}, metric=metric, mode="min", num_samples=1, verbose=0)
        assert tracking.get_run(result.best_run_id).data.metrics == {metric: 2.5}


@pytest.mark.parametrize("scalar", [False, True])
def test_scalar_and_dictionary_booleans_share_the_metric_filter(tracking, scalar):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run():
        integration = MLflowIntegration("tune")
        integration.record_trial(
            True if scalar else {"loss": True}, SimpleNamespace(config={}, trial_id="boolean"), "loss"
        )
        assert tracking.get_run(integration.manual_run_ids[0]).data.metrics == {}


@pytest.mark.parametrize("missing_getter", [False, True])
def test_ray_terminal_api_failures_are_explicit(monkeypatch, missing_getter):
    import importlib
    import sys
    from types import ModuleType

    module = importlib.import_module("flaml.tune.tune")
    ray = ModuleType("ray")
    ray.__version__ = "2.49.2"
    source = ModuleType("ray.tune")
    if not missing_getter:
        source.get_context = lambda: SimpleNamespace(get_trial_id=lambda: None)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.tune", source)
    for name in ("ray.train", "ray.air", "ray.air.session"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(module, "_use_ray", True)
    with pytest.raises(RuntimeError, match="identify.*Ray Tune trial"):
        module._ray_trial_id()
    with pytest.raises(ImportError, match="No compatible Ray reporting API"):
        module.report(loss=1.0)


def test_ray_1_reporting_delegates_the_legacy_signature(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    module = importlib.import_module("flaml.tune.tune")
    ray = ModuleType("ray")
    ray.__version__ = "1.13.0"
    ray.tune = SimpleNamespace(report=Mock())
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(module, "_use_ray", True)
    module.report(1.5, loss=0.5)
    ray.tune.report.assert_called_once_with(1.5, loss=0.5)


@pytest.mark.parametrize("lexicographic", [False, True])
def test_failed_trial_reports_are_ineligible_for_all_analysis_modes(lexicographic):
    from flaml.tune.trial_runner import SimpleTrial
    from flaml.tune.tune import ExperimentAnalysis

    trials = [SimpleTrial({"x": value}) for value in range(3)]
    for trial, loss in zip(trials, (0.0, 2.0, 1.0)):
        trial.set_status("RUNNING")
        trial.update_last_result({"loss": loss})
        trial.set_status("ERROR" if trial is trials[0] else "TERMINATED")
    objectives = {"metrics": ["loss"], "modes": ["min"], "tolerances": {"loss": 0}, "targets": {"loss": -float("inf")}}
    analysis = ExperimentAnalysis(
        trials, metric="loss", mode="min", lexico_objectives=objectives if lexicographic else None
    )
    assert analysis.best_trial is trials[2]
    assert analysis.best_config == {"x": 2}


@pytest.mark.parametrize("change", ["metric", "rename", "failure", "owner"])
def test_adoption_retains_runs_changed_after_search(tracking, monkeypatch, change):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run():
        integration = MLflowIntegration("tune")
        integration.autolog, integration.manual_log = True, False
        created = []
        integration.wrap_evaluation_function(lambda config: created.append(mlflow.active_run().info.run_id))(
            {}, _flaml_trial_id="unrecorded"
        )
        search = mlflow.search_runs

        def update_after_search(*args, **kwargs):
            result = search(*args, **kwargs)
            if change == "metric":
                tracking.log_metric(created[0], "diagnostic", 1.0)
            elif change == "rename":
                tracking.set_tag(created[0], "mlflow.runName", "user-annotated")
            elif change == "failure":
                tracking.set_terminated(created[0], status="FAILED")
            else:
                tracking.delete_tag(created[0], "flaml.owner_id")
            return result

        monkeypatch.setattr(mlflow, "search_runs", update_after_search)
        integration.adopt_children()
        assert tracking.get_run(created[0]).info.lifecycle_stage == "active"


def test_remote_manual_objective_hides_and_restores_the_last_active_run(tracking, monkeypatch, tmp_path):
    from flaml.fabric.mlflow import _mlflow_wrapper

    driver_uri = mlflow.get_tracking_uri()

    def objective(config):
        assert mlflow.last_active_run() is None
        return 1.0

    wrapped = _mlflow_wrapper(objective, "0")
    mlflow.set_tracking_uri((tmp_path / "worker").as_uri())
    try:
        with mlflow.start_run() as worker:
            worker_id = worker.info.run_id
        monkeypatch.setattr("socket.gethostname", lambda: "separate-worker")
        assert wrapped({}) == 1.0
        assert mlflow.last_active_run().info.run_id == worker_id
    finally:
        mlflow.set_tracking_uri(driver_uri)


@pytest.mark.parametrize("outcome", ["exception", "empty"])
@pytest.mark.parametrize("termination_fails", [False, True])
def test_remote_manual_failure_preserves_worker_state_and_primary_errors(
    tracking, monkeypatch, tmp_path, outcome, termination_fails
):
    from flaml.fabric.mlflow import _mlflow_wrapper

    driver_uri = mlflow.get_tracking_uri()
    created = []

    def objective(config):
        created.append(mlflow.start_run().info.run_id)
        if outcome == "exception":
            raise ValueError("primary manual failure")
        return {}

    wrapped = _mlflow_wrapper(objective, "0")
    mlflow.set_tracking_uri((tmp_path / "worker").as_uri())
    worker_client = mlflow.MlflowClient()
    try:
        with mlflow.start_run() as worker:
            monkeypatch.setattr("socket.gethostname", lambda: "manual-failure-worker")
            if termination_fails:
                monkeypatch.setattr(mlflow, "end_run", Mock(side_effect=RuntimeError("secondary termination failure")))
            if outcome == "exception":
                with pytest.raises(ValueError, match="primary manual failure"):
                    wrapped({})
            elif termination_fails:
                with pytest.raises(RuntimeError, match="secondary termination failure"):
                    wrapped({})
            else:
                assert wrapped({}) == {}
            assert mlflow.active_run().info.run_id == worker.info.run_id
            assert worker_client.get_run(worker.info.run_id).info.status == "RUNNING"
        assert tracking.get_run(created[0]).info.status == ("RUNNING" if termination_fails else "FAILED")
    finally:
        mlflow.set_tracking_uri(driver_uri)


def test_manual_summary_unwraps_ml_parameters_only_once(tracking):
    from flaml.fabric.mlflow import MLflowIntegration

    with mlflow.start_run() as parent:
        integration = MLflowIntegration("tune")
        trial = SimpleNamespace(
            trial_id="trial", status="TERMINATED", config={"ml": {"ml": 2, "depth": 3}}, last_result={"loss": 1.0}
        )
        integration.record_trial(trial.last_result, trial, "loss")
        analysis = SimpleNamespace(
            best_iteration=0, best_trial=trial, best_config=trial.config, best_result=trial.last_result, trials=[trial]
        )
        integration.log_tune(analysis, "loss")
        assert tracking.get_run(parent.info.run_id).data.params == {"ml": "2", "depth": "3"}


@pytest.mark.parametrize("failed_first", [False, True])
@pytest.mark.parametrize("trial_ids", [False, True])
def test_manual_summary_uses_the_recorded_winner_with_or_without_ids(tracking, failed_first, trial_ids):
    from flaml.fabric.mlflow import MLflowIntegration

    trials = [
        SimpleNamespace(
            trial_id=str(value) if trial_ids else None,
            status="TERMINATED",
            config={"x": value},
            last_result={"loss": float(value)},
            metric_analysis={"loss": {"last": float(value)}},
        )
        for value in (2, 1)
    ]
    if failed_first:
        trials.insert(
            0,
            SimpleNamespace(
                trial_id="0" if trial_ids else None,
                status="ERROR",
                config={"x": 0},
                last_result={"loss": 0.0},
                metric_analysis={"loss": {"last": 0.0}},
            ),
        )
    reported_best = trials[0] if failed_first else trials[-1]
    with mlflow.start_run() as parent:
        integration = MLflowIntegration("tune")
        for trial in trials:
            integration.record_trial(trial.last_result, trial, "loss")
        analysis = SimpleNamespace(
            trials=trials,
            best_trial=reported_best,
            best_iteration=0 if failed_first else 1,
            best_config=reported_best.config,
            best_result=reported_best.last_result,
            default_metric="loss",
            default_mode="min",
        )
        integration.log_tune(analysis, "loss")
        assert analysis.best_run_id == integration.manual_run_ids[1]
        assert tracking.get_run(parent.info.run_id).data.params["x"] == "1"
        assert tracking.get_run(parent.info.run_id).data.metrics["best_loss"] == 1.0


@pytest.mark.parametrize("accepted_attempt", [0, 1])
@pytest.mark.parametrize("page_size", [None, 1])
def test_spark_attempt_identity_adopts_only_the_accepted_run(tracking, monkeypatch, accepted_attempt, page_size):
    from pathlib import Path

    from flaml.fabric.mlflow import MLflowIntegration
    from flaml.tune.tune import _evaluate_spark_trial

    if page_size is not None:
        monkeypatch.setattr("flaml.fabric.mlflow.SEARCH_MAX_RESULTS", page_size)

    with mlflow.start_run() as parent:
        integration = MLflowIntegration("tune")
        integration.autolog, integration.manual_log = True, False
        run_ids = []

        def objective(config):
            attempt = len(run_ids)
            run_ids.append(mlflow.active_run().info.run_id)
            mlflow.log_param("attempt", attempt)
            mlflow.log_metric("diagnostic", float(attempt))
            mlflow.log_text(str(attempt), "attempt.txt")
            return {"loss": float(2 - attempt)}

        wrapped = integration.wrap_evaluation_function(objective)
        outcomes = [
            _evaluate_spark_trial(wrapped, {"x": 1}, "same-trial", {"_flaml_trial_id": "same-trial"}) for _ in range(2)
        ]
        accepted = outcomes[accepted_attempt]
        trial = SimpleNamespace(
            trial_id="same-trial", status="TERMINATED", config={"x": 1}, last_result=accepted.result
        )
        run_id = getattr(accepted, "mlflow_run_id", None)
        integration.record_trial(accepted.result, trial, "loss", **({"run_id": run_id} if run_id is not None else {}))
        analysis = SimpleNamespace(
            best_iteration=0, best_trial=trial, best_config=trial.config, best_result=trial.last_result, trials=[trial]
        )
        integration.log_tune(analysis, "loss")
        children = tracking.search_runs(["0"], filter_string="tags.`flaml.trial_id` = 'same-trial'")
        winners = [run for run in children if run.data.tags.get("flaml.best_run") == "True"]
        assert len(children) == 2 and len(winners) == 1
        assert winners[0].info.run_id == analysis.best_run_id == run_ids[accepted_attempt]
        assert accepted.mlflow_run_id == run_ids[accepted_attempt]
        assert tracking.get_run(parent.info.run_id).data.params["attempt"] == str(accepted_attempt)
        other = tracking.get_run(run_ids[1 - accepted_attempt])
        assert other.data.metrics == {"diagnostic": float(1 - accepted_attempt)}
        assert "flaml.child_counter" not in other.data.tags
        assert Path(tracking.download_artifacts(analysis.best_run_id, "attempt.txt")).read_text() == str(
            accepted_attempt
        )


@pytest.mark.parametrize("page_size", [None, 1])
def test_ambiguous_attempts_are_retained_without_blocking_a_unique_sibling(tracking, monkeypatch, page_size):
    import flaml.fabric.mlflow as module

    if page_size is not None:
        monkeypatch.setattr(module, "SEARCH_MAX_RESULTS", page_size)

    with mlflow.start_run() as parent:
        integration = module.MLflowIntegration("tune")
        integration.autolog, integration.manual_log = True, False
        wrapped = integration.wrap_evaluation_function(lambda config: mlflow.active_run().info.run_id)
        ambiguous = [wrapped({}, _flaml_trial_id="ambiguous")]
        sibling = wrapped({}, _flaml_trial_id="unique")
        ambiguous.append(wrapped({}, _flaml_trial_id="ambiguous"))
        trials = [
            SimpleNamespace(trial_id=name, config={"x": value}, status="TERMINATED", last_result={"loss": value})
            for name, value in (("ambiguous", 1.0), ("unique", 2.0))
        ]
        for trial in trials:
            integration.record_trial(trial.last_result, trial, "loss", run_id=None)
        analysis = SimpleNamespace(
            trials=trials,
            best_iteration=0,
            best_trial=trials[0],
            best_config=trials[0].config,
            best_result=trials[0].last_result,
            best_run_id=None,
            best_run_name=None,
        )
        warning = Mock(wraps=module.logger.warning)
        monkeypatch.setattr(module.logger, "warning", warning)
        integration.log_tune(analysis, "loss")
        assert analysis.best_run_id is None
        assert analysis.best_result == {"loss": 1.0}
        for run_id in ambiguous:
            stored = tracking.get_run(run_id)
            assert stored.info.lifecycle_stage == "active"
            assert not stored.data.metrics
            assert "flaml.child_counter" not in stored.data.tags
            assert stored.data.tags.get("flaml.best_run") != "True"
        assert tracking.get_run(sibling).data.metrics == {"loss": 2.0}
        assert tracking.get_run(sibling).data.tags["flaml.child_counter"] == "1"
        assert any("Cannot identify the accepted attempt" in call.args[0] for call in warning.call_args_list)
        assert "loss" not in tracking.get_run(parent.info.run_id).data.metrics


@pytest.mark.parametrize("run_id", ["", 0, [], object()])
def test_record_trial_rejects_invalid_attempt_ids_before_recording(tracking, run_id):
    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration("tune")
    with pytest.raises(ValueError, match="run ID.*nonempty string"):
        integration.record_trial({"loss": 1.0}, SimpleNamespace(config={}, trial_id="trial"), "loss", run_id=run_id)
    assert integration.infos == []
