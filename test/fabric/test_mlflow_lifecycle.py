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
def tracking(monkeypatch, tmp_path):
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
        yield mlflow.tracking.MlflowClient()
        while mlflow.active_run() is not None:
            mlflow.end_run()
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
                run_id=f"run-{index}", run_name=f"trial-{index}", start_time=(integration.start_time + index + 1) * 1000
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


def test_adoption_cleans_only_unrecorded_owned_empty_runs(tracking):
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
        created[name] = tracking.create_run(
            integration.experiment_id, start_time=int((integration.start_time + 1) * 1000), tags=tags
        ).info.run_id
    tracking.set_terminated(created["failed"], "FAILED")
    tracking.log_text(created["artifact"], "retain this diagnostic", "trace.txt")
    tracking.set_tag(created["annotated"], "mlflow.note.content", "retain this explanation")
    tracking.set_tag(created["retagged"], "extra_tag.sid", "caller-authored value")
    integration.adopt_children()
    assert tracking.get_run(created["recorded"]).info.lifecycle_stage == "active"
    assert tracking.get_run(created["stopped"]).info.lifecycle_stage == "deleted"
    assert tracking.get_run(created["unrelated"]).info.lifecycle_stage == "active"
    for name in ("failed", "artifact", "annotated", "retagged"):
        assert tracking.get_run(created[name]).info.lifecycle_stage == "active"
    assert tracking.list_artifacts(created["artifact"])[0].path == "trace.txt"


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
                {**result, "done": True, "config": trial.config, "config/x": trial.config["x"], "timestamp": 1234}
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
