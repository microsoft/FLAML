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

    from flaml.automl.pipeline import make_automl_pipeline
    from flaml.fabric.mlflow import MLflowIntegration

    X = pd.DataFrame({"value": np.arange(60.0), "squared": np.arange(60.0) ** 2})
    y = np.tile([0, 1], 30)
    expected = []
    original = MLflowIntegration.record_state

    def record(self, automl, state, estimator, *args, **kwargs):
        model = getattr(state, "trial_model", None) or state.trained_estimator
        expected.append(make_automl_pipeline(automl, model).predict(X))
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
                {"n_estimators": 4, "fe.numerical": "scaler_standard"},
                {"n_estimators": 8, "fe.numerical": "scaler_minmax"},
            ]
        },
    )
    with mlflow.start_run():
        automl.fit(X, y)
        run_ids = automl.mlflow_integration.manual_run_ids
        assert len(run_ids) == len(expected) == 2
        for run_id, predictions, trees in zip(run_ids, expected, (4, 8)):
            restored = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            np.testing.assert_array_equal(restored.predict(X), predictions)
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


def test_failed_fit_drains_artifact_producers_before_restoring_autolog():
    from flaml.automl.automl import _restore_mlflow_after_fit

    order = []
    integration = SimpleNamespace(resume_mlflow=lambda: order.append("restore"))
    instance = SimpleNamespace(mlflow_integration=integration, wait_futures=lambda: order.append("drain"))

    def fail(self):
        raise ValueError("original training error")

    with pytest.raises(ValueError, match="original training error"):
        _restore_mlflow_after_fit(fail)(instance)
    assert order == ["drain", "restore"]


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
