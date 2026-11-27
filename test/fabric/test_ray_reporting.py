import inspect
from tempfile import TemporaryDirectory

import pytest
from packaging.version import Version

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(Version(ray.__version__) < Version("1.13"), reason="Requires Ray >=1.13.")


def _ray_storage(tmp_path):
    from ray import tune as ray_tune

    key = "storage_path" if "storage_path" in inspect.signature(ray_tune.run).parameters else "local_dir"
    return {key: str(tmp_path / "ray-results")}


def _transactional_tracking(mlflow, monkeypatch, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = mlflow.create_experiment("native-ray", artifact_location=(tmp_path / "artifacts").as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment)
    return experiment


@pytest.fixture(scope="module")
def ray_workers():
    owned = not ray.is_initialized()
    if not owned:
        yield
        return
    # Windows's native driver logger can hold its diagnostic file until interpreter exit.
    with (
        TemporaryDirectory(prefix="flaml-ray-", ignore_cleanup_errors=True) as directory,
        pytest.MonkeyPatch.context() as environment,
    ):
        try:
            environment.delenv("RAY_ADDRESS", raising=False)
            ray.init(
                address="local" if Version(ray.__version__) >= Version("2") else None,
                _temp_dir=directory,
                num_cpus=1,
                num_gpus=0,
                include_dashboard=False,
                log_to_driver=False,
                object_store_memory=80 * 1024 * 1024,
                _memory=256 * 1024 * 1024,
            )
            yield
        finally:
            ray.shutdown()


@pytest.mark.parametrize("autolog", [False, True])
@pytest.mark.parametrize("reported", [False, True])
@pytest.mark.parametrize("failure", [None, "exception", "empty"])
def test_native_ray_mlflow_associates_returned_and_reported_objectives(
    ray_workers, tmp_path, monkeypatch, autolog, reported, failure
):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
    from ray import tune as ray_tune

    from flaml import tune
    from flaml.fabric.mlflow import MLflowIntegration
    from flaml.tune.tune import _MLFLOW_RUN_ID

    recorded = {}
    record_trial = MLflowIntegration.record_trial

    def capture_attempt(self, metrics, trial, metric, **kwargs):
        recorded[trial.trial_id] = kwargs.get("run_id")
        return record_trial(self, metrics, trial, metric, **kwargs)

    monkeypatch.setattr(MLflowIntegration, "record_trial", capture_attempt)

    failed_first = failure is not None

    def objective(config, offset):
        if failed_first and config["x"] == 1:
            tune.report(loss=0.0)
            if failure == "empty":
                return {}
            raise RuntimeError("trial failed after its first report")
        metrics = {"loss": float(config["x"]) + offset, "converged": True, "note": "not a metric"}
        if reported:
            tune.report(**metrics)
        else:
            return metrics

    if Version(ray.__version__) < Version("2") and (not reported or failure == "empty"):
        # Ray 1.x with_parameters discards returns; only its report-style contract can be exercised.
        def trainable(config):
            return objective(config, offset=0.0)

    else:
        trainable = ray_tune.with_parameters(objective, offset=0.0)

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            if autolog:
                mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id) as parent:
                result = tune.run(
                    trainable,
                    config={"x": tune.choice([1, 2]), "enabled": True},
                    points_to_evaluate=[{"x": 1, "enabled": True}, {"x": 2, "enabled": True}],
                    metric="loss",
                    mode="min",
                    num_samples=2,
                    use_ray=True,
                    raise_on_failed_trial=not failed_first,
                    time_budget_s=90,
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                client = mlflow.tracking.MlflowClient()
                children = client.search_runs(
                    [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
                )
                assert len(result.trials) == 2
                assert len(children) == (1 if failed_first and not autolog else 2)
                by_trial = {child.data.tags["flaml.trial_id"]: child for child in children}
                for trial in result.trials:
                    if trial.status == "ERROR":
                        if autolog:
                            assert by_trial[trial.trial_id].info.status == "FAILED"
                        continue
                    child = by_trial[trial.trial_id]
                    assert child.data.metrics == {"loss": float(trial.config["x"])}
                    assert child.data.params["enabled"] == "True"
                    assert child.data.params["x"] == str(trial.config["x"])
                    if autolog:
                        assert trial.last_result[_MLFLOW_RUN_ID] == child.info.run_id
                        assert recorded[trial.trial_id] == child.info.run_id
                    else:
                        assert recorded[trial.trial_id] is None
                assert result.best_run_id == by_trial[result.best_trial.trial_id].info.run_id
                assert result.best_config["x"] == (2 if failed_first else 1)
                assert mlflow.active_run().info.run_id == parent.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_native_ray_empty_results_fail_without_mlflow(ray_workers, tmp_path):
    from flaml import tune

    def objective(config):
        if config["x"] == 1:
            tune.report(loss=0.0)
            return {}
        return {"loss": 1.0}

    result = tune.run(
        objective,
        config={"x": tune.choice([1, 2])},
        points_to_evaluate=[{"x": 1}, {"x": 2}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=True,
        mlflow_logging=False,
        raise_on_failed_trial=False,
        time_budget_s=90,
        verbose=0,
        **_ray_storage(tmp_path),
    )
    assert [trial.status for trial in result.trials] == ["ERROR", "TERMINATED"]
    assert result.best_config["x"] == 2


def test_native_ray_scalar_result_preserves_attempt_identity(ray_workers, tmp_path, monkeypatch):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune
    from flaml.tune.result import DEFAULT_METRIC
    from flaml.tune.tune import _MLFLOW_RUN_ID

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id):
                result = tune.run(
                    lambda config: float(config["x"]),
                    config={"x": 2},
                    metric=DEFAULT_METRIC,
                    mode="min",
                    num_samples=1,
                    use_ray=True,
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                assert result.best_result[DEFAULT_METRIC] == 2.0
                assert result.best_result[_MLFLOW_RUN_ID] == result.best_run_id
                assert mlflow.MlflowClient().get_run(result.best_run_id).data.metrics == {DEFAULT_METRIC: 2.0}
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_native_ray_tune_result_supports_reporting(ray_workers, tmp_path):
    from ray.tune import ExperimentAnalysis

    from flaml import tune
    from flaml.fabric.visualization import get_param_importance, plot_optimization_history

    result = tune.run(
        lambda config: {"loss": float(config["x"])},
        config={"x": tune.randint(1, 4)},
        points_to_evaluate=[{"x": 1}, {"x": 3}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=True,
        mlflow_logging=False,
        verbose=0,
        **_ray_storage(tmp_path),
    )
    assert isinstance(result, ExperimentAnalysis)
    assert set(result.search_space) == {"x"}
    assert len(plot_optimization_history(result).data[0].y) == 2
    assert get_param_importance(result) == {"x": 1.0}


@pytest.mark.parametrize(
    "logging_mode,stop_early,failure",
    [(mode, stop, None) for mode in ("off", "manual", "autolog") for stop in (False, True)]
    + [("autolog", False, "empty"), ("autolog", False, "exception")],
)
def test_native_ray_generators_preserve_yields_and_tracking(
    ray_workers, tmp_path, monkeypatch, logging_mode, stop_early, failure
):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune
    from flaml.tune.tune import _MLFLOW_RUN_ID

    def objective(config):
        yield {"loss": 3.0}
        if failure == "empty":
            yield {}
        elif failure == "exception":
            raise RuntimeError("generator failed after reporting")
        else:
            yield {"loss": 2.0}

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            if logging_mode == "autolog":
                mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id) as parent:
                result = tune.run(
                    objective,
                    config={"x": 1},
                    metric="loss",
                    mode="min",
                    num_samples=1,
                    use_ray=True,
                    mlflow_logging=logging_mode != "off",
                    stop={"training_iteration": 1} if stop_early else None,
                    raise_on_failed_trial=failure is None,
                    time_budget_s=60,
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                if failure is not None:
                    assert result.trials[0].status == "ERROR"
                    assert result.best_run_id is None
                    children = mlflow.MlflowClient().search_runs(
                        [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
                    )
                    assert len(children) == 1 and children[0].info.status == "FAILED"
                    assert mlflow.active_run().info.run_id == parent.info.run_id
                    return
                assert result.best_result["loss"] == (3.0 if stop_early else 2.0)
                assert result.best_result["training_iteration"] == (1 if stop_early else 2)
                if logging_mode != "off":
                    child = mlflow.MlflowClient().get_run(result.best_run_id)
                    assert child.data.metrics == {"loss": result.best_result["loss"]}
                    assert child.info.status == "FINISHED"
                    if logging_mode == "autolog":
                        assert result.best_result[_MLFLOW_RUN_ID] == child.info.run_id
                assert mlflow.active_run().info.run_id == parent.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_native_only_ray_reports_preserve_results_without_guessing_attempts(
    ray_workers, tmp_path, monkeypatch, ambiguous
):
    from unittest.mock import Mock, patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    import flaml.fabric.mlflow as integration_module
    from flaml import tune
    from flaml.tune.tune import _MLFLOW_RUN_ID

    def objective(config):
        import ray

        metrics = {"loss": float(config["x"])}
        if ambiguous and config["x"] == 1:
            client = mlflow.MlflowClient()
            active = client.get_run(mlflow.active_run().info.run_id)
            duplicate = client.create_run(active.info.experiment_id, tags=active.data.tags.copy())
            client.set_terminated(duplicate.info.run_id, "FINISHED")
        if Version(ray.__version__) < Version("2"):
            from ray import tune as native

            native.report(**metrics)
        else:
            try:
                from ray.air import session as native
            except ImportError:
                from ray import train as native
            native.report(metrics)

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    warning = Mock(wraps=integration_module.logger.warning)
    monkeypatch.setattr(integration_module.logger, "warning", warning)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id) as parent:
                result = tune.run(
                    objective,
                    config={"x": tune.choice([1, 2])},
                    points_to_evaluate=[{"x": 1}, {"x": 2}],
                    metric="loss",
                    mode="min",
                    num_samples=2,
                    use_ray=True,
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                assert result.best_config["x"] == 1
                assert result.best_result["loss"] == 1.0
                assert all(_MLFLOW_RUN_ID not in trial.last_result for trial in result.trials)
                client = mlflow.MlflowClient()
                children = client.search_runs(
                    [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
                )
                assert len(children) == (3 if ambiguous else 2)
                assert all(child.info.lifecycle_stage == "active" for child in children)
                unique = [
                    child for child in children if child.data.tags["flaml.trial_id"] != result.best_trial.trial_id
                ]
                assert len(unique) == 1 and unique[0].data.metrics == {"loss": 2.0}
                if ambiguous:
                    assert result.best_run_id is None
                    assert all(child.data.tags.get("flaml.best_run") != "True" for child in children)
                    assert any(
                        "Cannot identify the accepted attempt" in call.args[0] for call in warning.call_args_list
                    )
                else:
                    assert client.get_run(result.best_run_id).data.metrics == {"loss": 1.0}
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("user_lagrange", [False, True])
def test_native_ray_report_metrics_exclude_optimizer_annotations(ray_workers, tmp_path, monkeypatch, user_lagrange):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune

    metrics = {"loss": 2.0, "memory": 2.0}
    if user_lagrange:
        metrics["loss_lagrange"] = 17.0

    def objective(config):
        tune.report(**metrics)

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id):
                result = tune.run(
                    objective,
                    config={"x": 1},
                    metric="loss",
                    mode="min",
                    metric_constraints=[("memory", "<=", 1.0)],
                    num_samples=1,
                    use_ray=True,
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                assert mlflow.MlflowClient().get_run(result.best_run_id).data.metrics == metrics
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_native_ray_report_stop_finishes_the_owned_run(ray_workers, tmp_path, monkeypatch):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune

    def objective(config):
        tune.report(loss=3.0)
        tune.report(loss=2.0)

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    experiment_id = _transactional_tracking(mlflow, monkeypatch, tmp_path)
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id=experiment_id) as parent:
                result = tune.run(
                    objective,
                    config={"x": 1},
                    metric="loss",
                    mode="min",
                    num_samples=1,
                    use_ray=True,
                    stop={"training_iteration": 1},
                    verbose=0,
                    **_ray_storage(tmp_path),
                )
                assert result.best_result["loss"] == 3.0
                assert result.best_result["training_iteration"] == 1
                run = mlflow.MlflowClient().get_run(result.best_run_id)
                assert run.info.status == "FINISHED"
                assert run.data.metrics == {"loss": 3.0}
                assert mlflow.active_run().info.run_id == parent.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_automl_native_ray_rejects_file_tracking_before_search(tmp_path, monkeypatch):
    import mlflow
    import numpy as np

    from flaml import AutoML

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    try:
        with mlflow.start_run(experiment_id="0"):
            automl = AutoML(max_iter=1, estimator_list=["rf"], verbose=0)
            with pytest.raises(ValueError, match="transactional tracking"):
                automl.fit(np.arange(40).reshape(20, 2), np.tile([0, 1], 10), use_ray=True)
            assert automl.model is None
    finally:
        mlflow.set_tracking_uri(previous_uri)
