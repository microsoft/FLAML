import inspect

import pytest
from packaging.version import Version

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(Version(ray.__version__) < Version("1.13"), reason="Requires Ray >=1.13.")


def _ray_storage(tmp_path):
    from ray import tune as ray_tune

    key = "storage_path" if "storage_path" in inspect.signature(ray_tune.run).parameters else "local_dir"
    return {key: str(tmp_path / "ray-results")}


@pytest.fixture(scope="module")
def ray_workers():
    owned = not ray.is_initialized()
    with pytest.MonkeyPatch.context() as environment:
        try:
            if owned:
                environment.delenv("RAY_ADDRESS", raising=False)
                ray.init(
                    address=None,
                    num_cpus=1,
                    num_gpus=0,
                    include_dashboard=False,
                    log_to_driver=False,
                    object_store_memory=80 * 1024 * 1024,
                    _memory=256 * 1024 * 1024,
                )
            yield
        finally:
            if owned:
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
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            if autolog:
                mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id="0") as parent:
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
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    try:
        with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(experiment_id="0"):
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
