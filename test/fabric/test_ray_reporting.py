import pytest

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(not ray.__version__.startswith("2."), reason="Declared Ray compatibility is 2.x.")


@pytest.fixture(scope="module")
def ray_workers():
    owned = not ray.is_initialized()
    try:
        if owned:
            ray.init(
                address="local",
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
def test_native_ray_mlflow_associates_returned_and_reported_objectives(
    ray_workers, tmp_path, monkeypatch, autolog, reported
):
    import inspect
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
    from ray import tune as ray_tune

    from flaml import tune

    def objective(config, offset):
        metrics = {"loss": float(config["x"]) + offset, "converged": True, "note": "not a metric"}
        if reported:
            tune.report(**metrics)
        else:
            return metrics

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
                storage_key = (
                    "storage_path" if "storage_path" in inspect.signature(ray_tune.run).parameters else "local_dir"
                )
                result = tune.run(
                    ray_tune.with_parameters(objective, offset=0.0),
                    config={"x": tune.choice([1, 2]), "enabled": True},
                    points_to_evaluate=[{"x": 1, "enabled": True}, {"x": 2, "enabled": True}],
                    metric="loss",
                    mode="min",
                    num_samples=2,
                    use_ray=True,
                    time_budget_s=90,
                    verbose=0,
                    **{storage_key: str(tmp_path / "ray-results")},
                )
                client = mlflow.tracking.MlflowClient()
                children = client.search_runs(
                    [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
                )
                assert len(result.trials) == len(children) == 2
                by_trial = {child.data.tags["flaml.trial_id"]: child for child in children}
                for trial in result.trials:
                    child = by_trial[trial.trial_id]
                    assert child.data.metrics == {"loss": float(trial.config["x"])}
                    assert child.data.params["enabled"] == "True"
                    assert child.data.params["x"] == str(trial.config["x"])
                assert result.best_run_id == by_trial[result.best_trial.trial_id].info.run_id
                assert mlflow.active_run().info.run_id == parent.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_native_ray_tune_result_supports_reporting(tmp_path):
    from ray.tune import ExperimentAnalysis

    from flaml import tune
    from flaml.fabric.visualization import get_param_importance, plot_optimization_history

    owned = not ray.is_initialized()
    try:
        if owned:
            ray.init(
                num_cpus=1,
                num_gpus=0,
                include_dashboard=False,
                log_to_driver=False,
                local_mode=True,
                object_store_memory=80 * 1024 * 1024,
                _memory=256 * 1024 * 1024,
            )
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
            local_dir=str(tmp_path / "ray-results"),
        )
        assert isinstance(result, ExperimentAnalysis)
        assert set(result.search_space) == {"x"}
        assert len(plot_optimization_history(result).data[0].y) == 2
        assert get_param_importance(result) == {"x": 1.0}
    finally:
        if owned:
            ray.shutdown()
