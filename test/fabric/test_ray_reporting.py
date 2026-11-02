import pytest

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(not ray.__version__.startswith("2."), reason="Declared Ray compatibility is 2.x.")


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
