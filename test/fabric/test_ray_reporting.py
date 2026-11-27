import inspect
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from packaging.version import Version

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(Version(ray.__version__) < Version("1.13"), reason="Requires Ray >=1.13.")


def _ray_storage(tmp_path):
    from ray import tune as ray_tune

    key = "storage_path" if "storage_path" in inspect.signature(ray_tune.run).parameters else "local_dir"
    return {key: str(tmp_path / "ray-results")}


@contextmanager
def _transactional_tracking(mlflow, monkeypatch, tmp_path, uri_kind="absolute"):
    from mlflow.tracking import fluent
    from mlflow.tracking._tracking_service import utils as tracking_utils

    previous_uri = tracking_utils._tracking_uri
    previous_environment_uri = os.environ.get("MLFLOW_TRACKING_URI")
    last = fluent._last_active_run_id
    previous_last = last.get() if callable(getattr(last, "get", None)) else last
    with monkeypatch.context() as state:
        if uri_kind != "absolute":
            state.chdir(tmp_path)
        state.setattr(fluent, "_active_experiment_id", None)
        state.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        state.delenv("MLFLOW_RUN_ID", raising=False)
        # MLflow caches stores by URI, not by the current working directory.
        relative_name = f"tracking-{uuid4().hex}.db"
        uri = {
            "absolute": f"sqlite:///{tmp_path / 'tracking.db'}",
            "relative": f"sqlite:///{relative_name}",
            "file": f"sqlite:///file:{relative_name}?mode=rwc&uri=true",
        }[uri_kind]
        try:
            mlflow.set_tracking_uri(uri)
            experiment = mlflow.create_experiment("native-ray", artifact_location=(tmp_path / "artifacts").as_uri())
            state.setenv("MLFLOW_EXPERIMENT_ID", experiment)
            yield experiment
        finally:
            mlflow.set_tracking_uri(previous_uri)
            if previous_environment_uri is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = previous_environment_uri
            if callable(getattr(last, "set", None)):
                last.set(previous_last)
            else:
                fluent._last_active_run_id = previous_last


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


@pytest.mark.parametrize("explicit_override", [False, True])
def test_transactional_tracking_restores_raw_configuration_after_setup_failure(
    tmp_path, monkeypatch, explicit_override
):
    from pathlib import Path
    from unittest.mock import Mock

    import mlflow
    from mlflow.tracking._tracking_service import utils as tracking_utils

    inherited = f"sqlite:///{tmp_path / 'inherited.db'}"
    override = f"sqlite:///{tmp_path / 'override.db'}" if explicit_override else None
    monkeypatch.setenv("MLFLOW_TRACKING_URI", inherited)
    monkeypatch.setattr(tracking_utils, "_tracking_uri", override)
    failure = RuntimeError("experiment setup failed")
    monkeypatch.setattr(mlflow, "create_experiment", Mock(side_effect=failure))
    original_directory = Path.cwd()
    with pytest.raises(RuntimeError) as caught:
        with _transactional_tracking(mlflow, monkeypatch, tmp_path, "relative"):
            pytest.fail("A failed setup must not enter the test body.")
    assert caught.value is failure
    assert tracking_utils._tracking_uri == override
    assert os.environ["MLFLOW_TRACKING_URI"] == inherited
    assert Path.cwd() == original_directory


@pytest.mark.parametrize("uri_kind", ["relative", "file"])
def test_transactional_tracking_keeps_sdk_cached_stores_distinct_between_directories(tmp_path, monkeypatch, uri_kind):
    import mlflow

    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        with _transactional_tracking(mlflow, monkeypatch, directory, uri_kind) as experiment:
            stored = mlflow.MlflowClient().get_experiment(experiment)
            assert stored.artifact_location == (directory / "artifacts").as_uri()


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

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment_id:
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


@pytest.mark.parametrize("uri_kind", ["absolute", "relative", "file"])
def test_native_ray_scalar_result_preserves_attempt_identity(ray_workers, tmp_path, monkeypatch, uri_kind):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune
    from flaml.tune.result import DEFAULT_METRIC
    from flaml.tune.tune import _MLFLOW_RUN_ID

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path, uri_kind) as experiment_id:
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

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment_id:
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

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    warning = Mock(wraps=integration_module.logger.warning)
    monkeypatch.setattr(integration_module.logger, "warning", warning)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment_id:
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

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment_id:
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


def test_native_ray_report_stop_finishes_the_owned_run(ray_workers, tmp_path, monkeypatch):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune

    def objective(config):
        tune.report(loss=3.0)
        tune.report(loss=2.0)

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    with _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment_id:
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


def test_notebook_automl_ray_branch_uses_transactional_tracking(ray_workers, tmp_path, monkeypatch):
    import sys
    from test.automl import test_notebook_example
    from types import ModuleType
    from unittest.mock import patch

    import mlflow
    from mlflow.tracking import fluent
    from mlflow.tracking._tracking_service import utils as tracking_utils
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from flaml import AutoML
    from flaml.automl import data

    X, y = load_iris(return_X_y=True, as_frame=True)
    splits = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    monkeypatch.setitem(sys.modules, "openml", ModuleType("openml"))
    monkeypatch.setattr(data, "load_openml_task", lambda **kwargs: splits)
    fit = AutoML.fit
    calls = []

    def bounded_fit(self, *args, **kwargs):
        call = (kwargs.get("use_ray", False), mlflow.get_tracking_uri())
        kwargs.update(n_jobs=1, max_iter=2, time_budget=30, estimator_list=["rf"])
        if call[0]:
            kwargs["use_ray"] = _ray_storage(tmp_path)
        result = fit(self, *args, **kwargs)
        calls.append(call)
        return result

    monkeypatch.setattr(AutoML, "fit", bounded_fit)
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    with (
        _transactional_tracking(mlflow, monkeypatch, tmp_path),
        patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True),
    ):
        names = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN_ID")

        def tracking_state():
            last = fluent._last_active_run_id
            return (
                tracking_utils._tracking_uri,
                {name: os.environ.get(name) for name in names},
                fluent._active_experiment_id,
                last.get() if callable(getattr(last, "get", None)) else last,
            )

        before = tracking_state()
        test_notebook_example.test_mlflow(tmp_path)
        assert [native for native, _ in calls] == [False, True]
        assert calls[-1][1].startswith("sqlite:///")
        assert tracking_state() == before
        client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'ray-tracking.db'}")
        experiment = client.get_experiment_by_name("ray-example")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
        assert len(parents) == 1 and parents[0].info.status == "FINISHED"
        children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parents[0].info.run_id]
        assert children and all(run.info.status == "FINISHED" for run in children)


def test_native_ray_worker_exit_before_reporting_finalizes_its_unique_run(ray_workers, tmp_path, monkeypatch):
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    from flaml import tune

    def objective(config):
        import os

        mlflow.log_param("worker_exit_probe", True)
        os._exit(23)

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    with (
        _transactional_tracking(mlflow, monkeypatch, tmp_path) as experiment,
        patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True),
    ):
        mlflow.autolog(log_models=False, silent=True)
        with mlflow.start_run(experiment_id=experiment) as parent:
            result = tune.run(
                objective,
                config={},
                metric="loss",
                mode="min",
                num_samples=1,
                use_ray=True,
                raise_on_failed_trial=False,
                max_failures=0,
                time_budget_s=60,
                verbose=0,
                **_ray_storage(tmp_path),
            )
            assert len(result.trials) == 1 and result.trials[0].status == "ERROR"
            assert "loss" not in (result.trials[0].last_result or {})
            runs = mlflow.MlflowClient().search_runs(
                [experiment], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
            )
            assert len(runs) == 1 and runs[0].info.status == "FAILED"
            assert runs[0].data.params["worker_exit_probe"] == "True"
            assert "loss" not in runs[0].data.metrics


def test_native_automl_worker_logging_uses_the_driver_store(ray_workers, tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import patch

    import mlflow
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
    from sklearn.datasets import load_iris

    from flaml import AutoML

    def logged_metric(X_val, y_val, estimator, labels, *args, **kwargs):
        mlflow.log_metric("worker_store_probe", 73.0)
        mlflow.log_param("worker_directory", os.getcwd())
        return float((estimator.predict(X_val) != y_val).mean()), {}

    X, y = load_iris(return_X_y=True)
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    with (
        _transactional_tracking(mlflow, monkeypatch, tmp_path, "relative") as experiment,
        patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True),
        mlflow.start_run(experiment_id=experiment) as parent,
    ):
        automl = AutoML()
        automl.fit(
            X,
            y,
            estimator_list=["rf"],
            max_iter=2,
            time_budget=60,
            n_jobs=1,
            metric=logged_metric,
            retrain_full=False,
            use_ray=_ray_storage(tmp_path),
            eval_method="holdout",
            verbose=0,
            starting_points={"rf": {"n_estimators": 4, "max_leaves": 4, "max_features": 1.0}},
        )
        assert automl.model is not None
        diagnostics = [
            run
            for run in mlflow.MlflowClient().search_runs([experiment])
            if run.data.metrics.get("worker_store_probe") == 73.0
        ]
        assert len(diagnostics) == 2
        assert all(run.info.status == "FINISHED" for run in diagnostics)
        assert all(Path(run.data.params["worker_directory"]).resolve() != tmp_path.resolve() for run in diagnostics)
        assert mlflow.active_run().info.run_id == parent.info.run_id


def test_native_ray_reuses_the_objective_inside_retained_native_limiters(ray_workers, tmp_path):
    from ray import tune as native

    if Version(ray.__version__) < Version("2"):
        from ray.tune.suggest import ConcurrencyLimiter
        from ray.tune.suggest.optuna import OptunaSearch
    else:
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.search.optuna import OptunaSearch

    from flaml import tune

    searcher = OptunaSearch(None, metric="training_iteration", mode="max")
    limiter = ConcurrencyLimiter(ConcurrencyLimiter(searcher, max_concurrent=1), max_concurrent=1)
    settings = dict(search_alg=limiter, num_samples=1, use_ray=True, mlflow_logging=False, verbose=0)
    tune.run(
        lambda config: {"loss": 1.0},
        config={"x": native.uniform(0, 1)},
        metric="loss",
        mode="min",
        **settings,
        **_ray_storage(tmp_path / "first"),
    )
    result = tune.run(lambda config: {"loss": 2.0}, **settings, **_ray_storage(tmp_path / "second"))
    assert result.best_result["loss"] == 2.0
    assert searcher.metric == "loss" and searcher.mode == "min"
    assert not limiter.live_trials and not limiter.searcher.live_trials
