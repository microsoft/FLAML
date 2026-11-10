import importlib
import os
import subprocess
import sys
import textwrap
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from flaml import AutoML, tune
from flaml.fabric import is_fabric_runtime


@pytest.fixture
def public_environment(monkeypatch):
    monkeypatch.delenv("FLAML_FEATURIZATION", raising=False)
    monkeypatch.delenv("MSNOTEBOOKUTILS_RUNTIME_TYPE", raising=False)
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")


@pytest.mark.parametrize(
    "runtime_type,context_file,expected",
    [
        ("", False, False),
        ("spark", False, False),
        ("jupyter", False, True),
        ("JupyterNotebook", False, True),
        ("", True, True),
    ],
)
def test_runtime_detection(monkeypatch, runtime_type, context_file, expected):
    monkeypatch.delenv("FLAML_FABRIC_RUNTIME", raising=False)
    monkeypatch.setenv("MSNOTEBOOKUTILS_RUNTIME_TYPE", runtime_type)
    monkeypatch.setattr("flaml.fabric.os.path.isfile", lambda path: context_file)
    assert is_fabric_runtime() is expected


@pytest.mark.parametrize("override,expected", [("true", True), ("1", True), (" FALSE ", False), ("0", False)])
def test_runtime_override_precedes_detection(monkeypatch, override, expected):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", override)
    monkeypatch.setattr("flaml.fabric.is_fabric_spark", Mock(side_effect=AssertionError("must not detect")))
    monkeypatch.setattr("flaml.fabric.is_pure_python_env", Mock(side_effect=AssertionError("must not detect")))
    assert is_fabric_runtime() is expected


def test_invalid_runtime_override_is_reported(monkeypatch):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "maybe")
    with pytest.raises(ValueError, match="FLAML_FABRIC_RUNTIME"):
        is_fabric_runtime()


@pytest.mark.parametrize("fabric", [False, True])
@pytest.mark.parametrize("explicit_history", [None, False, True])
def test_model_history_defaults_and_override(public_environment, monkeypatch, fabric, explicit_history):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    settings = {} if explicit_history is None else {"model_history": explicit_history}
    automl = AutoML(**settings)
    assert automl._settings["model_history"] is (fabric if explicit_history is None else explicit_history)
    assert automl._settings["featurization"] == "off"


@pytest.mark.parametrize("fabric", [False, True])
def test_explicit_featurization_precedes_environment(public_environment, monkeypatch, fabric):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    monkeypatch.setenv("FLAML_FEATURIZATION", "auto")
    assert AutoML()._settings["featurization"] == "auto"
    assert AutoML(featurization="off")._settings["featurization"] == "off"


def test_unsupported_featurization_dict_fails_before_data_validation(public_environment, monkeypatch):
    validation = Mock(side_effect=AssertionError("Invalid feature configuration reached training"))
    monkeypatch.setattr("flaml.automl.task.generic_task.GenericTask.validate_data", validation)
    automl = AutoML(featurization={"numerical": {"methods": "scaler_standard"}}, mlflow_logging=False)
    with pytest.raises(ValueError, match="Expect featurization to be one of 'auto', 'off', 'force';"):
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    validation.assert_not_called()


def test_automl_import_and_fit_without_mlflow(public_environment):
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockOptionalServices(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"mlflow", "synapse", "pyspark"}:
                    raise ModuleNotFoundError(fullname, name=fullname)

        sys.meta_path.insert(0, BlockOptionalServices())
        from flaml import AutoML, tune
        from flaml.automl import register_automl_pipeline
        import numpy as np

        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0)
        assert automl._settings["model_history"] is False
        assert automl._settings["featurization"] == "off"
        X = np.arange(80).reshape(40, 2)
        y = np.tile([0, 1], 20)
        automl.fit(X, y)
        assert len(automl.predict(X[:3])) == 3
        assert automl.mlflow_integration is None
        analysis = tune.run(lambda config: {"loss": 1.0}, config={"x": 1},
                            metric="loss", mode="min", num_samples=1, use_ray=False, verbose=0)
        assert len(analysis.trials) == 1
        assert not any(name.split(".")[0] in {"mlflow", "synapse", "pyspark"} for name in sys.modules)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def local_mlflow(public_environment, monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    integration_module = importlib.import_module("flaml.fabric.mlflow")
    tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    monkeypatch.setattr(integration_module, "AUTOLOGGING_INTEGRATIONS", {})
    monkeypatch.setattr(integration_module, "is_autolog_enabled", lambda: False)
    monkeypatch.setattr("flaml.automl.automl.is_autolog_enabled", lambda: False)
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    yield mlflow, integration_module
    mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)


@pytest.mark.parametrize("fabric", [False, True])
def test_automl_logging_policy(local_mlflow, monkeypatch, fabric):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    X = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0)
    automl.fit(X, y)
    if fabric:
        assert automl.mlflow_integration.only_history
        assert automl.mlflow_integration.infos
        assert "synapseml.flaml.version" in automl.mlflow_integration.infos[0]["tags"]
    else:
        assert automl.mlflow_integration is None


@pytest.mark.parametrize("experiment_type", ["automl", "tune"])
def test_history_only_does_not_contact_services_or_change_autolog(local_mlflow, monkeypatch, experiment_type):
    mlflow, module = local_mlflow
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "true")
    monkeypatch.setattr("flaml.tune.tune.is_autolog_enabled", lambda: False)
    no_service = Mock(side_effect=AssertionError("History-only must remain local"))
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", no_service)
    monkeypatch.setattr(mlflow, "set_experiment", no_service)
    monkeypatch.setattr(mlflow, "get_experiment", no_service)
    monkeypatch.setattr(mlflow, "autolog", no_service)
    monkeypatch.setattr(module, "_get_notebook_name", no_service)
    if experiment_type == "automl":
        result = AutoML(max_iter=1, estimator_list=["rf"], n_jobs=1, verbose=0)
        result.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
        assert result.mlflow_integration.only_history
        assert result.mlflow_integration.infos
    else:
        result = tune.run(
            lambda config: {"loss": 1.0},
            config={"x": 1},
            metric="loss",
            mode="min",
            num_samples=1,
            use_ray=False,
            verbose=0,
        )
        assert len(result.trials) == 1
    no_service.assert_not_called()


@pytest.mark.parametrize("active_run,log_models", [(False, True), (True, False), (True, True)])
def test_parallel_latency_probe_respects_logging_intent(local_mlflow, monkeypatch, active_run, log_models):
    mlflow, module = local_mlflow
    automl_module = importlib.import_module("flaml.automl.automl")
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "true")
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": {"disable": True, "log_models": log_models}})

    class ReachedParallelRunner(Exception):
        pass

    latency = Mock(return_value=0.0)
    runner = Mock(side_effect=ReachedParallelRunner)
    monkeypatch.setattr(automl_module, "get_mlflow_log_latency", latency)
    monkeypatch.setattr(automl_module.tune, "run", runner)
    monkeypatch.setattr(AutoML, "trainable", property(lambda self: Mock()))
    automl = AutoML(estimator_list=["rf"], max_iter=2, n_jobs=1, verbose=0)
    search = automl._search

    def parallel_search(*args, **kwargs):
        automl._use_spark = True
        automl._n_concurrent_trials = 2
        return search(*args, **kwargs)

    monkeypatch.setattr(automl, "_search", parallel_search)
    if active_run:
        mlflow.start_run()
    try:
        with pytest.raises(ReachedParallelRunner):
            automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
        runner.assert_called_once()
        if active_run:
            latency.assert_called_once_with(model_history=log_models)
        else:
            latency.assert_not_called()
    finally:
        mlflow.end_run()


@pytest.mark.parametrize("constructor_settings", [{}, {"task": "classification", "time_budget": 60}])
def test_configuration_metadata_tracks_each_fit(public_environment, constructor_settings):
    automl = AutoML(**constructor_settings)
    X = np.random.RandomState(7).normal(size=(60, 3))
    settings = {"estimator_list": ["lgbm"], "max_iter": 1, "n_jobs": 1, "verbose": 0, "mlflow_logging": False}
    automl.fit(X, X[:, 0], task="regression", time_budget=20, featurization="auto", **settings)
    assert automl._automl_user_configurations["task"] == "regression"
    assert automl._automl_user_configurations["time_budget"] == 20
    assert automl._automl_user_configurations["featurization"] == "auto"
    automl.fit(X, (X[:, 0] > 0).astype(int), task="classification", time_budget=30, featurization="off", **settings)
    assert automl._automl_user_configurations["task"] in ("classification", "binary")
    assert automl._automl_user_configurations["time_budget"] == 30
    assert automl._automl_user_configurations["featurization"] == "off"


def test_ensemble_reports_featurization_exclusion(public_environment, monkeypatch, caplog):
    class ReachedSearch(Exception):
        pass

    automl = AutoML(
        estimator_list=["rf", "extra_tree"],
        max_iter=1,
        n_jobs=1,
        verbose=2,
        mlflow_logging=False,
        ensemble=True,
        featurization="force",
    )
    monkeypatch.setattr(automl, "_search", Mock(side_effect=ReachedSearch))
    with pytest.raises(ReachedSearch):
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    assert automl._featurization == "off"
    assert automl._automl_user_configurations["featurization"] == "off"
    assert any("AutoFE" in record.message and "ensemble" in record.message for record in caplog.records)


def test_missing_autofe_does_not_disable_ordinary_automl(public_environment):
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import numpy as np

        class BlockAutoFE(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "flaml.fabric.autofe":
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, BlockAutoFE())
        import flaml
        assert flaml.has_automl
        automl = flaml.AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1,
                            mlflow_logging=False, verbose=0)
        X, y = np.arange(80).reshape(40, 2), np.tile([0, 1], 20)
        automl.fit(X, y)
        assert len(automl.predict(X[:2])) == 2
        try:
            automl.fit(X, y, featurization="auto")
        except ImportError as error:
            assert "flaml[autofe]" in str(error)
        else:
            raise AssertionError("Explicit AutoFE was silently skipped")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_telemetry_is_independent_of_mlflow(public_environment):
    code = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys
        import numpy as np

        class BlockMLflow(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "mlflow":
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, BlockMLflow())
        from flaml import AutoML, tune
        import flaml.fabric.telemetry as telemetry
        from flaml.fabric.logger import init_kusto_logger
        automl_module = importlib.import_module("flaml.automl.automl")
        tune_module = importlib.import_module("flaml.tune.tune")
        assert automl_module.kusto_logger is init_kusto_logger()
        assert tune_module.kusto_logger is init_kusto_logger()
        calls = []
        telemetry.is_fabric_runtime = lambda: True
        telemetry.report_usage_telemetry = lambda category, activity, **kwargs: calls.append(activity)
        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0)
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
        tune.run(lambda config: {"loss": 1.0}, config={"x": 1}, metric="loss",
                 mode="min", num_samples=1, use_ray=False, verbose=0)
        assert set(calls) == {"flaml-automl", "flaml-tune"}, calls
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_file_store_corruption_is_not_silently_skipped(tmp_path):
    pytest.importorskip("mlflow")
    import yaml
    from mlflow.store.tracking.file_store import FileStore

    (tmp_path / "meta.yaml").write_text("broken: [1,\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        FileStore._read_yaml(str(tmp_path), "meta.yaml", retries=0)


def test_local_file_store_retains_every_logged_trial(local_mlflow, monkeypatch):
    mlflow, module = local_mlflow
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": {"disable": True, "log_models": False}})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        model_history=False,
        starting_points={"rf": [{"n_estimators": 4}, {"n_estimators": 8}]},
    )
    with mlflow.start_run() as parent:
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
        expected = automl.mlflow_integration.manual_run_ids
        children = mlflow.tracking.MlflowClient().search_runs(
            [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
        )
    assert len(automl.mlflow_integration.infos) == len(expected) == 2
    assert {child.info.run_id for child in children} == set(expected)


def test_restore_failure_does_not_replace_training_error(caplog):
    import logging

    from flaml.automl.automl import _restore_mlflow_after_fit, logger

    caplog.set_level(logging.ERROR, logger=logger.name)

    def fail_training(self):
        raise ValueError("original training error")

    instance = SimpleNamespace(
        mlflow_integration=SimpleNamespace(resume_mlflow=Mock(side_effect=RuntimeError("restore failed")))
    )
    with pytest.raises(ValueError, match="original training error"):
        _restore_mlflow_after_fit(fail_training)(instance)
    assert any("restor" in record.message.lower() for record in caplog.records)


def test_restore_failure_after_success_remains_visible():
    from flaml.automl.automl import _restore_mlflow_after_fit

    instance = SimpleNamespace(
        mlflow_integration=SimpleNamespace(resume_mlflow=Mock(side_effect=RuntimeError("restore failed")))
    )
    with pytest.raises(RuntimeError, match="restore failed"):
        _restore_mlflow_after_fit(lambda self: "trained")(instance)


def test_usage_reporting_failure_is_nonfatal_and_visible(monkeypatch, caplog):
    module = importlib.import_module("flaml.fabric.telemetry")
    monkeypatch.setattr(module, "is_fabric_runtime", lambda: True)
    monkeypatch.setattr(module, "report_usage_telemetry", Mock(side_effect=RuntimeError("telemetry offline")))
    module.log_telemetry("test-activity")
    assert any("telemetry" in record.message.lower() for record in caplog.records)


def test_optional_diagnostic_logger_failure_is_nonfatal(monkeypatch, caplog):
    module = importlib.import_module("flaml.fabric.logger")
    factory = Mock(side_effect=RuntimeError("diagnostics offline"))
    monkeypatch.setattr(module, "no_synapse", False)
    monkeypatch.setattr(module, "get_mds_logger", factory, raising=False)
    monkeypatch.setattr(module, "default_scrubbers", [], raising=False)
    monkeypatch.setattr(module, "ArtifactNameScrubber", lambda: object(), raising=False)
    assert module.init_kusto_logger("test") is module.kusto_logger
    factory.assert_called_once()
    assert any("diagnostic" in record.message.lower() for record in caplog.records)


def test_signature_failure_diagnostics_do_not_include_sample_values(local_mlflow, monkeypatch):
    _, module = local_mlflow
    logger = Mock()
    monkeypatch.setattr(module, "kusto_logger", logger)
    monkeypatch.setattr(module.mlflow.models, "infer_signature", Mock(side_effect=ValueError("private-sample-value")))
    assert module.infer_signature(X_train=np.arange(6).reshape(3, 2), y_train=np.arange(3)) is None
    assert "private-sample-value" not in str(logger.mock_calls)


def test_automl_diagnostics_do_not_format_caller_objects(monkeypatch):
    from sklearn.model_selection import KFold

    import flaml.automl.automl as module
    from flaml.automl.model import RandomForestEstimator

    class PrivateMetric:
        def __repr__(self):
            return "private-metric-payload"

        def __call__(self, *args, **kwargs):
            return 0.5, {}

    class PrivateSplit(KFold):
        def __repr__(self):
            return "private-split-payload"

    logger = Mock()
    monkeypatch.setattr(module, "kusto_logger", logger)
    automl = AutoML(
        estimator_list=["private-learner-payload"],
        metric=PrivateMetric(),
        split_type=PrivateSplit(2),
        eval_method="cv",
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        entrypoint="code-first",
    )
    automl.add_learner("private-learner-payload", RandomForestEstimator)
    automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    assert "private-" not in str(logger.mock_calls)
    with pytest.raises(ValueError, match="entrypoint"):
        AutoML(entrypoint="private-entrypoint-payload")
    assert "private-entrypoint-payload" not in str(logger.mock_calls)


def test_telemetry_uses_normal_logging_without_an_extra_console_handler(public_environment):
    code = textwrap.dedent(
        """
        import logging
        from flaml.fabric import telemetry

        assert not telemetry.logger.handlers
        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)
        logging.getLogger().addHandler(Capture())
        telemetry.is_fabric_runtime = lambda: True
        def fail(*args, **kwargs):
            raise RuntimeError("diagnostic failure")
        telemetry.report_usage_telemetry = fail
        telemetry.log_telemetry("test")
        assert len([record for record in records if record.name == telemetry.logger.name]) == 1
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rejected_feature_trial_keeps_mlflow_history_complete(local_mlflow, monkeypatch):
    import pandas as pd
    from sklearn.model_selection import KFold

    mlflow, module = local_mlflow
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": {"disable": True, "log_models": True}})
    pipeline_log = Mock()
    monkeypatch.setattr(module.MLflowIntegration, "pickle_and_log_automl_artifacts", pipeline_log)
    monkeypatch.setattr(module.MLflowIntegration, "log_model", Mock())
    frame = pd.DataFrame({"feature": [0.0] * 10 + [1.0] * 10})
    settings = {"n_estimators": 4, "max_features": 1.0, "max_leaves": 4, "fe.extraction": "null"}
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        model_history=True,
        featurization="auto",
        auto_augment=False,
        eval_method="cv",
        split_type=KFold(2),
        starting_points={"rf": [{**settings, "fe.selection": "variance"}, {**settings, "fe.selection": "null"}]},
    )
    with mlflow.start_run():
        automl.fit(frame, np.tile([0, 1], 10))
    integration = automl.mlflow_integration
    assert len(integration.infos) == len(integration.manual_run_ids) == 2
    assert np.isinf(integration.infos[0]["metrics"]["validation_loss"])
    assert integration.best_run_id == integration.manual_run_ids[1]
    assert pipeline_log.call_count
    assert all(call.args[1] is not None for call in pipeline_log.call_args_list)


@pytest.mark.parametrize("fabric", [False, True])
def test_automl_logging_opt_out(local_mlflow, monkeypatch, fabric):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    automl_module = importlib.import_module("flaml.automl.automl")
    factory = Mock(side_effect=AssertionError("logging was disabled"))
    monkeypatch.setattr(automl_module, "MLflowIntegration", factory)
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, model_history=True)
    automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20), mlflow_logging=False, model_history=False)
    assert automl.mlflow_integration is None
    assert not automl._state.model_history
    factory.assert_not_called()


@pytest.mark.parametrize("fabric", [False, True])
@pytest.mark.parametrize("logging_enabled", [False, True])
def test_tune_logging_policy(local_mlflow, monkeypatch, fabric, logging_enabled):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    tune_module = importlib.import_module("flaml.tune.tune")
    factory = Mock(wraps=local_mlflow[1].MLflowIntegration)
    monkeypatch.setattr(tune_module, "MLflowIntegration", factory)
    monkeypatch.setattr(tune_module, "is_autolog_enabled", lambda: False, raising=False)
    analysis = tune.run(
        lambda config: {"loss": 1.0},
        config={"x": 1},
        metric="loss",
        mode="min",
        num_samples=1,
        use_ray=False,
        verbose=0,
        mlflow_logging=logging_enabled,
    )
    assert len(analysis.trials) == 1
    assert factory.call_count == int(fabric and logging_enabled)


@pytest.mark.parametrize("fabric,prefix", [(False, "flaml"), (True, "synapseml.flaml")])
@pytest.mark.parametrize("result,metric_tag", [({"loss": 0.1}, "['loss']"), (0.1, "loss")])
def test_mlflow_tag_namespace(local_mlflow, monkeypatch, fabric, prefix, result, metric_tag):
    from flaml.version import __version__

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    integration = local_mlflow[1].MLflowIntegration("tune")
    integration.record_trial(result, SimpleNamespace(config={"x": 1}), "loss")
    assert integration.infos[0]["tags"] == {
        f"{prefix}.best_run": False,
        f"{prefix}.iteration_number": 0,
        f"{prefix}.version": __version__,
        f"{prefix}.meric": metric_tag,
        f"{prefix}.run_source": "flaml-tune",
        f"{prefix}.log_type": integration.log_type,
    }


@pytest.mark.parametrize("fabric,prefix", [(False, "flaml"), (True, "synapseml.flaml")])
@pytest.mark.parametrize("experiment_type", ["automl", "tune"])
def test_active_runs_keep_logging_in_both_environments(local_mlflow, monkeypatch, fabric, prefix, experiment_type):
    mlflow, integration_module = local_mlflow
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    monkeypatch.setattr(integration_module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": {"log_models": False}})
    with mlflow.start_run():
        if experiment_type == "automl":
            result = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, model_history=False)
            result.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
        else:
            result = tune.run(
                lambda config: {"loss": 0.5},
                config={"x": 1},
                metric="loss",
                mode="min",
                num_samples=1,
                use_ray=False,
                verbose=0,
            )
        tags = mlflow.get_run(result.best_run_id).data.tags
        assert tags[f"{prefix}.best_run"] == "True"
        assert f"{prefix}.version" in tags
        assert tags[f"{prefix}.meric"] == (result._state.metric if experiment_type == "automl" else "['loss']")
        assert f"{prefix}.metric" not in tags


def test_corruption_surfaces_through_flaml_logging_client_path(local_mlflow, tmp_path):
    import yaml

    mlflow, module = local_mlflow
    with mlflow.start_run() as run:
        mlflow.log_metric("loss", 0.1)
        integration = module.MLflowIntegration("automl")
    integration.manual_run_ids = [run.info.run_id]
    metadata = tmp_path / "tracking" / run.info.experiment_id / run.info.run_id / "meta.yaml"
    assert metadata.is_file()
    metadata.write_text("broken: [1,\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        integration.log_automl(SimpleNamespace(_best_iteration=0))


@pytest.mark.parametrize("explicit_flavor", [False, True])
@pytest.mark.parametrize("active_parent", [False, True])
@pytest.mark.parametrize("training_failure", [False, True])
def test_real_global_autolog_registry_roundtrip(
    public_environment, tmp_path, explicit_flavor, active_parent, training_failure
):
    pytest.importorskip("mlflow")
    code = textwrap.dedent(
        """
        import copy
        import json
        import sys
        import mlflow
        import mlflow.sklearn
        import numpy as np
        from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
        from flaml import AutoML

        settings = json.loads(sys.argv[1])
        mlflow.autolog(log_models=False, silent=True)
        if settings["flavor"]:
            mlflow.sklearn.autolog(log_models=False, silent=True, log_input_examples=False)
        before = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
        parent = mlflow.start_run() if settings["parent"] else None
        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, model_history=True)
        if settings["failure"]:
            def fail():
                raise RuntimeError("expected training failure")
            automl._search = fail
        try:
            try:
                automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
            except RuntimeError as error:
                assert settings["failure"] and str(error) == "expected training failure"
            else:
                assert not settings["failure"]
                integration = automl.mlflow_integration
                assert integration.manual_run_ids
                client = mlflow.tracking.MlflowClient()
                for run_id in integration.manual_run_ids:
                    child = client.get_run(run_id)
                    if parent is not None:
                        assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
                    assert all(item.path != "model" for item in client.list_artifacts(run_id))
                if parent is not None:
                    assert all(item.path != "model" for item in client.list_artifacts(parent.info.run_id))
            assert AUTOLOGGING_INTEGRATIONS == before, (before, AUTOLOGGING_INTEGRATIONS)
        finally:
            mlflow.end_run()
        """
    )
    import json

    environment = {**os.environ, "MLFLOW_TRACKING_URI": (tmp_path / "tracking").as_uri()}
    environment.pop("MLFLOW_EXPERIMENT_ID", None)
    environment.pop("MLFLOW_EXPERIMENT_NAME", None)
    arguments = json.dumps({"flavor": explicit_flavor, "parent": active_parent, "failure": training_failure})
    result = subprocess.run(
        [sys.executable, "-c", code, arguments], env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_best_run_tag_waits_for_trial_metadata(monkeypatch):
    module = importlib.import_module("flaml.fabric.mlflow")
    trial_write = Future()
    integration = module.MLflowIntegration.__new__(module.MLflowIntegration)
    integration.autolog = False
    integration.manual_log = True
    integration.parent_run_id = None
    integration.manual_run_ids = ["winner"]
    integration.resume_params = {}
    integration.futures = {trial_write: "trial metadata"}
    integration._tag_prefix = "flaml"
    integration.mlflow_client = Mock()
    integration.mlflow_client.get_run.return_value.info.run_name = "winner-name"

    def complete_trial_writes(futures):
        assert trial_write in futures
        trial_write.set_result(None)
        return {trial_write}, set()

    def set_tag(*args):
        assert trial_write.done(), "Pending trial metadata can overwrite the winning tag"

    monkeypatch.setattr(module, "wait", complete_trial_writes)
    integration.mlflow_client.set_tag.side_effect = set_tag
    automl = SimpleNamespace(best_iteration=0, _best_iteration=0)
    integration.log_automl(automl)
    integration.mlflow_client.set_tag.assert_called_once_with("winner", "flaml.best_run", True)


@pytest.mark.parametrize("global_autolog", [False, True])
def test_flavor_autolog_configuration_is_restored(public_environment, monkeypatch, global_autolog):
    module = importlib.import_module("flaml.fabric.mlflow")
    flavor_settings = {"disable": False, "log_models": False, "silent": True}
    global_settings = {"disable": False, "log_models": True, "silent": True}
    registry = {"sklearn": flavor_settings.copy(), "pyspark.ml": {}}
    monkeypatch.setitem(sys.modules, "mlflow.pyspark.ml", None)
    if global_autolog:
        registry["mlflow"] = global_settings.copy()
        registry["sklearn"]["globally_configured"] = False
        registry["transformers"] = {"disable": False, "globally_configured": True}
        monkeypatch.setitem(sys.modules, "mlflow.transformers", None)
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", registry)
    monkeypatch.setattr(module, "is_autolog_enabled", lambda: True)
    monkeypatch.setattr(module.mlflow, "active_run", lambda: None)
    global_hook = Mock()
    flavor_hook = Mock()
    monkeypatch.setattr(module.mlflow, "autolog", global_hook)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", SimpleNamespace(autolog=flavor_hook))
    integration = module.MLflowIntegration.__new__(module.MLflowIntegration)
    integration.experiment_type = "automl"
    integration.resume_params = {}
    try:
        integration.update_autolog_state()
        assert not integration._do_log_model
        integration.resume_mlflow()
        if global_autolog:
            assert global_hook.call_args_list[-1] == call(**global_settings)
            assert flavor_hook.call_args_list == [call(**flavor_settings)]
            assert registry["sklearn"]["globally_configured"] is False
        else:
            global_hook.assert_not_called()
            assert flavor_hook.call_args_list == [
                call(**{**flavor_settings, "disable": True}),
                call(**flavor_settings),
            ]
        count = flavor_hook.call_count
        integration.resume_mlflow()
        assert flavor_hook.call_count == count
    finally:
        integration.resume_params = {}
        integration.resume_flavor_params = {}


@pytest.mark.parametrize("best_run_id", [None, "metrics-only-child"])
@pytest.mark.parametrize("experiment_name", [None, "experiment"])
def test_registration_uses_the_current_pipeline_artifact(local_mlflow, monkeypatch, best_run_id, experiment_name):
    mlflow, module = local_mlflow
    pipeline = object()
    automl = SimpleNamespace(
        automl_pipeline=pipeline,
        best_run_id=best_run_id,
        _state=SimpleNamespace(model_history=False),
        _mlflow_exp_name=experiment_name,
    )
    artifact = SimpleNamespace(model_uri="runs:/registration/current-pipeline")
    version = object()
    log_model = Mock(return_value=artifact)
    register_model = Mock(return_value=version)
    monkeypatch.setattr(
        module,
        "mlflow",
        SimpleNamespace(
            sklearn=SimpleNamespace(log_model=log_model),
            register_model=register_model,
            get_run=Mock(return_value=SimpleNamespace(info=SimpleNamespace(run_id=best_run_id))),
            search_model_versions=Mock(side_effect=AssertionError("must not select old versions")),
        ),
    )
    result = module.register_automl_pipeline(automl, artifact_path="current-pipeline")
    log_model.assert_called_once_with(pipeline, "current-pipeline", signature=None)
    register_model.assert_called_once_with(artifact.model_uri, (experiment_name or "flaml") + "_pipeline")
    assert result is version


@pytest.mark.parametrize("flavor", ["sklearn", "spark"])
def test_automatic_pipeline_logging_respects_model_optout(monkeypatch, flavor):
    module = importlib.import_module("flaml.fabric.mlflow")
    integration = module.MLflowIntegration.__new__(module.MLflowIntegration)
    integration._do_log_model = False
    integration.resume_params = {}
    no_logging = Mock(side_effect=AssertionError("model logging is disabled"))
    monkeypatch.setattr(module.mlflow, "active_run", no_logging)
    assert integration._log_pipeline(object(), flavor, "model", None, "parent", "rf") is None
    no_logging.assert_not_called()


def test_automl_artifact_logging_respects_model_optout(monkeypatch):
    module = importlib.import_module("flaml.fabric.mlflow")
    integration = module.MLflowIntegration.__new__(module.MLflowIntegration)
    integration._do_log_model = False
    integration.resume_params = {}
    no_logging = Mock(side_effect=AssertionError("model logging is disabled"))
    monkeypatch.setattr(integration, "_log_pipeline", no_logging)
    automl = SimpleNamespace(feature_transformer=None)
    model = SimpleNamespace(autofe=None)
    assert integration.pickle_and_log_automl_artifacts(automl, model, "rf", run_id="parent") is None
    no_logging.assert_not_called()


def test_disabled_global_autolog_preserves_artifact_optout(local_mlflow, monkeypatch):
    mlflow, module = local_mlflow
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": {"disable": True, "log_models": False}})
    with mlflow.start_run():
        integration = module.MLflowIntegration("automl")
        assert integration.manual_log
        assert not integration._do_log_model


def test_automl_restores_autolog_when_search_raises(local_mlflow, monkeypatch):
    mlflow, module = local_mlflow
    settings = {"disable": False, "log_models": False, "silent": True}
    monkeypatch.setattr(module, "AUTOLOGGING_INTEGRATIONS", {"mlflow": settings.copy()})
    monkeypatch.setattr(module, "is_autolog_enabled", lambda: True)
    monkeypatch.setattr("flaml.automl.automl.is_autolog_enabled", lambda: True)
    autolog = Mock()
    monkeypatch.setattr(mlflow, "autolog", autolog)
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0)
    monkeypatch.setattr(automl, "_search", Mock(side_effect=RuntimeError("training failed")))
    with pytest.raises(RuntimeError, match="training failed"):
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    assert autolog.call_args_list == [
        call(disable=True, silent=True, log_models=False),
        call(**settings),
    ]


def test_flavor_autolog_and_registered_pipeline_roundtrip(public_environment, tmp_path):
    pytest.importorskip("mlflow")
    code = textwrap.dedent(
        """
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
        from flaml import AutoML
        from flaml.automl import register_automl_pipeline

        AUTOLOGGING_INTEGRATIONS.pop("mlflow", None)
        mlflow.sklearn.autolog(log_models=False, silent=True)
        before = AUTOLOGGING_INTEGRATIONS["sklearn"].copy()
        X = pd.DataFrame(np.random.RandomState(42).normal(size=(60, 3)), columns=["a", "b", "c"])
        y = (X["a"] > 0).astype(int)
        starting_points = {"rf": {
            "n_estimators": 4, "max_features": 0.6, "max_leaves": 4,
            "fe.selection": "cardinality", "fe.categorical": "ordinal", "fe.extraction": "LDA",
        }}
        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0,
                       featurization="force", starting_points=starting_points)
        with mlflow.start_run() as parent:
            automl.fit(X, y)
            client = mlflow.tracking.MlflowClient()
            for run_id in [parent.info.run_id, *automl.mlflow_integration.manual_run_ids]:
                assert all(artifact.path != "model" for artifact in client.list_artifacts(run_id))
        assert AUTOLOGGING_INTEGRATIONS["sklearn"] == before
        assert "mlflow" not in AUTOLOGGING_INTEGRATIONS
        assert not automl._state.model_history
        assert automl.model.autofe is not None
        with mlflow.start_run():
            version = register_automl_pipeline(automl, model_name="current-pipeline")
            restored = mlflow.sklearn.load_model(f"models:/current-pipeline/{version.version}")
            np.testing.assert_array_equal(restored.predict(X), automl.automl_pipeline.predict(X))
        mlflow.sklearn.autolog(disable=True)
        plain = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0,
                       mlflow_logging=False, featurization="force", starting_points=starting_points)
        plain.fit(X, y)
        assert not hasattr(plain, "pipeline_signature")
        with mlflow.start_run():
            version = register_automl_pipeline(plain)
            assert version.name == "flaml_pipeline"
            restored = mlflow.sklearn.load_model(f"models:/flaml_pipeline/{version.version}")
            np.testing.assert_array_equal(restored.predict(X), plain.predict(X))
        """
    )
    environment = {**os.environ, "MLFLOW_TRACKING_URI": (tmp_path / "tracking").as_uri()}
    environment.pop("MLFLOW_EXPERIMENT_ID", None)
    environment.pop("MLFLOW_EXPERIMENT_NAME", None)
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
