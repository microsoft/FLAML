import importlib
import os
import subprocess
import sys
import textwrap
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
def test_mlflow_tag_namespace(local_mlflow, monkeypatch, fabric, prefix):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", str(fabric))
    integration = local_mlflow[1].MLflowIntegration("tune")
    integration.record_trial({"loss": 0.1}, SimpleNamespace(config={"x": 1}), "loss")
    assert f"{prefix}.version" in integration.infos[0]["tags"]
    assert f"{prefix}.best_run" in integration.infos[0]["tags"]
    assert all(key.startswith(prefix + ".") for key in integration.infos[0]["tags"])


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
        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, featurization="force")
        automl.fit(X, y)
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
                       mlflow_logging=False, featurization="force")
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
