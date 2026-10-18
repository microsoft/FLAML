import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from flaml import AutoML, tune
from flaml.version import __version__


@pytest.fixture
def telemetry_reporter(monkeypatch):
    module = importlib.import_module("flaml.fabric.telemetry")
    reporter = Mock()
    monkeypatch.setattr(module, "report_usage_telemetry", reporter)
    return module, reporter


@pytest.mark.parametrize("fabric_runtime", [False, True])
def test_automl_telemetry(monkeypatch, telemetry_reporter, fabric_runtime):
    module, reporter = telemetry_reporter
    monkeypatch.setattr(module, "is_fabric_runtime", lambda: fabric_runtime)
    monkeypatch.setattr("flaml.automl.automl.is_log_telemetry", True)
    monkeypatch.setattr("flaml.automl.automl.internal_mlflow", True)
    X = np.arange(40).reshape(20, 2)
    y = np.tile([0, 1], 10)
    for _ in range(2):
        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
        automl.fit(X, y)
    if fabric_runtime:
        reporter.assert_called_once_with(
            "PyLibraryImport",
            "flaml-automl",
            attributes={"version": __version__, "ImportType": "EXPLICIT_IMPORTED_BY_USER"},
        )
    else:
        reporter.assert_not_called()


@pytest.mark.parametrize("fabric_runtime", [False, True])
def test_tune_telemetry(monkeypatch, telemetry_reporter, fabric_runtime):
    module, reporter = telemetry_reporter
    monkeypatch.setattr(module, "is_fabric_runtime", lambda: fabric_runtime)
    monkeypatch.setattr("flaml.tune.tune.is_log_telemetry_tune", True)
    monkeypatch.setattr("flaml.tune.tune.internal_mlflow", True)
    for exponent in (2, 3):
        tune.run(
            lambda config: {"metric": config["x"] ** exponent},
            config={"x": 1},
            num_samples=1,
            metric="metric",
            mode="min",
            mlflow_logging=False,
            verbose=0,
        )
    if fabric_runtime:
        reporter.assert_called_once_with(
            "PyLibraryImport",
            "flaml-tune",
            attributes={"version": __version__, "ImportType": "EXPLICIT_IMPORTED_BY_USER"},
        )
    else:
        reporter.assert_not_called()


def test_tune_central_logs_do_not_include_user_configuration(monkeypatch):
    module = importlib.import_module("flaml.tune.tune")
    central_logger = Mock()
    monkeypatch.setattr(module, "kusto_logger", central_logger)
    private_value = "private-configuration-value"
    analysis = tune.run(
        lambda config: {"metric": 1.0},
        config={"api_key": private_value},
        extra_tag={"user_tag": private_value},
        mlflow_exp_name=private_value,
        num_samples=1,
        metric="metric",
        mode="min",
        mlflow_logging=False,
        verbose=0,
    )
    assert len(analysis.trials) == 1
    central_logger.info.assert_called_once()
    assert private_value not in str(central_logger.mock_calls)
    assert "api_key" not in str(central_logger.mock_calls)
    assert "user_tag" not in str(central_logger.mock_calls)
