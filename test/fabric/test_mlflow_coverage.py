"""Tests to improve coverage for flaml/fabric/mlflow.py.

Covers exception handling, version-specific paths, synapse integration,
requirements management, model logging/loading, and helper functions.
"""

import json
import os
import pickle
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import mlflow
import pytest
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from sklearn import tree
from sklearn.pipeline import Pipeline


@pytest.fixture(autouse=True)
def _end_mlflow_runs():
    """Ensure no MLflow run leaks between tests."""
    yield
    while mlflow.active_run():
        mlflow.end_run()


def _mock_active_run_none_then_mock(run_id="r1"):
    """Return None first (no active run), then a mock for subsequent calls."""
    mock = MagicMock()
    mock.info.run_id = run_id
    first = [True]

    def side_effect():
        if first[0]:
            first[0] = False
            return None
        return mock

    return side_effect


# ---------------------------------------------------------------------------
# SparkPipelineModel fallback import (lines 24-28)
# ---------------------------------------------------------------------------
class TestSparkPipelineModelFallback:
    def test_fallback_class_defined_when_pyspark_missing(self):
        """When pyspark is not installed the stub SparkPipelineModel is used.

        Run the import in a fresh subprocess so we don't pollute the global
        module state of the parent test process. Using importlib.reload here
        previously caused class-identity mismatches in unrelated AutoML tests
        when they ran later in the same worker (e.g. under pytest-xdist),
        because joblib/pickle would look up
        flaml.fabric.mlflow.MLflowIntegration and find the post-reload class
        object, while existing AutoML instances still referenced the
        pre-reload one.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            sys.modules['pyspark.ml'] = None  # force ImportError on `from pyspark.ml import ...`
            import flaml.fabric.mlflow as mod
            assert mod.SparkPipelineModel is not None, "stub SparkPipelineModel not defined"
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"subprocess failed (rc={result.returncode}):\n" f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# convert_requirement (lines 77, 82)
# ---------------------------------------------------------------------------
class TestConvertRequirement:
    def test_convert_requirement_old_mlflow(self):
        from flaml.fabric.mlflow import convert_requirement

        with patch("flaml.fabric.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.16.0"
            result = convert_requirement(["numpy>=1.0", "pandas"])
            assert len(result) == 2

    def test_convert_requirement_new_mlflow(self):
        from flaml.fabric.mlflow import convert_requirement

        with patch("flaml.fabric.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.18.0"
            result = convert_requirement(["numpy>=1.0", "pandas"])
            assert result == ["numpy>=1.0", "pandas"]


# ---------------------------------------------------------------------------
# time_it decorator (lines 117-123)
# ---------------------------------------------------------------------------
class TestTimeIt:
    def test_time_it_with_none_returns_decorator(self):
        from flaml.fabric.mlflow import time_it

        decorator = time_it(None)
        assert callable(decorator)

        @decorator
        def dummy():
            return 42

        assert dummy() == 42

    def test_time_it_with_string_code(self):
        from flaml.fabric.mlflow import time_it

        # Execute a simple code string
        time_it("x = 1 + 1")


# ---------------------------------------------------------------------------
# get_mlflow_log_latency (lines 144-147)
# ---------------------------------------------------------------------------
class TestGetMlflowLogLatency:
    def test_invalid_env_var_returns_zero_fallback(self):
        from flaml.fabric.mlflow import get_mlflow_log_latency

        with patch.dict(os.environ, {"FLAML_MLFLOW_LOG_LATENCY": "not_a_number"}):
            # ValueError path -> FLAML_MLFLOW_LOG_LATENCY = 0, falls through
            result = get_mlflow_log_latency(model_history=False, delete_run=True)
            assert isinstance(result, float)

    def test_high_latency_env_var_returns_early(self):
        from flaml.fabric.mlflow import get_mlflow_log_latency

        with patch.dict(os.environ, {"FLAML_MLFLOW_LOG_LATENCY": "5.0"}):
            result = get_mlflow_log_latency()
            assert result == 5.0


# ---------------------------------------------------------------------------
# update_and_install_requirements (lines 213, 218, 238, 251, 258)
# ---------------------------------------------------------------------------
class TestUpdateAndInstallRequirements:
    def test_raises_without_run_id_or_model_info(self):
        from flaml.fabric.mlflow import update_and_install_requirements

        with pytest.raises(ValueError, match="Please provide"):
            update_and_install_requirements()

    def test_with_model_name_and_version(self):
        from flaml.fabric.mlflow import update_and_install_requirements

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir, exist_ok=True)
            req_path = os.path.join(model_dir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("numpy>=1.0\nscikit-learn>=0.24\nflaml>=1.0\npyspark>=3.0\n")

            mock_client = MagicMock()
            mock_client.get_model_version.return_value = SimpleNamespace(run_id="fake_run_id")
            mock_client.download_artifacts.side_effect = lambda *a, **kw: None

            with patch("flaml.fabric.mlflow.mlflow") as mock_mlflow:
                mock_mlflow.MlflowClient.return_value = mock_client
                result = update_and_install_requirements(model_name="test_model", model_version="1", dst_path=tmpdir)
            assert result == req_path
            with open(req_path) as f:
                content = f.read()
            assert "flaml" not in content
            assert "pyspark" not in content

    def test_install_with_ipython(self):
        from flaml.fabric.mlflow import update_and_install_requirements

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir, exist_ok=True)
            req_path = os.path.join(model_dir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("numpy>=1.0\n")

            mock_client = MagicMock()
            mock_client.download_artifacts.side_effect = lambda *a, **kw: None

            mock_ipython = MagicMock()
            with patch("flaml.fabric.mlflow.mlflow") as mock_mlflow, patch.dict(
                "sys.modules", {"IPython": MagicMock(get_ipython=lambda: mock_ipython)}
            ):
                mock_mlflow.MlflowClient.return_value = mock_client
                result = update_and_install_requirements(run_id="fake_run", dst_path=tmpdir, install_with_ipython=True)
            assert result == req_path
            mock_ipython.run_line_magic.assert_called_once()


# ---------------------------------------------------------------------------
# _mlflow_wrapper (lines 267-272, 279)
# ---------------------------------------------------------------------------
class TestMlflowWrapper:
    def test_wrapper_with_synapse_config(self):
        from flaml.fabric.mlflow import _mlflow_wrapper

        mock_func = MagicMock(return_value="result")
        mock_config = MagicMock()

        mock_set_config = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "synapse": MagicMock(),
                "synapse.ml": MagicMock(),
                "synapse.ml.mlflow": MagicMock(set_mlflow_env_config=mock_set_config),
            },
        ):
            wrapped = _mlflow_wrapper(mock_func, None, mlflow_config=mock_config, autolog=False)
            result = wrapped("arg1")
        assert result == "result"
        mock_func.assert_called_once_with("arg1")

    def test_wrapper_synapse_import_fails(self):
        from flaml.fabric.mlflow import _mlflow_wrapper

        mock_func = MagicMock(return_value="ok")
        wrapped = _mlflow_wrapper(mock_func, None, mlflow_config="some_config", autolog=False)
        # synapse.ml.mlflow not available, exception caught
        result = wrapped()
        assert result == "ok"


# ---------------------------------------------------------------------------
# _get_notebook_name (lines 293-306)
# ---------------------------------------------------------------------------
class TestGetNotebookName:
    def test_success_path(self):
        from flaml.fabric.mlflow import _get_notebook_name

        mock_config = MagicMock()
        mock_config.artifact_id = "test_id"
        mock_artifact = MagicMock()
        mock_artifact.displayName = "My Notebook! (v2)"

        with patch.dict(
            "sys.modules",
            {
                "synapse": MagicMock(),
                "synapse.ml": MagicMock(),
                "synapse.ml.mlflow": MagicMock(get_mlflow_env_config=MagicMock(return_value=mock_config)),
                "synapse.ml.mlflow.shared_platform_utils": MagicMock(
                    get_artifact=MagicMock(return_value=mock_artifact)
                ),
            },
        ):
            result = _get_notebook_name()
        assert result == "My-Notebook-v2-"

    def test_failure_returns_none(self):
        from flaml.fabric.mlflow import _get_notebook_name

        # No synapse module -> exception -> None
        result = _get_notebook_name()
        assert result is None


# ---------------------------------------------------------------------------
# MLflowIntegration.__init__ synapse path (lines 321-323)
# ---------------------------------------------------------------------------
class TestMLflowIntegrationInit:
    def test_init_with_synapse(self):
        from flaml.fabric.mlflow import MLflowIntegration

        mock_config = MagicMock()
        mock_set = MagicMock()
        mock_get = MagicMock(return_value=mock_config)

        with patch.dict(
            "sys.modules",
            {
                "synapse": MagicMock(),
                "synapse.ml": MagicMock(),
                "synapse.ml.mlflow": MagicMock(
                    get_mlflow_env_config=mock_get,
                    set_mlflow_env_config=mock_set,
                ),
                "synapse.ml.mlflow.shared_platform_utils": MagicMock(
                    get_artifact=MagicMock(return_value=MagicMock(displayName="nb"))
                ),
            },
        ), patch("flaml.fabric.mlflow.is_fabric_runtime", return_value=True):
            with mlflow.start_run():
                integration = MLflowIntegration()
                assert integration._on_internal is True
                assert integration.driver_mlflow_env_config == mock_config

    def test_init_unknown_experiment_type_raises(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            with pytest.raises(ValueError, match="Unknown experiment type"):
                MLflowIntegration(experiment_type="unknown")

    def test_init_empty_parent_run_name(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run(run_name=""):
            integration = MLflowIntegration()
            assert integration.parent_run_name is not None


# ---------------------------------------------------------------------------
# set_mlflow_config (lines 381-387)
# ---------------------------------------------------------------------------
class TestSetMlflowConfig:
    def test_set_mlflow_config_with_synapse(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.driver_mlflow_env_config = MagicMock()

        mock_set = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "synapse": MagicMock(),
                "synapse.ml": MagicMock(),
                "synapse.ml.mlflow": MagicMock(set_mlflow_env_config=mock_set),
            },
        ):
            integration.set_mlflow_config()
        mock_set.assert_called_once()

    def test_set_mlflow_config_synapse_fails(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.driver_mlflow_env_config = MagicMock()
        # No synapse module -> exception caught silently
        integration.set_mlflow_config()


# ---------------------------------------------------------------------------
# copy_mlflow_run exception paths (lines 436-437, 444, 453)
# ---------------------------------------------------------------------------
class TestCopyMlflowRun:
    def test_copy_run_param_exception_suppressed(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()

        mock_run = MagicMock()
        mock_run.data.params = {"p1": "v1"}
        mock_run.data.metrics = {}
        mock_run.data.tags = {}
        integration.mlflow_client.get_run = MagicMock(return_value=mock_run)
        integration.mlflow_client.log_param = MagicMock(side_effect=MlflowException("dup"))

        result = integration.copy_mlflow_run("src_id", "target_id", components=["param"])
        assert "Successfully" in result

    def test_copy_run_no_metric_no_tag(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()

        mock_run = MagicMock()
        mock_run.data.params = {}
        mock_run.data.metrics = {"m1": 0.5}
        mock_run.data.tags = {"flaml.test": "val"}
        integration.mlflow_client.get_run = MagicMock(return_value=mock_run)

        result = integration.copy_mlflow_run("s", "t", components=[])
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# log_model paths (lines 509, 525, 537-545, 550-563)
# ---------------------------------------------------------------------------
class TestLogModel:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_log_model_do_log_model_false(self):
        integration = self._make_integration()
        integration._do_log_model = False
        result = integration.log_model(MagicMock(), "xgboost", run_id="r1")
        assert result is None

    def test_log_model_sklearn(self):
        integration = self._make_integration()
        with mlflow.start_run() as run:
            integration.parent_run_id = run.info.run_id
            with patch.object(mlflow.sklearn, "log_model"):
                with patch.object(mlflow.models.model, "update_model_requirements"):
                    result = integration.log_model(tree.DecisionTreeClassifier(), "xgboost", run_id=run.info.run_id)
        assert "Successfully" in result

    def test_log_model_lgbm(self):
        integration = self._make_integration()
        integration.parent_run_id = "other_id"
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.lightgbm.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "lgbm", run_id="r1")
        assert "Successfully" in result

    def test_log_model_spark(self):
        integration = self._make_integration()
        mock_run = MagicMock()
        mock_run.info.run_id = "spark_run"
        integration.parent_run_id = "spark_run"  # match parent_run_id to trigger first branch
        with patch.object(mlflow, "active_run", return_value=mock_run):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.spark.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "lgbm_spark", run_id="r1")
        assert "Successfully" in result

    def test_log_model_wrong_active_run(self):
        integration = self._make_integration()
        mock_run = MagicMock()
        mock_run.info.run_id = "active_id"
        integration.parent_run_id = "parent_id"
        with patch.object(mlflow, "active_run", return_value=mock_run):
            with patch("mlflow.sklearn.log_model"):
                with patch.object(mlflow.models.model, "update_model_requirements"):
                    result = integration.log_model(MagicMock(), "xgboost", run_id="target_id")
        assert "Error" in result

    def test_log_model_transformer(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.transformers.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "transformer", run_id="r1")
        assert "Successfully" in result

    def test_log_model_statsmodels(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.statsmodels.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "arima", run_id="r1")
        assert "Successfully" in result

    def test_log_model_pytorch(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.pytorch.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "tcn", run_id="r1")
        assert "Successfully" in result

    def test_log_model_prophet(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.prophet.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration.log_model(MagicMock(), "prophet", run_id="r1")
        assert "Successfully" in result

    def test_log_model_orbit_unsupported(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch.object(mlflow.models.model, "update_model_requirements"):
                        result = integration.log_model(MagicMock(), "orbit", run_id="r1")
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# _pickle_and_log_artifact (lines 578-590)
# ---------------------------------------------------------------------------
class TestPickleAndLogArtifact:
    def test_pickle_success(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.mlflow_client.log_artifact = MagicMock()
        result = integration._pickle_and_log_artifact({"key": "val"}, "test_artifact", run_id="r1")
        assert result is True

    def test_pickle_failure(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.mlflow_client.log_artifact = MagicMock(side_effect=Exception("fail"))
        result = integration._pickle_and_log_artifact({"key": "val"}, "test_artifact", run_id="r1")
        assert result is False

    def test_pickle_skip_when_no_log_model(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration._do_log_model = False
        result = integration._pickle_and_log_artifact({"key": "val"}, "test_artifact", run_id="r1")
        assert result is True


# ---------------------------------------------------------------------------
# _log_pipeline (lines 604-610, 616-619)
# ---------------------------------------------------------------------------
class TestLogPipeline:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_log_pipeline_wrong_active_run(self):
        integration = self._make_integration()
        mock_run = MagicMock()
        mock_run.info.run_id = "wrong_id"
        integration.parent_run_id = "parent_id"
        with patch.object(mlflow, "active_run", return_value=mock_run):
            with patch.object(mlflow.models.model, "update_model_requirements"):
                result = integration._log_pipeline(MagicMock(), "sklearn", "model", None, "target_id", "xgboost")
        assert "Error" in result

    def test_log_pipeline_no_active_run(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.sklearn.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration._log_pipeline(MagicMock(), "sklearn", "model", None, "r1")
        assert "Successfully" in result

    def test_log_pipeline_spark_flavor(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch("mlflow.spark.log_model"):
                        with patch.object(mlflow.models.model, "update_model_requirements"):
                            result = integration._log_pipeline(MagicMock(), "spark", "model", None, "r1")
        assert "Successfully" in result

    def test_log_pipeline_unsupported_flavor(self):
        integration = self._make_integration()
        with patch.object(mlflow, "active_run", side_effect=_mock_active_run_none_then_mock("r1")):
            with patch.object(mlflow, "start_run"):
                with patch.object(mlflow, "end_run"):
                    with patch.object(mlflow.models.model, "update_model_requirements"):
                        result = integration._log_pipeline(MagicMock(), "unknown_flavor", "model", None, "r1")
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# pickle_and_log_automl_artifacts (lines 650, 656-657, 665-674)
# ---------------------------------------------------------------------------
class TestPickleAndLogAutomlArtifacts:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_spark_estimator_returns_early(self):
        integration = self._make_integration()
        automl = MagicMock()
        result = integration.pickle_and_log_automl_artifacts(automl, MagicMock(), "lgbm_spark", run_id="r1")
        assert result is None

    def test_spark_pipeline_model_path(self):
        from flaml.fabric.mlflow import SparkPipelineModel

        integration = self._make_integration()
        automl = MagicMock()
        mock_spark_pipeline = MagicMock(spec=SparkPipelineModel)
        mock_spark_pipeline.stages = []
        automl.feature_transformer = mock_spark_pipeline

        model = MagicMock()
        with patch.object(integration, "_log_pipeline", return_value="ok"):
            integration.pickle_and_log_automl_artifacts(automl, model, "lgbm_spark_fake", run_id="r1")

    def test_no_feature_transformer_no_spark(self):
        integration = self._make_integration()
        automl = MagicMock()
        automl.feature_transformer = "not_a_pipeline"  # neither Pipeline nor SparkPipelineModel

        model = MagicMock()
        model.autofe = MagicMock()  # has autofe

        with patch.object(integration, "_log_pipeline", return_value="ok"):
            result = integration.pickle_and_log_automl_artifacts(automl, model, "xgboost", run_id="r1")
        assert "Successfully" in result

    def test_no_feature_transformer_no_autofe(self):
        integration = self._make_integration()
        automl = MagicMock()
        automl.feature_transformer = "not_a_pipeline"

        model = MagicMock()
        model.autofe = None

        with patch.object(integration, "_log_pipeline", return_value="ok"):
            result = integration.pickle_and_log_automl_artifacts(automl, model, "xgboost", run_id="r1")
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# log_automl autolog path (lines 799-801, 808, 813-814, 821, 824, 826)
# ---------------------------------------------------------------------------
class TestLogAutoml:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_log_automl_autolog_with_parent_and_model(self):
        integration = self._make_integration()
        integration.autolog = True
        integration.manual_log = False
        integration.has_model = False
        integration.has_summary = False

        automl = MagicMock()
        automl.best_iteration = 0
        automl._best_iteration = 0
        automl._state.best_loss = 0.1
        automl._trained_estimator = MagicMock()
        automl._trained_estimator._model = MagicMock()
        automl.best_estimator = "xgboost"
        automl.model = MagicMock()
        automl.pipeline_signature = None
        automl.estimator_signature = None

        with patch.object(mlflow, "start_run"), patch.object(mlflow, "log_metrics"), patch.object(
            integration, "adopt_children"
        ), patch.object(integration, "pickle_and_log_automl_artifacts"):
            integration.log_automl(automl)
        assert integration.has_model is True

    def test_log_automl_autolog_spark_model(self):
        integration = self._make_integration()
        integration.autolog = True
        integration.manual_log = False
        integration.has_model = False

        automl = MagicMock()
        automl.best_iteration = 0
        automl._best_iteration = 0
        automl._state.best_loss = 0.1
        automl._trained_estimator = MagicMock()
        automl._trained_estimator._model = MagicMock()
        automl.best_estimator = "lgbm_spark"
        automl.estimator_signature = None

        with patch.object(mlflow, "start_run"), patch.object(mlflow, "log_metrics"), patch.object(
            integration, "adopt_children"
        ), patch.object(integration, "log_model"):
            integration.log_automl(automl)
        assert integration.has_model is True


# ---------------------------------------------------------------------------
# log_automl manual_log spark path (lines 861, 869)
# ---------------------------------------------------------------------------
class TestLogAutomlManualSpark:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_manual_log_spark_best_model(self):
        integration = self._make_integration()
        integration.autolog = False
        integration.manual_log = True
        integration.has_model = False
        integration.has_summary = False
        integration.manual_run_ids = ["child_run_1"]
        integration.futures_log_model = {}

        mock_child_run = MagicMock()
        mock_child_run.info.run_name = "child"
        integration.mlflow_client = MagicMock()
        integration.mlflow_client.get_run = MagicMock(return_value=mock_child_run)

        automl = MagicMock()
        automl.best_iteration = 0
        automl._best_iteration = 0
        automl._best_estimator = "lgbm_spark"
        automl.best_estimator = "lgbm_spark"
        automl._config_history = {0: ("learner", {"ml": {"n_estimators": 100}})}
        automl._trained_estimator = MagicMock()
        automl._trained_estimator._model = MagicMock()
        automl.estimator_signature = None

        with patch.object(integration, "copy_mlflow_run"), patch.object(
            integration, "_log_automl_configurations"
        ), patch.object(integration, "log_model"):
            integration.log_automl(automl)
        assert integration.has_model is True


# ---------------------------------------------------------------------------
# _log_automl_configurations (lines 890, 895, 900)
# ---------------------------------------------------------------------------
class TestLogAutomlConfigurations:
    def test_log_automl_configurations(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.automl_user_configurations = '{"task": "classification"}'
        integration.automl_display_configurations = '{"metric": "accuracy"}'
        integration.mlflow_client.log_text = MagicMock()

        result = integration._log_automl_configurations("run_id")
        assert "Successfully" in result
        assert integration.mlflow_client.log_text.call_count == 2


# ---------------------------------------------------------------------------
# _log_info_to_run with submetrics (lines 915-921)
# ---------------------------------------------------------------------------
class TestLogInfoToRun:
    def test_log_info_with_submetrics(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()

        info = {
            "metrics": {"m1": 0.5},
            "tags": {"flaml.tag": "v1"},
            "params": {"learner": "lgbm", "sample_size": 100},
            "submetrics": {
                "iter_counter": 1,
                "values": [{"sub_metric_1": 0.3}],
            },
        }
        integration.mlflow_client.log_batch = MagicMock()
        with patch.object(mlflow, "start_run") as mock_start:
            mock_run = MagicMock()
            mock_run.__enter__ = MagicMock(return_value=MagicMock(info=MagicMock(run_id="sub_run_id")))
            mock_run.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = mock_run
            result = integration._log_info_to_run(info, "run_id", log_params=True)
        assert "Successfully" in result


# ---------------------------------------------------------------------------
# adopt_children (lines 933, 958, 961-962, 975, 986-987, 1002, 1021-1028)
# ---------------------------------------------------------------------------
class TestAdoptChildren:
    def _make_integration(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        return integration

    def test_adopt_children_with_empty_run_deletion(self):
        integration = self._make_integration()
        integration.autolog = True
        integration.best_iteration = 0
        integration.has_summary = False
        integration.mlflow_client = MagicMock()

        empty_run = MagicMock()
        empty_run.info.run_id = "empty_run"
        empty_run.info.start_time = (time.time() + 10) * 1000
        empty_run.data.params = {}
        empty_run.data.metrics = {}
        empty_run.data.tags = {}

        with patch.object(mlflow, "search_runs", return_value=[empty_run]):
            integration.adopt_children()
        integration.mlflow_client.delete_run.assert_called_with("empty_run")

    def test_adopt_children_dedup_cv_runs(self):
        integration = self._make_integration()
        integration.autolog = True
        integration.best_iteration = 0
        integration.has_summary = False
        integration.experiment_type = "automl"
        integration.mlflow_client = MagicMock()

        run1 = MagicMock()
        run1.info.run_id = "run1"
        run1.info.start_time = (time.time() + 10) * 1000
        run1.data.params = {"n_estimators": "100", "lr": "0.1"}
        run1.data.metrics = {"m1": 0.5}
        run1.data.tags = {}

        # Duplicate (same params after removing n_estimators)
        run2 = MagicMock()
        run2.info.run_id = "run2"
        run2.info.start_time = (time.time() + 11) * 1000
        run2.data.params = {"n_estimators": "200", "lr": "0.1"}
        run2.data.metrics = {"m1": 0.6}
        run2.data.tags = {}

        integration.infos = [
            {
                "metrics": {"m1": 0.5},
                "tags": {"synapseml.flaml.best_run": False},
                "params": {"learner": "lgbm", "sample_size": 100},
                "submetrics": {"iter_counter": 0, "values": []},
            }
        ]

        mock_child_run = MagicMock()
        mock_child_run.data.params = {"lr": "0.1"}
        integration.mlflow_client.get_run = MagicMock(return_value=mock_child_run)

        with patch.object(mlflow, "search_runs", return_value=[run1, run2]):
            with patch.object(integration, "_log_info_to_run"):
                integration.adopt_children()
        integration.mlflow_client.delete_run.assert_any_call("run2")

    def test_adopt_children_before_start_time(self):
        integration = self._make_integration()
        integration.autolog = True
        integration.best_iteration = None  # triggers warning

        old_run = MagicMock()
        old_run.info.run_id = "old_run"
        old_run.info.start_time = (integration.start_time - 100) * 1000  # before start
        old_run.data.params = {"p": "v"}
        old_run.data.metrics = {"m": 1}
        old_run.data.tags = {}

        with patch.object(mlflow, "search_runs", return_value=[old_run]):
            integration.adopt_children()

    def test_adopt_children_automl_log_params(self):
        """Cover lines 1021-1028: log learner/sample_size when missing."""
        integration = self._make_integration()
        integration.autolog = True
        integration.best_iteration = 0
        integration.has_summary = False
        integration.experiment_type = "automl"
        integration.mlflow_client = MagicMock()

        child = MagicMock()
        child.info.run_id = "child1"
        child.info.start_time = (time.time() + 10) * 1000
        child.data.params = {"n_estimators": "10"}
        child.data.metrics = {"m1": 0.5}
        child.data.tags = {}

        integration.infos = [
            {
                "metrics": {"m1": 0.5},
                "tags": {"synapseml.flaml.best_run": False},
                "params": {"learner": "lgbm", "sample_size": 100},
                "submetrics": {"iter_counter": 0, "values": []},
            }
        ]

        mock_child_run = MagicMock()
        mock_child_run.data.params = {}  # missing learner and sample_size
        mock_child_run.info.run_name = "child_name"
        integration.mlflow_client.get_run = MagicMock(return_value=mock_child_run)

        result_obj = MagicMock()
        with patch.object(mlflow, "search_runs", return_value=[child]):
            with patch.object(integration, "_log_info_to_run"):
                with patch.object(integration, "copy_mlflow_run"):
                    integration.adopt_children(result_obj)
        # learner and sample_size should be logged
        assert integration.mlflow_client.log_param.call_count >= 2

    def test_adopt_children_no_flaml_info(self):
        """Cover line 1028: child_counter >= num_infos."""
        integration = self._make_integration()
        integration.autolog = True
        integration.best_iteration = 0
        integration.has_summary = False
        integration.infos = []  # no infos
        integration.mlflow_client = MagicMock()

        child = MagicMock()
        child.info.run_id = "child1"
        child.info.start_time = (time.time() + 10) * 1000
        child.data.params = {"p": "v"}
        child.data.metrics = {"m": 1}
        child.data.tags = {}

        with patch.object(mlflow, "search_runs", return_value=[child]):
            integration.adopt_children()


# ---------------------------------------------------------------------------
# retrain (lines 1052-1056)
# ---------------------------------------------------------------------------
class TestRetrain:
    def test_retrain_autolog(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()

        integration.autolog = True
        mock_train = MagicMock()
        with patch.object(integration, "set_mlflow_config"):
            with patch.object(mlflow, "start_run"):
                integration.retrain(mock_train, {"lr": 0.1})
        mock_train.assert_called_once_with({"lr": 0.1})


# ---------------------------------------------------------------------------
# resume_mlflow shutdown safety (defensive ``globals().get("mlflow")`` lookup)
# ---------------------------------------------------------------------------
class TestResumeMlflowShutdownSafety:
    """Regression guards for the 'NoneType has no attribute autolog' crash.

    During Python interpreter shutdown the module-level ``mlflow`` reference
    inside ``flaml/fabric/mlflow.py`` may be cleared to ``None`` by the
    garbage collector before ``MLflowIntegration.__del__`` runs.
    ``resume_mlflow`` reads the reference via ``globals().get("mlflow")``
    and short-circuits when it is missing or does not expose ``autolog``,
    so the destructor never propagates a shutdown-time exception regardless
    of whether ``mlflow`` is still bound, has been set to ``None``, or no
    longer exposes ``autolog``.
    """

    def test_resume_mlflow_noop_when_no_resume_params(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.resume_params = {}

        with patch("flaml.fabric.mlflow.mlflow", None):
            integration.resume_mlflow()  # must not raise

    def test_resume_mlflow_handles_mlflow_set_to_none(self):
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.resume_params = {"disable": False, "silent": True}

        with patch("flaml.fabric.mlflow.mlflow", None):
            integration.resume_mlflow()  # must not raise

    def test_resume_mlflow_handles_mlflow_missing_autolog(self):
        """A non-None mlflow without an ``autolog`` attribute is treated like None."""
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.resume_params = {"disable": False}

        # A bare ``object()`` has no ``autolog`` attribute, so ``hasattr``
        # returns ``False`` and ``resume_mlflow`` must short-circuit
        # instead of attempting ``object().autolog(...)``.
        with patch("flaml.fabric.mlflow.mlflow", object()):
            integration.resume_mlflow()  # must not raise

    def test_resume_mlflow_calls_autolog_normally(self):
        """The happy path still calls ``mlflow.autolog`` with ``resume_params``."""
        from flaml.fabric.mlflow import MLflowIntegration

        with mlflow.start_run():
            integration = MLflowIntegration()
        integration.resume_params = {"disable": False, "silent": True}

        fake = MagicMock()
        with patch("flaml.fabric.mlflow.mlflow", fake):
            integration.resume_mlflow()
        fake.autolog.assert_called_once_with(disable=False, silent=True)


# ---------------------------------------------------------------------------
# register_automl_pipeline (lines 1064-1071, 1077, 1080, 1082-1084)
# ---------------------------------------------------------------------------
class TestRegisterAutomlPipeline:
    def test_pipeline_none(self):
        from flaml.fabric.mlflow import register_automl_pipeline

        automl = MagicMock()
        automl.automl_pipeline = None
        result = register_automl_pipeline(automl)
        assert result is None

    def test_no_best_run_id(self):
        from flaml.fabric.mlflow import register_automl_pipeline

        automl = MagicMock()
        automl.automl_pipeline = Pipeline([("clf", tree.DecisionTreeClassifier())])
        automl.best_run_id = None
        automl._mlflow_exp_name = "test"
        automl.pipeline_signature = None

        mock_mv = MagicMock()
        with patch.object(mlflow.sklearn, "log_model"), patch.object(
            mlflow, "search_model_versions", return_value=[mock_mv]
        ):
            result = register_automl_pipeline(automl)
        assert result == mock_mv

    def test_with_best_run_id(self):
        from flaml.fabric.mlflow import register_automl_pipeline

        automl = MagicMock()
        automl.automl_pipeline = Pipeline([("clf", tree.DecisionTreeClassifier())])
        automl.best_run_id = "best_run"
        automl._mlflow_exp_name = "test"

        mock_run = MagicMock()
        mock_run.info.run_id = "best_run"
        mock_mv = MagicMock()
        with patch.object(mlflow, "get_run", return_value=mock_run), patch.object(
            mlflow, "register_model", return_value=mock_mv
        ):
            result = register_automl_pipeline(automl, model_name="custom_name")
        assert result == mock_mv

    def test_with_custom_signature(self):
        from flaml.fabric.mlflow import register_automl_pipeline

        automl = MagicMock()
        automl.automl_pipeline = Pipeline([("clf", tree.DecisionTreeClassifier())])
        automl.best_run_id = None
        automl._mlflow_exp_name = "test"

        mock_mv = MagicMock()
        custom_sig = MagicMock()
        with patch.object(mlflow.sklearn, "log_model") as mock_log, patch.object(
            mlflow, "search_model_versions", return_value=[mock_mv]
        ):
            register_automl_pipeline(automl, signature=custom_sig)
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs[1].get("signature", call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None) is not None


# ---------------------------------------------------------------------------
# flatten_dict and safe_json_dumps helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_flatten_dict_empty(self):
        from flaml.fabric.mlflow import flatten_dict

        assert flatten_dict({}) == {}

    def test_safe_json_dumps_with_non_serializable(self):
        from flaml.fabric.mlflow import safe_json_dumps

        result = safe_json_dumps({"obj": object()})
        assert isinstance(result, str)

    def test_is_autolog_enabled(self):
        from flaml.fabric.mlflow import is_autolog_enabled

        result = is_autolog_enabled()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# infer_signature error paths
# ---------------------------------------------------------------------------
class TestInferSignature:
    def test_infer_signature_exception(self):
        from flaml.fabric.mlflow import infer_signature

        with patch.object(mlflow.models, "infer_signature", side_effect=TypeError("bad")):
            result = infer_signature(X_train="bad_data", y_train="bad_label")
        assert result is None

    def test_infer_signature_dataframe_exception(self):
        import pandas as pd

        from flaml.fabric.mlflow import infer_signature

        df = pd.DataFrame({"a": [1], "b": [2]})
        with patch.object(mlflow.models, "infer_signature", side_effect=MlflowException("fail")):
            result = infer_signature(dataframe=df, label="b")
        assert result is None
