"""Unit coverage for Spark test configuration without importing FLAML or starting a JVM."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

TEST_ROOT = Path(__file__).resolve().parent
COMMON_JARS = [
    "com.microsoft.azure:synapseml_2.12:1.0.14",
    "org.apache.hadoop:hadoop-azure:3.3.5",
    "com.microsoft.azure:azure-storage:8.6.6",
]


def _load_spark_test(name):
    spec = importlib.util.spec_from_file_location(name, TEST_ROOT / "spark" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def spark_dependencies(monkeypatch):
    """Replace optional imports, including the learner broadcast that writes a module."""
    builder = Mock()
    builder.appName.return_value = builder
    builder.master.return_value = builder
    builder.config.return_value = builder
    mlflow = SimpleNamespace(__version__="2.22.1")
    utils = SimpleNamespace(_spark_major_minor_version=(3, 5), check_spark=Mock(return_value=(True, None)))
    automl = Mock()
    learner = Mock()
    datasets = SimpleNamespace(load_wine=Mock(return_value=(object(), object())))
    modules = {
        "flaml": SimpleNamespace(__path__=[], AutoML=automl),
        "flaml.automl": SimpleNamespace(__path__=[]),
        "flaml.automl.spark": SimpleNamespace(disable_spark_ansi_mode=Mock(), restore_spark_ansi_mode=Mock()),
        "flaml.tune": SimpleNamespace(__path__=[]),
        "flaml.tune.spark": SimpleNamespace(__path__=[]),
        "flaml.tune.spark.utils": utils,
        "flaml.tune.spark.mylearner": SimpleNamespace(MyRegularizedGreedyForest=learner),
        "mlflow": mlflow,
        "pyspark": SimpleNamespace(sql=SimpleNamespace(SparkSession=SimpleNamespace(builder=builder))),
        "sklearn": SimpleNamespace(__path__=[]),
        "sklearn.datasets": datasets,
        "test": SimpleNamespace(__path__=[]),
        "test.spark": SimpleNamespace(__path__=[]),
        "test.spark.custom_mylearner": SimpleNamespace(__all__=[]),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.chdir(TEST_ROOT.parent)
    monkeypatch.setenv("FLAML_MAX_CONCURRENT", "2")
    return SimpleNamespace(
        builder=builder, mlflow=mlflow, utils=utils, automl=automl, learner=learner, datasets=datasets
    )


@pytest.mark.parametrize(
    "mlflow_version,spark_version,mlflow_artifact",
    [
        ("2.8.2", (3, 5), "mlflow-spark"),
        ("2.9.0", (3, 5), "mlflow-spark_2.12"),
        ("2.10.0", (3, 5), "mlflow-spark_2.12"),
        ("2.22.1", (3, 5), "mlflow-spark_2.12"),
        ("2.8.2", (4, 0), "mlflow-spark"),
        ("2.9.1", (4, 0), "mlflow-spark_2.13"),
        ("2.22.1", (4, 0), "mlflow-spark_2.13"),
        ("2.8.2", (4, 1), "mlflow-spark"),
        ("2.22.1", (4, 1), "mlflow-spark_2.13"),
    ],
)
def test_spark_jars_preserve_common_coordinates(spark_dependencies, mlflow_version, spark_version, mlflow_artifact):
    spark_dependencies.mlflow.__version__ = mlflow_version
    spark_dependencies.utils._spark_major_minor_version = spark_version
    module = _load_spark_test("_init_spark")

    assert module._spark_jars_packages().split(",") == COMMON_JARS + [f"org.mlflow:{mlflow_artifact}:{mlflow_version}"]
    spark_dependencies.builder.getOrCreate.assert_not_called()
    spark_dependencies.utils.check_spark.assert_not_called()


def test_init_spark_session_configures_jars_and_preserves_options(spark_dependencies):
    module = _load_spark_test("_init_spark")

    spark = module.init_spark_session(app_name="jar-test", master="local[1]")

    builder = spark_dependencies.builder
    assert spark is builder.getOrCreate.return_value
    builder.appName.assert_called_once_with("jar-test")
    builder.master.assert_called_once_with("local[1]")
    assert builder.config.call_args_list == [
        call("spark.jars.packages", ",".join(COMMON_JARS + ["org.mlflow:mlflow-spark_2.12:2.22.1"])),
        call("spark.jars.repositories", "https://mmlspark.azureedge.net/maven"),
        call("spark.sql.debug.maxToStringFields", "100"),
        call("spark.driver.extraJavaOptions", "-Xss1m"),
        call("spark.executor.extraJavaOptions", "-Xss1m"),
    ]
    builder.getOrCreate.assert_called_once_with()
    spark.sparkContext._conf.set.assert_called_once_with(
        "spark.mlflow.pysparkml.autolog.logModelAllowlistFile",
        "https://mmlspark.blob.core.windows.net/publicwasb/log_model_allowlist.txt",
    )


@pytest.mark.parametrize("python_version", [(3, 10), (3, 11), (3, 12), (3, 13)])
def test_ensemble_is_not_skipped_on_supported_python(monkeypatch, spark_dependencies, python_version):
    with monkeypatch.context() as version_patch:
        version_patch.setattr(sys, "version_info", (*python_version, 0, "final", 0))
        module = _load_spark_test("test_ensemble")

    result = unittest.TestResult()
    module.TestEnsemble("test_ensemble").run(result)

    assert result.testsRun == 1
    assert not result.errors
    assert not result.failures
    assert not result.skipped
    spark_dependencies.automl.assert_called_once_with()
    automl = spark_dependencies.automl.return_value
    automl.add_learner.assert_called_once_with(learner_name="RGF", learner_class=spark_dependencies.learner)
    automl.fit.assert_called_once()
    settings = automl.fit.call_args.kwargs
    assert settings["X_train"] is spark_dependencies.datasets.load_wine.return_value[0]
    assert settings["y_train"] is spark_dependencies.datasets.load_wine.return_value[1]
    assert settings["use_spark"] is True
    assert settings["n_concurrent_trials"] == 2
    assert settings["estimator_list"] == ["rf", "xgboost", "catboost"]
    assert settings["ensemble"] == {
        "final_estimator": spark_dependencies.learner.return_value,
        "passthrough": False,
    }


@pytest.mark.parametrize("missing_dependency", ["spark", "custom_learner", "repository_root"])
def test_ensemble_still_skips_missing_prerequisites(monkeypatch, spark_dependencies, missing_dependency):
    if missing_dependency == "spark":
        spark_dependencies.utils.check_spark.return_value = (False, ImportError("PySpark is not installed"))
    elif missing_dependency == "custom_learner":
        monkeypatch.setitem(sys.modules, "flaml.tune.spark.mylearner", None)
    else:
        monkeypatch.chdir(TEST_ROOT)
    module = _load_spark_test("test_ensemble")

    result = unittest.TestResult()
    module.TestEnsemble("test_ensemble").run(result)

    assert result.testsRun == 1
    assert not result.errors
    assert not result.failures
    assert len(result.skipped) == 1
    spark_dependencies.automl.assert_not_called()
