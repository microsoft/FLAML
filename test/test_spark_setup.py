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
    spec = importlib.util.spec_from_file_location(Path(name).stem, TEST_ROOT / "spark" / f"{name}.py")
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
        "flaml.automl.spark": SimpleNamespace(
            disable_spark_ansi_mode=Mock(return_value=(None, None, False)), restore_spark_ansi_mode=Mock()
        ),
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


@pytest.mark.parametrize("entrypoint", ["init_spark_session", "setup_spark_for_tests"])
@pytest.mark.parametrize(
    "configured_master,explicit_master,expected_master",
    [
        (None, None, "local[2]"),
        ("spark://master:7077", None, "spark://master:7077"),
        ("spark://master:7077", "local[1]", "local[1]"),
    ],
)
def test_spark_setup_honors_cluster_without_changing_local_defaults(
    monkeypatch, spark_dependencies, entrypoint, configured_master, explicit_master, expected_master
):
    module = _load_spark_test("_init_spark")
    environment = {"FLAML_TEST_SPARK_MASTER": configured_master} if configured_master else {}
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", environ=environment))
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(module, "atexit", SimpleNamespace(register=Mock()))

    kwargs = {"master": explicit_master} if explicit_master else {}
    getattr(module, entrypoint)(**kwargs)

    spark_dependencies.builder.master.assert_called_once_with(expected_master)


@pytest.mark.parametrize(
    "master,hosts",
    [
        ("local[2]", ["worker-1", "worker-2"]),
        ("spark://master:7077", ["worker-1", "worker-1"]),
        ("spark://master:7077", ["driver", "worker-2"]),
    ],
)
def test_container_guard_rejects_local_or_incomplete_execution(monkeypatch, master, hosts):
    module = _load_spark_test(Path("containers") / "run_tests")
    spark = Mock()
    spark.sparkContext.master = master
    spark.sparkContext.parallelize.return_value.barrier.return_value.mapPartitions.return_value.collect.return_value = [
        (index, host) for index, host in enumerate(hosts)
    ]
    monkeypatch.setattr(module, "_wait_for_workers", Mock())

    with pytest.raises(RuntimeError, match="Container tests require master|Expected execution on"):
        module.verify_cluster(spark, "spark://master:7077", 2)
    if master.startswith("local"):
        spark.sparkContext.parallelize.assert_not_called()


@pytest.mark.parametrize("master,workers", [("", "0"), ("spark://master:7077", "1")])
def test_container_runner_requires_an_explicit_multiworker_cluster(monkeypatch, master, workers):
    module = _load_spark_test(Path("containers") / "run_tests")
    monkeypatch.setenv("FLAML_TEST_SPARK_MASTER", master)
    monkeypatch.setenv("FLAML_TEST_SPARK_WORKERS", workers)
    with pytest.raises(RuntimeError, match="Container tests require"):
        module.main()


@pytest.mark.parametrize("measured_context", ["spark-worker:worker-1:123", "driver"])
def test_container_coverage_gate_requires_actual_worker_context(tmp_path, measured_context):
    import coverage

    from flaml.tune.spark.utils import get_broadcast_data

    module = _load_spark_test(Path("containers") / "run_tests")
    filename = str(tmp_path / "worker.coverage")
    data = coverage.CoverageData(basename=filename)
    data.set_context(measured_context)
    data.add_lines({"flaml/tune/spark/utils.py": [get_broadcast_data.__code__.co_firstlineno + 1]})
    data.write()
    results = [
        (
            0,
            "worker-1",
            (3, 11),
            "3.5.1",
            "2.8.0",
            {"file": filename, "context": "spark-worker:worker-1:123", "pid": 123},
        )
    ]
    if measured_context == "driver":
        with pytest.raises(RuntimeError, match="executed FLAML code"):
            module._verify_worker_coverage(results)
    else:
        module._verify_worker_coverage(results)


def test_container_coverage_gate_rejects_missing_shards(tmp_path):
    module = _load_spark_test(Path("containers") / "run_tests")
    with pytest.raises(RuntimeError, match="coverage shard"):
        module._verify_worker_coverage(
            [
                (
                    0,
                    "worker",
                    (3, 11),
                    "3.5.1",
                    "2.8.0",
                    {"file": str(tmp_path / "missing"), "context": "spark-worker:worker:1", "pid": 1},
                )
            ]
        )


def test_executor_guard_flushes_real_flaml_coverage(tmp_path):
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import sys
        from types import ModuleType, SimpleNamespace
        import coverage
        import flaml
        from test.spark.containers.run_tests import _executor_identity, _verify_worker_coverage

        fake = ModuleType("pyspark")
        fake.__version__ = "test-runtime"
        fake.TaskContext = SimpleNamespace(get=lambda: object())
        sys.modules["pyspark"] = fake
        collector = coverage.Coverage(
            source=["flaml"], branch=True, data_file={str(tmp_path / "worker.coverage")!r}, data_suffix=True
        )
        collector.start()
        try:
            result = _executor_identity(7)
            assert result[0] == 7
            _verify_worker_coverage([result])
        finally:
            collector.stop()
            collector.save()
        print("worker-coverage-verified")
    """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker-coverage-verified" in result.stdout


@pytest.mark.parametrize("invalid", ["spark-runtime", "joblib-host", "joblib-runtime"])
def test_container_guard_rejects_runtime_or_joblib_mismatch(monkeypatch, invalid):
    from contextlib import nullcontext

    import flaml

    module = _load_spark_test(Path("containers") / "run_tests")
    runtime = (sys.version_info[:2], "3.5.1", flaml.__version__)
    results = [(index, f"worker-{index}", *runtime) for index in range(2)]
    joblib_results = list(results)
    if invalid == "spark-runtime":
        results[0] = (0, "worker-0", (0, 0), *runtime[1:])
    elif invalid == "joblib-host":
        joblib_results[0] = (0, "driver", *runtime)
    else:
        joblib_results[0] = (0, "worker-0", (0, 0), *runtime[1:])
    spark = Mock()
    spark.sparkContext.master = "spark://master:7077"
    spark.sparkContext.parallelize.return_value.barrier.return_value.mapPartitions.return_value.collect.return_value = (
        results
    )
    monkeypatch.setattr(module, "_wait_for_workers", Mock())
    monkeypatch.setattr(module, "_verify_worker_coverage", Mock())
    monkeypatch.setitem(sys.modules, "pyspark", SimpleNamespace(__version__="3.5.1"))
    monkeypatch.setitem(sys.modules, "joblibspark", SimpleNamespace(register_spark=lambda: None))
    monkeypatch.setitem(
        sys.modules,
        "joblib",
        SimpleNamespace(
            Parallel=lambda **kwargs: lambda jobs: joblib_results,
            delayed=lambda function: lambda value: (function, value),
            parallel_backend=lambda name: nullcontext(),
        ),
    )
    with pytest.raises(RuntimeError, match="runtime mismatch|Joblib did not execute"):
        module.verify_cluster(spark, "spark://master:7077", 2)


@pytest.mark.parametrize("hosts", [("worker-a", "worker-b"), ("worker-a", "worker-a")])
def test_container_registration_wait_requires_distinct_alive_workers(monkeypatch, hosts):
    import io
    import json

    module = _load_spark_test(Path("containers") / "run_tests")
    payload = json.dumps({"workers": [{"state": "ALIVE", "host": host} for host in hosts]}).encode()
    request = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(payload))
    clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 1, 121]), sleep=Mock())
    monkeypatch.setattr(module, "urlopen", request)
    monkeypatch.setattr(module, "time", clock)
    if len(set(hosts)) == 1:
        with pytest.raises(RuntimeError, match="distinct registered Spark workers"):
            module._wait_for_workers(2)
        clock.sleep.assert_called_once_with(1)
    else:
        module._wait_for_workers(2)
        clock.sleep.assert_not_called()
    request.assert_called_once_with("http://master:8080/json/", timeout=5)


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
