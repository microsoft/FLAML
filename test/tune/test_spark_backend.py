import importlib
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pytest
from joblib import parallel_backend
from joblib._parallel_backends import SequentialBackend, ThreadingBackend
from joblib.parallel import BACKENDS, get_active_backend
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from flaml import AutoML, tune


@pytest.fixture
def spark_backend(monkeypatch):
    configured_jobs = []

    class SparkBackend(ThreadingBackend):
        def effective_n_jobs(self, n_jobs):
            # Spark can report zero task capacity while an inline trial runs on the driver.
            return 0 if n_jobs == -1 else super().effective_n_jobs(n_jobs)

        def configure(self, n_jobs=1, parallel=None, **backend_args):
            self.parallel = parallel
            self._n_jobs = self.effective_n_jobs(n_jobs)
            configured_jobs.append(self._n_jobs)
            return self._n_jobs

        def get_nested_backend(self):
            return SequentialBackend(nesting_level=self.nesting_level + 1), None

    sc = SimpleNamespace(statusTracker=lambda: SimpleNamespace(getExecutorInfos=lambda: []))
    spark_context = SimpleNamespace(
        defaultParallelism=2,
        broadcast=lambda value: value,
        getConf=lambda: SimpleNamespace(get=lambda key: None),
    )
    spark = SimpleNamespace(_jsc=SimpleNamespace(sc=lambda: sc), sparkContext=spark_context)
    spark_session = SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: spark))
    pyspark_module = ModuleType("pyspark")
    pyspark_module.__path__ = []
    sql_module = ModuleType("pyspark.sql")
    sql_module.SparkSession = spark_session
    joblibspark_module = ModuleType("joblibspark")
    joblibspark_module.register_spark = lambda: None
    monkeypatch.setitem(sys.modules, "pyspark", pyspark_module)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql_module)
    monkeypatch.setitem(sys.modules, "joblibspark", joblibspark_module)
    monkeypatch.setitem(BACKENDS, "spark", SparkBackend)
    for module_name in ("flaml.tune.tune", "flaml.tune.spark.utils", "flaml.automl.automl"):
        monkeypatch.setattr(importlib.import_module(module_name), "check_spark", lambda: (True, None))
    monkeypatch.setattr(importlib.import_module("flaml.tune.spark.utils"), "SparkSession", spark_session, raising=False)
    monkeypatch.setattr(importlib.import_module("flaml.tune.tune"), "internal_mlflow", False)
    monkeypatch.delenv("FLAML_MAX_CONCURRENT", raising=False)
    return configured_jobs


@pytest.mark.parametrize("estimator_class", [RandomForestClassifier, ExtraTreesClassifier])
@pytest.mark.parametrize("n_concurrent_trials", [1, 2])
def test_spark_tuning_forest_prediction(spark_backend, estimator_class, n_concurrent_trials):
    X, y = make_classification(n_samples=80, n_features=6, random_state=41)
    model = estimator_class(n_estimators=4, n_jobs=-1, random_state=41).fit(X, y)
    expected_loss = 1 - roc_auc_score(y, model.predict_proba(X)[:, 1])
    workers_ready = Barrier(n_concurrent_trials)

    def evaluate(config):
        workers_ready.wait(timeout=10)
        return {"loss": 1 - roc_auc_score(y, model.predict_proba(X)[:, 1])}

    with parallel_backend("threading", n_jobs=2):
        original_backend = get_active_backend()[0]
        analysis = tune.run(
            evaluate,
            config={"trial": tune.randint(0, 100)},
            metric="loss",
            mode="min",
            search_alg="RandomSearch",
            num_samples=2,
            use_spark=True,
            n_concurrent_trials=n_concurrent_trials,
            verbose=0,
        )
        assert get_active_backend()[0] is original_backend

    assert spark_backend == [n_concurrent_trials]
    assert len(analysis.trials) == 2
    assert analysis.best_result["loss"] == pytest.approx(expected_loss)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_spark_tuning_restores_backend_after_error(spark_backend, monkeypatch, error_type):
    sleep = Mock()
    monkeypatch.setattr(importlib.import_module("flaml.tune.tune").time, "sleep", sleep)
    calls = []

    def evaluate(config):
        calls.append(config)
        raise error_type("trial failed")

    with parallel_backend("threading", n_jobs=2):
        original_backend = get_active_backend()[0]
        with pytest.raises(error_type, match="trial failed"):
            tune.run(evaluate, config={}, num_samples=1, use_spark=True, n_concurrent_trials=1, verbose=0)
        assert get_active_backend()[0] is original_backend
    assert len(calls) == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("learner", ["rf", "extra_tree"])
def test_automl_spark_single_worker(spark_backend, learner):
    X, y = make_classification(n_samples=80, n_features=6, random_state=41)
    automl = AutoML(mlflow_logging=False)
    automl.fit(
        X,
        y,
        task="classification",
        metric="roc_auc",
        estimator_list=[learner],
        eval_method="cv",
        n_splits=2,
        max_iter=2,
        time_budget=30,
        n_jobs=-1,
        use_spark=True,
        n_concurrent_trials=3,
        retrain_full=False,
        verbose=0,
    )

    assert spark_backend == [1]
    assert np.isfinite(automl.best_loss)
    assert automl.predict_proba(X).shape == (80, 2)


@pytest.fixture
def broadcast_files(monkeypatch, tmp_path):
    package = importlib.import_module("flaml.tune.spark")
    utils = importlib.import_module("flaml.tune.spark.utils")
    for name in ("sys", "os", "check_spark", "get_n_cpus", "with_parameters", "broadcast_code", "_SparkCodePaths"):
        if hasattr(package, name):
            monkeypatch.setattr(package, name, getattr(package, name))
    driver = tmp_path / "driver"
    spark_files = tmp_path / "spark-files"
    driver.mkdir()
    spark_files.mkdir()
    context = Mock()
    monkeypatch.setattr(utils, "__file__", str(driver / "utils.py"))
    monkeypatch.setattr(utils, "check_spark", lambda: (True, None))
    monkeypatch.setattr(
        utils,
        "SparkSession",
        SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: SimpleNamespace(sparkContext=context))),
        raising=False,
    )
    monkeypatch.setattr(
        utils,
        "pyspark",
        SimpleNamespace(SparkFiles=SimpleNamespace(getRootDirectory=lambda: str(spark_files))),
        raising=False,
    )
    monkeypatch.setattr(package, "__path__", type(package.__path__)([str(tmp_path / "isolated-worker")]))
    return utils, context, driver


def test_broadcast_code_supports_isolated_and_late_workers(monkeypatch, broadcast_files):
    utils, context, driver = broadcast_files
    code = """
    class Learner:
        def predict(self, value):
            return value * 3
    """
    name = "_test_distributed_learner"
    qualified_name = f"flaml.tune.spark.{name}"
    path = utils.broadcast_code(code, name)
    assert Path(path).parent == driver
    assert Path(path).read_text(encoding="utf-8") == textwrap.dedent(code)
    archive = context.addPyFile.call_args.args[0]
    context.parallelize.assert_not_called()

    # The package was imported before Spark added this archive, as on reused workers.
    monkeypatch.syspath_prepend(archive)
    try:
        module = importlib.import_module(qualified_name)
        payload = pickle.dumps(module.Learner())
        del sys.modules[qualified_name]
        assert pickle.loads(payload).predict(7) == 21
    finally:
        for key in list(sys.modules):
            if key == qualified_name or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


def test_broadcast_code_updates_use_distinct_archives(monkeypatch, broadcast_files):
    utils, context, _ = broadcast_files
    name = "_test_updated_learner"
    qualified_name = f"flaml.tune.spark.{name}"
    try:
        code = "class Learner:\n    def predict(self):\n        return 1\n"
        utils.broadcast_code(code, name)
        first_archive = context.addPyFile.call_args.args[0]
        monkeypatch.syspath_prepend(first_archive)
        module = importlib.import_module(qualified_name)
        first = module.Learner
        assert first().predict() == 1

        utils._published_code_aliases.discard(name)
        utils.broadcast_code(code.replace("return 1", "return 2"), name)
        second_archive = context.addPyFile.call_args.args[0]
        assert first_archive != second_archive
        monkeypatch.syspath_prepend(second_archive)
        second = importlib.import_module(qualified_name).Learner
        assert second().predict() == 2
        assert first.__module__ != second.__module__
        assert [learner().predict() for learner in pickle.loads(pickle.dumps((first, second)))] == [1, 2]

        archive_bytes = Path(second_archive).read_bytes()
        utils.broadcast_code(code.replace("return 1", "return 2"), name)
        assert context.addPyFile.call_args.args[0] == second_archive
        assert Path(second_archive).read_bytes() == archive_bytes
    finally:
        for key in list(sys.modules):
            if key == qualified_name or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("worker_kind", ["reused", "late"])
def test_worker_alias_imports_follow_publication_order_and_reverts(broadcast_files, worker_kind):
    utils, context, _ = broadcast_files
    name = "_test_worker_public_alias"
    archives = []
    for value in (1, 2, 1):
        utils.broadcast_code(f"class Learner:\n    def predict(self):\n        return {value}\n", name)
        archives.append(context.addPyFile.call_args.args[0])
    if worker_kind == "reused":
        requests = [archives[:1], archives[:2], archives]
        expected = [1, 2, 1]
    else:
        requests = [[archives[1], archives[2], archives[0]]]
        expected = [1]
    script = """
import importlib
import json
import os
import pickle
import sys
retained = []
module = None
for line in sys.stdin:
    request = json.loads(line)
    sys.path[:] = request["paths"] + [path for path in sys.path if not path.endswith("_flaml_custom_code.zip")]
    previous = module
    module = importlib.import_module(request["module"])
    retained.append(pickle.dumps(module.Learner()))
    print(json.dumps({
        "value": module.Learner().predict(), "retained": [pickle.loads(item).predict() for item in retained],
        "pid": os.getpid(), "same_module": previous is None or previous is module,
    }), flush=True)
"""
    messages = "\n".join(json.dumps({"paths": paths, "module": f"flaml.tune.spark.{name}"}) for paths in requests)
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=messages + "\n",
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "FLAML_FABRIC_RUNTIME": "false"},
        timeout=30,
        check=True,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert [row["value"] for row in responses] == expected
    assert responses[-1]["retained"] == expected
    assert len({row["pid"] for row in responses}) == 1
    assert all(row["same_module"] for row in responses)


def test_broadcast_code_without_spark_preserves_local_file(monkeypatch, broadcast_files):
    utils, context, _ = broadcast_files
    monkeypatch.setattr(utils, "check_spark", lambda: (False, ImportError("no pyspark")))
    path = utils.broadcast_code("value = 42\n", "_test_local_learner")
    assert Path(path).read_text(encoding="utf-8") == "value = 42\n"
    context.addPyFile.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "nested/module",
        "",
        None,
        "class",
        "utils",
        "__init__",
        "UTILS",
        "sys",
        "os",
        "check_spark",
        "get_n_cpus",
        "with_parameters",
        "broadcast_code",
        "_SparkCodePaths",
        "__path__",
        "_flaml_custom_previous_revision",
    ],
)
def test_broadcast_code_rejects_nonmodule_and_reserved_names(broadcast_files, name):
    utils, context, driver = broadcast_files
    with pytest.raises(ValueError, match="module"):
        utils.broadcast_code("value = 42", name)
    assert not list(driver.iterdir())
    context.addPyFile.assert_not_called()


def test_broadcast_archive_does_not_publish_top_level_imports(monkeypatch, broadcast_files):
    utils, context, _ = broadcast_files
    name = "_test_private_archive_learner"
    qualified = f"flaml.tune.spark.{name}"
    utils.broadcast_code("value = 7\n", name)
    archive_path = context.addPyFile.call_args.args[0]
    with ZipFile(archive_path) as archive:
        assert all(path.startswith("_flaml_code/") for path in archive.namelist())
    monkeypatch.syspath_prepend(archive_path)
    try:
        assert importlib.util.find_spec(name) is None
        assert importlib.import_module(qualified).value == 7
    finally:
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


def test_failed_distribution_does_not_publish_a_new_driver_alias(monkeypatch, broadcast_files):
    utils, context, driver = broadcast_files
    name = "_test_failed_update"
    qualified = f"flaml.tune.spark.{name}"
    utils.broadcast_code("value = 1\n", name)
    monkeypatch.syspath_prepend(context.addPyFile.call_args.args[0])
    original = importlib.import_module(qualified)
    alias = (driver / f"{name}.py").read_bytes()
    context.addPyFile.side_effect = RuntimeError("distribution unavailable")
    try:
        with pytest.raises(RuntimeError, match="distribution unavailable"):
            utils.broadcast_code("value = 2\n", name)
        assert (driver / f"{name}.py").read_bytes() == alias
        assert importlib.import_module(qualified) is original
        assert original.value == 1
    finally:
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("failure", ["alias", "bytecode"])
def test_fallible_alias_preparation_does_not_publish_code_to_workers(monkeypatch, broadcast_files, failure):
    utils, context, driver = broadcast_files
    name = "_test_alias_preparation_failure"
    qualified = f"flaml.tune.spark.{name}"
    context.addPyFile.side_effect = monkeypatch.syspath_prepend
    utils.broadcast_code("value = 1\n", name)
    original = importlib.import_module(qualified)
    alias_path = driver / f"{name}.py"
    before = alias_path.read_bytes()
    publish, unlink = utils._publish_code_file, utils.os.unlink
    if failure == "alias":

        def fail_alias(path, contents):
            if Path(path) == alias_path:
                raise PermissionError("alias locked")
            return publish(path, contents)

        monkeypatch.setattr(utils, "_publish_code_file", fail_alias)
    else:
        cache_path = Path(importlib.util.cache_from_source(str(alias_path)))
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_bytes(b"old cache")

        def fail_cache(path, *args, **kwargs):
            if Path(path) == cache_path:
                raise PermissionError("bytecode locked")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(utils.os, "unlink", fail_cache)
    try:
        with pytest.raises(PermissionError, match="locked"):
            utils.broadcast_code("value = 2\n", name)
        assert (context.addPyFile.call_count, original.value) == (1, 1)
        assert alias_path.read_bytes() == before
    finally:
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)
