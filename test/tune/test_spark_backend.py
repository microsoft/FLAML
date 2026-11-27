import importlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
import time
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


@pytest.mark.parametrize("report_after_nested", [False, True])
def test_inline_spark_reports_do_not_cross_trial_or_nested_tune_contexts(spark_backend, report_after_nested):
    ready = Barrier(2)

    def objective(config):
        tune.report(loss=config["x"])
        ready.wait(timeout=10)
        if config["x"] == 1:
            inner = tune.run(
                lambda inner_config: tune.report(loss=100),
                config={"x": 1},
                metric="loss",
                mode="min",
                num_samples=1,
                use_ray=False,
                use_spark=False,
                mlflow_logging=False,
                verbose=0,
            )
            assert inner.best_result["loss"] == 100
        ready.wait(timeout=10)
        if report_after_nested:
            tune.report(loss=config["x"] + 10)

    result = tune.run(
        objective,
        config={"x": tune.choice([1, 2])},
        points_to_evaluate=[{"x": 1}, {"x": 2}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_spark=True,
        n_concurrent_trials=2,
        mlflow_logging=False,
        verbose=0,
    )
    assert [trial.last_result["loss"] for trial in result.trials] == ([11, 12] if report_after_nested else [1, 2])


@pytest.mark.parametrize("delivery", ["report", "return", "both"])
def test_spark_scheduler_stop_preserves_siblings_and_completes_searcher_trials(spark_backend, monkeypatch, delivery):
    from flaml.tune.trial_runner import SparkTrialRunner

    runners = []
    initialize = SparkTrialRunner.__init__

    def capture_runner(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._search_alg.on_trial_complete = Mock(wraps=self._search_alg.on_trial_complete)
        runners.append(self)

    monkeypatch.setattr(SparkTrialRunner, "__init__", capture_runner)

    class StopFirst:
        def set_search_properties(self, **kwargs):
            return True

        def on_trial_add(self, runner, trial):
            pass

        def on_trial_result(self, runner, trial, result):
            return "STOP" if trial.config["x"] == 1 else "CONTINUE"

        def on_trial_remove(self, runner, trial):
            pass

        def on_trial_complete(self, *args):
            pass

    def objective(config):
        if delivery != "return":
            tune.report(loss=float(config["x"]))
        if delivery != "report":
            return {"loss": float(config["x"]) + 10}

    result = tune.run(
        objective,
        config={"x": tune.choice([1, 2, 3, 4])},
        points_to_evaluate=[{"x": value} for value in (1, 2, 3, 4)],
        metric="loss",
        mode="min",
        num_samples=4,
        use_spark=True,
        n_concurrent_trials=2,
        scheduler=StopFirst(),
        mlflow_logging=False,
        verbose=0,
    )
    expected = [1.0, 2.0, 3.0, 4.0] if delivery == "report" else [11.0, 12.0, 13.0, 14.0]
    if delivery == "both":
        expected[0] = 1.0
    assert [trial.last_result["loss"] for trial in result.trials] == expected
    assert all(trial.status == "TERMINATED" for trial in result.trials)
    assert runners[0].running_trials == []
    assert runners[0]._search_alg.on_trial_complete.call_count == 4


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


@pytest.mark.parametrize("previous", [False, True])
def test_first_driver_import_cannot_observe_a_failed_pending_publication(monkeypatch, broadcast_files, previous):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    utils, context, driver = broadcast_files
    package = importlib.import_module("flaml.tune.spark")
    monkeypatch.setattr(package, "__path__", type(package.__path__)([str(driver)]))
    name = f"_test_concurrent_alias_{int(previous)}"
    qualified = f"flaml.tune.spark.{name}"
    if previous:
        utils.broadcast_code("value = 1\n", name)
    distributing, importing, release = Event(), Event(), Event()
    install = package._install_code_alias

    def observed_install(*args):
        importing.set()
        return install(*args)

    def blocked_distribution(path):
        distributing.set()
        assert release.wait(10)
        raise RuntimeError("distribution rejected")

    monkeypatch.setattr(package, "_install_code_alias", observed_install)
    context.addPyFile.side_effect = blocked_distribution
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            publication = executor.submit(utils.broadcast_code, "value = 2\n", name)
            assert distributing.wait(10)
            imported = executor.submit(importlib.import_module, qualified)
            try:
                assert importing.wait(10)
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="distribution rejected"):
                publication.result(timeout=10)
            assert not Path(importlib.util.cache_from_source(str(driver / f"{name}.py"))).exists()
            if previous:
                module = imported.result(timeout=10)
                assert module.value == 1
                assert importlib.import_module(qualified).value == 1
            else:
                with pytest.raises(ModuleNotFoundError):
                    imported.result(timeout=10)
                assert not (driver / f"{name}.py").exists()
    finally:
        release.set()
        package.__dict__.pop(name, None)
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


def test_secondary_alias_rollback_failure_preserves_the_distribution_error(monkeypatch, broadcast_files):
    utils, context, driver = broadcast_files
    name = "_test_rollback_error"
    utils.broadcast_code("value = 1\n", name)
    before = (driver / f"{name}.py").read_text()
    publish = utils._publish_code_file

    def fail_rollback(path, content):
        if Path(path) == driver / f"{name}.py" and content == before:
            raise PermissionError("rollback blocked")
        return publish(path, content)

    monkeypatch.setattr(utils, "_publish_code_file", fail_rollback)
    context.addPyFile.side_effect = RuntimeError("original distribution error")
    logged = Mock()
    monkeypatch.setattr(utils.logger, "exception", logged)
    with pytest.raises(RuntimeError, match="original distribution error"):
        utils.broadcast_code("value = 2\n", name)
    logged.assert_called_once()


@pytest.mark.parametrize("failure", ["alias", "distribution"])
def test_restarted_driver_keeps_its_on_disk_publication_after_failure(monkeypatch, broadcast_files, failure):
    utils, context, driver = broadcast_files
    package = importlib.import_module("flaml.tune.spark")
    monkeypatch.setattr(package, "__path__", type(package.__path__)([str(driver)]))
    name = f"_test_restarted_{failure}"
    qualified = f"flaml.tune.spark.{name}"
    utils.broadcast_code("value = 1\n", name)
    alias_path = driver / f"{name}.py"
    original = alias_path.read_bytes()
    archives = set(driver.parent.joinpath("spark-files").glob("*_flaml_custom_code.zip"))
    state_file = driver / ".flaml-code-state" / f"{name}.json"
    if state_file.exists():
        state_file.unlink()
    utils._published_code_revisions.pop(name, None)
    utils._published_code_aliases.discard(name)
    publish = utils._publish_code_file
    if failure == "alias":

        def fail_alias(path, content):
            if Path(path) == alias_path:
                raise RuntimeError("publication failed")
            return publish(path, content)

        monkeypatch.setattr(utils, "_publish_code_file", fail_alias)
    else:
        context.addPyFile.side_effect = RuntimeError("publication failed")
    try:
        with pytest.raises(RuntimeError, match="publication failed"):
            utils.broadcast_code("value = 2\n", name)
        assert alias_path.read_bytes() == original
        assert importlib.import_module(qualified).value == 1
        retained = set(driver.parent.joinpath("spark-files").glob("*_flaml_custom_code.zip"))
        assert archives.issubset(retained)
        assert len(retained - archives) == (1 if failure == "distribution" else 0)
    finally:
        package.__dict__.pop(name, None)
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("previous", [False, True])
def test_separate_interpreter_cannot_import_an_uncommitted_alias(broadcast_files, previous):
    utils, context, driver = broadcast_files
    name = f"_test_shared_alias_{int(previous)}"
    if previous:
        utils.broadcast_code("value = 1\n", name)
    observed = []
    script = """
import importlib
import json
import sys
package = importlib.import_module("flaml.tune.spark")
package.__path__ = type(package.__path__)([sys.argv[1]])
try:
    module = importlib.import_module("flaml.tune.spark." + sys.argv[2])
    result = {"value": module.value}
except ModuleNotFoundError:
    result = {"unavailable": True}
print(json.dumps(result))
"""

    def reject_distribution(path):
        result = subprocess.run(
            [sys.executable, "-c", script, str(driver), name],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        )
        observed.append(json.loads(result.stdout.strip().splitlines()[-1]))
        raise RuntimeError("distribution failed")

    context.addPyFile.side_effect = reject_distribution
    with pytest.raises(RuntimeError, match="distribution failed"):
        utils.broadcast_code("value = 2\n", name)
    assert observed == ([{"value": 1}] if previous else [{"unavailable": True}])


def test_alias_resolution_does_not_hold_the_publication_lock_during_module_import(broadcast_files):
    utils, _, driver = broadcast_files
    dependency = "_test_import_dependency"
    name = "_test_import_primary"
    utils.broadcast_code("value = 7\n", dependency)
    revision = Path(
        utils.broadcast_code(
            "from _flaml_test_import_sync import entered, release\n"
            "entered.set()\n"
            "assert release.wait(10)\n"
            f"from flaml.tune.spark.{dependency} import value\n",
            name,
        )
    ).stem
    script = """
import importlib
import json
import sys
import threading
from types import ModuleType
package = importlib.import_module("flaml.tune.spark")
package.__path__ = type(package.__path__)([sys.argv[1]])
sync = ModuleType("_flaml_test_import_sync")
sync.entered, sync.release = threading.Event(), threading.Event()
sys.modules[sync.__name__] = sync
waiting = threading.Event()
original = importlib.import_module
revision = "flaml.tune.spark." + sys.argv[3]
values = []
def observe(name, *args, **kwargs):
    if name == revision and threading.current_thread().name == "alias":
        waiting.set()
    return original(name, *args, **kwargs)
package.importlib.import_module = observe
direct = threading.Thread(target=lambda: values.append(original(revision).value), daemon=True)
alias = threading.Thread(
    target=lambda: values.append(original("flaml.tune.spark." + sys.argv[2]).value),
    name="alias", daemon=True,
)
direct.start()
assert sync.entered.wait(10)
alias.start()
assert waiting.wait(10)
sync.release.set()
direct.join(10)
alias.join(10)
print(json.dumps({"values": values, "blocked": direct.is_alive() or alias.is_alive()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(driver), name, revision],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"values": [7, 7], "blocked": False}


@pytest.mark.parametrize("replacement_completed", [False, True])
def test_state_publication_error_keeps_delivered_code_recoverable(monkeypatch, broadcast_files, replacement_completed):
    utils, context, driver = broadcast_files
    package = importlib.import_module("flaml.tune.spark")
    monkeypatch.setattr(package, "__path__", type(package.__path__)([str(driver)]))
    name = f"_test_state_error_{int(replacement_completed)}"
    qualified = f"flaml.tune.spark.{name}"
    utils.broadcast_code("value = 1\n", name)
    state_path = Path(utils._publication_file(str(driver), name))
    publish = utils._publish_code_file

    def fail_state(path, contents):
        if Path(path) == state_path:
            if replacement_completed:
                publish(path, contents)
            raise RuntimeError("state publication failed")
        return publish(path, contents)

    monkeypatch.setattr(utils, "_publish_code_file", fail_state)
    try:
        with pytest.raises(RuntimeError, match="state publication failed"):
            utils.broadcast_code("value = 2\n", name)
        assert Path(context.addPyFile.call_args.args[0]).exists()
        assert importlib.import_module(qualified).value == (2 if replacement_completed else 1)
    finally:
        package.__dict__.pop(name, None)
        for key in list(sys.modules):
            if key == qualified or key.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("restarted", [False, True])
@pytest.mark.parametrize("worker_kind", ["reused", "late"])
@pytest.mark.parametrize("failure_point", ["state", "delivery"])
def test_revert_after_failed_publication_outranks_delivered_archives(
    monkeypatch, broadcast_files, restarted, worker_kind, failure_point
):
    utils, context, driver = broadcast_files
    name = "_test_delivered_revert"
    worker = driver.parent / "delivered-worker"
    worker.mkdir()
    archives = []

    def deliver(path):
        target = worker / Path(path).name
        shutil.copyfile(path, target)
        archives.append(str(target))
        if failure_point == "delivery" and len(archives) == 2:
            raise RuntimeError("publication acknowledgement lost")

    context.addPyFile.side_effect = deliver
    utils.broadcast_code("value = 1\n", name)
    spark_root = Path(context.addPyFile.call_args.args[0]).parent
    state_path = Path(utils._publication_file(str(driver), name))
    publish = utils._publish_code_file

    def reject_commit(path, contents):
        if Path(path) == state_path:
            raise RuntimeError("publication acknowledgement lost")
        return publish(path, contents)

    with monkeypatch.context() as failure:
        if failure_point == "state":
            failure.setattr(utils, "_publish_code_file", reject_commit)
        with pytest.raises(RuntimeError, match="publication acknowledgement lost"):
            utils.broadcast_code("value = 2\n", name)
    if restarted:
        script = """
import importlib
import sys
from types import SimpleNamespace
utils = importlib.import_module("flaml.tune.spark.utils")
utils.__file__ = sys.argv[1]
utils.check_spark = lambda: (True, None)
context = SimpleNamespace(addPyFile=print)
utils.SparkSession = SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: SimpleNamespace(sparkContext=context)))
utils.pyspark = SimpleNamespace(SparkFiles=SimpleNamespace(getRootDirectory=lambda: sys.argv[2]))
utils.time.time_ns = lambda: 0
utils.broadcast_code("value = 1\\n", sys.argv[3])
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(driver / "utils.py"), str(spark_root), name],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        )
        deliver(result.stdout.strip().splitlines()[-1])
    else:
        utils.broadcast_code("value = 1\n", name)
    requests = [archives[:1], archives[:2], archives] if worker_kind == "reused" else [archives]
    script = """
import importlib
import json
import sys
package = importlib.import_module("flaml.tune.spark")
package.__path__ = type(package.__path__)([])
values = []
for archives in json.loads(sys.argv[2]):
    sys.path[:0] = archives
    values.append(importlib.import_module("flaml.tune.spark." + sys.argv[1]).value)
print(json.dumps(values))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, name, json.dumps(requests)],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == ([1, 2, 1] if worker_kind == "reused" else [1])


def test_worker_archives_supersede_a_copied_older_installation(broadcast_files):
    utils, context, driver = broadcast_files
    name = "_test_copied_installation"
    old_revision = Path(utils.broadcast_code("value = 1\n", name))
    old_alias = (driver / f"{name}.py").read_bytes()
    old_state = Path(utils._publication_file(str(driver), name)).read_bytes()
    utils.broadcast_code("value = 2\n", name)
    archive = context.addPyFile.call_args.args[0]
    worker = driver.parent / "copied-installation"
    worker.mkdir()
    (worker / f"{name}.py").write_bytes(old_alias)
    (worker / old_revision.name).write_bytes(old_revision.read_bytes())
    state = Path(utils._publication_file(str(worker), name))
    state.parent.mkdir()
    state.write_bytes(old_state)
    script = """
import importlib
import sys
package = importlib.import_module("flaml.tune.spark")
package.__path__ = type(package.__path__)([sys.argv[1]])
sys.path.insert(0, sys.argv[3])
print(importlib.import_module("flaml.tune.spark." + sys.argv[2]).value)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(worker), name, archive],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "2"


def test_publication_holds_a_process_lock_through_distribution(broadcast_files):
    utils, context, driver = broadcast_files
    name = "_test_process_lock"
    observed = []
    script = """
import errno
import os
import sys
with open(sys.argv[1], "r+b") as stream:
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise
        print("blocked")
    else:
        print("acquired")
"""

    def probe_lock(path):
        result = subprocess.run(
            [sys.executable, "-c", script, utils._publication_file(str(driver), name) + ".lock"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        observed.append(result.stdout.strip())

    context.addPyFile.side_effect = probe_lock
    utils.broadcast_code("value = 1\n", name)
    utils.broadcast_code("value = 2\n", name)
    assert observed == ["blocked", "blocked"]


def test_competing_publication_helpers_wait_for_distribution_to_finish(broadcast_files):
    utils, context, driver = broadcast_files
    name = "_test_contended_helper"
    contenders = []
    script = """
import sys
from pathlib import Path
from flaml.tune.spark.utils import _publication_file_lock
Path(sys.argv[3]).write_text("ready")
with _publication_file_lock(sys.argv[1], sys.argv[2]):
    print("acquired", flush=True)
"""

    def contend(path):
        ready = driver / f"ready-{len(contenders)}"
        process = subprocess.Popen(
            [sys.executable, "-c", script, utils._publication_file(str(driver), name), name, str(ready)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        contenders.append(process)
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "The competing publisher did not reach lock acquisition."
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.3)

    context.addPyFile.side_effect = contend
    try:
        for value in (1, 2):
            utils.broadcast_code(f"value = {value}\n", name)
            output, errors = contenders[-1].communicate(timeout=30)
            assert contenders[-1].returncode == 0, errors
            assert output.strip().splitlines()[-1] == "acquired"
    finally:
        for process in contenders:
            if process.poll() is None:
                process.kill()
                process.communicate()
