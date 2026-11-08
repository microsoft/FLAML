import importlib
import sys
from threading import Barrier
from types import ModuleType, SimpleNamespace

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


def test_spark_tuning_restores_backend_after_error(spark_backend):
    def evaluate(config):
        raise ValueError("trial failed")

    with parallel_backend("threading", n_jobs=2):
        original_backend = get_active_backend()[0]
        with pytest.raises(ValueError, match="trial failed"):
            tune.run(evaluate, config={}, num_samples=1, use_spark=True, n_concurrent_trials=1, verbose=0)
        assert get_active_backend()[0] is original_backend


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
