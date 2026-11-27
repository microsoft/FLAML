import atexit
import importlib
import os
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from timeit import timeit
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

try:
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    import pyspark
    import pyspark.pandas as ps
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import SparkSession

    from flaml.automl.ml import sklearn_metric_loss_score
    from flaml.automl.spark import disable_spark_ansi_mode, restore_spark_ansi_mode
    from flaml.automl.spark.metrics import spark_metric_loss_score
    from flaml.automl.spark.utils import (
        iloc_pandas_on_spark,
        len_labels,
        to_pandas_on_spark,
        train_test_split_pyspark,
        unique_pandas_on_spark,
        unique_value_first_index,
    )
    from flaml.tune.spark.utils import (
        _spark_major_minor_version,
        check_spark,
        get_broadcast_data,
        get_n_cpus,
        with_parameters,
    )

    spark_available, _ = check_spark()
    skip_spark = not spark_available
except ImportError:
    print("Spark is not installed. Skip all spark tests.")
    skip_spark = True
    _spark_major_minor_version = (0, 0)


pytestmark = [pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests."), pytest.mark.spark]


@pytest.mark.skipif(_spark_major_minor_version[0] < 4, reason="Requires Spark 4.0+")
def test_to_pandas_on_spark_temp_override():
    import pyspark.pandas as ps
    from pyspark.sql import Row

    from flaml.automl.spark.utils import to_pandas_on_spark

    spark_session = SparkSession.builder.getOrCreate()
    spark, ansi_conf, adjusted = disable_spark_ansi_mode()
    atexit.register(restore_spark_ansi_mode, spark, ansi_conf, adjusted)

    # Ensure we can toggle options
    orig = ps.get_option("compute.fail_on_ansi_mode")

    try:
        spark_session.conf.set("spark.sql.ansi.enabled", "true")
        ps.set_option("compute.fail_on_ansi_mode", True)

        # create tiny spark df
        sdf = spark_session.createDataFrame([Row(a=1, b=2)])
        # Should not raise as our function temporarily disables fail_on_ansi_mode
        pds = to_pandas_on_spark(sdf)
        assert "a" in pds.columns
    finally:
        # restore test environment
        ps.set_option("compute.fail_on_ansi_mode", orig)
        spark_session.conf.set("spark.sql.ansi.enabled", "false")


def test_with_parameters_spark():
    def train(config, data=None):
        if isinstance(data, pyspark.broadcast.Broadcast):
            data = data.value
        print(config, len(data))

    data = ["a"] * 10**6

    with_parameters_train = with_parameters(train, data=data)
    partial_train = partial(train, data=data)

    spark = SparkSession.builder.getOrCreate()
    rdd = spark.sparkContext.parallelize(list(range(2)))

    t_partial = timeit(lambda: rdd.map(lambda x: partial_train(config=x)).collect(), number=5)
    print("python_partial_train: " + str(t_partial))

    t_spark = timeit(
        lambda: rdd.map(lambda x: with_parameters_train(config=x)).collect(),
        number=5,
    )
    print("spark_with_parameters_train: " + str(t_spark))

    # assert t_spark < t_partial


def test_get_n_cpus_spark():
    n_cpus = get_n_cpus()
    assert isinstance(n_cpus, int)


@contextmanager
def _publication_namespace(tmp_path):
    package = importlib.import_module("flaml.tune.spark")
    utils = importlib.import_module("flaml.tune.spark.utils")
    name = f"_test_broadcast_learner_{uuid4().hex}"
    directory = tmp_path / name
    directory.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(utils, "__file__", str(directory / "utils.py"))
        patch.setattr(package, "__path__", type(package.__path__)([*package.__path__, str(directory)]))
        try:
            yield name, directory
        finally:
            package.__dict__.pop(name, None)
            utils._published_code_revisions.pop(name, None)
            utils._published_code_aliases.discard(name)
            for module in list(sys.modules):
                if module == f"flaml.tune.spark.{name}" or module.startswith(f"flaml.tune.spark._flaml_custom_{name}_"):
                    sys.modules.pop(module, None)


@pytest.fixture
def publication_namespace(tmp_path):
    with _publication_namespace(tmp_path) as namespace:
        yield namespace


def test_broadcast_code(publication_namespace):
    from flaml.automl.model import LGBMEstimator
    from flaml.tune.spark.utils import broadcast_code

    custom_code = """
    from flaml.automl.model import LGBMEstimator
    from flaml import tune

    class MyLargeLGBM(LGBMEstimator):
        @classmethod
        def search_space(cls, **params):
            return {
                "n_estimators": {
                    "domain": tune.lograndint(lower=4, upper=32768),
                    "init_value": 32768,
                    "low_cost_init_value": 4,
                },
                "num_leaves": {
                    "domain": tune.lograndint(lower=4, upper=32768),
                    "init_value": 32768,
                    "low_cost_init_value": 4,
                },
            }
    """

    file_name, _ = publication_namespace
    qualified_name = f"flaml.tune.spark.{file_name}"
    custom_path = broadcast_code(custom_code=custom_code, file_name=file_name)
    paths = [custom_path, os.path.join(os.path.dirname(custom_path), file_name + ".py")]
    broadcast_learner = None
    broadcast_revisions = None
    try:
        module = importlib.import_module(qualified_name)
        first_learner = module.MyLargeLGBM
        assert isinstance(first_learner(), LGBMEstimator)
        context = SparkSession.builder.getOrCreate().sparkContext
        assert context.getConf().get("spark.python.worker.reuse", "true").lower() == "true"
        broadcast_learner = context.broadcast(first_learner)
        reuse_marker = f"_flaml_test_reuse_{file_name}"

        def verify_import_on_worker(_):
            import importlib
            import os
            import socket
            import sys

            from pyspark import BarrierTaskContext

            BarrierTaskContext.get().barrier()
            learner = broadcast_learner.value
            assert isinstance(learner(), LGBMEstimator)
            public = importlib.import_module(qualified_name)
            assert public.MyLargeLGBM is learner
            setattr(sys.modules["flaml.tune.spark"], reuse_marker, (learner.__module__, id(public)))
            return [(socket.gethostname(), os.getpid(), learner.__module__)]

        workers = os.environ.get("FLAML_TEST_SPARK_WORKERS")
        partitions = min(context.defaultParallelism, int(workers or "2"))
        results = (
            context.parallelize(range(partitions), partitions)
            .barrier()
            .mapPartitions(verify_import_on_worker)
            .collect()
        )
        assert len(results) == partitions
        assert all(module_name == first_learner.__module__ for _, _, module_name in results)
        if workers:
            assert len({host for host, _, _ in results}) == int(workers)
        updated_path = broadcast_code(custom_code.replace("32768", "16384"), file_name)
        paths.append(updated_path)
        updated = importlib.import_module(qualified_name).MyLargeLGBM
        broadcast_revisions = context.broadcast((first_learner, updated))

        def verify_revisions_on_reused_worker(_, cleanup=False):
            import importlib
            import os
            import socket
            import sys

            from pyspark import BarrierTaskContext

            BarrierTaskContext.get().barrier()
            before, after = broadcast_revisions.value
            marker = vars(sys.modules["flaml.tune.spark"])[reuse_marker]
            public = importlib.import_module(qualified_name)
            assert marker == (before.__module__, id(public))
            if cleanup:
                vars(sys.modules["flaml.tune.spark"]).pop(reuse_marker)
            return [
                (
                    socket.gethostname(),
                    os.getpid(),
                    before.search_space()["n_estimators"]["init_value"],
                    after.search_space()["n_estimators"]["init_value"],
                    public.MyLargeLGBM.search_space()["n_estimators"]["init_value"],
                )
            ]

        revisions = (
            context.parallelize(range(partitions), partitions)
            .barrier()
            .mapPartitions(verify_revisions_on_reused_worker)
            .collect()
        )
        assert all((before, after, current) == (32768, 16384, 16384) for _, _, before, after, current in revisions)
        assert {(host, pid) for host, pid, _, _, _ in revisions} == {(host, pid) for host, pid, _ in results}
        if workers:
            assert {host for host, _, _, _, _ in revisions} == {host for host, _, _ in results}

        broadcast_code(custom_code, file_name)
        assert importlib.import_module(qualified_name).MyLargeLGBM is first_learner
        reverted = (
            context.parallelize(range(partitions), partitions)
            .barrier()
            .mapPartitions(partial(verify_revisions_on_reused_worker, cleanup=True))
            .collect()
        )
        assert all((before, after, current) == (32768, 16384, 32768) for _, _, before, after, current in reverted)
        assert {(host, pid) for host, pid, _, _, _ in reverted} == {(host, pid) for host, pid, _ in results}
    finally:
        if broadcast_learner is not None:
            broadcast_learner.destroy()
        if broadcast_revisions is not None:
            broadcast_revisions.destroy()
        sys.modules.pop(qualified_name, None)
        for path in paths:
            for owned_path in (path, importlib.util.cache_from_source(path)):
                if os.path.exists(owned_path):
                    os.remove(owned_path)


@pytest.mark.parametrize("prior_success", [False, True])
def test_broadcast_failure_cannot_pollute_the_installed_package(tmp_path, monkeypatch, prior_success):
    package = importlib.import_module("flaml.tune.spark")
    utils = importlib.import_module("flaml.tune.spark.utils")
    installed_directory = Path(utils.__file__).parent
    context = SparkSession.builder.getOrCreate().sparkContext
    with pytest.raises(RuntimeError, match="injected distribution failure"):
        with _publication_namespace(tmp_path) as (name, directory):
            if prior_success:
                monkeypatch.setattr(context, "addPyFile", lambda path: None)
                utils.broadcast_code("value = 1\n", name)
                assert importlib.import_module(f"flaml.tune.spark.{name}").value == 1

            def fail(path):
                assert list(directory.glob(f"_flaml_custom_{name}_*.py"))
                raise RuntimeError("injected distribution failure")

            monkeypatch.setattr(context, "addPyFile", fail)
            utils.broadcast_code("value = 2\n", name)
    assert not list(installed_directory.glob(f"*{name}*.py"))
    assert name not in vars(package)
    assert name not in utils._published_code_revisions
    assert name not in utils._published_code_aliases
    assert not any(
        module == f"flaml.tune.spark.{name}" or module.startswith(f"flaml.tune.spark._flaml_custom_{name}_")
        for module in sys.modules
    )


def test_get_broadcast_data():
    data = ["a"] * 10
    spark = SparkSession.builder.getOrCreate()
    bc_data = spark.sparkContext.broadcast(data)
    assert get_broadcast_data(bc_data) == data


def test_to_pandas_on_spark(capsys):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    psdf = to_pandas_on_spark(pdf)
    print(psdf)
    captured = capsys.readouterr()
    assert captured.out == "   a  b\n0  1  4\n1  2  5\n2  3  6\n"
    assert isinstance(psdf, ps.DataFrame)

    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf)
    psdf = to_pandas_on_spark(sdf)
    print(psdf)
    captured = capsys.readouterr()
    assert captured.out == "   a  b\n0  1  4\n1  2  5\n2  3  6\n"
    assert isinstance(psdf, ps.DataFrame)

    pds = pd.Series([1, 2, 3])
    pss = to_pandas_on_spark(pds)
    print(pss)
    captured = capsys.readouterr()
    assert captured.out == "0    1\n1    2\n2    3\ndtype: int64\n"
    assert isinstance(pss, ps.Series)


def test_train_test_split_pyspark():
    pdf = pd.DataFrame({"x": range(2000), "y": [0, 1] * 1000})
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf).coalesce(1).cache()
    try:
        # Both lazy split branches must sample the same ordered input snapshot.
        sdf.count()
        psdf = to_pandas_on_spark(sdf)
        train_sdf, test_sdf = train_test_split_pyspark(sdf, test_fraction=0.5, to_pandas_spark=False, seed=1)
        train_psdf, test_psdf = train_test_split_pyspark(psdf, test_fraction=0.5, stratify_column="y", seed=1)
        assert isinstance(train_sdf, pyspark.sql.dataframe.DataFrame)
        assert isinstance(test_sdf, pyspark.sql.dataframe.DataFrame)
        assert isinstance(train_psdf, ps.DataFrame)
        assert isinstance(test_psdf, ps.DataFrame)
        expected_rows = set(zip(pdf["x"], pdf["y"]))
        # Spark sampling is approximate, not an exact-size split of each stratum.
        for kind, train, test in [
            ("spark", train_sdf, test_sdf),
            ("pandas", train_psdf.to_spark(), test_psdf.to_spark()),
        ]:
            train_rows = [(row.x, row.y) for row in train.select("x", "y").collect()]
            test_rows = [(row.x, row.y) for row in test.select("x", "y").collect()]
            assert len(train_rows) + len(test_rows) == len(pdf), kind
            assert set(train_rows).isdisjoint(test_rows), kind
            assert set(train_rows) | set(test_rows) == expected_rows, kind
            for label in (0, 1):
                assert sum(row[1] == label for row in test_rows) / 1000 == pytest.approx(0.5, abs=0.1), kind
    finally:
        sdf.unpersist()


def test_unique_pandas_on_spark():
    pdf = pd.DataFrame({"x": [1, 2, 2, 3], "y": [0, 1, 1, 0]})
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf)
    psdf = to_pandas_on_spark(sdf)
    label_set, counts = unique_pandas_on_spark(psdf)
    assert len(label_set) == len(counts) == 3
    assert dict(zip(label_set, counts)) == {1: 1, 2: 2, 3: 1}
    assert np.array_equal(counts, np.array([2, 1, 1]))


def test_len_labels():
    y1 = np.array([1, 2, 5, 4, 5])
    y2 = ps.Series([1, 2, 5, 4, 5])
    assert len_labels(y1) == 4
    ll, la = len_labels(y2, return_labels=True)
    assert ll == 4
    assert set(la.to_numpy()) == {1, 2, 5, 4}


def test_lightweight_label_counter_keeps_distributed_work_in_spark(monkeypatch):
    from flaml.automl.utils import len_labels as count_labels

    frame = SparkSession.builder.getOrCreate().sql("SELECT id % 3 AS label FROM range(1000)")
    labels = to_pandas_on_spark(frame)["label"]
    original = np.unique

    def unique(values, *args, **kwargs):
        assert values is not labels, "The complete distributed label series was sent to NumPy."
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "unique", unique)
    assert count_labels(labels) == 3
    count, distinct = count_labels(labels, return_labels=True)
    assert count == 3 and isinstance(distinct, type(labels))
    assert set(distinct.to_pandas()) == {0, 1, 2}


def test_unique_value_first_index():
    y1 = np.array([1, 2, 5, 4, 5])
    y2 = ps.Series([1, 2, 5, 4, 5])
    l1, f1 = unique_value_first_index(y1)
    l2, f2 = unique_value_first_index(y2)
    assert np.array_equal(l1, np.array([1, 2, 4, 5]))
    assert np.array_equal(f1, np.array([0, 1, 3, 2]))
    assert np.array_equal(l2, np.array([1, 2, 5, 4]))
    assert np.array_equal(f2, np.array([0, 1, 2, 3]))


def test_n_current_trials():
    spark = SparkSession.builder.getOrCreate()
    sc = spark._jsc.sc()
    num_executors = len([executor.host() for executor in sc.statusTracker().getExecutorInfos()]) - 1

    def get_n_current_trials(n_concurrent_trials=0, num_executors=num_executors):
        try:
            FLAML_MAX_CONCURRENT = int(os.getenv("FLAML_MAX_CONCURRENT", 0))
        except ValueError:
            FLAML_MAX_CONCURRENT = 0
        num_executors = max(num_executors, FLAML_MAX_CONCURRENT, 1)
        max_spark_parallelism = max(spark.sparkContext.defaultParallelism, FLAML_MAX_CONCURRENT)
        max_concurrent = max(1, max_spark_parallelism)
        n_concurrent_trials = min(
            n_concurrent_trials if n_concurrent_trials > 0 else num_executors,
            max_concurrent,
        )
        print("n_concurrent_trials:", n_concurrent_trials)
        return n_concurrent_trials

    os.environ["FLAML_MAX_CONCURRENT"] = "invlaid"
    assert get_n_current_trials() == max(num_executors, 1)
    tmp_max = spark.sparkContext.defaultParallelism
    assert get_n_current_trials(1) == 1
    assert get_n_current_trials(2) == min(2, tmp_max)
    assert get_n_current_trials(50) == min(50, tmp_max)
    assert get_n_current_trials(200) == min(200, tmp_max)
    os.environ["FLAML_MAX_CONCURRENT"] = "0"
    assert get_n_current_trials() == max(num_executors, 1)
    os.environ["FLAML_MAX_CONCURRENT"] = "4"
    tmp_max = max(4, spark.sparkContext.defaultParallelism)
    assert get_n_current_trials() == min(4, tmp_max)
    os.environ["FLAML_MAX_CONCURRENT"] = "9999999"
    assert get_n_current_trials() == 9999999
    os.environ["FLAML_MAX_CONCURRENT"] = "100"
    tmp_max = max(100, spark.sparkContext.defaultParallelism)
    assert get_n_current_trials(1) == 1
    assert get_n_current_trials(2) == min(2, tmp_max)
    assert get_n_current_trials(50) == min(50, tmp_max)
    assert get_n_current_trials(200) == min(200, tmp_max)
    del os.environ["FLAML_MAX_CONCURRENT"]


def test_iloc_pandas_on_spark():
    psdf = ps.DataFrame({"x": [1, 2, 2, 3], "y": [0, 1, 1, 0]}, index=[0, 1, 2, 3])
    psds = ps.Series([1, 2, 2, 3], index=[0, 1, 2, 3])
    assert iloc_pandas_on_spark(psdf, 0).tolist() == [1, 0]
    d1 = iloc_pandas_on_spark(psdf, slice(1, 3)).to_pandas()
    d2 = pd.DataFrame({"x": [2, 2], "y": [1, 1]}, index=[1, 2])
    assert d1.equals(d2)
    d1 = iloc_pandas_on_spark(psdf, [1, 3]).to_pandas()
    d2 = pd.DataFrame({"x": [2, 3], "y": [1, 0]}, index=[0, 1])
    assert d1.equals(d2)
    assert iloc_pandas_on_spark(psds, 0) == 1
    assert iloc_pandas_on_spark(psds, slice(1, 3)).tolist() == [2, 2]
    assert iloc_pandas_on_spark(psds, [0, 3]).tolist() == [1, 3]


def test_spark_metric_loss_score():
    spark = SparkSession.builder.getOrCreate()
    scoreAndLabels = map(
        lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
        [
            (0.1, 0.0),
            (0.1, 1.0),
            (0.4, 0.0),
            (0.6, 0.0),
            (0.6, 1.0),
            (0.6, 1.0),
            (0.8, 1.0),
        ],
    )
    dataset = spark.createDataFrame(scoreAndLabels, ["raw", "label"])
    dataset = to_pandas_on_spark(dataset)
    # test pr_auc
    metric = spark_metric_loss_score(
        "pr_auc",
        dataset["raw"],
        dataset["label"],
    )
    print("pr_auc: ", metric)
    assert str(metric)[:5] == "0.166"
    # test roc_auc
    metric = spark_metric_loss_score(
        "roc_auc",
        dataset["raw"],
        dataset["label"],
    )
    print("roc_auc: ", metric)
    assert str(metric)[:5] == "0.291"

    scoreAndLabels = [
        (-28.98343821, -27.0),
        (20.21491975, 21.5),
        (-25.98418959, -22.0),
        (30.69731842, 33.0),
        (74.69283752, 71.0),
    ]
    dataset = spark.createDataFrame(scoreAndLabels, ["raw", "label"])
    dataset = to_pandas_on_spark(dataset)
    # test rmse
    metric = spark_metric_loss_score(
        "rmse",
        dataset["raw"],
        dataset["label"],
    )
    print("rmse: ", metric)
    assert str(metric)[:5] == "2.842"
    # test mae
    metric = spark_metric_loss_score(
        "mae",
        dataset["raw"],
        dataset["label"],
    )
    print("mae: ", metric)
    assert str(metric)[:5] == "2.649"
    # test r2
    metric = spark_metric_loss_score(
        "r2",
        dataset["raw"],
        dataset["label"],
    )
    print("r2: ", metric)
    assert str(metric)[:5] == "0.006"
    # test mse
    metric = spark_metric_loss_score(
        "mse",
        dataset["raw"],
        dataset["label"],
    )
    print("mse: ", metric)
    assert str(metric)[:5] == "8.079"
    # test var
    metric = spark_metric_loss_score(
        "var",
        dataset["raw"],
        dataset["label"],
    )
    print("var: ", metric)
    assert str(metric)[:5] == "-1489"

    predictionAndLabelsWithProbabilities = [
        (1.0, 1.0, 1.0, [0.1, 0.8, 0.1]),
        (0.0, 2.0, 1.0, [0.9, 0.05, 0.05]),
        (0.0, 0.0, 1.0, [0.8, 0.2, 0.0]),
        (1.0, 1.0, 1.0, [0.3, 0.65, 0.05]),
    ]
    dataset = spark.createDataFrame(
        predictionAndLabelsWithProbabilities,
        ["prediction", "label", "weight", "probability"],
    )
    dataset = to_pandas_on_spark(dataset)
    # test logloss
    metric = spark_metric_loss_score(
        "log_loss",
        dataset["probability"],
        dataset["label"],
    )
    print("log_loss: ", metric)
    assert str(metric)[:5] == "0.968"
    # test accuracy
    metric = spark_metric_loss_score(
        "accuracy",
        dataset["prediction"],
        dataset["label"],
    )
    print("accuracy: ", metric)
    assert str(metric)[:5] == "0.25"
    # test f1
    metric = spark_metric_loss_score(
        "f1",
        dataset["prediction"],
        dataset["label"],
    )
    print("f1: ", metric)
    assert str(metric)[:5] == "0.333"

    scoreAndLabels = [
        ([0.0, 1.0], [0.0, 2.0]),
        ([0.0, 2.0], [0.0, 1.0]),
        ([], [0.0]),
        ([2.0], [2.0]),
        ([2.0, 0.0], [2.0, 0.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0]),
        ([1.0], [1.0, 2.0]),
    ]
    dataset = spark.createDataFrame(scoreAndLabels, ["prediction", "label"])
    dataset = to_pandas_on_spark(dataset)
    # test micro_f1
    metric = spark_metric_loss_score(
        "micro_f1",
        dataset["prediction"],
        dataset["label"],
    )
    print("micro_f1: ", metric)
    assert str(metric)[:5] == "0.304"
    # test macro_f1
    metric = spark_metric_loss_score(
        "macro_f1",
        dataset["prediction"],
        dataset["label"],
    )
    print("macro_f1: ", metric)
    assert str(metric)[:5] == "0.111"

    scoreAndLabels = [
        (
            [1.0, 6.0, 2.0, 7.0, 8.0, 3.0, 9.0, 10.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ),
        ([4.0, 1.0, 5.0, 6.0, 2.0, 7.0, 3.0, 8.0, 9.0, 10.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], []),
    ]
    dataset = spark.createDataFrame(scoreAndLabels, ["prediction", "label"])
    dataset = to_pandas_on_spark(dataset)
    # test ap
    metric = spark_metric_loss_score(
        "ap",
        dataset["prediction"],
        dataset["label"],
    )
    print("ap: ", metric)
    assert str(metric)[:5] == "0.644"
    # test ndcg
    # ndcg is tested in synapseML rank tests, so we don't need to test it here


if __name__ == "__main__":
    # test_with_parameters_spark()
    # test_get_n_cpus_spark()
    # test_broadcast_code()
    # test_get_broadcast_data()
    # test_train_test_split_pyspark()
    test_n_current_trials()
    # test_len_labels()
    # test_iloc_pandas_on_spark()
    test_spark_metric_loss_score()
