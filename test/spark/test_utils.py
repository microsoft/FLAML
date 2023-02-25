from flaml.tune.spark.utils import (
    with_parameters,
    check_spark,
    get_n_cpus,
    get_broadcast_data,
)
from flaml.automl.spark.utils import (
    to_pandas_on_spark,
    train_test_split_pyspark,
    unique_pandas_on_spark,
    len_labels,
    unique_value_first_index,
)
import numpy as np
import pandas as pd
from functools import partial
from timeit import timeit
import pytest
import os

try:
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    from pyspark.sql import SparkSession
    import pyspark
    import pyspark.pandas as ps

    spark_available, _ = check_spark()
    skip_spark = not spark_available
except ImportError:
    print("Spark is not installed. Skip all spark tests.")
    skip_spark = True

pytestmark = pytest.mark.skipif(
    skip_spark, reason="Spark is not installed. Skip all spark tests."
)


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

    t_partial = timeit(
        lambda: rdd.map(lambda x: partial_train(config=x)).collect(), number=5
    )
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


def test_broadcast_code():
    from flaml.tune.spark.utils import broadcast_code
    from flaml.automl.model import LGBMEstimator

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

    _ = broadcast_code(custom_code=custom_code)
    from flaml.tune.spark.mylearner import MyLargeLGBM

    assert isinstance(MyLargeLGBM(), LGBMEstimator)


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
    pdf = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 1, 0]})
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf)
    psdf = to_pandas_on_spark(sdf)
    train_sdf, test_sdf = train_test_split_pyspark(
        sdf, test_fraction=0.5, to_pandas_spark=False, seed=1
    )
    train_psdf, test_psdf = train_test_split_pyspark(
        psdf, test_fraction=0.5, startify_column="y", seed=1
    )

    out1 = pd.DataFrame({"x": [2, 3, 4], "y": [1, 1, 0]})
    out2 = pd.DataFrame({"x": [1], "y": [0]})
    assert train_sdf.toPandas().equals(out1)
    assert test_sdf.toPandas().equals(out2)
    assert train_psdf.to_pandas().equals(out2)
    assert test_psdf.to_pandas().equals(out1)


def test_unique_pandas_on_spark():
    pdf = pd.DataFrame({"x": [1, 2, 2, 3], "y": [0, 1, 1, 0]})
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf)
    psdf = to_pandas_on_spark(sdf)
    label_set, counts = unique_pandas_on_spark(psdf)
    assert np.array_equal(label_set, np.array([2, 1, 3]))
    assert np.array_equal(counts, np.array([2, 1, 1]))


def test_len_labels():
    y1 = np.array([1, 2, 5, 4, 5])
    y2 = ps.Series([1, 2, 5, 4, 5])
    assert len_labels(y1) == 4
    ll, la = len_labels(y2, return_labels=True)
    assert ll == 4
    assert np.array_equal(la.to_numpy(), np.array([1, 2, 5, 4]))


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
    num_executors = (
        len([executor.host() for executor in sc.statusTracker().getExecutorInfos()]) - 1
    )

    def get_n_current_trials(n_concurrent_trials=0, num_executors=num_executors):
        try:
            FLAML_MAX_CONCURRENT = int(os.getenv("FLAML_MAX_CONCURRENT", 0))
            num_executors = max(num_executors, FLAML_MAX_CONCURRENT, 1)
        except ValueError:
            FLAML_MAX_CONCURRENT = 0
        max_spark_parallelism = (
            min(spark.sparkContext.defaultParallelism, FLAML_MAX_CONCURRENT)
            if FLAML_MAX_CONCURRENT > 0
            else spark.sparkContext.defaultParallelism
        )
        max_concurrent = max(1, max_spark_parallelism)
        n_concurrent_trials = min(
            n_concurrent_trials if n_concurrent_trials > 0 else num_executors,
            max_concurrent,
        )
        print("n_concurrent_trials:", n_concurrent_trials)
        return n_concurrent_trials

    os.environ["FLAML_MAX_CONCURRENT"] = "invlaid"
    assert get_n_current_trials() == num_executors
    os.environ["FLAML_MAX_CONCURRENT"] = "0"
    assert get_n_current_trials() == max(num_executors, 1)
    os.environ["FLAML_MAX_CONCURRENT"] = "4"
    assert get_n_current_trials() == max(num_executors, 4)
    os.environ["FLAML_MAX_CONCURRENT"] = "9999999"
    assert get_n_current_trials() == spark.sparkContext.defaultParallelism
    os.environ["FLAML_MAX_CONCURRENT"] = "100"
    tmp_max = min(100, spark.sparkContext.defaultParallelism)
    assert get_n_current_trials(1) == 1
    assert get_n_current_trials(2) == min(2, tmp_max)
    assert get_n_current_trials(50) == min(50, tmp_max)
    assert get_n_current_trials(200) == min(200, tmp_max)


if __name__ == "__main__":
    test_with_parameters_spark()
    test_get_n_cpus_spark()
    test_broadcast_code()
    test_get_broadcast_data()
    test_train_test_split_pyspark()
    test_n_current_trials()
