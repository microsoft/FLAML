from flaml.tune.spark.utils import (
    with_parameters,
    check_spark,
    get_n_cpus,
    get_broadcast_data,
)
from flaml.automl.spark.utils import to_pandas_on_spark
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


if __name__ == "__main__":
    test_with_parameters_spark()
    test_get_n_cpus_spark()
    test_broadcast_code()
    test_get_broadcast_data()
