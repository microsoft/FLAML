from flaml.utils import with_parameters, check_spark, get_n_cpus
from functools import partial
import sys
from timeit import timeit

try:
    check_spark()
except Exception:
    print("Spark is not installed. Skip all tests in test_utils.py")
    sys.exit(0)

from pyspark.sql import SparkSession
import pyspark


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

    assert t_spark < t_partial


def test_get_n_cpus_spark():
    n_cpus = get_n_cpus()
    assert isinstance(n_cpus, int)


if __name__ == "__main__":
    test_with_parameters_spark()
