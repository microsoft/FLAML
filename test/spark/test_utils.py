from flaml.utils import with_parameters, check_spark, get_n_cpus
from functools import partial
from timeit import timeit
import pytest

try:
    check_spark()
    skip_spark = False
except Exception:
    print("Spark is not installed. Skip all spark tests.")
    skip_spark = True

from pyspark.sql import SparkSession
import pyspark


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
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


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_get_n_cpus_spark():
    n_cpus = get_n_cpus()
    assert isinstance(n_cpus, int)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_customize_learner():
    from flaml.utils import customize_learner
    from flaml.model import LGBMEstimator

    learner_code = """
    from flaml.model import LGBMEstimator
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

    customize_learner(learner_code=learner_code)
    from flaml.mylearner import MyLargeLGBM

    assert isinstance(MyLargeLGBM(), LGBMEstimator)


if __name__ == "__main__":
    test_with_parameters_spark()
    test_get_n_cpus_spark()
    test_customize_learner()
