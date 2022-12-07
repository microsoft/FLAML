import os
import logging
from functools import partial
import textwrap

logger = logging.getLogger(__name__)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)

try:
    from pyspark.sql import SparkSession
    from pyspark.util import VersionUtils
    import pyspark

    _have_spark = True
    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)
except ImportError as e:
    logger.debug("Could not import pyspark: %s", e)
    _have_spark = False
    _spark_major_minor_version = (0, 0)


def check_spark():
    if not _have_spark:
        raise ImportError(
            "use_spark=True requires installation of PySpark. "
            "Please run pip install flaml[spark]. More details about installing "
            "[PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)"
        )

    if _spark_major_minor_version[0] < 3:
        raise Exception("Spark version must be >= 3.0 to use flaml[spark]")

    try:
        spark = SparkSession.builder.getOrCreate()
        assert isinstance(spark, SparkSession)
    except Exception:
        raise Exception(
            "Can't start SparkSession. Please check your Spark installation. "
            "More details about installing "
            "[PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)."
        )

    return True


def get_n_cpus():
    """Return the number of CPUs each trial can use."""
    try:
        n_cpus = int(
            SparkSession.builder.getOrCreate()
            .sparkContext.getConf()
            .get("spark.executor.cores")
        )
    except Exception:
        n_cpus = os.cpu_count()
    return n_cpus


def with_parameters(trainable, **kwargs):
    """Wrapper for trainables to pass arbitrary large data objects.

    This wrapper function will store all passed parameters in the Spark
    Broadcast variable.

    Args:
        trainable: Trainable to wrap.
        **kwargs: parameters to store in object store.

    ```python
    import pyspark
    import flaml
    from sklearn.datasets import load_iris
    def train(config, data=None):
        if isinstance(data, pyspark.broadcast.Broadcast):
            data = data.value
        print(config, data)

    data = load_iris()
    with_parameters_train = flaml.utils.with_parameters(train, data=data)
    with_parameters_train(config=1)
    train(config=1)
    ```
    """

    if not callable(trainable):
        raise ValueError(
            f"`with_parameters() only works with function trainables`. "
            f"Got type: "
            f"{type(trainable)}."
        )

    check_spark()
    spark = SparkSession.builder.getOrCreate()

    bc_kwargs = dict()
    for k, v in kwargs.items():
        bc_kwargs[k] = spark.sparkContext.broadcast(v)

    return partial(trainable, **bc_kwargs)


def customize_learner(learner_code="", file_name="mylearner", use_spark=True):
    """Write customized learner code contents to a file for importing.
    It is necessary for using the customized learner in spark backend.

    Args:
        learner_code: str, default="" | code contents of the customized learner.
        file_name: str, default="mylearner" | file name of the customized learner.
        use_spark: bool, default=True | whether to use spark backend. If True, this
            function will write the customized learner code to all the executors.

    ```python
    import pyspark
    from flaml.utils import customize_learner
    from flaml.model import LGBMEstimator

    learner_code = '''
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
    '''

    customize_learner(learner_code=learner_code)
    from flaml.mylearner import MyLargeLGBM
    assert isinstance(MyLargeLGBM(), LGBMEstimator)
    ```
    """
    flaml_path = os.path.dirname(os.path.abspath(__file__))
    learner_code = textwrap.dedent(learner_code)
    learner_path = os.path.join(flaml_path, file_name + ".py")

    def writefile(_):
        with open(learner_path, "w") as f:
            f.write(learner_code)

    if use_spark:
        check_spark()
        spark = SparkSession.builder.getOrCreate()
        sc = spark._jsc.sc()
        num_executors = (
            len([executor.host() for executor in sc.statusTracker().getExecutorInfos()])
            - 1
        )
        sc = spark.sparkContext
        # do two times to make sure all executors have the learner file
        sc.parallelize(range(num_executors * 2)).map(writefile).collect()
    # write learner file to local/driver
    writefile(None)
