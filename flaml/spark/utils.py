import os
import logging
from functools import partial, lru_cache
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


@lru_cache(maxsize=2)
def check_spark():
    """Check if Spark is installed and running.
    Result of the function will be cached since test once is enough.

    Returns:
        Return True if the check passes, otherwise raise an exception.
    """
    logger.debug("\ncheck Spark installation...This line should appear only once.\n")
    if not _have_spark:
        raise ImportError(
            "use_spark=True requires installation of PySpark. "
            "Please run pip install flaml[spark] and check "
            "[here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html) "
            "for more details about installing Spark."
        )

    if _spark_major_minor_version[0] < 3:
        raise ImportError("Spark version must be >= 3.0 to use flaml[spark]")

    # Raise the original exception if SparkSession is not available
    SparkSession.builder.getOrCreate()

    return True


def get_n_cpus(node="driver"):
    """Get the number of CPU cores of the given type of node.

    Args:
        node: string | The type of node to get the number of cores. Can be 'driver' or 'executor'.
            Default is 'driver'.

    Returns:
        An int of the number of CPU cores.
    """
    assert node in ["driver", "executor"]
    try:
        n_cpus = int(
            SparkSession.builder.getOrCreate()
            .sparkContext.getConf()
            .get(f"spark.{node}.cores")
        )
    except (TypeError, RuntimeError):
        n_cpus = os.cpu_count()
    return n_cpus


def with_parameters(trainable, **kwargs):
    """Wrapper for trainables to pass arbitrary large data objects.

    This wrapper function will store all passed parameters in the Spark
    Broadcast variable.

    Args:
        trainable: Trainable to wrap.
        **kwargs: parameters to store in object store.

    Returns:
        A new function with partial application of the given arguments
        and keywords. The given arguments and keywords will be broadcasted
        to all the executors.


    ```python
    import pyspark
    import flaml
    from sklearn.datasets import load_iris
    def train(config, data=None):
        if isinstance(data, pyspark.broadcast.Broadcast):
            data = data.value
        print(config, data)

    data = load_iris()
    with_parameters_train = flaml.spark.utils.with_parameters(train, data=data)
    with_parameters_train(config=1)
    train(config={"metric": "accuracy"})
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


def customize_learner(learner_code="", file_name="mylearner"):
    """Write customized learner/metric code contents to a file for importing.
    It is necessary for using the customized learner in spark backend.
    The path of the learner file will be returned.

    Args:
        learner_code: str, default="" | code contents of the customized learner.
        file_name: str, default="mylearner" | file name of the customized learner.

    Returns:
        The path of the customized learner file.
    ```python
    import pyspark
    from flaml.spark.utils import customize_learner
    from flaml.automl.model import LGBMEstimator

    learner_code = '''
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
    '''

    customize_learner(learner_code=learner_code)
    from flaml.spark.mylearner import MyLargeLGBM
    assert isinstance(MyLargeLGBM(), LGBMEstimator)
    ```
    """
    flaml_path = os.path.dirname(os.path.abspath(__file__))
    learner_code = textwrap.dedent(learner_code)
    learner_path = os.path.join(flaml_path, file_name + ".py")

    def writefile(_):
        with open(learner_path, "w") as f:
            f.write(learner_code)

    # write learner into a file
    writefile(None)
    return learner_path
