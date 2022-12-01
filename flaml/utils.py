import os
import logging
from functools import partial

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
            "Please run pip install flaml[spark]"
        )

    if _spark_major_minor_version[0] < 3:
        raise Exception("Spark version must be >= 3.0 to use flaml[spark]")


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
