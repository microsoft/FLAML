import atexit
import logging
import os

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
try:
    import pyspark
    import pyspark.pandas as ps
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.pandas import DataFrame as psDataFrame
    from pyspark.pandas import Series as psSeries
    from pyspark.pandas import set_option
    from pyspark.sql import DataFrame as sparkDataFrame
    from pyspark.sql import SparkSession
    from pyspark.util import VersionUtils
except ImportError:

    class psDataFrame:
        pass

    F = T = ps = sparkDataFrame = SparkSession = psSeries = psDataFrame
    _spark_major_minor_version = set_option = None
    ERROR = ImportError(
        """Please run pip install flaml[spark]
    and check [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
    for more details about installing Spark."""
    )
else:
    ERROR = None
    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)

try:
    import pandas as pd
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = pd = None


logger = logging.getLogger(__name__)


def disable_spark_ansi_mode():
    """Disable Spark ANSI mode if it is enabled."""
    spark = SparkSession.getActiveSession() if hasattr(SparkSession, "getActiveSession") else None
    adjusted = False
    try:
        ps_conf = ps.get_option("compute.fail_on_ansi_mode")
    except Exception:
        ps_conf = None
    ansi_conf = [None, ps_conf]  # ansi_conf and ps_conf original values
    # Spark may store the config as string 'true'/'false' (or boolean in some contexts)
    if spark is not None:
        ansi_conf[0] = spark.conf.get("spark.sql.ansi.enabled")
        ansi_enabled = (
            (isinstance(ansi_conf[0], str) and ansi_conf[0].lower() == "true")
            or (isinstance(ansi_conf[0], bool) and ansi_conf[0] is True)
            or ansi_conf[0] is None
        )
        try:
            if ansi_enabled:
                logger.debug("Adjusting spark.sql.ansi.enabled to false")
                spark.conf.set("spark.sql.ansi.enabled", "false")
                adjusted = True
        except Exception:
            # If reading/setting options fail for some reason, keep going and let
            # pandas-on-Spark raise a meaningful error later.
            logger.exception("Failed to set spark.sql.ansi.enabled")

    if ansi_conf[1]:
        logger.debug("Adjusting pandas-on-Spark compute.fail_on_ansi_mode to False")
        ps.set_option("compute.fail_on_ansi_mode", False)
        adjusted = True

    return spark, ansi_conf, adjusted


def restore_spark_ansi_mode(spark, ansi_conf, adjusted):
    """Restore Spark ANSI mode to its original setting."""
    # Restore the original spark.sql.ansi.enabled to avoid persistent side-effects.
    if adjusted and spark and ansi_conf[0] is not None:
        try:
            logger.debug(f"Restoring spark.sql.ansi.enabled to {ansi_conf[0]}")
            spark.conf.set("spark.sql.ansi.enabled", ansi_conf[0])
        except Exception:
            logger.exception("Failed to restore spark.sql.ansi.enabled")

    if adjusted and ansi_conf[1]:
        logger.debug(f"Restoring pandas-on-Spark compute.fail_on_ansi_mode to {ansi_conf[1]}")
        ps.set_option("compute.fail_on_ansi_mode", ansi_conf[1])


spark, ansi_conf, adjusted = disable_spark_ansi_mode()
atexit.register(restore_spark_ansi_mode, spark, ansi_conf, adjusted)
