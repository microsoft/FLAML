"""Single source of truth for setting up the Spark session shared by FLAML's
Spark-dependent test modules.

Several test files (``test/spark/test_0sparkml.py``,
``test/spark/test_internal_mlflow.py``, ``test/automl/test_extra_models.py``)
historically duplicated the same large ``SparkSession.builder`` configuration
(synapseml + hadoop-azure + mlflow-spark JARs, MMLSpark Maven repo,
log_model_allowlistFile, ANSI-mode handling). Keeping the duplicated copies
in sync turned out to be brittle — this helper centralises it.

The module name starts with an underscore so pytest will not try to collect
it as a test module.
"""

from __future__ import annotations

import atexit
import os
import sys
import warnings

import mlflow
from packaging.version import Version

from flaml.automl.spark import disable_spark_ansi_mode, restore_spark_ansi_mode
from flaml.tune.spark.utils import check_spark


def _spark_jars_packages() -> str:
    """Maven coordinates for the JARs added to the Spark session."""
    return (
        "com.microsoft.azure:synapseml_2.12:1.0.14,"
        "org.apache.hadoop:hadoop-azure:3.3.5,"
        "com.microsoft.azure:azure-storage:8.6.6,"
        f"org.mlflow:mlflow-spark_2.12:{mlflow.__version__}"
        if Version(mlflow.__version__) >= Version("2.9.0")
        else f"org.mlflow:mlflow-spark:{mlflow.__version__}"
    )


def init_spark_session(app_name: str = "MyApp", master: str = "local[2]"):
    """Build (or fetch) the SparkSession used by FLAML's Spark tests.

    Imports ``pyspark`` lazily so this module remains importable in
    environments where pyspark is not installed (collection-time safety).
    """
    import pyspark

    spark = (
        pyspark.sql.SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.jars.packages", _spark_jars_packages())
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.sql.debug.maxToStringFields", "100")
        .config("spark.driver.extraJavaOptions", "-Xss1m")
        .config("spark.executor.extraJavaOptions", "-Xss1m")
        .getOrCreate()
    )
    spark.sparkContext._conf.set(
        "spark.mlflow.pysparkml.autolog.logModelAllowlistFile",
        "https://mmlspark.blob.core.windows.net/publicwasb/log_model_allowlist.txt",
    )
    return spark


def setup_spark_for_tests(
    app_name: str = "MyApp",
    master: str = "local[2]",
) -> tuple[object | None, bool]:
    """Initialise Spark for the importing test module.

    Returns ``(spark, skip_spark)``:

    * ``spark``      -- the configured ``SparkSession`` (or ``None`` when
                        pyspark is not available, e.g. on macOS / Windows
                        runners or when the package is not installed).
    * ``skip_spark`` -- ``True`` when the Spark-dependent tests in the caller
                        should be skipped. Suitable for use with
                        ``pytest.mark.skipif(skip_spark, ...)``.

    ANSI-mode handling matches the prior duplicated setup: it is invoked
    unconditionally and ``restore_spark_ansi_mode`` is registered with
    ``atexit`` (this is a no-op when no active Spark session exists).
    """
    warnings.simplefilter(action="ignore")

    spark: object | None = None
    skip_spark = False

    if sys.platform == "darwin" or "nt" in os.name:
        skip_spark = True
    else:
        try:
            spark = init_spark_session(app_name=app_name, master=master)
            spark_available, _ = check_spark()
            skip_spark = not spark_available
        except ImportError:
            skip_spark = True

    active_spark, ansi_conf, adjusted = disable_spark_ansi_mode()
    atexit.register(restore_spark_ansi_mode, active_spark, ansi_conf, adjusted)

    return spark, skip_spark
