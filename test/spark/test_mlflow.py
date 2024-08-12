import importlib
import os
import sys
import time
import warnings

import mlflow
import pytest
from packaging.version import Version
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import flaml
from flaml.automl.spark.utils import to_pandas_on_spark

try:
    import pyspark
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import VectorAssembler
except ImportError:
    pass
warnings.filterwarnings("ignore")

skip_spark = importlib.util.find_spec("pyspark") is None
client = mlflow.tracking.MlflowClient()

"""
The spark used in below tests should be initiated in test_0sparkml.py when run with pytest.
"""


def _sklearn_tune(config):
    is_autolog = config.pop("is_autolog")
    is_parent_run = config.pop("is_parent_run")
    is_parallel = config.pop("is_parallel")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    rf = RandomForestRegressor(**config)
    rf.fit(train_x, train_y)
    pred = rf.predict(test_x)
    r2 = r2_score(test_y, pred)
    if not is_autolog and not is_parent_run and not is_parallel:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("r2", r2)
    return {"r2": r2}


def _test_tune(is_autolog, is_parent_run, is_parallel):
    mlflow_exp_name = f"test_mlflow_integration_{int(time.time())}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    params = {
        "n_estimators": flaml.tune.randint(100, 1000),
        "min_samples_leaf": flaml.tune.randint(1, 10),
        "is_autolog": is_autolog,
        "is_parent_run": is_parent_run,
        "is_parallel": is_parallel,
    }
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"tune_autolog_{is_autolog}_sparktrial_{is_parallel}")
    flaml.tune.run(
        _sklearn_tune,
        params,
        metric="r2",
        mode="max",
        num_samples=3,
        use_spark=True if is_parallel else False,
        n_concurrent_trials=2 if is_parallel else 1,
        mlflow_exp_name=mlflow_exp_name,
    )
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


def _check_mlflow_logging(possible_num_runs, metric, is_parent_run, experiment_id, is_automl=False, skip_tags=False):
    if isinstance(possible_num_runs, int):
        possible_num_runs = [possible_num_runs]
    if is_parent_run:
        parent_run = mlflow.last_active_run()
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        )
    else:
        child_runs = client.search_runs(experiment_ids=[experiment_id])
    experiment_name = client.get_experiment(experiment_id).name
    metrics = [metric in run.data.metrics for run in child_runs]
    tags = ["synapseml.flaml.version" in run.data.tags for run in child_runs]
    params = ["learner" in run.data.params for run in child_runs]
    assert (
        len(child_runs) in possible_num_runs
    ), f"The number of child runs is not correct on experiment {experiment_name}."
    if possible_num_runs[0] > 0:
        assert all(metrics), f"The metrics are not logged correctly on experiment {experiment_name}."
        assert (
            all(tags) if not skip_tags else True
        ), f"The tags are not logged correctly on experiment {experiment_name}."
        assert (
            all(params) if is_automl else True
        ), f"The params are not logged correctly on experiment {experiment_name}."
    # mlflow.delete_experiment(experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_autolog_parentrun_parallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=True, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id)


def test_tune_autolog_parentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=True, is_parallel=False)
    _check_mlflow_logging(3, "r2", True, experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_autolog_noparentrun_parallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=False, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", False, experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_noautolog_parentrun_parallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=True, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id)


def test_tune_autolog_noparentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=False, is_parallel=False)
    _check_mlflow_logging(3, "r2", False, experiment_id)


def test_tune_noautolog_parentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=True, is_parallel=False)
    _check_mlflow_logging(3, "r2", True, experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_noautolog_noparentrun_parallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=False, is_parallel=True)
    _check_mlflow_logging(0, "r2", False, experiment_id)


def test_tune_noautolog_noparentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=False, is_parallel=False)
    _check_mlflow_logging(3, "r2", False, experiment_id, skip_tags=True)


def _test_automl_sparkdata(is_autolog, is_parent_run):
    mlflow_exp_name = f"test_mlflow_integration_{int(time.time())}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"automl_sparkdata_autolog_{is_autolog}")
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    pd_df = load_diabetes(as_frame=True).frame
    df = spark.createDataFrame(pd_df)
    df = df.repartition(4).cache()
    train, test = df.randomSplit([0.8, 0.2], seed=1)
    feature_cols = df.columns[:-1]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["target", "features"]
    featurizer.transform(test)["target", "features"]
    automl = flaml.AutoML()
    settings = {
        "max_iter": 3,
        "metric": "mse",
        "task": "regression",  # task type
        "log_file_name": "flaml_experiment.log",  # flaml log file
        "mlflow_exp_name": mlflow_exp_name,
        "log_type": "all",
        "n_splits": 2,
        "model_history": True,
    }
    df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))
    automl.fit(
        dataframe=df,
        label="target",
        **settings,
    )
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


def _test_automl_nonsparkdata(is_autolog, is_parent_run):
    mlflow_exp_name = f"test_mlflow_integration_{int(time.time())}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"automl_nonsparkdata_autolog_{is_autolog}")
    automl_experiment = flaml.AutoML()
    automl_settings = {
        "max_iter": 3,
        "metric": "r2",
        "task": "regression",
        "n_concurrent_trials": 2,
        "use_spark": True,
        "mlflow_exp_name": None if is_parent_run else mlflow_exp_name,
        "log_type": "all",
        "n_splits": 2,
        "model_history": True,
    }
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    automl_experiment.fit(X_train=train_x, y_train=train_y, **automl_settings)
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_autolog_parentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=True, is_parent_run=True)
    _check_mlflow_logging(3, "mse", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_autolog_noparentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=True, is_parent_run=False)
    _check_mlflow_logging(3, "mse", False, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_noautolog_parentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=False, is_parent_run=True)
    _check_mlflow_logging(3, "mse", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_noautolog_noparentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=False, is_parent_run=False)
    _check_mlflow_logging(0, "mse", False, experiment_id, is_automl=True)  # no logging


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_autolog_parentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=True, is_parent_run=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_autolog_noparentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=True, is_parent_run=False)
    _check_mlflow_logging([4, 3], "r2", False, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_noautolog_parentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=False, is_parent_run=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_noautolog_noparentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=False, is_parent_run=False)
    _check_mlflow_logging(0, "r2", False, experiment_id, is_automl=True)  # no logging


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_exit_pyspark_autolog():
    import pyspark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext._gateway.shutdown_callback_server()  # this is to avoid stucking
    mlflow.autolog(disable=True)


def _init_spark_for_main():
    import pyspark

    spark = (
        pyspark.sql.SparkSession.builder.appName("MyApp")
        .master("local[2]")
        .config(
            "spark.jars.packages",
            (
                "com.microsoft.azure:synapseml_2.12:1.0.4,"
                "org.apache.hadoop:hadoop-azure:3.3.5,"
                "com.microsoft.azure:azure-storage:8.6.6,"
                f"org.mlflow:mlflow-spark_2.12:{mlflow.__version__}"
                if Version(mlflow.__version__) >= Version("2.9.0")
                else f"org.mlflow:mlflow-spark:{mlflow.__version__}"
            ),
        )
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


if __name__ == "__main__":
    _init_spark_for_main()

    test_tune_autolog_parentrun_parallel()
    # test_tune_autolog_parentrun_nonparallel()
    # test_tune_autolog_noparentrun_parallel()  # TODO: runs not removed
    # test_tune_noautolog_parentrun_parallel()
    # test_tune_autolog_noparentrun_nonparallel()
    # test_tune_noautolog_parentrun_nonparallel()
    # test_tune_noautolog_noparentrun_parallel()
    # test_tune_noautolog_noparentrun_nonparallel()
    # test_automl_sparkdata_autolog_parentrun()
    # test_automl_sparkdata_autolog_noparentrun()
    # test_automl_sparkdata_noautolog_parentrun()
    # test_automl_sparkdata_noautolog_noparentrun()
    # test_automl_nonsparkdata_autolog_parentrun()
    # test_automl_nonsparkdata_autolog_noparentrun()  # TODO: runs not removed
    # test_automl_nonsparkdata_noautolog_parentrun()
    # test_automl_nonsparkdata_noautolog_noparentrun()

    test_exit_pyspark_autolog()
