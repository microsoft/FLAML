import os
import sys
import time
import warnings
import mlflow
import pytest
import flaml
import importlib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

skip_spark = importlib.util.find_spec("pyspark") is None
client = mlflow.tracking.MlflowClient()


"""
The spark used in below tests should be initiated in test_0sparkml.py when run with pytest.
"""


def _sklearn_tune(config):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    rf = RandomForestRegressor(**config)
    rf.fit(train_x, train_y)
    pred = rf.predict(test_x)
    r2 = r2_score(test_y, pred)
    return {"r2": r2}


def test_tune_nonspark_autolog():
    mlflow_exp_name = "test_mlflow_integration"
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.autolog()
    params = {
        "n_estimators": flaml.tune.randint(100, 1000),
        "min_samples_leaf": flaml.tune.randint(1, 10),
    }
    n_child_runs = 3
    with mlflow.start_run(nested=True, run_name=f"nonspark_auto_trials_{int(time.time())}"):
        analysis = flaml.tune.run(
            _sklearn_tune,
            params,
            metric="r2",
            mode="max",
            num_samples=n_child_runs,
            verbose=5,
        )

    best_conf = analysis.best_config
    sklearn_r2 = analysis.best_result["r2"]
    print(f"Best config: {best_conf}")
    print(f"R^2: {sklearn_r2}")

    parent_run = mlflow.last_active_run()
    print(parent_run.info)
    mlflow.end_run()

    child_runs = client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string="tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id),
    )
    print(child_runs[0].data.metrics)
    metrics = [len(run.data.metrics) > 0 for run in child_runs]
    mlflow.autolog(disable=True)
    assert all(metrics), "The metrics are not logged correctly."


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_spark_autolog():
    mlflow_exp_name = "test_mlflow_integration"
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.autolog()
    n_child_runs = 5
    params = {
        "n_estimators": flaml.tune.randint(100, 1000),
        "min_samples_leaf": flaml.tune.randint(1, 10),
    }
    with mlflow.start_run(nested=True, run_name=f"spark_auto_trials_{int(time.time())}"):
        analysis = flaml.tune.run(
            _sklearn_tune,
            params,
            metric="r2",
            mode="max",
            num_samples=n_child_runs,
            verbose=5,
            use_spark=True,
            n_concurrent_trials=2,
        )
    best_conf = analysis.best_config
    sklearn_r2 = analysis.best_result["r2"]
    print(f"Best config: {best_conf}")
    print(f"R^2: {sklearn_r2}")

    parent_run = mlflow.last_active_run()
    print(parent_run.info)

    child_runs = client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string="tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id),
    )
    print(child_runs[0].data.metrics)
    metrics = [len(run.data.metrics) > 0 for run in child_runs]
    mlflow.autolog(disable=True)
    assert all(metrics), "The metrics are not logged correctly."


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparktrial_autolog():
    mlflow_exp_name = "test_mlflow_integration"
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.autolog()
    automl_experiment = flaml.AutoML()
    automl_settings = {
        "max_iter": 3,
        "metric": "r2",
        "task": "regression",
        "n_concurrent_trials": 2,
        "use_spark": True,
        "estimator_list": [
            "lgbm",
            "rf",
            "xgboost",
            "extra_tree",
            "xgb_limitdepth",
        ],  # catboost is not yet support mlflow autologging
    }
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    with mlflow.start_run(nested=True, run_name=f"automl_spark_trials_{int(time.time())}"):
        automl_experiment.fit(X_train=train_x, y_train=train_y, **automl_settings)
    print(automl_experiment.model)
    print(automl_experiment.config_history)
    print(automl_experiment.best_iteration)
    print(automl_experiment.best_estimator)
    mlflow.autolog(disable=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_synapseml_autolog():
    import pyspark
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import RegressionEvaluator
    from flaml.automl.spark.utils import to_pandas_on_spark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    exp_name = "test_mlflow_integration"
    # spark.sparkContext.setLogLevel("ERROR")
    mlflow.set_experiment(exp_name)
    mlflow.autolog()
    pd_df = load_diabetes(as_frame=True).frame
    df = spark.createDataFrame(pd_df)
    df = df.repartition(4).cache()
    df.count()
    train, test = df.randomSplit([0.8, 0.2], seed=1)
    feature_cols = df.columns[:-1]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["target", "features"]
    test_data = featurizer.transform(test)["target", "features"]
    automl = flaml.AutoML()
    settings = {
        "max_iter": 3,
        "metric": "mse",
        "estimator_list": ["lgbm_spark"],  # list of ML learners; we tune lightgbm in this example
        "task": "regression",  # task type
        "log_file_name": "flaml_experiment.log",  # flaml log file
        "seed": 7654321,  # random seed
    }
    df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))
    with mlflow.start_run(nested=True, run_name=f"automl_synapseml_{int(time.time())}"):
        automl.fit(
            dataframe=df,
            label="target",
            labelCol="target",
            # isUnbalance=True,
            **settings,
        )

    model = automl.model.estimator
    predictions = model.transform(test_data)
    predictions.show(10)

    evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="mse")
    metric = evaluator.evaluate(predictions)
    print(f"mse: {metric}")
    mlflow.autolog(disable=True)


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
                "com.microsoft.azure:synapseml_2.12:0.10.2,"
                "org.apache.hadoop:hadoop-azure:3.3.5,"
                "com.microsoft.azure:azure-storage:8.6.6,"
                f"org.mlflow:mlflow-spark:{mlflow.__version__}"
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

    test_tune_nonspark_autolog()
    test_tune_spark_autolog()
    test_automl_sparktrial_autolog()
    test_automl_synapseml_autolog()

    test_exit_pyspark_autolog()
