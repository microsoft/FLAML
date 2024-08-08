import os
import sys
import unittest
import warnings
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import scipy
from packaging.version import Version
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score
from flaml.tune.spark.utils import check_spark

leaderboard = defaultdict(dict)

warnings.simplefilter(action="ignore")
if sys.platform == "darwin" or "nt" in os.name:
    # skip this test if the platform is not linux
    skip_spark = True
else:
    try:
        import pyspark
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
        from pyspark.ml.feature import VectorAssembler

        from flaml.automl.spark.utils import to_pandas_on_spark

        spark = (
            pyspark.sql.SparkSession.builder.appName("MyApp")
            .master("local[2]")
            .config(
                "spark.jars.packages",
                (
                    "com.microsoft.azure:synapseml_2.12:1.0.2,"
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
        # spark.sparkContext.setLogLevel("ERROR")
        spark_available, _ = check_spark()
        skip_spark = not spark_available
    except ImportError:
        skip_spark = True


def _test_regular_models(estimator_list, task):
    if isinstance(estimator_list, str):
        estimator_list = [estimator_list]
    if task == "classification":
        load_dataset_func = load_iris
        metric = "accuracy"
    else:
        load_dataset_func = load_diabetes
        metric = "r2"

    x, y = load_dataset_func(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7654321)

    automl_experiment = AutoML()
    automl_settings = {
        "max_iter": 5,
        "task": task,
        "estimator_list": estimator_list,
        "metric": metric,
    }
    automl_experiment.fit(X_train=x_train, y_train=y_train, **automl_settings)
    predictions = automl_experiment.predict(x_test)
    score = sklearn_metric_loss_score(metric, predictions, y_test)
    for estimator_name in estimator_list:
        leaderboard[task][estimator_name] = score


def _test_spark_models(estimator_list, task):
    if isinstance(estimator_list, str):
        estimator_list = [estimator_list]
    if task == "classification":
        load_dataset_func = load_iris
        evaluator = MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="accuracy"
        )
        metric = "accuracy"

    elif task == "regression":
        load_dataset_func = load_diabetes
        evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="r2")
        metric = "r2"

    elif task == "binary":
        load_dataset_func = load_breast_cancer
        evaluator = MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="accuracy"
        )
        metric = "accuracy"

    final_cols = ["target", "features"]
    extra_args = {}

    if estimator_list is not None and "aft_spark" in estimator_list:
        # survival analysis task
        pd_df = pd.read_csv(
            "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/datasets/rossi.csv"
        )
        pd_df.rename(columns={"week": "target"}, inplace=True)
        final_cols += ["arrest"]
        extra_args["censorCol"] = "arrest"
    else:
        pd_df = load_dataset_func(as_frame=True).frame

    rename = {}
    for attr in pd_df.columns:
        rename[attr] = attr.replace(" ", "_")
    pd_df = pd_df.rename(columns=rename)
    df = spark.createDataFrame(pd_df)
    df = df.repartition(4)
    train, test = df.randomSplit([0.8, 0.2], seed=7654321)
    feature_cols = [col for col in df.columns if col not in ["target", "arrest"]]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)[final_cols]
    test_data = featurizer.transform(test)[final_cols]
    automl = AutoML()
    settings = {
        "max_iter": 1,
        "estimator_list": estimator_list,  # ML learner we intend to test
        "task": task,  # task type
        "metric": metric,  # metric to optimize
    }
    settings.update(extra_args)
    df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))

    automl.fit(
        dataframe=df,
        label="target",
        **settings,
    )

    model = automl.model.estimator
    predictions = model.transform(test_data)
    predictions.show(5)

    score = evaluator.evaluate(predictions)
    if estimator_list is not None:
        for estimator_name in estimator_list:
            leaderboard[task][estimator_name] = score


def _test_sparse_matrix_classification(estimator):
    automl_experiment = AutoML()
    automl_settings = {
        "estimator_list": [estimator],
        "time_budget": 2,
        "metric": "auto",
        "task": "classification",
        "log_file_name": "test/sparse_classification.log",
        "split_type": "uniform",
        "n_jobs": 1,
        "model_history": True,
    }
    X_train = scipy.sparse.random(1554, 21, dtype=int)
    y_train = np.random.randint(3, size=1554)
    automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)


def load_multi_dataset():
    """multivariate time series forecasting dataset"""
    import pandas as pd

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/nyc_energy_consumption.csv"
    )
    # preprocessing data
    df["timeStamp"] = pd.to_datetime(df["timeStamp"])
    df = df.set_index("timeStamp")
    df = df.resample("D").mean()
    df["temp"] = df["temp"].fillna(method="ffill")
    df["precip"] = df["precip"].fillna(method="ffill")
    df = df[:-2]  # last two rows are NaN for 'demand' column so remove them
    df = df.reset_index()

    return df


def _test_forecast(estimator_list, budget=10):
    if isinstance(estimator_list, str):
        estimator_list = [estimator_list]
    df = load_multi_dataset()
    # split data into train and test
    time_horizon = 180
    num_samples = df.shape[0]
    split_idx = num_samples - time_horizon
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    # test dataframe must contain values for the regressors / multivariate variables
    X_test = test_df[["timeStamp", "precip", "temp"]]
    y_test = test_df["demand"]
    # return
    automl = AutoML()
    settings = {
        "time_budget": budget,  # total running time in seconds
        "metric": "mape",  # primary metric
        "task": "ts_forecast",  # task type
        "log_file_name": "test/energy_forecast_numerical.log",  # flaml log file
        "log_dir": "logs/forecast_logs",  # tcn/tft log folder
        "eval_method": "holdout",
        "log_type": "all",
        "label": "demand",
        "estimator_list": estimator_list,
    }
    """The main flaml automl API"""
    automl.fit(dataframe=train_df, **settings, period=time_horizon)
    print(automl.best_config)
    pred_y = automl.predict(X_test)
    mape = sklearn_metric_loss_score("mape", pred_y, y_test)
    for estimator_name in estimator_list:
        leaderboard["forecast"][estimator_name] = mape


class TestExtraModel(unittest.TestCase):
    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_rf_spark(self):
        tasks = ["classification", "regression"]
        for task in tasks:
            _test_spark_models("rf_spark", task)

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_nb_spark(self):
        _test_spark_models("nb_spark", "classification")

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_glr(self):
        _test_spark_models("glr_spark", "regression")

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_lr(self):
        _test_spark_models("lr_spark", "regression")

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_svc_spark(self):
        _test_spark_models("svc_spark", "binary")

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_gbt_spark(self):
        tasks = ["binary", "regression"]
        for task in tasks:
            _test_spark_models("gbt_spark", task)

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_aft(self):
        _test_spark_models("aft_spark", "regression")

    @unittest.skipIf(skip_spark, reason="Spark is not installed. Skip all spark tests.")
    def test_default_spark(self):
        _test_spark_models(None, "classification")

    def test_svc(self):
        _test_regular_models("svc", "classification")
        _test_sparse_matrix_classification("svc")

    def test_sgd(self):
        tasks = ["classification", "regression"]
        for task in tasks:
            _test_regular_models("sgd", task)
        _test_sparse_matrix_classification("sgd")

    def test_enet(self):
        _test_regular_models("enet", "regression")

    def test_lassolars(self):
        _test_regular_models("lassolars", "regression")
        _test_forecast("lassolars")

    def test_seasonal_naive(self):
        _test_forecast("snaive")

    def test_naive(self):
        _test_forecast("naive")

    def test_seasonal_avg(self):
        _test_forecast("savg")

    def test_avg(self):
        _test_forecast("avg")

    def test_tcn(self):
        _test_forecast("tcn")


if __name__ == "__main__":
    unittest.main()
    print(leaderboard)
