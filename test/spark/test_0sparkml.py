import os
import sys
import warnings

import mlflow
import pytest
import sklearn.datasets as skds
from packaging.version import Version

from flaml import AutoML
from flaml.tune.spark.utils import check_spark

warnings.simplefilter(action="ignore")
if sys.platform == "darwin" or "nt" in os.name:
    # skip this test if the platform is not linux
    skip_spark = True
else:
    try:
        import pyspark
        from pyspark.ml.feature import VectorAssembler

        from flaml.automl.spark.utils import to_pandas_on_spark

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
            # .config("spark.executor.memory", "48G")
            # .config("spark.driver.memory", "48G")
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

if sys.version_info >= (3, 11):
    skip_py311 = True
else:
    skip_py311 = False

pytestmark = pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")


def _test_spark_synapseml_lightgbm(spark=None, task="classification"):
    if task == "classification":
        metric = "accuracy"
        X_train, y_train = skds.load_iris(return_X_y=True, as_frame=True)
    elif task == "regression":
        metric = "r2"
        X_train, y_train = skds.load_diabetes(return_X_y=True, as_frame=True)
    elif task == "rank":
        metric = "ndcg@5"
        sdf = spark.read.format("parquet").load(
            "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_test.parquet"
        )
        df = to_pandas_on_spark(sdf)
        X_train = df.drop(["labels"], axis=1)
        y_train = df["labels"]

    automl_experiment = AutoML()
    automl_settings = {
        "time_budget": 10,
        "metric": metric,
        "task": task,
        "estimator_list": ["lgbm_spark"],
        "log_training_metric": True,
        "log_file_name": "test_spark_synapseml.log",
        "model_history": True,
        "verbose": 5,
    }

    y_train.name = "label"
    X_train = to_pandas_on_spark(X_train)
    y_train = to_pandas_on_spark(y_train)

    if task == "rank":
        automl_settings["groupCol"] = "query"
        automl_settings["evalAt"] = [1, 3, 5]
        automl_settings["groups"] = X_train["query"]
        automl_settings["groups"].name = "groups"
        X_train = X_train.to_spark(index_col="index")
    else:
        columns = X_train.columns
        feature_cols = [col for col in columns if col != "label"]
        featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
        X_train = featurizer.transform(X_train.to_spark(index_col="index"))["index", "features"]
    X_train = to_pandas_on_spark(X_train)

    automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
    if task == "classification":
        print(automl_experiment.classes_)
    print(automl_experiment.model)
    print(automl_experiment.config_history)
    print(automl_experiment.best_model_for_estimator("lgbm_spark"))
    print(automl_experiment.best_iteration)
    print(automl_experiment.best_estimator)
    print(automl_experiment.best_loss)
    if task != "rank":
        print(automl_experiment.score(X_train, y_train, metric=metric))
    del automl_settings["metric"]
    del automl_settings["model_history"]
    del automl_settings["log_training_metric"]
    del automl_settings["verbose"]
    del automl_settings["estimator_list"]
    automl_experiment = AutoML(task=task)
    try:
        duration = automl_experiment.retrain_from_log(
            X_train=X_train,
            y_train=y_train,
            train_full=True,
            record_id=0,
            **automl_settings,
        )
        print(duration)
        print(automl_experiment.model)
        print(automl_experiment.predict(X_train)[:5])
        print(y_train.to_numpy()[:5])
    except ValueError:
        return


def test_spark_synapseml_classification():
    _test_spark_synapseml_lightgbm(spark, "classification")


def test_spark_synapseml_regression():
    _test_spark_synapseml_lightgbm(spark, "regression")


def test_spark_synapseml_rank():
    _test_spark_synapseml_lightgbm(spark, "rank")


def test_spark_input_df():
    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/company_bankruptcy_prediction_data.csv")
    )
    train, test = df.randomSplit([0.8, 0.2], seed=1)
    feature_cols = df.columns[1:]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["Bankrupt?", "features"]
    test_data = featurizer.transform(test)["Bankrupt?", "features"]
    automl = AutoML()
    settings = {
        "time_budget": 30,  # total running time in seconds
        "metric": "roc_auc",
        # "estimator_list": ["lgbm_spark"],  # list of ML learners; we tune lightgbm in this example
        "task": "classification",  # task type
        "log_file_name": "flaml_experiment.log",  # flaml log file
        "seed": 7654321,  # random seed
        "eval_method": "holdout",
    }
    df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))

    automl.fit(
        dataframe=df,
        label="Bankrupt?",
        isUnbalance=True,
        **settings,
    )

    try:
        model = automl.model.estimator
        predictions = model.transform(test_data)

        from synapse.ml.train import ComputeModelStatistics

        if not skip_py311:
            # ComputeModelStatistics doesn't support python 3.11
            metrics = ComputeModelStatistics(
                evaluationMetric="classification",
                labelCol="Bankrupt?",
                scoredLabelsCol="prediction",
            ).transform(predictions)
            metrics.show()
    except AttributeError:
        print("No fitted model because of too short training time.")

    # test invalid params
    settings = {
        "time_budget": 10,  # total running time in seconds
        "metric": "roc_auc",
        "estimator_list": ["lgbm"],  # list of ML learners; we tune lightgbm in this example
        "task": "classification",  # task type
    }
    with pytest.raises(ValueError) as excinfo:
        automl.fit(
            dataframe=df,
            label="Bankrupt?",
            isUnbalance=True,
            **settings,
        )
    assert "No estimator is left." in str(excinfo.value)


def _test_spark_large_df():
    """Test with large dataframe, should not run in pipeline."""
    import os
    import time

    import pandas as pd
    from pyspark.sql import functions as F

    import flaml

    os.environ["FLAML_MAX_CONCURRENT"] = "8"
    start_time = time.time()

    def load_higgs():
        # 11M rows, 29 columns, 1.1GB
        df = (
            spark.read.format("csv")
            .option("header", False)
            .option("inferSchema", True)
            .load("/datadrive/datasets/HIGGS.csv")
            .withColumnRenamed("_c0", "target")
            .withColumn("target", F.col("target").cast("integer"))
            .limit(1000000)
            .fillna(0)
            .na.drop(how="any")
            .repartition(64)
            .cache()
        )
        print("Number of rows in data: ", df.count())
        return df

    def load_bosch():
        # 1.184M rows, 969 cols, 1.5GB
        df = (
            spark.read.format("csv")
            .option("header", True)
            .option("inferSchema", True)
            .load("/datadrive/datasets/train_numeric.csv")
            .withColumnRenamed("Response", "target")
            .withColumn("target", F.col("target").cast("integer"))
            .limit(1000000)
            .fillna(0)
            .drop("Id")
            .repartition(64)
            .cache()
        )
        print("Number of rows in data: ", df.count())
        return df

    def prepare_data(dataset_name="higgs"):
        df = load_higgs() if dataset_name == "higgs" else load_bosch()
        train, test = df.randomSplit([0.75, 0.25], seed=7654321)
        feature_cols = [col for col in df.columns if col not in ["target", "arrest"]]
        final_cols = ["target", "features"]
        featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_data = featurizer.transform(train)[final_cols]
        test_data = featurizer.transform(test)[final_cols]
        train_data = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))
        return train_data, test_data

    train_data, test_data = prepare_data("higgs")
    end_time = time.time()
    print("time cost in minutes for prepare data: ", (end_time - start_time) / 60)
    automl = flaml.AutoML()
    automl_settings = {
        "max_iter": 3,
        "time_budget": 7200,
        "metric": "accuracy",
        "task": "classification",
        "seed": 1234,
        "eval_method": "holdout",
    }
    automl.fit(dataframe=train_data, label="target", ensemble=False, **automl_settings)
    model = automl.model.estimator
    predictions = model.transform(test_data)
    predictions.show(5)
    end_time = time.time()
    print("time cost in minutes: ", (end_time - start_time) / 60)


if __name__ == "__main__":
    test_spark_synapseml_classification()
    test_spark_synapseml_regression()
    test_spark_synapseml_rank()
    test_spark_input_df()

    # import cProfile
    # import pstats
    # from pstats import SortKey

    # cProfile.run("_test_spark_large_df()", "_test_spark_large_df.profile")
    # p = pstats.Stats("_test_spark_large_df.profile")
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)
