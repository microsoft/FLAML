import warnings
import os
import pytest
import numpy as np
import pandas as pd
import sklearn.datasets as skds
from flaml import AutoML
from flaml.tune.spark.utils import check_spark
from flaml.automl.spark.utils import to_pandas_on_spark
import pyspark
from pyspark.ml.feature import VectorAssembler

warnings.simplefilter(action="ignore")

os.environ["FLAML_MAX_CONCURRENT"] = "2"

spark = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    .config(
        "spark.jars.packages",
        f"com.microsoft.azure:synapseml_2.12:0.10.2,org.apache.hadoop:hadoop-azure:{pyspark.__version__},com.microsoft.azure:azure-storage:8.6.6",
    )
    .config("spark.sql.debug.maxToStringFields", "100")
    .getOrCreate()
)
spark_available, _ = check_spark()
skip_spark = not spark_available

pytestmark = pytest.mark.skipif(
    skip_spark, reason="Spark is not installed. Skip all spark tests."
)


def _test_spark_synapseml_lightgbm(task="classification"):
    if task == "classification":
        metric = "accuracy"
        X_train, y_train = skds.load_iris(return_X_y=True, as_frame=True)
    elif task == "regression":
        metric = "r2"
        X_train, y_train = skds.load_diabetes(return_X_y=True, as_frame=True)
    elif task == "rank":
        metric = "ndcg"
        sdf = spark.read.format("parquet").load(
            "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_train.parquet"
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
        automl_settings["groups"] = X_train["query"].to_pandas()
        X_train = X_train.to_spark(index_col="index")
    else:
        columns = X_train.columns
        feature_cols = [col for col in columns if col != "label"]
        featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
        X_train = featurizer.transform(X_train.to_spark(index_col="index"))[
            "index", "features"
        ]
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
    del automl_settings["metric"]
    del automl_settings["model_history"]
    del automl_settings["log_training_metric"]
    automl_experiment = AutoML(task=task)
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


def test_spark_synapseml():
    for task in ["classification", "regression", "rank"]:
        _test_spark_synapseml_lightgbm(task)


if __name__ == "__main__":
    test_spark_synapseml()
