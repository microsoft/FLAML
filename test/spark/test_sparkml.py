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
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.10.2")
    .config("spark.sql.debug.maxToStringFields", "100")
    .getOrCreate()
)
spark_available, _ = check_spark()
skip_spark = not spark_available

pytestmark = pytest.mark.skipif(
    skip_spark, reason="Spark is not installed. Skip all spark tests."
)


def test_spark_synapseml_classification():
    automl_experiment = AutoML()
    automl_settings = {
        "time_budget": 10,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm_spark"],
        "log_training_metric": True,
        "log_file_name": "test_spark_synapseml.log",
        "model_history": True,
        "verbose": 5,
    }
    X_train, y_train = skds.load_iris(return_X_y=True, as_frame=True)
    X_train = to_pandas_on_spark(X_train)
    y_train = to_pandas_on_spark(y_train)

    columns = X_train.columns
    feature_cols = [col for col in columns if col != "label"]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    X_train = featurizer.transform(X_train.to_spark(index_col="index"))[
        "index", "features"
    ]
    X_train = to_pandas_on_spark(X_train)

    automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
    print(automl_experiment.classes_)
    print(automl_experiment.predict(X_train).to_numpy().astype(int))
    print(y_train.to_numpy())
    print(automl_experiment.model)
    print(automl_experiment.config_history)
    print(automl_experiment.best_model_for_estimator("lgbm_spark"))
    print(automl_experiment.best_iteration)
    print(automl_experiment.best_estimator)
    del automl_settings["metric"]
    del automl_settings["model_history"]
    del automl_settings["log_training_metric"]
    automl_experiment = AutoML(task="classification")
    duration = automl_experiment.retrain_from_log(
        log_file_name=automl_settings["log_file_name"],
        X_train=X_train,
        y_train=y_train,
        train_full=True,
        record_id=0,
    )
    print(duration)
    print(automl_experiment.model)
    print(automl_experiment.predict(X_train)[:5])


def test_spark_synapseml_regression():
    automl_experiment = AutoML()
    automl_settings = {
        "time_budget": 30,
        "metric": "rmse",
        "task": "regression",
        "estimator_list": ["lgbm_spark"],
        "log_training_metric": True,
        "log_file_name": "test_spark_synapseml.log",
        "model_history": True,
        "verbose": 5,
    }
    X_train, y_train = skds.load_diabetes(return_X_y=True, as_frame=True)
    X_train = to_pandas_on_spark(X_train)
    y_train = to_pandas_on_spark(y_train)

    columns = X_train.columns
    feature_cols = [col for col in columns if col != "label"]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    X_train = featurizer.transform(X_train.to_spark(index_col="index"))[
        "index", "features"
    ]
    X_train = to_pandas_on_spark(X_train)

    automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
    print(automl_experiment.model)
    print(automl_experiment.config_history)
    print(automl_experiment.best_model_for_estimator("lgbm_spark"))
    print(automl_experiment.best_iteration)
    print(automl_experiment.best_estimator)
    del automl_settings["metric"]
    del automl_settings["model_history"]
    del automl_settings["log_training_metric"]
    automl_experiment = AutoML(task="regression")
    duration = automl_experiment.retrain_from_log(
        log_file_name=automl_settings["log_file_name"],
        X_train=X_train,
        y_train=y_train,
        train_full=True,
        record_id=0,
    )
    print(duration)
    print(automl_experiment.model)
    print(automl_experiment.predict(X_train)[:5])
    print(y_train.to_numpy()[:5])


if __name__ == "__main__":
    test_spark_synapseml_classification()
    test_spark_synapseml_regression()
