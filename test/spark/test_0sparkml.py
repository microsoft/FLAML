import os
import sys

import numpy as np
import pytest
import sklearn.datasets as skds

from flaml import AutoML
from flaml.automl.data import auto_convert_dtypes_pandas, auto_convert_dtypes_spark, get_random_dataframe

if sys.platform == "darwin" or "nt" in os.name:
    # skip this test if the platform is not linux
    skip_spark = True
else:
    try:
        from test.spark._init_spark import setup_spark_for_tests

        from pyspark.ml.feature import VectorAssembler

        from flaml.automl.spark.utils import to_pandas_on_spark

        spark, skip_spark = setup_spark_for_tests("MyApp")
    except ImportError:
        skip_spark = True

if sys.version_info >= (3, 11):
    skip_py311 = True
else:
    skip_py311 = False

pytestmark = [pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests."), pytest.mark.spark]


def _test_spark_synapseml_lightgbm(spark=None, task="classification"):
    # TODO: remove the estimator assignment once SynapseML supports spark 4+.
    from flaml.automl.spark.utils import _spark_major_minor_version

    if _spark_major_minor_version[0] >= 4:
        # skip synapseml lightgbm test for spark 4+
        return

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


def test_spark_input_df_and_pickle():
    import pandas as pd

    import flaml.visualization as fviz

    file_url = "https://mmlspark.blob.core.windows.net/publicwasb/company_bankruptcy_prediction_data.csv"
    df = pd.read_csv(file_url)
    df = spark.createDataFrame(df)
    train, test = df.randomSplit([0.8, 0.2], seed=1)
    feature_cols = df.columns[1:]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["Bankrupt?", "features"]
    test_data = featurizer.transform(test)["Bankrupt?", "features"]
    automl = AutoML()

    # TODO: remove the estimator assignment once SynapseML supports spark 4+.
    from flaml.automl.spark.utils import _spark_major_minor_version

    estimator_list = ["rf_spark"] if _spark_major_minor_version[0] >= 4 else None

    settings = {
        "time_budget": 30,  # total running time in seconds
        "metric": "roc_auc",
        "task": "classification",  # task type
        "log_file_name": "flaml_experiment.log",  # flaml log file
        "seed": 7654321,  # random seed
        "eval_method": "holdout",
        "estimator_list": estimator_list,  # TODO: remove once SynapseML supports spark 4+
    }
    df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))

    automl.fit(
        dataframe=df,
        label="Bankrupt?",
        isUnbalance=True,
        **settings,
    )

    # test pickle and load_pickle, should work for vizualization and prediction
    automl.pickle("automl_spark.pkl")
    automl_loaded = AutoML().load_pickle("automl_spark.pkl", load_spark_models=False)
    assert automl_loaded.best_estimator == automl.best_estimator
    assert automl_loaded.best_loss == automl.best_loss
    # automl_loaded.predict(df)
    # automl_loaded.model.estimator.transform(test_data)

    fig1 = fviz.plot_optimization_history(automl)
    fig2 = fviz.plot_optimization_history(automl_loaded)
    assert fig1.to_json() == fig2.to_json()
    fviz.plot_feature_importance(automl_loaded)
    fviz.plot_parallel_coordinate(automl_loaded)
    fviz.plot_contour(automl_loaded)
    fviz.plot_edf(automl_loaded)
    fviz.plot_timeline(automl_loaded)
    fviz.plot_slice(automl_loaded)
    fviz.plot_param_importance(automl_loaded)

    if estimator_list == ["rf_spark"]:
        return

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


def test_get_random_dataframe():
    # Test with default parameters
    df = get_random_dataframe(n_rows=50, ratio_none=0.2, seed=123)
    assert df.shape == (50, 14)  # Default is 200 rows and 14 columns

    # Test column types
    assert "timestamp" in df.columns and np.issubdtype(df["timestamp"].dtype, np.datetime64)
    assert "id" in df.columns and np.issubdtype(df["id"].dtype, np.integer)
    assert "score" in df.columns and np.issubdtype(df["score"].dtype, np.floating)
    assert "category" in df.columns and df["category"].dtype.name == "category"


def test_auto_convert_dtypes_pandas():
    # Create a test DataFrame with various types
    import pandas as pd

    test_df = pd.DataFrame(
        {
            "int_col": ["1", "2", "3", "4", "5", "6", "6"],
            "float_col": ["1.1", "2.2", "3.3", "NULL", "5.5", "6.6", "6.6"],
            "date_col": ["2021-01-01", "2021-02-01", "NA", "2021-04-01", "2021-05-01", "2021-06-01", "2021-06-01"],
            "cat_col": ["A", "B", "A", "A", "B", "A", "B"],
            "string_col": ["text1", "text2", "text3", "text4", "text5", "text6", "text7"],
        }
    )

    # Convert dtypes
    converted_df, schema = auto_convert_dtypes_pandas(test_df)

    # Check conversions
    assert schema["int_col"] == "int"
    assert schema["float_col"] == "double"
    assert schema["date_col"] == "timestamp"
    assert schema["cat_col"] == "category"
    assert schema["string_col"] == "string"


def test_auto_convert_dtypes_spark():
    """Test auto_convert_dtypes_spark function with various data types."""
    import pandas as pd

    # Create a test DataFrame with various types
    test_pdf = pd.DataFrame(
        {
            "int_col": ["1", "2", "3", "4", "NA"],
            "float_col": ["1.1", "2.2", "3.3", "NULL", "5.5"],
            "date_col": ["2021-01-01", "2021-02-01", "NA", "2021-04-01", "2021-05-01"],
            "cat_col": ["A", "B", "A", "C", "B"],
            "string_col": ["text1", "text2", "text3", "text4", "text5"],
        }
    )

    # Convert pandas DataFrame to Spark DataFrame
    test_df = spark.createDataFrame(test_pdf)

    # Convert dtypes
    converted_df, schema = auto_convert_dtypes_spark(test_df)

    # Check conversions
    assert schema["int_col"] == "int"
    assert schema["float_col"] == "double"
    assert schema["date_col"] == "timestamp"
    assert schema["cat_col"] == "string"  # Conceptual category in schema
    assert schema["string_col"] == "string"

    # Verify the actual data types from the Spark DataFrame
    spark_dtypes = dict(converted_df.dtypes)
    assert spark_dtypes["int_col"] == "int"
    assert spark_dtypes["float_col"] == "double"
    assert spark_dtypes["date_col"] == "timestamp"
    assert spark_dtypes["cat_col"] == "string"  # In Spark, categories are still strings
    assert spark_dtypes["string_col"] == "string"


def test_auto_convert_dtypes_skip_columns():
    """Both pandas and spark variants must leave ``skip_columns`` untouched."""
    import pandas as pd

    # Pandas
    pdf = pd.DataFrame(
        {
            "id": [str(i) for i in range(10)],  # would normally be int
            "raw": ["A", "B", "NULL", "C", "A"] * 2,  # NA-token preserved
            "lab": ["X", "Y", "Z", "X", "Y"] * 2,  # still inferred
        }
    )
    p_out, p_schema = auto_convert_dtypes_pandas(pdf, skip_columns=["id", "raw"])
    assert p_schema["id"] == str(pdf["id"].dtype)
    assert p_out["id"].tolist() == pdf["id"].tolist()
    assert p_schema["raw"] == str(pdf["raw"].dtype)
    assert p_out["raw"].tolist() == pdf["raw"].tolist()
    assert p_schema["lab"] == "category"

    # Spark
    sdf = spark.createDataFrame(pdf)
    s_out, s_schema = auto_convert_dtypes_spark(sdf, skip_columns=["id", "raw"])
    s_dtypes = dict(s_out.dtypes)
    assert s_dtypes["id"] == "string"  # untouched
    assert s_dtypes["raw"] == "string"  # untouched, NA-token preserved
    # "raw" values are not normalized when skipped.
    raw_vals = [r[0] for r in s_out.select("raw").collect()]
    assert "NULL" in raw_vals


def test_auto_convert_dtypes_boolean_to_int_pandas():
    """Pandas: a boolean column is cast to ``Int64`` (nullable int).

    True→1, False→0, NA→pd.NA. This keeps the column numeric so a
    downstream sklearn ``ColumnTransformer`` that also handles string-typed
    categoricals does not trip on the ``is_bool_dtype`` early-conversion
    path.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "flag_ext": pd.array(
                [True, False, True, False, True, False, True, False, True, pd.NA],
                dtype="boolean",
            ),
            "flag_np": [True, False, True, False, True, False, True, False, True, False],
        }
    )
    out, schema = auto_convert_dtypes_pandas(df)
    assert schema["flag_ext"] == "int"
    assert schema["flag_np"] == "int"
    assert isinstance(out["flag_ext"].dtype, pd.Int64Dtype)
    assert isinstance(out["flag_np"].dtype, pd.Int64Dtype)
    # NA round-trips, values map cleanly.
    assert out["flag_ext"].iloc[0] == 1 and out["flag_ext"].iloc[1] == 0
    assert pd.isna(out["flag_ext"].iloc[-1])


def test_auto_convert_dtypes_boolean_to_int_spark():
    """Spark: a ``BooleanType`` column is cast to ``IntegerType`` (1/0/null)."""
    import pandas as pd
    from pyspark.sql.types import BooleanType, StructField, StructType

    schema_in = StructType([StructField("flag", BooleanType(), nullable=True)])
    rows = [(True,), (False,), (True,), (None,), (False,)] * 4
    sdf = spark.createDataFrame(rows, schema=schema_in)

    out, sch = auto_convert_dtypes_spark(sdf)
    assert sch["flag"] == "int"
    assert dict(out.dtypes)["flag"] == "int"

    # Value mapping preserved (True→1, False→0, null→null).
    vals = [r[0] for r in out.select("flag").collect()]
    assert vals.count(1) == 8
    assert vals.count(0) == 8
    assert vals.count(None) == 4


def test_auto_convert_dtypes_spark_non_numeric_string_no_throw():
    """Non-numeric string columns must not raise under Spark ANSI mode.

    Regression test for a ``NumberFormatException`` / ``DateTimeException``
    that surfaced when ``auto_convert_dtypes_spark`` called ``cast("double")``
    or ``to_timestamp`` against a high-cardinality string column (e.g.
    customer ids like ``CUST_006255``) while Spark ANSI mode was enabled
    (default in Spark 4.x). The fix routes both numeric and timestamp casts
    through ``try_cast`` so unparseable values become NULL instead of raising.
    """
    import pandas as pd

    # Mix a numeric-looking int column with a free-text id column whose values
    # cannot be parsed as numbers OR as timestamps. Without ``try_cast`` the
    # numeric-probe AND the timestamp-probe would raise under ANSI mode.
    pdf = pd.DataFrame(
        {
            "cust_id": [f"CUST_{i:06d}" for i in range(120)],
            "amount": [str(i) for i in range(120)],
        }
    )
    sdf = spark.createDataFrame(pdf)

    out, sch = auto_convert_dtypes_spark(sdf)

    # cust_id is high-cardinality non-numeric → keep as string (not int/double)
    assert sch["cust_id"] == "string"
    assert dict(out.dtypes)["cust_id"] == "string"
    # amount is purely numeric strings → int
    assert sch["amount"] == "int"
    assert dict(out.dtypes)["amount"] == "int"
    # Forcing materialization must not raise.
    assert out.count() == 120


def test_auto_convert_dtypes_spark_outlier_survives_inference():
    """Apply-phase ``try_cast`` must NULL-out non-numeric outliers, not raise.

    The inference samples a fraction of the data. If the sample happens to
    contain only numeric values but the full data has a stray non-numeric
    string, the apply-phase cast must not raise under ANSI mode.
    """
    import pandas as pd

    # 99 numeric strings plus one outlier that cannot be cast to a number.
    pdf = pd.DataFrame({"val": [str(i) for i in range(99)] + ["not_a_number"]})
    sdf = spark.createDataFrame(pdf)

    out, sch = auto_convert_dtypes_spark(sdf)

    # The column is overwhelmingly numeric, so we should infer int.
    assert sch["val"] == "int"
    assert dict(out.dtypes)["val"] == "int"

    # The outlier must materialize as NULL (try_cast behavior) instead of
    # crashing the .collect() call.
    vals = [r[0] for r in out.select("val").collect()]
    assert vals.count(None) == 1
    assert sum(v is not None for v in vals) == 99


@pytest.mark.parametrize("ansi_enabled", ["true", "false"])
def test_auto_convert_dtypes_spark_works_in_both_ansi_modes(ansi_enabled):
    """``auto_convert_dtypes_spark`` must succeed with ANSI on AND off.

    Spark 3.x defaults ANSI to off (lenient cast → NULL); Spark 4.x defaults
    it to on (strict cast → exception). The ``try_cast`` calls used by
    ``auto_convert_dtypes_spark`` return NULL for unparseable values in both
    modes, so behavior must be identical regardless of the runtime setting.
    """
    import pandas as pd

    original = spark.conf.get("spark.sql.ansi.enabled", "false")
    try:
        spark.conf.set("spark.sql.ansi.enabled", ansi_enabled)

        pdf = pd.DataFrame(
            {
                # Non-numeric, non-timestamp free-text → would trip strict cast.
                "cust_id": [f"CUST_{i:06d}" for i in range(120)],
                # Purely numeric string → int.
                "age": [str(20 + i % 40) for i in range(120)],
                # Mostly-numeric with a stray outlier → int + NULL for outlier.
                "score": [str(i) for i in range(119)] + ["bad"],
            }
        )
        sdf = spark.createDataFrame(pdf)

        out, sch = auto_convert_dtypes_spark(sdf)

        # Inferred schema is invariant to ANSI mode.
        assert sch["cust_id"] == "string"
        assert sch["age"] == "int"
        assert sch["score"] == "int"

        # Materialization must succeed (no exception thrown).
        assert out.count() == 120

        # Outlier in ``score`` must become NULL (try_cast semantics).
        score_vals = [r[0] for r in out.select("score").collect()]
        assert score_vals.count(None) == 1
    finally:
        spark.conf.set("spark.sql.ansi.enabled", original)


if __name__ == "__main__":
    # test_spark_synapseml_classification()
    # test_spark_synapseml_regression()
    # test_spark_synapseml_rank()
    test_spark_input_df_and_pickle()
    # test_get_random_dataframe()
    # test_auto_convert_dtypes_pandas()
    # test_auto_convert_dtypes_spark()

    # import cProfile
    # import pstats
    # from pstats import SortKey

    # cProfile.run("_test_spark_large_df()", "_test_spark_large_df.profile")
    # p = pstats.Stats("_test_spark_large_df.profile")
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)
