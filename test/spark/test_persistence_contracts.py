import os
from pathlib import Path
from test.spark.test_dtype_contracts import dtype_spark as dtype_spark

import numpy as np
import pytest

from flaml import AutoML

pytestmark = pytest.mark.spark


@pytest.fixture
def persistence_spark(dtype_spark):
    original = dtype_spark.conf.get("spark.sql.ansi.enabled")
    dtype_spark.conf.set("spark.sql.ansi.enabled", "false")
    try:
        yield dtype_spark
    finally:
        dtype_spark.conf.set("spark.sql.ansi.enabled", original)


@pytest.mark.skipif(
    os.name == "nt" and not (Path(os.environ.get("HADOOP_HOME", "")) / "bin" / "winutils.exe").is_file(),
    reason="Native Spark ML persistence on Windows requires Hadoop winutils; maintained Spark CI runs on Linux.",
)
def test_native_spark_model_roundtrip_restores_predictions(persistence_spark, tmp_path):
    from pyspark.ml.feature import VectorAssembler

    rows = persistence_spark.sql(
        "SELECT id, CAST(id % 2 AS DOUBLE) AS label, CAST(id AS DOUBLE) AS x, "
        "CAST(id % 3 AS DOUBLE) AS z FROM range(40)"
    )
    vector_rows = VectorAssembler(inputCols=["x", "z"], outputCol="features").transform(rows)
    data = vector_rows.select("label", "features").pandas_api()
    automl = AutoML(
        estimator_list=["rf_spark"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        model_history=False,
        auto_augment=False,
        starting_points={"rf_spark": {"numTrees": 3, "maxDepth": 2, "featureSubsetStrategy": "all", "seed": 42}},
    )
    automl.fit(dataframe=data, label="label", task="classification", metric="accuracy")
    expected = automl.model.estimator.transform(vector_rows).select("id", "prediction").orderBy("id").collect()
    filename = str(tmp_path / "native-spark.pkl")
    automl.pickle(filename)
    restored = AutoML.load_pickle(filename)
    assert restored.model.estimator is not None
    actual = restored.model.estimator.transform(vector_rows).select("id", "prediction").orderBy("id").collect()
    np.testing.assert_array_equal([row.prediction for row in actual], [row.prediction for row in expected])
    assert len(restored.predict(data)) == len(expected)
