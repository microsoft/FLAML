from importlib import import_module

import pytest

from flaml.automl.data import auto_convert_dtypes_spark

pytest.importorskip("pyspark")
SparkSession = import_module("pyspark.sql").SparkSession
F = import_module("pyspark.sql.functions")

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def dtype_spark():
    session = SparkSession.getActiveSession()
    owned = session is None
    if owned:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("flaml-dtype-contracts")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
    yield session
    if owned:
        session.stop()


@pytest.mark.parametrize("ansi", ["true", "false"])
@pytest.mark.parametrize(
    "values,expected,dtype",
    [
        (["0", "1.0", "-2e0"], [0, 1, -2], "int"),
        (["-2147483649", "3000000000", "3e9"], [-2147483649, 3000000000, 3000000000], "bigint"),
        (["9.007199254740993e15", "9007199254740994.0"], [9007199254740993, 9007199254740994], "bigint"),
        (["-9223372036854775808", "9223372036854775807"], [-(2**63), 2**63 - 1], "bigint"),
    ],
)
def test_inferred_integer_width_preserves_values(dtype_spark, ansi, values, expected, dtype):
    original_ansi = dtype_spark.conf.get("spark.sql.ansi.enabled")
    dtype_spark.conf.set("spark.sql.ansi.enabled", ansi)
    try:
        rows = ", ".join(f"({index}, '{value}')" for index, value in enumerate(values))
        frame = dtype_spark.sql(f"SELECT * FROM VALUES {rows} AS input(row_id, value)")
        converted, schema = auto_convert_dtypes_spark(frame, sample_ratio=1.0, skip_columns=["row_id"])
        assert schema["value"] == dtype
        assert dict(converted.dtypes)["value"] == dtype
        assert [row.value for row in converted.orderBy("row_id").collect()] == expected
    finally:
        dtype_spark.conf.set("spark.sql.ansi.enabled", original_ansi)


def test_integer_width_validation_covers_unsampled_values(dtype_spark, monkeypatch):
    frame = dtype_spark.sql("SELECT * FROM VALUES ('1'), ('2'), ('3000000000') AS input(value)")
    monkeypatch.setattr(
        type(frame), "sample", lambda self, *args, **kwargs: self.filter(F.col("value") != "3000000000")
    )
    converted, schema = auto_convert_dtypes_spark(frame)
    assert schema["value"] == "bigint"
    assert sorted(row.value for row in converted.collect()) == [1, 2, 3000000000]


def test_out_of_int64_range_is_not_silently_nulled(dtype_spark):
    frame = dtype_spark.sql("SELECT * FROM VALUES ('1e40'), ('2e40') AS input(value)")
    converted, schema = auto_convert_dtypes_spark(frame, sample_ratio=1.0)
    assert schema["value"] == "string"
    assert sorted(row.value for row in converted.collect()) == ["1e40", "2e40"]


def test_unsampled_incompatible_finite_value_raises(dtype_spark, monkeypatch):
    frame = dtype_spark.sql("SELECT * FROM VALUES ('1'), ('2'), ('1e40') AS input(value)")
    monkeypatch.setattr(type(frame), "sample", lambda self, *args, **kwargs: self.filter(F.col("value") != "1e40"))
    with pytest.raises(ValueError, match="finite numeric values incompatible"):
        auto_convert_dtypes_spark(frame)


def test_integer_validation_retains_boolean_and_malformed_value_behavior(dtype_spark):
    frame = dtype_spark.sql(
        "SELECT * FROM VALUES (true, '1'), (false, '2'), (null, 'not_a_number') AS input(flag, value)"
    )
    converted, schema = auto_convert_dtypes_spark(frame, sample_ratio=1.0, convert_threshold=0.6)
    assert schema["flag"] == schema["value"] == "int"
    assert sorted((row.flag, row.value) for row in converted.collect() if row.flag is not None) == [(0, 2), (1, 1)]
    assert converted.filter(F.col("flag").isNull() & F.col("value").isNull()).count() == 1


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan")])
def test_invalid_sampling_ratio_does_not_scan_spark_data(dtype_spark, monkeypatch, ratio):
    frame = dtype_spark.range(3)

    def fail_count(self):
        raise AssertionError("Invalid input started a Spark scan")

    monkeypatch.setattr(type(frame), "count", fail_count)
    with pytest.raises(ValueError, match="sample_ratio"):
        auto_convert_dtypes_spark(frame, sample_ratio=ratio)


def test_empty_spark_dataframe_has_consistent_boolean_schema(dtype_spark):
    frame = dtype_spark.sql("SELECT CAST(NULL AS BOOLEAN) AS flag, CAST(NULL AS STRING) AS value WHERE false")
    converted, schema = auto_convert_dtypes_spark(frame)
    assert converted.count() == 0
    assert schema == {"flag": "int", "value": "string"}
    assert dict(converted.dtypes) == schema


def test_spark_conversion_preserves_quoted_column_names(dtype_spark):
    frame = dtype_spark.sql("SELECT * FROM VALUES ('1'), ('2') AS input(`a``b`)")
    converted, schema = auto_convert_dtypes_spark(frame, sample_ratio=1.0)
    assert schema == {"a`b": "int"}
    assert sorted(row[0] for row in converted.collect()) == [1, 2]
