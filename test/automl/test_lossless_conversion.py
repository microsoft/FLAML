import numpy as np
import pandas as pd
import pytest

from flaml.automl.data import add_time_idx_col, auto_convert_dtypes_pandas


@pytest.mark.parametrize(
    "values,expected",
    [
        (["1e16", "2e16", "3e16"], [10**16, 2 * 10**16, 3 * 10**16]),
        (["9.007199254740993e15", "9007199254740994.0", "-1e16"], [9007199254740993, 9007199254740994, -(10**16)]),
    ],
)
def test_scientific_notation_integers_are_preserved(values, expected):
    result, schema = auto_convert_dtypes_pandas(pd.DataFrame({"value": values}))
    assert schema["value"] == "int"
    assert result["value"].tolist() == expected


def test_unrepresentable_integer_notation_is_not_rounded_or_dropped():
    source = ["1e40", "2e40", "3e40"]
    result, schema = auto_convert_dtypes_pandas(pd.DataFrame({"value": source}))
    assert schema["value"] == "string"
    assert result["value"].tolist() == source


@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)])
def test_large_integer_conversion_preserves_boolean_cells(boolean):
    frame = pd.DataFrame({"value": pd.Series([9007199254740993, 9007199254740994, boolean], dtype=object)})
    converted, schema = auto_convert_dtypes_pandas(frame)
    assert schema["value"] == "int"
    assert converted["value"].tolist() == [9007199254740993, 9007199254740994, int(boolean)]


@pytest.mark.parametrize("days,expected", [([0, 1, 3], [0, 1, 3]), ([0, 2, 5, 7], [0, 2, 5, 7])])
def test_tied_timestamp_steps_do_not_merge_observations(days, expected):
    timestamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D")
    result = add_time_idx_col(pd.DataFrame({"ds": timestamps, "value": np.arange(len(days))}))
    assert result["time_idx"].tolist() == expected
    assert result["time_idx"].is_unique


@pytest.mark.parametrize("repeated_hour_only", [False, True])
def test_fall_dst_keeps_distinct_elapsed_observations(repeated_hour_only):
    timestamps = pd.date_range("2024-11-03 00:00", periods=5, freq="h", tz="America/New_York")
    if repeated_hour_only:
        timestamps = timestamps[1:3]
    result = add_time_idx_col(pd.DataFrame({"ds": timestamps}))
    assert result["time_idx"].tolist() == list(range(len(timestamps)))
    assert result["time_idx"].is_unique


@pytest.mark.parametrize("time_named_index", [False, True])
def test_shuffled_training_rows_have_chronological_validation_boundaries(time_named_index):
    from flaml.automl.time_series import TimeSeriesDataset

    frame = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=8), "y": np.arange(8), "group": list("aabbccdd")})
    if time_named_index:
        frame = frame.set_index("ds", drop=False)
    frame = frame.iloc[[3, 0, 7, 5, 1, 6, 2, 4]]
    original = frame.copy(deep=True)
    data = TimeSeriesDataset(frame, "ds", "y")
    assert data.train_data["ds"].is_monotonic_increasing
    assert data.train_data["y"].tolist() == list(range(8))
    assert data.train_data["group"].tolist() == list("aabbccdd")
    splits = [data.move_validation_boundary(-2), *data.cv_train_val_sets(2, 2, 1)]
    for split in splits:
        assert split.train_data["ds"].max() < split.test_data["ds"].min()
    assert data.end_date == pd.Timestamp("2025-01-08")
    pd.testing.assert_frame_equal(frame, original)


def test_explicit_forecast_request_order_is_preserved():
    from flaml.automl.time_series import TimeSeriesDataset

    frame = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=8), "y": np.arange(8)})
    future = pd.DataFrame(
        {"ds": pd.to_datetime(["2025-01-11", "2025-01-09", "2025-01-10"]), "y": [10, 8, 9]},
        index=["last", "first", "middle"],
    )
    data = TimeSeriesDataset(frame.iloc[::-1], "ds", "y", test_data=future)
    pd.testing.assert_frame_equal(data.test_data, future)


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_timestamp_coercion_replaces_string_dtype(dtype):
    frame = pd.DataFrame({"ds": pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"], dtype=dtype)})
    result = add_time_idx_col(frame)
    assert pd.api.types.is_datetime64_any_dtype(result["ds"])
    assert result["time_idx"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("values", ["NA", b"NA", [None], [1], [b"NA"]])
def test_invalid_na_value_collections_are_rejected(values):
    with pytest.raises(TypeError, match="na_values"):
        auto_convert_dtypes_pandas(pd.DataFrame({"value": ["NA", "1"]}), na_values=values)


def test_na_value_generator_is_materialized_once():
    result, _ = auto_convert_dtypes_pandas(
        pd.DataFrame({"value": ["NULL", "NA", "3"]}), na_values=(value for value in ["NULL", "NA"])
    )
    assert result["value"].isna().tolist() == [True, True, False]


@pytest.mark.parametrize("columns", ["value", b"value", [["value"]]])
def test_invalid_skip_column_collections_are_rejected(columns):
    with pytest.raises(TypeError, match="skip_columns"):
        auto_convert_dtypes_pandas(pd.DataFrame({"value": ["1", "2"]}), skip_columns=columns)


def test_spark_skip_columns_requires_string_names():
    from flaml.automl.data import _validate_skip_columns

    with pytest.raises(TypeError, match="must be a str"):
        _validate_skip_columns([1], require_str=True)


def test_spark_cast_type_and_identifier_guards(monkeypatch):
    from types import SimpleNamespace

    import flaml.automl.data as module

    monkeypatch.setattr(module, "F", SimpleNamespace(expr=lambda expression: expression))
    with pytest.raises(ValueError, match="unsupported sql_type"):
        module._spark_try_cast("value", "int); SELECT 1")
    assert module._spark_try_cast("a`b", "bigint") == "try_cast(`a``b` as bigint)"


@pytest.mark.parametrize("values", [[1, 2, 3], ["not-a-date", "still-not-a-date", "never-a-date"]])
def test_invalid_timestamp_inputs_raise_explicitly(values):
    with pytest.raises(TypeError, match="datetime"):
        add_time_idx_col(pd.DataFrame({"ds": values}))


def test_nanosecond_span_and_index_overflow_are_lossless():
    dates = [pd.Timestamp.min, pd.Timestamp.min + pd.Timedelta(1, "ns"), pd.Timestamp.max]
    result = add_time_idx_col(pd.DataFrame({"ds": dates}))
    assert result["time_idx"].tolist() == [0, 1, pd.Timestamp.max.value - pd.Timestamp.min.value]
    assert result["time_idx"].dtype == object


@pytest.mark.parametrize("frequency", ["MS", "BMS", "BYS", "BYE"])
def test_calendar_index_values_are_calendar_based(frequency):
    dates = pd.date_range("2020-01-01", periods=8, freq=frequency)
    result = add_time_idx_col(pd.DataFrame({"ds": dates}))
    expected = dates.year if frequency.startswith("BY") else dates.year * 12 + dates.month
    assert result["time_idx"].tolist() == expected.tolist()


@pytest.mark.parametrize("step", [0, -1])
def test_nonpositive_computed_timestamp_step_is_rejected(monkeypatch, step):
    monkeypatch.setattr(pd.Series, "mode", lambda self, *args, **kwargs: pd.Series([step]))
    with pytest.raises(ValueError, match="positive indexing interval"):
        add_time_idx_col(pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3)}))


@pytest.mark.parametrize("sample_ratio", [0.1, 1.0])
def test_complete_two_digit_year_dates_are_timestamps(sample_ratio):
    source = ["01/02/24", "01/03/24", "01/04/24"] * 50
    result, schema = auto_convert_dtypes_pandas(pd.DataFrame({"value": source}), sample_ratio=sample_ratio)
    assert schema["value"] == "timestamp"
    expected = pd.to_datetime(pd.Series(source), format="%m/%d/%y")
    pd.testing.assert_series_equal(result["value"], expected, check_names=False)


def test_all_null_inference_sample_still_validates_full_timedelta_column():
    frame = pd.DataFrame({"duration": ["PT1H"] * 1000}, index=np.arange(1000) + 500)
    frame.loc[frame.sample(n=100, random_state=0).index, "duration"] = None
    sampled, sampled_schema = auto_convert_dtypes_pandas(frame)
    full, full_schema = auto_convert_dtypes_pandas(frame, sample_ratio=1.0)
    assert sampled_schema == full_schema == {"duration": "timedelta"}
    pd.testing.assert_frame_equal(sampled, full)
    assert sampled["duration"].isna().sum() == 100
    assert (sampled["duration"].dropna() == pd.Timedelta(hours=1)).all()
