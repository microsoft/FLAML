"""Regression tests for ts_forecast input where the timestamp is the DataFrame index.

FLAML expects the timestamp to live in a *column* (`time_col`). When a user passes timestamps
as a `DatetimeIndex`, `validate_data` promotes the index to a column, preserves explicit index
names, ensures label-based alignment for DatetimeIndex targets, and maintains backwards
compatibility for explicit columns and NumPy array inputs across training, validation, and prediction.
"""

import numpy as np
import pandas as pd
import pytest

from flaml import AutoML


def _make_train_df(periods=120):
    return pd.DataFrame(
        {"y": np.sin(np.arange(periods) / 6) * 10 + 50},
        index=pd.date_range("2018-01-01", periods=periods, freq="MS"),
    )


def _fit(automl, **kwargs):
    settings = {
        "task": "ts_forecast",
        "label": "y",
        "period": 12,
        "time_budget": 5,
        "estimator_list": ["lgbm"],
        "metric": "mape",
        "log_file_name": False,
        "verbose": 0,
    }
    settings.update(kwargs)
    automl.fit(**settings)


def test_unnamed_datetime_index_becomes_the_time_column():
    automl = AutoML()
    _fit(automl, dataframe=_make_train_df())
    assert automl._state.task.time_col == "ds"


def test_named_datetime_index_keeps_its_name():
    df = _make_train_df()
    df.index.name = "month"
    automl = AutoML()
    _fit(automl, dataframe=df)
    assert automl._state.task.time_col == "month"


def test_promoted_index_keeps_every_training_row():
    """Pre-promotion the label column was read as epoch nanoseconds, collapsing the 120 real
    timestamps onto a handful of 1970-01-01 values and dropping the duplicates."""
    df = _make_train_df()
    automl = AutoML()
    _fit(automl, dataframe=df)
    assert automl._state.data_size[0] == len(df)


def test_explicit_time_col_still_wins_over_the_index():
    df = _make_train_df()
    df["ts"] = df.index
    automl = AutoML()
    _fit(automl, dataframe=df, time_col="ts")
    assert automl._state.task.time_col == "ts"


def test_genuine_missing_timestamp_still_raises():
    df = pd.DataFrame({"note": ["not-a-date"] * 120, "y": np.arange(120, dtype=float)})
    automl = AutoML()
    with pytest.raises(ValueError) as exc:
        _fit(automl, dataframe=df)
    assert "must contain timestamp values" in str(exc.value)


def test_timestamp_column_input_is_unchanged():
    df = _make_train_df().reset_index().rename(columns={"index": "ds"})
    automl = AutoML()
    _fit(automl, dataframe=df, time_col="ds")
    assert automl._state.task.time_col == "ds"
    assert len(automl.predict(df[["ds"]].tail(12))) == 12


def test_index_name_collision_resolves_to_unique_column():
    """When dataframe already has a column named 'ds' (or 'index') that is not datetime,
    promoting an unnamed DatetimeIndex must not overwrite or conflict with the existing column."""
    df = _make_train_df()
    df["ds"] = np.arange(len(df), dtype=float)  # Collision candidate
    df["ds_1"] = np.arange(len(df), dtype=float)  # Double collision candidate
    automl = AutoML()
    _fit(automl, dataframe=df)
    # Target column should be uniquely chosen as ds_2 avoiding collision
    assert automl._state.task.time_col == "ds_2"
    assert "ds" in automl._feature_names_in_
    assert "ds_1" in automl._feature_names_in_
    assert "ds_2" in automl._feature_names_in_
    assert automl.data_size_full == len(df)


def test_validation_data_with_datetime_index():
    """Verify that validation data passed as DataFrame with DatetimeIndex is properly promoted."""
    df_train = _make_train_df(periods=100)
    val_index = pd.date_range("2026-05-01", periods=20, freq="MS")
    df_val = pd.DataFrame({"y": np.sin(np.arange(100, 120) / 6) * 10 + 50}, index=val_index)
    X_val = pd.DataFrame(index=val_index)
    y_val = df_val["y"]
    automl = AutoML()
    _fit(
        automl,
        dataframe=df_train,
        X_val=X_val,
        y_val=y_val,
    )
    assert automl._state.task.time_col == "ds"
    assert automl._state.eval_method == "holdout"
    assert len(automl.predict(X_val)) == len(val_index)


def test_validation_data_with_dataframe_target_and_datetime_index():
    """Verify that validation data with DataFrame y_val and DatetimeIndex preserves aligned indexes."""
    df_train = _make_train_df(periods=100)
    val_index = pd.date_range("2026-05-01", periods=20, freq="MS")
    df_val = pd.DataFrame({"y": np.sin(np.arange(100, 120) / 6) * 10 + 50}, index=val_index)
    X_val = pd.DataFrame(index=val_index)
    y_val = df_val[["y"]]  # DataFrame target with DatetimeIndex
    automl = AutoML()
    _fit(
        automl,
        dataframe=df_train,
        X_val=X_val,
        y_val=y_val,
    )
    assert automl._state.task.time_col == "ds"
    assert automl._state.eval_method == "holdout"
    assert len(automl.predict(X_val)) == len(val_index)


def test_xtrain_ytrain_with_datetime_index():
    """Verify that (X_train, y_train) inputs with DatetimeIndex are promoted and work seamlessly."""
    train_idx = pd.date_range("2018-01-01", periods=100, freq="MS")
    val_idx = pd.date_range("2026-05-01", periods=20, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(100, dtype=float)}, index=train_idx)
    y_train = pd.Series(np.sin(np.arange(100) / 6) * 10 + 50, index=train_idx, name="y")
    X_val = pd.DataFrame({"feat": np.arange(100, 120, dtype=float)}, index=val_idx)
    y_val = pd.DataFrame({"y": np.sin(np.arange(100, 120) / 6) * 10 + 50}, index=val_idx)
    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    assert automl._state.task.time_col == "ds"
    assert len(automl.predict(X_val)) == 20


def test_predict_with_datetime_index():
    """Verify that future prediction data passed with DatetimeIndex works without explicit time_col."""
    df = _make_train_df(periods=120)
    automl = AutoML()
    _fit(automl, dataframe=df)

    # Future prediction input using DatetimeIndex
    future_index = pd.date_range("2028-01-01", periods=12, freq="MS")
    future_df = pd.DataFrame(index=future_index)
    preds = automl.predict(future_df)
    assert len(preds) == 12


def test_training_with_reordered_dataframe_target_aligns_by_label():
    """Verify that when y_train is passed with shuffled timestamps, values align by label rather than position."""
    train_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=train_idx)
    # Shuffled index for y_train
    shuffled_idx = train_idx[::-1]
    y_values = np.sin(np.arange(60)[::-1] / 6) * 10 + 50
    y_train = pd.DataFrame({"y": y_values}, index=shuffled_idx)

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, keep_search_state=True)

    # When y_train is reordered, AutoML must correctly align y by index rather than crashing or distorting data
    preds = automl.predict(X_train.tail(12))
    assert len(preds) == 12
    assert automl._state.task.time_col == "ds"
    expected_y = y_train.reindex(train_idx)["y"].values
    actual_y = automl._state.X_train_all.train_data["y"].values
    np.testing.assert_allclose(actual_y, expected_y)


def test_validation_with_reordered_dataframe_target_aligns_by_label():
    """Verify that when y_val is passed with shuffled timestamps, values align by label rather than position."""
    df_train = _make_train_df(periods=80)
    val_idx = pd.date_range("2024-09-01", periods=20, freq="MS")
    X_val = pd.DataFrame(index=val_idx)
    # Shuffled index for y_val
    shuffled_val_idx = val_idx[::-1]
    y_val = pd.DataFrame({"y": (np.arange(20)[::-1] + 100.0)}, index=shuffled_val_idx)

    automl = AutoML()
    _fit(automl, dataframe=df_train, X_val=X_val, y_val=y_val, keep_search_state=True)

    preds = automl.predict(X_val)
    assert len(preds) == 20
    assert automl._state.eval_method == "holdout"
    expected_val_y = y_val.reindex(val_idx)["y"].values
    actual_val_y = automl._state.X_val.test_data["y"].values
    np.testing.assert_allclose(actual_val_y, expected_val_y)


def test_mismatched_target_index_raises_value_error():
    """Verify that passing targets with non-matching labels raises ValueError instead of silent corruption."""
    train_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    mismatched_idx = pd.date_range("2019-01-01", periods=60, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=train_idx)
    y_train = pd.DataFrame({"y": np.arange(60, dtype=float)}, index=mismatched_idx)

    automl = AutoML()
    with pytest.raises(ValueError, match="Target index labels do not match feature index labels"):
        _fit(automl, X_train=X_train, y_train=y_train)


def test_training_with_range_index_series_target():
    """Verify positional pairing when X_train has DatetimeIndex and y_train is a RangeIndex Series."""
    train_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=train_idx)
    y_train = pd.Series(np.sin(np.arange(60) / 6) * 10 + 50, name="y")  # Default RangeIndex

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train)

    assert automl._state.task.time_col == "ds"
    assert automl._state.data_size[0] == len(X_train)
    preds = automl.predict(X_train.tail(12))
    assert len(preds) == 12


def test_validation_with_range_index_series_target():
    """Verify positional pairing when validation data has DatetimeIndex X_val and RangeIndex Series y_val."""
    df_train = _make_train_df(periods=80)
    val_idx = pd.date_range("2024-09-01", periods=20, freq="MS")
    X_val = pd.DataFrame({"feat": np.arange(80, 100, dtype=float)}, index=val_idx)
    y_val = pd.Series(np.sin(np.arange(80, 100) / 6) * 10 + 50, name="y")  # Default RangeIndex

    automl = AutoML()
    _fit(automl, dataframe=df_train, X_val=X_val, y_val=y_val)

    assert automl._state.eval_method == "holdout"
    preds = automl.predict(X_val)
    assert len(preds) == 20


def test_training_and_validation_with_range_index_series_targets():
    """Verify positional pairing when both training and validation targets are RangeIndex Series."""
    train_idx = pd.date_range("2018-01-01", periods=80, freq="MS")
    val_idx = pd.date_range("2024-09-01", periods=20, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(80, dtype=float)}, index=train_idx)
    y_train = pd.Series(np.sin(np.arange(80) / 6) * 10 + 50, name="y")
    X_val = pd.DataFrame({"feat": np.arange(80, 100, dtype=float)}, index=val_idx)
    y_val = pd.Series(np.sin(np.arange(80, 100) / 6) * 10 + 50, name="y")

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    assert automl._state.task.time_col == "ds"
    assert automl._state.eval_method == "holdout"
    preds = automl.predict(X_val)
    assert len(preds) == 20


def test_training_with_reordered_series_target_aligns_by_label():
    """Verify that when y_train is a Series with shuffled timestamps, values align by label."""
    train_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=train_idx)
    shuffled_idx = train_idx[::-1]
    y_values = np.sin(np.arange(60)[::-1] / 6) * 10 + 50
    y_train = pd.Series(y_values, index=shuffled_idx, name="y")

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, keep_search_state=True)

    preds = automl.predict(X_train.tail(12))
    assert len(preds) == 12
    assert automl._state.task.time_col == "ds"
    expected_y = y_train.reindex(train_idx).values
    actual_y = automl._state.X_train_all.train_data["y"].values
    np.testing.assert_allclose(actual_y, expected_y)


def test_training_with_range_index_x_datetime_col_and_datetime_index_series_target():
    """Verify positional pairing when X_train is RangeIndex with a datetime column and y_train is DatetimeIndex Series."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    X_train = pd.DataFrame({"ds": dates, "feat": np.arange(60, dtype=float)})
    y_train = pd.Series(np.sin(np.arange(60) / 6) * 10 + 50, index=dates, name="y")

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, time_col="ds")

    assert automl._state.task.time_col == "ds"
    assert automl._state.data_size[0] == len(X_train)
    preds = automl.predict(X_train.tail(12))
    assert len(preds) == 12


def test_validation_with_range_index_x_datetime_col_and_datetime_index_series_target():
    """Verify positional pairing when X_val is RangeIndex with a datetime column and y_val is DatetimeIndex Series."""
    train_dates = pd.date_range("2018-01-01", periods=80, freq="MS")
    val_dates = pd.date_range("2024-09-01", periods=20, freq="MS")
    X_train = pd.DataFrame({"ds": train_dates, "feat": np.arange(80, dtype=float)})
    y_train = pd.Series(np.sin(np.arange(80) / 6) * 10 + 50, name="y")
    X_val = pd.DataFrame({"ds": val_dates, "feat": np.arange(80, 100, dtype=float)})
    y_val = pd.Series(np.sin(np.arange(80, 100) / 6) * 10 + 50, index=val_dates, name="y")

    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, time_col="ds")

    assert automl._state.eval_method == "holdout"
    preds = automl.predict(X_val)
    assert len(preds) == 20


def test_index_only_training_input_with_separate_target():
    """Verify that X_train with 0 feature columns and DatetimeIndex promotes correctly without IndexError."""
    dates = pd.date_range("2018-01-01", periods=120, freq="MS")
    X_train = pd.DataFrame(index=dates)
    y_train = pd.Series(np.sin(np.arange(120) / 6) * 10 + 50, index=dates, name="y")
    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train)
    assert automl._state.task.time_col == "ds"
    assert len(automl.predict(X_train.tail(12))) == 12


def test_length_mismatch_raises_value_error():
    """Verify that when X has fewer rows than y, extra rows in y are NOT silently truncated."""
    train_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    extra_idx = pd.date_range("2018-01-01", periods=61, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=train_idx)
    y_train = pd.Series(np.arange(61, dtype=float), index=extra_idx, name="y")
    automl = AutoML()
    with pytest.raises(ValueError, match="does not match feature length"):
        _fit(automl, X_train=X_train, y_train=y_train)


def test_validation_length_mismatch_raises_value_error():
    """Verify that validation data with mismatched X_val and y_val raises ValueError."""
    df_train = _make_train_df(periods=80)
    val_idx = pd.date_range("2024-09-01", periods=20, freq="MS")
    val_extra_idx = pd.date_range("2024-09-01", periods=21, freq="MS")
    X_val = pd.DataFrame(index=val_idx)
    y_val = pd.Series(np.arange(21, dtype=float), index=val_extra_idx, name="y")
    automl = AutoML()
    with pytest.raises(ValueError, match="does not match feature length"):
        _fit(automl, dataframe=df_train, X_val=X_val, y_val=y_val)


def test_unnamed_datetime_index_colliding_with_target_named_ds():
    """Verify that when target is named 'ds', an unnamed index promotes to 'ds_1' to avoid duplicate columns."""
    dates = pd.date_range("2018-01-01", periods=120, freq="MS")
    X_train = pd.DataFrame({"feat": np.arange(120, dtype=float)}, index=dates)
    y_train = pd.Series(np.sin(np.arange(120) / 6) * 10 + 50, index=dates, name="ds")
    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, label="ds")
    assert automl._state.task.time_col == "ds_1"
    assert len(automl.predict(X_train.tail(12))) == 12


def test_named_datetime_index_colliding_with_target_named_y():
    """Verify that when index and target are both named 'y', index promotes to 'y_1' to avoid duplicate columns."""
    dates = pd.date_range("2018-01-01", periods=120, freq="MS")
    dates.name = "y"
    X_train = pd.DataFrame({"feat": np.arange(120, dtype=float)}, index=dates)
    y_train = pd.Series(np.sin(np.arange(120) / 6) * 10 + 50, index=dates, name="y")
    automl = AutoML()
    _fit(automl, X_train=X_train, y_train=y_train, label="y")
    assert automl._state.task.time_col == "y_1"
    assert len(automl.predict(X_train.tail(12))) == 12


def test_valid_date_string_column_preserved_over_incidental_datetime_index():
    """Verify that a valid ISO date string column is preserved over an incidental DatetimeIndex during inference and fit."""
    dates_str = pd.date_range("2018-01-01", periods=120, freq="MS").strftime("%Y-%m-%d").tolist()
    incidental_idx = pd.date_range("2030-01-01", periods=120, freq="MS")
    df = pd.DataFrame(
        {
            "ds": dates_str,
            "y": np.sin(np.arange(120) / 6) * 10 + 50,
        },
        index=incidental_idx,
    )
    automl = AutoML()
    # Inferred time_col (no explicit time_col provided)
    _fit(automl, dataframe=df)
    assert automl._state.task.time_col == "ds"
    # Ensure the timeline corresponds to 2018, not 2030 (120 monthly periods end in 2027)
    assert automl.model.end_date.year == 2027
    future_df = pd.DataFrame({"ds": pd.date_range("2028-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")})
    assert len(automl.predict(future_df)) == 12


def test_positional_series_contract_preserved_when_timestamp_column_present():
    """Verify that when X has an existing timestamp column, Series target is paired positionally even if X has a DatetimeIndex."""
    from flaml.automl.task.time_series_task import TimeSeriesTask

    # X has explicit timestamp column 'ds' in order, but an incidental reverse DatetimeIndex
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    X = pd.DataFrame(
        {"ds": dates, "feat": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=dates[::-1],
    )
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=dates[::-1])

    needs_promotion = TimeSeriesTask._needs_datetime_index_promotion(X, time_col="ds", is_inferred=False)
    assert not needs_promotion

    X_aligned, y_aligned = TimeSeriesTask._align_y_to_X_and_promote(X, y, needs_promotion=needs_promotion)
    # Target values must stay in original order (positional pairing), NOT re-indexed
    assert list(y_aligned.values) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_label_mismatch_with_existing_nan_raises_value_error():
    """Verify that index label mismatch is rejected even when dropped row contains NaN (which defeated NaN-count check)."""
    from flaml.automl.task.time_series_task import TimeSeriesTask

    # X has Jan 1 and Jan 2; y has Jan 2 and Jan 3 (with value NaN)
    x_idx = pd.date_range("2020-01-01", periods=2, freq="D")
    y_idx = pd.date_range("2020-01-02", periods=2, freq="D")

    X = pd.DataFrame({"feat": [1.0, 2.0]}, index=x_idx)
    y = pd.Series([10.0, np.nan], index=y_idx)

    # When promotion is needed, mismatched index labels must raise ValueError
    with pytest.raises(ValueError, match="Target index labels do not match feature index labels"):
        TimeSeriesTask._align_y_to_X_and_promote(X, y, needs_promotion=True)


def test_predict_restores_datetime_index_preventing_assignment_nans():
    """Verify that predictions on future_df with DatetimeIndex retain future_df.index, preventing assignment NaNs."""
    df = _make_train_df(periods=120)
    automl = AutoML()
    _fit(automl, dataframe=df)

    future_index = pd.date_range("2028-01-01", periods=12, freq="MS")
    future_df = pd.DataFrame(index=future_index)

    preds = automl.predict(future_df)
    assert isinstance(preds, (pd.Series, pd.DataFrame, np.ndarray))
    # When assigned back to future_df, pandas aligns by index. If index was RangeIndex, this produced all NaNs.
    future_df["prediction"] = preds
    assert not future_df["prediction"].isna().any()
    assert len(future_df["prediction"]) == 12


def test_is_valid_timestamp_col_boundary_cases():
    """Directly test _is_valid_timestamp_col with various dtypes."""
    from flaml.automl.task.time_series_task import TimeSeriesTask

    df = pd.DataFrame(
        {
            "num_int": [1, 2, 3],
            "num_float": [1.0, 2.0, 3.0],
            "num_str": ["100", "200", "300"],
            "invalid_str": ["not-a-date", "foo", "bar"],
            "iso_str": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "dt64": pd.date_range("2020-01-01", periods=3),
            "bool_col": [True, False, True],
        }
    )

    assert not TimeSeriesTask._is_valid_timestamp_col(df, "num_int")
    assert not TimeSeriesTask._is_valid_timestamp_col(df, "num_float")
    assert not TimeSeriesTask._is_valid_timestamp_col(df, "num_str")
    assert not TimeSeriesTask._is_valid_timestamp_col(df, "invalid_str")
    assert not TimeSeriesTask._is_valid_timestamp_col(df, "bool_col")
    assert not TimeSeriesTask._is_valid_timestamp_col(df, "non_existent_col")
    assert TimeSeriesTask._is_valid_timestamp_col(df, "iso_str")
    assert TimeSeriesTask._is_valid_timestamp_col(df, "dt64")


def test_reordered_dataframe_target_with_existing_time_col_aligns_by_value():
    """Verify that when X has an existing timestamp column, DataFrame target with reversed index
    aligns by index label to preserve target-feature mapping (reproducing maintainer thinkall's case)."""
    n = 40
    X = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feat": np.arange(n),
        }
    )
    y = pd.DataFrame({"y": np.arange(n, dtype=float)[::-1]}, index=np.arange(n)[::-1])

    automl = AutoML()
    _fit(
        automl,
        X_train=X,
        y_train=y,
        period=4,
        time_budget=3,
        keep_search_state=True,
    )
    head_df = automl._state.X_train_all.train_data.head()
    assert list(head_df["y"].values) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_reordered_dataframe_target_with_datetime_index_promotion_aligns_by_value():
    """Verify that when DatetimeIndex is promoted to time_col, DataFrame target with reversed
    DatetimeIndex aligns by timestamp to preserve target-feature mapping."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"feat": np.arange(40)}, index=dates)
    y = pd.DataFrame({"y": np.arange(40, dtype=float)[::-1]}, index=dates[::-1])

    automl = AutoML()
    _fit(
        automl,
        X_train=X,
        y_train=y,
        period=4,
        time_budget=3,
        keep_search_state=True,
    )
    head_df = automl._state.X_train_all.train_data.head()
    assert list(head_df["y"].values) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_explicit_missing_time_col_raises_error_even_with_datetime_index():
    """Verify that explicitly specifying a non-existent time_col raises ValueError rather than
    silently promoting DatetimeIndex during training (Copilot review discussion_r4308583028)."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"feat": np.arange(40)}, index=dates)
    y = pd.Series(np.arange(40, dtype=float), index=dates, name="y")

    automl = AutoML()
    with pytest.raises(ValueError, match="must contain timestamp values"):
        _fit(automl, X_train=X, y_train=y, time_col="non_existent_time_col")

    # Also test dataframe input with explicit missing time_col
    df = pd.DataFrame({"feat": np.arange(40), "y": np.arange(40, dtype=float)}, index=dates)
    automl_df = AutoML()
    with pytest.raises(ValueError, match="must contain timestamp values"):
        _fit(automl_df, dataframe=df, time_col="non_existent_time_col")


def test_mismatched_dataframe_target_labels_raise_value_error():
    """Verify that DataFrame targets with mismatched labels raise ValueError instead of silent corruption."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    wrong_dates = pd.date_range("2021-01-01", periods=40, freq="D")
    X = pd.DataFrame({"ds": dates, "feat": np.arange(40)})
    y = pd.DataFrame({"y": np.arange(40, dtype=float)}, index=wrong_dates)

    automl = AutoML()
    with pytest.raises(ValueError, match="Target index labels do not match feature index labels"):
        _fit(automl, X_train=X, y_train=y, time_col="ds")


def test_explicit_time_col_missing_from_features_rejected_even_if_present_in_target_dataframe():
    """Verify that an explicitly specified time_col must be present in features X, and cannot be
    satisfied by a numeric column in DataFrame target y (reproducing maintainer thinkall's case)."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"feat": np.arange(40)}, index=dates)
    y = pd.DataFrame(
        {"y": 3 + 7 * np.arange(40, dtype=float), "ds": np.arange(40)},
        index=dates,
    )

    automl = AutoML()
    with pytest.raises(ValueError, match="must contain timestamp values"):
        _fit(
            automl,
            X_train=X,
            y_train=y,
            label="y",
            time_col="ds",
            period=4,
            keep_search_state=True,
        )


def test_numpy_array_training_input_with_explicit_time_col():
    """Verify that a NumPy array X_train with timestamps in column 0 is accepted when time_col is explicit."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    X_arr = np.column_stack([dates.astype(object), np.arange(40, dtype=float)])
    y_arr = np.arange(40, dtype=float)

    automl = AutoML()
    _fit(
        automl,
        X_train=X_arr,
        y_train=y_arr,
        time_col="ds",
        period=4,
    )
    assert automl._state.task.time_col == "ds"
    assert len(automl.predict(X_arr[-12:])) == 12


def test_numpy_array_validation_input_accepted():
    """Verify that a NumPy array X_val with timestamps in column 0 is accepted for holdout validation."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df_train = pd.DataFrame({"ds": dates[:30], "y": np.arange(30, dtype=float)})
    X_val_arr = dates[30:].values.reshape(-1, 1)
    y_val_arr = np.arange(30, 40, dtype=float)

    automl = AutoML()
    _fit(
        automl,
        dataframe=df_train,
        label="y",
        X_val=X_val_arr,
        y_val=y_val_arr,
        period=4,
    )
    assert automl._state.eval_method == "holdout"
    assert len(automl.predict(X_val_arr)) == 10


@pytest.mark.parametrize("name", ["", 0, "when"])
def test_explicit_index_name_prediction_form_equivalence(name):
    """Verify that explicit index names (including empty string '' and integer 0) are preserved
    as the time column, and raw-index future frames produce identical predictions to reset_index() frames."""
    df = _make_train_df(periods=120)
    df.index.name = name
    automl = AutoML()
    _fit(automl, dataframe=df)
    assert automl._state.task.time_col == name

    future_index = pd.date_range("2028-01-01", periods=12, freq="MS", name=name)
    future_raw = pd.DataFrame(index=future_index)
    future_reset = future_raw.reset_index()

    pred_raw = automl.predict(future_raw)
    pred_reset = automl.predict(future_reset)

    assert len(pred_raw) == 12
    assert len(pred_reset) == 12
    np.testing.assert_allclose(pred_raw.values, pred_reset.values)
