"""Regression tests for ts_forecast input where the timestamp is the DataFrame index.

Flaml expects the timestamp to live in a *column* (`time_col`). When a user passes the
idiomatic pandas shape instead -- timestamps as a `DatetimeIndex` -- `validate_data`
defaults `time_col` to `dataframe.columns[0]`, which is a value column (often the label
itself), so the real timestamps are never looked at.
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
