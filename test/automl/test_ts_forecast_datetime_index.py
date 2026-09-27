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
