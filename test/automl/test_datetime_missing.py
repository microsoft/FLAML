"""Missing datetime features must use the learned numeric imputer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from flaml import AutoML
from flaml.automl.data import DataTransformer
from flaml.automl.task.factory import task_factory


def _frame(values, utc=False):
    return pd.DataFrame({"timestamp": pd.to_datetime(values, utc=utc), "numeric": np.arange(len(values), dtype=float)})


def _fit(frame):
    transformer = DataTransformer()
    fitted, _ = transformer.fit_transform(frame, np.arange(len(frame), dtype=float), "regression")
    return transformer, fitted


@pytest.mark.parametrize("utc", [False, True], ids=["naive", "utc"])
def test_fit_imputes_missing_datetime_from_valid_ordinals(utc):
    frame = _frame(["2020-02-28 01:00", "2020-02-29 03:00", None, "2020-03-02 05:00"], utc)
    original = frame.copy(deep=True)
    transformer, fitted = _fit(frame)
    expected = [
        date(2020, 2, 28).toordinal(),
        date(2020, 2, 29).toordinal(),
        date(2020, 2, 29).toordinal(),
        date(2020, 3, 2).toordinal(),
    ]
    np.testing.assert_array_equal(fitted["timestamp"], expected)
    assert fitted.loc[2, "hour_timestamp"] == 3
    pd.testing.assert_frame_equal(transformer.transform(frame), fitted)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("utc", [False, True], ids=["naive", "utc"])
@pytest.mark.parametrize("all_missing", [False, True], ids=["mixed", "all_missing"])
def test_transform_uses_training_datetime_median(utc, all_missing):
    train = _frame(["2020-02-28 01:00", "2020-02-29 03:00", "2020-03-02 05:00"], utc)
    transformer, _ = _fit(train)
    values = [None, None] if all_missing else ["2021-01-01 07:00", None]
    frame = _frame(values, utc)
    original = frame.copy(deep=True)
    transformed = transformer.transform(frame)
    median = date(2020, 2, 29).toordinal()
    expected = [median, median] if all_missing else [date(2021, 1, 1).toordinal(), median]
    np.testing.assert_array_equal(transformed["timestamp"], expected)
    assert transformed.loc[1, "hour_timestamp"] == 3
    assert np.isfinite(transformed.to_numpy()).all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("utc", [False, True], ids=["naive", "utc"])
def test_valid_datetimes_keep_their_ordinals(utc):
    frame = _frame(["2020-02-28 01:00", "2020-02-29 03:00", "2020-03-02 05:00"], utc)
    transformer, fitted = _fit(frame)
    np.testing.assert_array_equal(
        fitted["timestamp"],
        [date(2020, 2, 28).toordinal(), date(2020, 2, 29).toordinal(), date(2020, 3, 2).toordinal()],
    )
    pd.testing.assert_frame_equal(transformer.transform(frame), fitted)


def test_all_missing_training_datetime_is_still_dropped():
    frame = _frame([None] * 4)
    transformer, fitted = _fit(frame)
    assert list(fitted.columns) == ["numeric"]
    pd.testing.assert_frame_equal(transformer.transform(frame), fitted)


@pytest.mark.parametrize("task_name", ["regression", "classification"])
def test_task_validation_and_preprocess_impute_missing_datetimes(task_name):
    train = _frame(["2020-02-28 01:00", "2020-02-29 03:00", None, "2020-03-02 05:00"])
    labels = np.array([0, 1, 0, 1])
    validation = _frame([None, "2021-01-01 07:00"])
    automl = AutoML()
    task = task_factory(task_name, train, labels)
    automl._skip_transform = False
    automl._state.fit_kwargs = {}
    automl._state.task = task
    task.validate_data(automl, automl._state, train, labels, None, None, validation, np.array([0, 1]))
    assert automl._X_train_all.loc[2, "timestamp"] == date(2020, 2, 29).toordinal()
    np.testing.assert_array_equal(
        automl._state.X_val["timestamp"], [date(2020, 2, 29).toordinal(), date(2021, 1, 1).toordinal()]
    )
    pd.testing.assert_frame_equal(automl.preprocess(validation), automl._state.X_val)
    np.testing.assert_array_equal(automl._y_train_all, labels)
