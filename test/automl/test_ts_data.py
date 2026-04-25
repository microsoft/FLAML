import numpy as np
import pandas as pd

from flaml.automl.time_series.ts_data import TimeSeriesDataset, create_forward_frame


def test_prettify_prediction_generates_timestamps_without_test_data():
    train_data = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = TimeSeriesDataset(train_data, time_col="ds", target_names="y")
    expected_times = pd.date_range("2020-01-05", periods=2, freq="D")

    for y_pred in (
        pd.DataFrame({"y": [5.0, 6.0]}, index=[10, 11]),
        pd.Series([5.0, 6.0]),
        np.array([5.0, 6.0]),
    ):
        prediction = dataset.prettify_prediction(y_pred)
        assert isinstance(prediction, pd.DataFrame)
        pd.testing.assert_series_equal(prediction["ds"], pd.Series(expected_times, name="ds"), check_index=False)
        assert prediction["y"].tolist() == [5.0, 6.0]


def test_prettify_prediction_generates_monthly_timestamps_without_test_data():
    train_data = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = TimeSeriesDataset(train_data, time_col="ds", target_names="y")

    prediction = dataset.prettify_prediction(pd.DataFrame({"y": [5.0, 6.0]}))

    pd.testing.assert_series_equal(
        prediction["ds"],
        pd.Series(pd.date_range("2020-05-01", periods=2, freq="MS"), name="ds"),
        check_index=False,
    )
    assert prediction["y"].tolist() == [5.0, 6.0]


def test_create_forward_frame_uses_next_frequency_offset():
    # Pandas 3 uses QE-DEC while older supported versions use Q-DEC.
    quarter_end_freq = "QE-DEC"
    try:
        pd.tseries.frequencies.to_offset(quarter_end_freq)
    except ValueError:
        quarter_end_freq = "Q-DEC"

    weekly_frame = create_forward_frame("W-SUN", 2, pd.Timestamp("2020-01-05"), "ds")
    quarterly_frame = create_forward_frame(quarter_end_freq, 2, pd.Timestamp("2020-03-31"), "ds")

    pd.testing.assert_series_equal(
        weekly_frame["ds"], pd.Series(pd.date_range("2020-01-12", periods=2, freq="W-SUN"), name="ds")
    )
    pd.testing.assert_series_equal(
        quarterly_frame["ds"], pd.Series(pd.date_range("2020-06-30", periods=2, freq=quarter_end_freq), name="ds")
    )
