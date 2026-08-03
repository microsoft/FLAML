import numpy as np
import pandas as pd
import pytest

from flaml import AutoML
from flaml.automl.time_series.ts_data import TimeSeriesDataset, create_forward_frame


def _make_stock_data(start="2024-01-01", end="2024-12-31", holidays=None):
    """Create a synthetic daily stock price dataset on business days,
    optionally dropping specific holiday dates."""
    dates = pd.bdate_range(start, end)
    if holidays is not None:
        dates = dates[~dates.isin(pd.to_datetime(holidays))]

    rng = np.random.RandomState(42)
    price = 100 + np.cumsum(rng.randn(len(dates)) * 0.5)
    volume = rng.randint(1_000, 10_000, size=len(dates)).astype(float)

    return pd.DataFrame({"ds": dates, "price": price, "volume": volume})


US_HOLIDAYS_2024 = [
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents' Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
]


def test_regular_daily():
    """Regular daily data should be inferred directly by pd.infer_freq."""
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.arange(90, dtype=float)})
    ts = TimeSeriesDataset(df, time_col="ds", target_names="y")
    assert ts.frequency == "D"


def test_regular_monthly():
    """Regular month-start data should be inferred as MS."""
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"ds": dates, "y": np.arange(24, dtype=float)})
    ts = TimeSeriesDataset(df, time_col="ds", target_names="y")
    assert ts.frequency == "MS"


def test_business_day_no_holidays():
    """Business day data without holiday gaps should be inferred as B."""
    dates = pd.bdate_range("2024-01-01", "2024-06-30")
    df = pd.DataFrame({"ds": dates, "y": np.arange(len(dates), dtype=float)})
    ts = TimeSeriesDataset(df, time_col="ds", target_names="y")
    # ``"24h"`` is what ``pd.infer_freq`` returns on some pandas/OS combos
    # (notably pandas 3.x on Windows) for the same daily cadence that yields
    # ``"D"`` elsewhere. Accept all three forms of "daily-ish".
    assert ts.frequency in ("B", "D", "24h")


def test_business_day_with_holidays():
    """Business day data with holidays removed should still succeed
    via the fallback frequency inference."""
    df = _make_stock_data(holidays=US_HOLIDAYS_2024)
    ts = TimeSeriesDataset(df, time_col="ds", target_names="price")
    assert ts.frequency is not None
    # See ``test_business_day_no_holidays`` for why ``"24h"`` is accepted.
    assert ts.frequency in ("D", "B", "24h")


def test_stock_data_multivariate():
    """Multivariate stock data (price + volume) with holidays should work."""
    df = _make_stock_data(holidays=US_HOLIDAYS_2024)
    ts = TimeSeriesDataset(df, time_col="ds", target_names=["price"])
    assert ts.frequency is not None
    assert "volume" in ts.time_varying_known_reals


def test_unsorted_input():
    """Shuffled rows should still yield a valid frequency."""
    df = _make_stock_data(holidays=US_HOLIDAYS_2024).sample(frac=1, random_state=0)
    ts = TimeSeriesDataset(df, time_col="ds", target_names="price")
    assert ts.frequency is not None


def test_single_timestamp_raises():
    """A single unique timestamp cannot have a frequency inferred."""
    df = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01"]), "y": [1.0]})
    with pytest.raises(AssertionError, match="regular frequency"):
        TimeSeriesDataset(df, time_col="ds", target_names="y")


def test_forecast_stock_data_without_holidays(budget=30):
    """End-to-end AutoML forecast: train and predict on synthetic daily
    stock price data that has business-day dates with US holidays removed."""
    df = _make_stock_data(holidays=US_HOLIDAYS_2024)[["ds", "price"]]
    time_horizon = 20
    split_idx = len(df) - time_horizon

    train_df = df.iloc[:split_idx]
    y_test = df.iloc[split_idx:]["price"]

    automl = AutoML()
    settings = {
        "time_budget": budget,
        "metric": "mape",
        "task": "ts_forecast",
        "log_file_name": "test/stock_forecast.log",
        "eval_method": "holdout",
        "label": "price",
        "estimator_list": ["arima", "sarimax"],
    }

    automl.fit(dataframe=train_df, **settings, period=time_horizon)

    assert automl.best_estimator is not None, "AutoML should find a best estimator"
    assert automl.best_config is not None, "AutoML should produce a best config"

    y_pred = automl.predict(time_horizon)
    assert len(y_pred) == time_horizon, f"Expected {time_horizon} predictions, got {len(y_pred)}"
    assert not np.any(np.isnan(y_pred)), "Predictions should not contain NaN"

    from flaml.automl.ml import sklearn_metric_loss_score

    mape = sklearn_metric_loss_score("mape", y_pred, y_test.values)
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print(f"Test MAPE: {mape}")
    assert np.isfinite(mape), "MAPE should be finite"


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
