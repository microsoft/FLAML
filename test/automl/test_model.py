from datetime import datetime

import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_classification

from flaml.automl.contrib.histgb import HistGradientBoostingEstimator
from flaml.automl.model import (
    BaseEstimator,
    CatBoostEstimator,
    KNeighborsEstimator,
    LGBMEstimator,
    LRL2Classifier,
    RandomForestEstimator,
    XGBoostEstimator,
)
from flaml.automl.time_series import ARIMA, LGBM_TS, Prophet, TimeSeriesDataset


def test_lrl2():
    BaseEstimator.search_space(1, "")
    X, y = make_classification(100000, 1000)
    print("start")
    lr = LRL2Classifier()
    lr.predict(X)
    lr.fit(X, y, budget=1e-5)


def test_prep():
    X = np.array(
        list(
            zip(
                [
                    3.0,
                    16.0,
                    10.0,
                    12.0,
                    3.0,
                    14.0,
                    11.0,
                    12.0,
                    5.0,
                    14.0,
                    20.0,
                    16.0,
                    15.0,
                    11.0,
                ],
                [
                    "a",
                    "b",
                    "a",
                    "c",
                    "c",
                    "b",
                    "b",
                    "b",
                    "b",
                    "a",
                    "b",
                    1.0,
                    1.0,
                    "a",
                ],
            )
        ),
        dtype=object,
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    lr = LRL2Classifier()
    lr.fit(X, y)
    lr.predict(X)
    print(lr.feature_names_in_)
    print(lr.feature_importances_)
    lgbm = LGBMEstimator(n_estimators=4)
    lgbm.fit(X, y)
    print(lgbm.feature_names_in_)
    print(lgbm.feature_importances_)
    cat = CatBoostEstimator(n_estimators=4)
    cat.fit(X, y)
    print(cat.feature_names_in_)
    print(cat.feature_importances_)
    knn = KNeighborsEstimator(task="regression")
    knn.fit(X, y)
    print(knn.feature_names_in_)
    print(knn.feature_importances_)
    xgb = XGBoostEstimator(n_estimators=4, max_leaves=4)
    xgb.fit(X, y)
    xgb.predict(X)
    print(xgb.feature_names_in_)
    print(xgb.feature_importances_)
    rf = RandomForestEstimator(task="regression", n_estimators=4, criterion="gini")
    rf.fit(X, y)
    print(rf.feature_names_in_)
    print(rf.feature_importances_)
    hgb = HistGradientBoostingEstimator(task="regression", n_estimators=4, max_leaves=4)
    hgb.fit(X, y)
    hgb.predict(X)
    print(hgb.feature_names_in_)
    print(hgb.feature_importances_)

    prophet = Prophet()
    try:
        prophet.predict(4)
    except ValueError:
        # predict() with steps is only supported for arima/sarimax.
        pass
    prophet.predict(X)

    # What's the point of callin ARIMA without parameters, or calling predict before fit?
    arima = ARIMA(p=1, q=1, d=0)
    arima.predict(X)
    arima._model = False
    try:
        arima.predict(X)
    except ValueError:
        # X_test needs to be either a pandas Dataframe with dates as the first column or an int number of periods for predict().
        pass
    lgbm = LGBM_TS(lags=1)
    X = DataFrame(
        {
            "A": [
                datetime(1900, 3, 1),
                datetime(1900, 3, 2),
                datetime(1900, 3, 3),
                datetime(1900, 3, 4),
                datetime(1900, 3, 4),
                datetime(1900, 3, 4),
                datetime(1900, 3, 5),
                datetime(1900, 3, 6),
            ],
        }
    )
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    lgbm.predict(X[:2])
    df = X.copy()
    df["y"] = y
    tsds = TimeSeriesDataset(df, time_col="A", target_names="y")
    lgbm.fit(tsds, period=2)
    lgbm.predict(X[:2])
    print(lgbm.feature_names_in_)
    print(lgbm.feature_importances_)


def test_prettify_prediction_auto_timestamps():
    """Test that prettify_prediction auto-generates timestamps when test_data is None.

    This tests the fix for the TODO that previously raised NotImplementedError.
    """
    import pandas as pd

    # Create training data with daily frequency
    n = 30
    train_df = DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=n, freq="D"),
            "value": np.random.randn(n),
        }
    )

    # Create TimeSeriesDataset without test_data
    tsds = TimeSeriesDataset(train_df, time_col="date", target_names="value")

    # Verify test_data is empty (not None, but empty DataFrame per __init__)
    assert len(tsds.test_data) == 0

    # Test with np.ndarray input
    pred_steps = 5
    y_pred_array = np.random.randn(pred_steps)
    result_array = tsds.prettify_prediction(y_pred_array)

    assert isinstance(result_array, pd.DataFrame)
    assert "date" in result_array.columns
    assert "value" in result_array.columns
    assert len(result_array) == pred_steps
    # Verify timestamps start from training end + 1 period
    expected_start = pd.date_range(start=train_df["date"].max(), periods=2, freq="D")[1]
    assert result_array["date"].iloc[0] == expected_start

    # Test with pd.Series input
    y_pred_series = pd.Series(np.random.randn(pred_steps))
    result_series = tsds.prettify_prediction(y_pred_series)

    assert isinstance(result_series, pd.DataFrame)
    assert "date" in result_series.columns
    assert "value" in result_series.columns
    assert len(result_series) == pred_steps
    assert result_series["date"].iloc[0] == expected_start

    # Test with pd.DataFrame input
    y_pred_df = pd.DataFrame({"value": np.random.randn(pred_steps)})
    result_df = tsds.prettify_prediction(y_pred_df)

    assert isinstance(result_df, pd.DataFrame)
    assert "date" in result_df.columns
    assert "value" in result_df.columns
    assert len(result_df) == pred_steps
    assert result_df["date"].iloc[0] == expected_start

    # Verify the generated timestamps follow the correct frequency
    expected_dates = pd.date_range(start=expected_start, periods=pred_steps, freq="D")
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(result_array["date"]),
        expected_dates,
        check_names=False,
    )

    print("test_prettify_prediction_auto_timestamps passed!")


def test_prettify_prediction_auto_timestamps_monthly():
    """Test auto-timestamp generation with monthly frequency."""
    import pandas as pd

    # Create training data with monthly frequency
    n = 24
    train_df = DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="MS"),
            "value": np.random.randn(n),
        }
    )

    tsds = TimeSeriesDataset(train_df, time_col="date", target_names="value")
    assert len(tsds.test_data) == 0

    pred_steps = 6
    y_pred = np.random.randn(pred_steps)
    result = tsds.prettify_prediction(y_pred)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == pred_steps

    # For monthly frequency, verify timestamps are monthly
    expected_dates = pd.date_range(
        start=train_df["date"].max(),
        periods=pred_steps + 1,
        freq="MS",
    )[1:]  # Skip first (which is train_end_date)

    pd.testing.assert_index_equal(
        pd.DatetimeIndex(result["date"]),
        expected_dates,
        check_names=False,
    )

    print("test_prettify_prediction_auto_timestamps_monthly passed!")


if __name__ == "__main__":
    test_lrl2()
    test_prep()
    test_prettify_prediction_auto_timestamps()
    test_prettify_prediction_auto_timestamps_monthly()
