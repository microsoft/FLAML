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


if __name__ == "__main__":
    test_prep()
