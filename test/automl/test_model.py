import platform
import sys
from datetime import datetime

import numpy as np
import pytest
import scipy.sparse
from pandas import DataFrame
from sklearn.datasets import make_classification

from flaml.automl.contrib.histgb import HistGradientBoostingEstimator
from flaml.automl.contrib.sefr import SEFRBoostEstimator, SEFRClassifier, SEFREstimator
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


@pytest.mark.skipif(
    sys.platform == "win32" and platform.machine() == "ARM64", reason="catboost is not available on win-arm64 machine"
)
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


def test_sefr_matches_closed_form():
    """SEFR is a closed form; check the fitted model against it directly."""
    X, y = make_classification(200, 8, random_state=0)
    clf = SEFRClassifier(scaling="none", calibration="sigmoid").fit(X, y)

    avg_pos = X[y == 1].mean(axis=0)
    avg_neg = X[y == 0].mean(axis=0)
    expected_coef = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1e-7)
    scores = X @ expected_coef
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    expected_bias = (n_neg * scores[y == 1].mean() + n_pos * scores[y == 0].mean()) / (n_pos + n_neg)

    np.testing.assert_allclose(clf.coef_.ravel(), expected_coef)
    np.testing.assert_allclose(clf.intercept_[0], expected_bias)
    assert clf.coef_.shape == (1, 8)
    # the model is n_features + 1 floats
    assert clf.coef_.size + clf.intercept_.size == 9


def test_sefr_sample_weight():
    X, y = make_classification(200, 8, random_state=0)
    base = SEFRClassifier().fit(X, y)
    uniform = SEFRClassifier().fit(X, y, sample_weight=np.full(len(y), 3.0))
    # a constant weight rescales both class means identically, so it is a no-op
    np.testing.assert_allclose(base.coef_, uniform.coef_)
    np.testing.assert_allclose(base.intercept_, uniform.intercept_)

    weights = np.random.RandomState(0).uniform(0.1, 5.0, len(y))
    weighted = SEFRClassifier().fit(X, y, sample_weight=weights)
    assert not np.allclose(base.coef_, weighted.coef_)


def test_sefr_multiclass_and_proba():
    X, y = make_classification(300, 8, n_classes=3, n_informative=5, random_state=0)
    for calibration in ("platt", "sigmoid"):
        clf = SEFRClassifier(calibration=calibration).fit(X, y)
        assert clf.coef_.shape == (3, 8)
        proba = clf.predict_proba(X)
        assert proba.shape == (300, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert set(np.unique(clf.predict(X))) <= set(clf.classes_)


def test_sefr_sparse():
    X, y = make_classification(200, 8, random_state=0)
    X = np.abs(X)
    dense = SEFRClassifier(scaling="maxabs").fit(X, y)
    sparse = SEFRClassifier(scaling="maxabs").fit(scipy.sparse.csr_matrix(X), y)
    np.testing.assert_allclose(dense.coef_, sparse.coef_)
    np.testing.assert_allclose(dense.decision_function(X), sparse.decision_function(scipy.sparse.csr_matrix(X)))


def test_sefr_estimators():
    X, y = make_classification(200, 8, random_state=0)
    assert set(SEFREstimator.search_space()) == {
        "scaling",
        "class_weight",
        "threshold",
        "threshold_shift",
        "calibration",
    }
    assert "n_estimators" in SEFRBoostEstimator.search_space(data_size=(200, 8))

    sefr = SEFREstimator(task="binary")
    sefr.fit(X, y)
    assert sefr.predict(X).shape == (200,)
    assert sefr.predict_proba(X).shape == (200, 2)
    assert sefr.feature_importances_ is not None

    boost = SEFRBoostEstimator(task="binary", n_estimators=4)
    boost.fit(X, y)
    assert boost.predict_proba(X).shape == (200, 2)


if __name__ == "__main__":
    test_lrl2()
    test_prep()
