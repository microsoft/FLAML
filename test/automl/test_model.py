import inspect
import platform
import sys
from datetime import datetime

import numpy as np
import pytest
import scipy.sparse
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.utils.estimator_checks import check_estimator

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
    X = np.abs(X)
    clf = SEFRClassifier(scaling="none").fit(X, y)

    avg_pos = X[y == 1].mean(axis=0)
    avg_neg = X[y == 0].mean(axis=0)
    expected_coef = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1e-7)
    scores = X @ expected_coef
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    expected_bias = (n_neg * scores[y == 1].mean() + n_pos * scores[y == 0].mean()) / (n_pos + n_neg)

    np.testing.assert_allclose(clf.coef_.ravel(), expected_coef)
    np.testing.assert_allclose(clf.intercept_[0], expected_bias)
    # a binary head is n_features + 1 floats
    assert clf.coef_.size + clf.intercept_.size == 9


def test_sefr_sample_weight():
    X, y = make_classification(200, 8, random_state=0)
    for calibration in ("sigmoid", "platt"):
        base = SEFRClassifier(calibration=calibration).fit(X, y)
        uniform = SEFRClassifier(calibration=calibration).fit(X, y, sample_weight=np.full(len(y), 3.0))
        # a constant weight rescales both class means identically, so it is a no-op
        np.testing.assert_allclose(base.coef_, uniform.coef_)
        np.testing.assert_allclose(base.intercept_, uniform.intercept_)

        weights = np.random.RandomState(0).uniform(0.1, 5.0, len(y))
        weighted = SEFRClassifier(calibration=calibration).fit(X, y, sample_weight=weights)
        assert not np.allclose(base.coef_, weighted.coef_)
        # the weights reach the calibration too, not only the SEFR head
        assert base.calibration_scale_ != weighted.calibration_scale_

        # a zero weight is the same as dropping the sample, feature range included
        weights[:50] = 0
        zeroed = SEFRClassifier(calibration=calibration).fit(X, y, sample_weight=weights)
        dropped = SEFRClassifier(calibration=calibration).fit(X[50:], y[50:], sample_weight=weights[50:])
        np.testing.assert_allclose(zeroed.coef_, dropped.coef_)
        np.testing.assert_allclose(zeroed.calibration_scale_, dropped.calibration_scale_, rtol=1e-5)


def test_sefr_invalid_sample_weight():
    X, y = make_classification(200, 8, random_state=0)
    weights = np.ones(len(y))
    weights[y == 0] = 0
    with pytest.raises(ValueError, match="sum to zero"):
        SEFRClassifier(class_weight="balanced").fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="non-negative"):
        SEFRClassifier().fit(X, y, sample_weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match="finite"):
        SEFRClassifier().fit(X, y, sample_weight=np.full(len(y), np.nan))


def test_sefr_multiclass_and_proba():
    X, y = make_classification(300, 8, n_classes=3, n_informative=5, random_state=0)
    for calibration in ("sigmoid", "platt"):
        clf = SEFRClassifier(calibration=calibration).fit(X, y)
        assert clf.coef_.shape == (3, 8)
        proba = clf.predict_proba(X)
        assert proba.shape == (300, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(X))
        np.testing.assert_array_equal(clf.decision_function(X).argmax(axis=1), proba.argmax(axis=1))


def test_sefr_multiclass_calibration():
    X, y = make_classification(600, 8, n_classes=4, n_informative=6, weights=[0.55, 0.25, 0.15], random_state=0)
    closed = SEFRClassifier().fit(X, y)
    fitted = SEFRClassifier(calibration="platt").fit(X, y)
    # the closed form uses the log class prior as the per-class bias
    prior = np.bincount(y) / len(y)
    np.testing.assert_allclose(closed.calibration_bias_, np.log(prior))
    # maximum likelihood starts from the closed form, so it can only improve on it
    assert log_loss(y, fitted.predict_proba(X)) < log_loss(y, closed.predict_proba(X))
    assert fitted.calibration_scale_ > 0

    # sample weights reach the fitted biases
    weights = np.where(y == 0, 5.0, 1.0)
    weighted = SEFRClassifier(calibration="platt").fit(X, y, sample_weight=weights)
    assert weighted.calibration_bias_[0] - weighted.calibration_bias_[1] > (
        fitted.calibration_bias_[0] - fitted.calibration_bias_[1]
    )


def test_sefr_calibration_subsample(monkeypatch):
    import flaml.automl.contrib.sefr as sefr

    X, y = make_classification(2000, 8, n_classes=4, n_informative=6, random_state=0)
    full = SEFRClassifier(calibration="platt").fit(X, y)
    monkeypatch.setattr(sefr, "_CALIBRATION_MAX_ENTRIES", 1000)
    sub = SEFRClassifier(calibration="platt").fit(X, y)
    np.testing.assert_allclose(full.coef_, sub.coef_)
    assert np.all(np.isfinite(sub.calibration_bias_))
    assert abs(log_loss(y, sub.predict_proba(X)) - log_loss(y, full.predict_proba(X))) < 0.05


def test_sefr_sparse():
    X, y = make_classification(200, 8, random_state=0)
    X = np.where(np.abs(X) > 0.5, np.abs(X), 0.0)
    # every column holds a zero, so dense min-max scaling and sparse max scaling agree
    dense = SEFRClassifier().fit(X, y)
    sparse = SEFRClassifier().fit(scipy.sparse.csr_matrix(X), y)
    np.testing.assert_allclose(dense.coef_, sparse.coef_)
    np.testing.assert_allclose(dense.decision_function(X), sparse.decision_function(scipy.sparse.csr_matrix(X)))

    # sparse input cannot be shifted into [0, 1] without densifying, so negatives are rejected
    signed = scipy.sparse.csr_matrix(np.where(y[:, None] == 1, X, -X))
    with pytest.raises(ValueError, match="Negative values"):
        SEFRClassifier().fit(signed, y)
    with pytest.raises(ValueError, match="Negative values"):
        SEFRClassifier(scaling="none").fit(-X, y)
    # out-of-range values at prediction time are clipped into [0, 1]
    assert np.all(sparse._scale(-signed).data >= 0)
    assert np.all(dense._scale(X * 3 - 1) >= 0) and np.all(dense._scale(X * 3 - 1) <= 1)


def test_sefr_sklearn_estimator_checks():
    # Per-class weights cancel in the class means of Eq. 3-4 and only move the
    # threshold of Eq. 9, so SEFR cannot reach the 87% majority this check expects.
    expected_failed = {"check_class_weight_classifiers": "SEFR class weights only move the Eq. 9 threshold"}
    for estimator in (SEFRClassifier(), SEFRClassifier(calibration="platt")):
        if "on_fail" in inspect.signature(check_estimator).parameters:  # scikit-learn >= 1.6
            results = check_estimator(estimator, expected_failed_checks=expected_failed, on_fail=None)
            failed = [r["check_name"] for r in results if r["status"] == "failed"]
        else:
            failed = []
            for est, check in check_estimator(estimator, generate_only=True):
                name = getattr(check, "func", check).__name__
                if name in expected_failed:
                    continue
                try:
                    check(est)
                except Exception:
                    failed.append(name)
        assert not failed, failed


def test_sefr_estimators():
    X, y = make_classification(200, 8, random_state=0)
    assert set(SEFREstimator.search_space()) == {
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
