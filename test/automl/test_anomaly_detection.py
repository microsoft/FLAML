import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from flaml import AutoML


def test_anomaly_detection_rejects_unsupported_builtin_estimator():
    automl = AutoML()

    with pytest.raises(ValueError, match="do not support anomaly_detection"):
        automl.fit(
            X_train=np.random.randn(20, 2),
            y_train=np.zeros(20),
            task="anomaly_detection",
            estimator_list=["lgbm"],
            time_budget=1,
        )
def test_isolation_forest_rejects_non_anomaly_task():
    automl = AutoML()

    with pytest.raises(
        ValueError,
        match="only supports the anomaly_detection task",
    ):
        automl.fit(
            X_train=np.random.randn(20, 2),
            y_train=np.array([0, 1] * 10),
            task="classification",
            estimator_list=["isolation_forest"],
            time_budget=1,
        )

def test_automl_anomaly_detection_e2e():
    X_normal, _ = make_blobs(
        n_samples=100,
        centers=1,
        cluster_std=0.5,
        random_state=42,
    )

    rng = np.random.RandomState(42)
    X_anomaly = rng.uniform(low=6, high=8, size=(20, 2))

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    X_train = X_normal
    y_train = np.zeros(len(X_normal))

    X_val = X
    y_val = y

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="anomaly_detection",
        estimator_list=["isolation_forest"],
        metric="ap",
        time_budget=3,
        max_iter=2,
    )

    preds = automl.predict(X_val)
    scores = automl.score_samples(X_val)
    decision_scores = automl.decision_function(X_val)
    anomaly_scores = -scores
    auc = roc_auc_score(y_val, anomaly_scores)

    assert preds.shape == y_val.shape
    assert scores.shape == y_val.shape
    assert decision_scores.shape == y_val.shape
    assert set(preds).issubset({-1, 1})
    assert auc > 0.9
    assert automl.best_estimator == "isolation_forest"


def test_anomaly_detection_x_only_fit():
    X = np.random.RandomState(42).randn(100, 2)

    automl = AutoML()
    automl.fit(
        X_train=X,
        task="anomaly_detection",
        estimator_list=["isolation_forest"],
    )

    preds = automl.predict(X)
    scores = automl.score_samples(X)
    decision_scores = automl.decision_function(X)

    assert automl.best_estimator == "isolation_forest"
    assert preds.shape == (len(X),)
    assert scores.shape == (len(X),)
    assert decision_scores.shape == (len(X),)
    assert set(preds).issubset({-1, 1})


@pytest.mark.parametrize(
    "fit_kwargs",
    [
        {"max_iter": 2},
        {"time_budget": 1},
    ],
)
def test_anomaly_detection_x_only_rejects_hpo(fit_kwargs):
    X = np.random.RandomState(42).randn(100, 2)

    with pytest.raises(ValueError, match="requires labeled validation data"):
        AutoML().fit(
            X_train=X,
            task="anomaly_detection",
            estimator_list=["isolation_forest"],
            **fit_kwargs,
        )


def test_anomaly_detection_rejects_unsupported_metric():
    X = np.random.RandomState(42).randn(100, 2)

    with pytest.raises(ValueError, match="limited to 'ap' and 'roc_auc'"):
        AutoML().fit(
            X_train=X,
            task="anomaly_detection",
            estimator_list=["isolation_forest"],
            metric="accuracy",
            max_iter=1,
        )


def test_anomaly_detection_public_score_uses_continuous_scores():
    X_normal, _ = make_blobs(
        n_samples=100,
        centers=1,
        cluster_std=0.5,
        random_state=42,
    )
    rng = np.random.RandomState(42)
    X_anomaly = rng.uniform(low=6, high=8, size=(20, 2))

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    automl = AutoML()
    automl.fit(
        X_train=X_normal,
        y_train=np.zeros(len(X_normal)),
        X_val=X,
        y_val=y,
        task="anomaly_detection",
        estimator_list=["isolation_forest"],
        metric="roc_auc",
        max_iter=1,
    )

    anomaly_scores = -automl.score_samples(X)
    expected_auc = roc_auc_score(y, anomaly_scores)

    score_01 = automl.score(X, y)

    y_iforest = np.where(y == 1, -1, 1)
    score_iforest = automl.score(X, y_iforest)

    assert score_01 == pytest.approx(expected_auc)
    assert score_iforest == pytest.approx(expected_auc)
    assert score_01 > 0.9


def test_normalize_anomaly_labels_single_class():
    from flaml.automl.ml import normalize_anomaly_labels

    np.testing.assert_array_equal(
        normalize_anomaly_labels(np.array([1, 1, 1])),
        np.array([1, 1, 1]),
    )
    np.testing.assert_array_equal(
        normalize_anomaly_labels(np.array([-1, 1, 1])),
        np.array([1, 0, 0]),
    )
