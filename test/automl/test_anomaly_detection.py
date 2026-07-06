import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from flaml import AutoML
from flaml.automl.model import IsolationForestEstimator


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

    def anomaly_metric(
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        *args,
        **kwargs,
    ):
        scores = -estimator.score_samples(X_val)
        auc = roc_auc_score(y_val, scores)

        return (
            1.0 - auc,
            {"roc_auc": auc},
        )

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="anomaly_detection",
        estimator_list=["isolation_forest"],
        metric=anomaly_metric,
        time_budget=3,
        max_iter=2,
    )

    preds = automl.predict(X_val)
    scores = automl.model.score_samples(X_val)

    assert preds.shape == y_val.shape
    assert scores.shape == y_val.shape
    assert set(preds).issubset({-1, 1})
    assert automl.best_estimator == "isolation_forest"
