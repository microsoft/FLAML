import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from flaml.automl.model import IsolationForestEstimator


def test_isolation_forest_anomaly_estimator():
    X_normal, _ = make_blobs(
        n_samples=100,
        centers=1,
        cluster_std=0.5,
        random_state=42,
    )

    rng = np.random.RandomState(42)
    X_anomaly = rng.uniform(low=6, high=8, size=(20, 2))

    X_test = np.vstack([X_normal, X_anomaly])
    y_test = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    model = IsolationForestEstimator(
        n_estimators=50,
        contamination=0.15,
        random_state=42,
    )

    model.fit(X_normal)

    preds = model.predict(X_test)
    scores = -model.decision_function(X_test)

    assert preds.shape == y_test.shape
    assert scores.shape == y_test.shape
    assert set(preds).issubset({-1, 1})

    auc = roc_auc_score(y_test, scores)
    assert auc > 0.9
