import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import log_loss, roc_auc_score

from flaml.automl.model import LGBMEstimator


def test_score_uses_probabilities_for_roc_auc():
    X, y = load_breast_cancer(return_X_y=True)
    estimator = LGBMEstimator(task="binary", n_estimators=4)
    estimator.fit(X, y)

    expected = roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    assert estimator.score(X, y, metric="roc_auc") == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["log_loss", "roc_auc_ovr"])
def test_score_multiclass_probability_metrics(metric):
    X, y = load_iris(return_X_y=True)
    estimator = LGBMEstimator(task="multiclass", n_estimators=4)
    estimator.fit(X, y)
    proba = estimator.predict_proba(X)

    if metric == "log_loss":
        expected = log_loss(y, proba)
    else:
        expected = roc_auc_score(y, proba, multi_class="ovr")
    assert estimator.score(X, y, metric=metric) == pytest.approx(expected)
