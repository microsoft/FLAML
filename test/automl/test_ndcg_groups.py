import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from flaml.automl.ml import sklearn_metric_loss_score

# Two queries, each ranked perfectly, but the scores of the second query are all lower.
Y_TRUE = np.array([3, 2, 1, 0, 3, 2, 1, 0])
Y_PRED = np.array([10, 9, 8, 7, 4, 3, 2, 1])
GROUPS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize("metric", ["ndcg", "ndcg@3", "ndcg@8"])
def test_ndcg_averages_over_query_groups(metric):
    assert sklearn_metric_loss_score(metric, Y_PRED, Y_TRUE, groups=GROUPS) == pytest.approx(0.0)


@pytest.mark.parametrize("metric, k", [("ndcg", None), ("ndcg@3", 3)])
def test_ndcg_without_groups_uses_one_query(metric, k):
    expected = 1 - ndcg_score([Y_TRUE], [Y_PRED], k=k)
    assert sklearn_metric_loss_score(metric, Y_PRED, Y_TRUE) == pytest.approx(expected)
