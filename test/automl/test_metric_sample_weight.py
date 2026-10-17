import numpy as np
import pytest
from sklearn.metrics import mean_absolute_percentage_error

from flaml.automl.ml import sklearn_metric_loss_score


def test_mape_uses_sample_weight():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.7])
    sample_weight = np.array([1.0, 2.0, 1.0, 1.0])

    score = sklearn_metric_loss_score("mape", y_pred, y_true, sample_weight=sample_weight)

    expected = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
    assert score == pytest.approx(expected)
    assert score != pytest.approx(mean_absolute_percentage_error(y_true, y_pred))
