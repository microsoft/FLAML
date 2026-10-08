import pytest
from sklearn.datasets import load_iris

from flaml import AutoML


@pytest.mark.parametrize("metric", ["roc_auc", "roc_auc_weighted", "ap", "f1"])
def test_binary_only_metric_on_multiclass_raises(metric):
    X, y = load_iris(return_X_y=True)
    automl = AutoML()
    with pytest.raises(ValueError, match="only supports binary classification"):
        automl.fit(X, y, task="classification", metric=metric, time_budget=1, verbose=0)
