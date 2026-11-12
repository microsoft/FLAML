import pytest
from sklearn.datasets import make_classification

from flaml import AutoML


@pytest.mark.parametrize("max_iter", [0, 1])
@pytest.mark.parametrize("estimator", ["rf", "extra_tree"])
def test_no_search_with_time_budget(estimator, max_iter):
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    automl = AutoML()
    # max_iter < 2 skips the search and retrains the initial config on a fresh estimator
    # (with the time budget still set) without any earlier trial of this estimator.
    automl.fit(X, y, task="classification", time_budget=5, max_iter=max_iter, estimator_list=[estimator], verbose=0)
    assert automl.best_estimator == estimator
    assert automl.predict(X).shape == (200,)
    assert automl.model.estimator.n_estimators == automl.best_config["n_estimators"]


def test_zero_shot_data_starting_points_with_time_budget():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    automl = AutoML()
    automl.fit(
        X,
        y,
        task="classification",
        time_budget=60,
        max_iter=0,
        starting_points="data",
        estimator_list=["rf", "lgbm"],
        verbose=0,
    )
    assert automl.predict(X).shape == (200,)
