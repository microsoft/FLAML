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
        estimator_list=["rf", "extra_tree"],
        verbose=0,
    )
    assert automl.predict(X).shape == (200,)


@pytest.mark.parametrize("estimator", ["rf", "extra_tree"])
def test_fresh_retrain_respects_finite_budget(estimator):
    from flaml.automl.model import ExtraTreesEstimator, RandomForestEstimator

    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    cls = RandomForestEstimator if estimator == "rf" else ExtraTreesEstimator
    capped = cls(task="classification", n_estimators=1000)
    capped.fit(X, y, budget=1e-6, is_retrain=True)
    assert capped.params["n_estimators"] < 1000
    # no budget: retraining keeps the full configured size
    full = cls(task="classification", n_estimators=50)
    full.fit(X, y, is_retrain=True)
    assert full.params["n_estimators"] == 50
