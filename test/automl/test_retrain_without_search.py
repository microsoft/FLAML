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


@pytest.mark.parametrize("estimator", ["rf", "extra_tree"])
def test_no_search_best_config_reports_fitted_size(estimator):
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    automl = AutoML()
    automl.fit(
        X,
        y,
        task="classification",
        time_budget=1e-6,
        max_iter=0,
        starting_points={estimator: {"n_estimators": 100}},
        estimator_list=[estimator],
        verbose=0,
    )
    fitted = automl.model.estimator.n_estimators
    assert fitted < 100
    assert fitted == len(automl.model.estimator.estimators_)
    assert automl.best_config["n_estimators"] == fitted
    # the caller's starting point is left untouched
    assert automl.best_config_per_estimator[estimator]["n_estimators"] == fitted


def test_no_search_best_config_unchanged_without_cap():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    automl = AutoML()
    automl.fit(
        X,
        y,
        task="classification",
        time_budget=60,
        max_iter=0,
        starting_points={"rf": {"n_estimators": 20}},
        estimator_list=["rf"],
        verbose=0,
    )
    assert automl.model.estimator.n_estimators == 20
    assert automl.best_config["n_estimators"] == 20


def test_fresh_retrain_does_not_probe_past_budget(monkeypatch):
    import time

    from flaml.automl.model import RandomForestEstimator

    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    est = RandomForestEstimator(task="classification", n_estimators=40)
    requested = []

    def fake_fit(self, X_train, y_train, **kwargs):
        requested.append(self.params["n_estimators"])
        return 0.9 if self.params["n_estimators"] == 1 else 0.01

    monkeypatch.setattr(RandomForestEstimator, "_fit", fake_fit)
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    def advancing_fit(self, X_train, y_train, **kwargs):
        d = fake_fit(self, X_train, y_train, **kwargs)
        clock["now"] += d
        return d

    monkeypatch.setattr(RandomForestEstimator, "_fit", advancing_fit)
    est.fit(X, y, budget=1.0, is_retrain=True)
    # after a one-tree probe used 0.9s of 1s, no four-tree probe is requested
    assert 4 not in requested
    assert est.params["n_estimators"] < 40


def test_fresh_retrain_memory_cap_is_applied_to_final_fit(monkeypatch):
    from flaml.automl import model as model_module
    from flaml.automl.model import RandomForestEstimator

    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    est = RandomForestEstimator(task="classification", n_estimators=40)
    avail = {"v": 250}

    class VM:
        def __init__(self, a):
            self.available = a
            self.total = 10000

    monkeypatch.setattr(model_module.psutil, "virtual_memory", lambda: VM(avail["v"]))
    fits = []

    def fake_fit(self, X_train, y_train, **kwargs):
        n = self.params["n_estimators"]
        fits.append(n)
        avail["v"] -= 100 * n  # every tree retains 100 units
        return 0.0001

    monkeypatch.setattr(RandomForestEstimator, "_fit", fake_fit)
    est.fit(X, y, budget=60, is_retrain=True)
    # 250 units available at 100 per tree: the model kept may not exceed the 2-tree cap
    assert est.params["n_estimators"] <= 2
    assert fits[-1] == est.params["n_estimators"]
