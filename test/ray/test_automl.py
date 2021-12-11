import ray
from flaml import AutoML


def _test_ray_classification():
    from sklearn.datasets import make_classification

    ray.init(address="auto", _redis_password="NOPASS")
    X, y = make_classification(1000, 10)
    automl = AutoML()
    automl.fit(X, y, time_budget=10, task="classification", n_concurrent_trials=2)


if __name__ == "__main__":
    _test_ray_classification()
