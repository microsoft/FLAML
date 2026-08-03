from pathlib import Path

import pytest
from sklearn.datasets import load_iris

import flaml
from flaml import AutoML
from flaml.fabric.visualization import get_param_importance


@pytest.mark.conda
def test_package_minimum():
    # Initialize an AutoML instance
    automl = AutoML()
    # Specify automl goal and constraint
    automl_settings = {
        # "estimator_list": ["sgd", "svc"],
        "time_budget": 10,  # in seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": "iris.log",
    }
    X_train, y_train = load_iris(return_X_y=True)
    # Train with labeled input data
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    # Check that `best_config` is created, the log was created and best model is accessible
    assert hasattr(automl, "best_config")
    assert Path("iris.log").exists()
    assert automl.model is not None
    print(automl.model)
    # Predict and check that the prediction shape is as expected
    preds = automl.predict_proba(X_train)
    assert preds.shape == (150, 3)
    print(preds)


def objective(config):
    target = config["x"] ** 2 + config["y"] * config["eps"]
    if config["cat"] == "b":
        target += 10
    return {"target": target}


@pytest.mark.conda
def test_param_importance():
    search_space = {
        "x": flaml.tune.randint(1, 10),
        "y": flaml.tune.randint(-10, 10),
        "eps": flaml.tune.uniform(1e-5, 1),
        "cat": flaml.tune.choice(["a", "b"]),
    }
    analysis = flaml.tune.run(
        objective,
        search_space,
        metric="target",
        mode="max",
        num_samples=10,
    )
    importance = get_param_importance(analysis)
    importance_sum = sum(importance.values())
    assert (
        abs(1.0 - importance_sum) < 1e-4
    ), f"Sum of hyperparameter importance should be close to 1.0, but get {importance_sum}"


if __name__ == "__main__":
    test_package_minimum()
    test_param_importance()
