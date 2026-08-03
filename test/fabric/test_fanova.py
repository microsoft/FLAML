import numpy as np
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution, IntUniformDistribution, UniformDistribution

import flaml
from flaml.fabric.fanova import FanovaImportanceEvaluator
from flaml.fabric.visualization import get_param_importance


def objective(config):
    target = config["x"] ** 2 + config["y"] * config["eps"]
    if config["cat"] == "b":
        target += 10
    return {"target": target}


def test_optuna_backed_evaluator():
    hp_df = pd.DataFrame(
        {
            "x": [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)],
            "y": [np.int64(-5), np.int64(-4), np.int64(-3), np.int64(-2), np.int64(-1), np.int64(0)],
            "eps": [
                np.float64(0.1),
                np.float64(0.2),
                np.float64(0.3),
                np.float64(0.4),
                np.float64(0.5),
                np.float64(0.6),
            ],
            "cat": [np.str_("a"), np.str_("b"), np.str_("a"), np.str_("b"), np.str_("a"), np.str_("b")],
        }
    )
    scores = pd.Series([1.0, 2.5, 4.0, 6.5, 9.0, 12.5])
    search_space = {
        "x": IntUniformDistribution(1, 10),
        "y": IntUniformDistribution(-10, 10),
        "eps": UniformDistribution(1e-5, 1.0),
        "cat": CategoricalDistribution(["a", "b"]),
    }

    importance = FanovaImportanceEvaluator(seed=0).evaluate(hp_df, scores, search_space)

    assert set(importance) == set(search_space)
    assert pytest.approx(1.0, abs=1e-4) == sum(importance.values())


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
    test_param_importance()
