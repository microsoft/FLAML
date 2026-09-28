import numpy as np
import pandas as pd
import pytest
from optuna import distributions
from optuna.distributions import CategoricalDistribution, IntUniformDistribution, UniformDistribution

import flaml
from flaml.fabric.fanova import FanovaImportanceEvaluator
from flaml.fabric.fanova import evaluator as fanova_evaluator
from flaml.fabric.fanova.evaluator import _expand_distribution
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


def test_optuna_backed_evaluator_expands_space_to_observed_values():
    hp_df = pd.DataFrame(
        {
            "n_estimators": [2, 4, 8, 16, 32, 64],
            "max_leaves": [4, 6, 8, 10, 12, 14],
        }
    )
    scores = pd.Series([0.1, 0.2, 0.35, 0.5, 0.65, 0.8])
    search_space = {
        "n_estimators": IntUniformDistribution(4, 120),
        "max_leaves": IntUniformDistribution(4, 120),
    }

    importance = FanovaImportanceEvaluator(seed=0).evaluate(hp_df, scores, search_space)

    assert set(importance) == set(search_space)
    assert pytest.approx(1.0, abs=1e-4) == sum(importance.values())


@pytest.mark.parametrize(
    "distribution_name, kwargs, values, expected_step, expected_log",
    [
        ("IntUniformDistribution", {"low": 5, "high": 15, "step": 5}, [0, 20], 5, False),
        ("IntUniformDistribution", {"low": 5, "high": 15, "step": 5}, [2, 17], 1, False),
        ("IntDistribution", {"low": 5, "high": 15, "step": 5}, [0, 20], 5, False),
        ("IntDistribution", {"low": 5, "high": 15, "step": 5}, [2, 17], 1, False),
        ("IntDistribution", {"low": 5, "high": 15, "step": 5}, [7], 1, False),
        ("IntDistribution", {"low": 4, "high": 120, "log": True}, [2, 240], 1, True),
        ("IntDistribution", {"low": 4, "high": 120, "log": True}, [0, 2], 1, False),
        ("FloatDistribution", {"low": 0.1, "high": 0.9, "step": 0.2}, [-0.1, 1.1], 0.2, False),
        ("FloatDistribution", {"low": 0.0, "high": 0.2, "step": 0.1}, [0.3], 0.1, False),
        ("FloatDistribution", {"low": 0.0, "high": 0.2, "step": 0.1}, [0.1 + 0.2], 0.1, False),
        ("FloatDistribution", {"low": 0.1, "high": 0.9, "step": 0.2}, [0.0, 1.0], None, False),
        ("FloatDistribution", {"low": 0.1, "high": 0.9, "step": 0.2}, [0.2], None, False),
        ("FloatDistribution", {"low": 0.1, "high": 1.0, "log": True}, [0.01, 10.0], None, True),
        ("FloatDistribution", {"low": 0.1, "high": 1.0, "log": True}, [0.0], None, False),
        ("FloatDistribution", {"low": 0.1, "high": 1.0, "log": True}, [-0.1], None, False),
    ],
)
def test_expand_distribution_preserves_supported_semantics(
    distribution_name, kwargs, values, expected_step, expected_log
):
    distribution_class = getattr(distributions, distribution_name, None)
    if distribution_class is None:
        pytest.skip(f"{distribution_name} is unavailable in this Optuna version.")
    distribution = distribution_class(**kwargs)
    original_attributes = distribution.__dict__.copy()

    expanded = _expand_distribution(distribution, values)

    assert expanded.step == expected_step
    assert getattr(expanded, "log", False) == expected_log
    assert expanded.low <= min(distribution.low, *values)
    assert expanded.high >= max(distribution.high, *values)
    original_values = [distribution.low, distribution.high]
    if kwargs.get("step") is not None:
        original_values.extend(
            distribution.low + index * distribution.step
            for index in range(round((distribution.high - distribution.low) / distribution.step) + 1)
        )
    for value in values + original_values:
        assert expanded._contains(expanded.to_internal_repr(value))
    assert distribution.__dict__ == original_attributes


@pytest.mark.parametrize(
    "distribution_name, kwargs, values",
    [
        ("IntUniformDistribution", {"low": 5, "high": 15, "step": 5}, [5, 10, 15]),
        ("IntDistribution", {"low": 5, "high": 15, "step": 5}, [5, 10, 15]),
        ("IntDistribution", {"low": 4, "high": 120, "log": True}, [4, 8, 120]),
        ("FloatDistribution", {"low": 0.1, "high": 0.9, "step": 0.2}, [0.1, 0.5, 0.9]),
        ("FloatDistribution", {"low": 0.1, "high": 1.0, "log": True}, [0.1, 0.5, 1.0]),
    ],
)
def test_expand_distribution_keeps_valid_distribution(distribution_name, kwargs, values):
    distribution_class = getattr(distributions, distribution_name, None)
    if distribution_class is None:
        pytest.skip(f"{distribution_name} is unavailable in this Optuna version.")
    distribution = distribution_class(**kwargs)

    assert _expand_distribution(distribution, values) is distribution


def test_expand_distribution_preserves_legacy_integer_step(monkeypatch):
    monkeypatch.setattr(fanova_evaluator, "IntDistribution", None)
    distribution = IntUniformDistribution(5, 15, step=5)

    expanded = _expand_distribution(distribution, [0, 20])

    assert isinstance(expanded, IntUniformDistribution)
    assert expanded.step == 5
    assert (expanded.low, expanded.high) == (0, 20)


def test_expand_categorical_distribution_preserves_original_choices():
    original = CategoricalDistribution(["a", "b"])
    expanded = _expand_distribution(original, ["b", "new", "new"])
    assert expanded.choices == ("a", "b", "new")
    assert original.choices == ("a", "b")


def test_replay_ignores_invalid_scores_when_expanding_bounds(monkeypatch):
    original = IntUniformDistribution(4, 10)
    recorded = []
    expand = fanova_evaluator._expand_distribution

    def record_expansion(distribution, values):
        result = expand(distribution, values)
        recorded.append(result)
        return result

    monkeypatch.setattr(fanova_evaluator, "_expand_distribution", record_expansion)
    frame = pd.DataFrame({"trees": [4, 6, 8, 10, 10000]})
    scores = pd.Series([0.1, 0.3, 0.5, 0.7, np.nan])
    importance = FanovaImportanceEvaluator(seed=0).evaluate(frame, scores, {"trees": original})
    assert recorded == [original]
    assert importance == {"trees": 1.0}


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
