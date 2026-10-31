import numpy as np
import pytest

from flaml import tune
from flaml.tune.searcher.flow2 import FLOW2
from flaml.tune.space import denormalize, normalize


@pytest.mark.parametrize("sampler, lower", [("randint", 0), ("randint", -3), ("randint", 7), ("lograndint", 7)])
def test_singleton_integer_normalization(sampler, lower):
    domain = getattr(tune, sampler)(lower, lower + 1)
    space = {"width": domain}
    # The exclusive upper bound leaves exactly one legal integer.
    np.testing.assert_array_equal(domain.sample(size=16, random_state=20), np.full(16, lower))
    with np.errstate(all="raise"):
        normalized = normalize({"width": lower}, space, {}, {})
    assert np.isfinite(normalized["width"])
    assert 0 <= normalized["width"] <= 1
    for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
        actual = denormalize({"width": position}, space, {}, {}, np.random.RandomState(20))
        assert actual == {"width": lower}


@pytest.mark.parametrize("sampler, lower", [("randint", 0), ("randint", -3), ("randint", 7), ("lograndint", 7)])
def test_tune_with_singleton_integer(sampler, lower):
    evaluated = []

    def objective(config):
        evaluated.append(config.copy())
        return {"loss": (config["rate"] - 0.3) ** 2}

    with np.errstate(divide="raise", invalid="raise"):
        analysis = tune.run(
            objective,
            config={"width": getattr(tune, sampler)(lower, lower + 1), "rate": tune.uniform(0.0, 1.0)},
            low_cost_partial_config={"width": lower, "rate": 0.5},
            metric="loss",
            mode="min",
            num_samples=6,
            use_ray=False,
            verbose=0,
        )
    assert len(analysis.trials) == len(evaluated) == 6
    assert all(config["width"] == lower and 0 <= config["rate"] <= 1 for config in evaluated)
    assert len({config["rate"] for config in evaluated}) > 1
    assert np.isfinite(analysis.best_result["loss"])


@pytest.mark.parametrize("sampler", ["randint", "lograndint"])
def test_singleton_integer_does_not_bound_varying_dimension_step(sampler):
    searcher = FLOW2(
        init_config={"width": 7, "rate": 0.5},
        metric="loss",
        mode="min",
        space={"width": getattr(tune, sampler)(7, 8), "rate": tune.uniform(0.0, 1.0)},
    )

    assert searcher.step_lower_bound == pytest.approx(FLOW2.STEP_LOWER_BOUND)
    assert searcher.step > searcher.step_lower_bound


def test_non_singleton_integer_still_bounds_step():
    searcher = FLOW2(
        init_config={"width": 7, "rate": 0.5},
        metric="loss",
        mode="min",
        space={"width": tune.randint(7, 9), "rate": tune.uniform(0.0, 1.0)},
    )

    assert searcher.step_lower_bound == pytest.approx(np.sqrt(2) / 2)


@pytest.mark.parametrize("sampler", ["randint", "lograndint", "qrandint", "qlograndint"])
def test_multi_integer_normalization_endpoints(sampler):
    factory = getattr(tune, sampler)
    domain = factory(7, 9, 1) if sampler.startswith("q") else factory(7, 10)
    space = {"width": domain}
    for endpoint, expected in [(7, 0.0), (9, 1.0)]:
        normalized = normalize({"width": endpoint}, space, {}, {})
        assert normalized["width"] == pytest.approx(expected)
        actual = denormalize(normalized, space, {}, {}, np.random.RandomState(20))
        assert actual == {"width": endpoint}
