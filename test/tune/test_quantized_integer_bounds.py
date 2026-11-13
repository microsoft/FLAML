import numpy as np
import pytest

from flaml import CFO, tune
from flaml.tune.space import denormalize


@pytest.mark.parametrize(
    "domain, expected",
    [
        (tune.qrandint(1, 11, 3), [3, 9]),
        (tune.qrandint(2, 10, 4), [4, 8]),
        (tune.qlograndint(5, 30, 4), [8, 28]),
    ],
)
def test_denormalize_quantized_integer_stays_in_bounds(domain, expected):
    space = {"x": domain}
    low, high = (denormalize({"x": p}, space, {}, {}, np.random.RandomState(0))["x"] for p in (0.0, 1.0))
    assert [low, high] == expected
    for p in np.linspace(0, 1, 101):
        value = denormalize({"x": p}, space, {}, {}, np.random.RandomState(0))["x"]
        assert domain.lower <= value <= domain.upper
        assert value % domain.get_sampler().q == 0


def test_tune_run_quantized_integer_stays_in_bounds():
    evaluated = []

    def objective(config):
        evaluated.append(config["x"])
        return {"loss": (config["x"] - 6) ** 2}

    space = {"x": tune.qrandint(1, 11, 3)}
    tune.run(
        objective,
        config=space,
        search_alg=CFO(space=space, metric="loss", mode="min"),
        metric="loss",
        mode="min",
        num_samples=40,
        use_ray=False,
        verbose=0,
    )
    assert evaluated
    assert set(evaluated) <= {3, 6, 9}


def test_blendsearch_global_search_quantized_log_integer_stays_in_bounds():
    pytest.importorskip("optuna")
    from flaml import BlendSearch

    evaluated = []

    def objective(config):
        evaluated.append(config["x"])
        return {"loss": abs(config["x"] - 17)}

    space = {"x": tune.qlograndint(5, 30, 4)}
    tune.run(
        objective,
        config=space,
        search_alg=BlendSearch(space=space, metric="loss", mode="min"),
        metric="loss",
        mode="min",
        num_samples=30,
        use_ray=False,
        verbose=0,
    )
    assert evaluated
    assert all(5 <= x <= 30 and x % 4 == 0 for x in evaluated), sorted(set(evaluated))


def test_tune_run_default_qlograndint_stays_in_bounds():
    pytest.importorskip("optuna")
    evaluated = []

    def objective(config):
        evaluated.append(config["x"])
        return {"loss": abs(config["x"] - 17)}

    tune.run(
        objective,
        config={"x": tune.qlograndint(5, 30, 4)},
        metric="loss",
        mode="min",
        num_samples=30,
        use_ray=False,
        verbose=0,
    )
    assert all(5 <= x <= 30 and x % 4 == 0 for x in evaluated), sorted(set(evaluated))
