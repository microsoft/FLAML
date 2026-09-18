import pytest

from flaml import tune
from flaml.tune.result import DEFAULT_METRIC


@pytest.mark.parametrize("value", [0, 0.0, False, 1.0, -1.0])
@pytest.mark.parametrize("report_directly", [False, True])
def test_anonymous_zero_metric_is_recorded(value, report_directly):
    def evaluate(config):
        if report_directly:
            tune.report(value)
        else:
            return value

    analysis = tune.run(evaluate, config={}, mode="min", num_samples=1, use_ray=False, verbose=0)
    assert analysis.trials[0].last_result[DEFAULT_METRIC] == value
    assert analysis.best_trial is analysis.trials[0]


@pytest.mark.parametrize("callback_name", ["on_trial_complete", "on_trial_result"])
@pytest.mark.parametrize("mode, first_value", [("min", 1.0), ("max", -1.0)])
def test_flow2_callback_updates_incumbent_to_zero(callback_name, mode, first_value):
    from flaml.tune.searcher.flow2 import FLOW2

    searcher = FLOW2(
        init_config={"x": 0.5},
        metric=DEFAULT_METRIC,
        mode=mode,
        space={"x": tune.uniform(0.0, 1.0)},
    )
    notify = getattr(searcher, callback_name)

    first_config = searcher.suggest("nonzero")
    assert first_config is not None
    notify("nonzero", {DEFAULT_METRIC: first_value})
    # Both parameter sets normalize to an internal objective of +1.
    assert searcher.best_obj == 1.0
    assert searcher.best_config == first_config

    zero_config = searcher.suggest("zero")
    assert zero_config is not None
    assert zero_config != first_config
    notify("zero", {DEFAULT_METRIC: 0.0})
    assert searcher.best_obj == 0.0
    assert searcher.best_config == zero_config
    assert searcher.incumbent == searcher.normalize(zero_config)


def test_flow2_incumbent_tracks_zero_metric():
    from flaml.tune.searcher.flow2 import FLOW2

    calls = {"n": 0}

    def evaluate(config):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 0.0

    searcher = FLOW2(
        init_config={"x": 0.5},
        metric=DEFAULT_METRIC,
        mode="min",
        space={"x": tune.uniform(0.0, 1.0)},
    )
    analysis = tune.run(evaluate, config={}, mode="min", search_alg=searcher, num_samples=2, use_ray=False, verbose=0)
    assert analysis.best_trial.last_result[DEFAULT_METRIC] == 0.0
    assert searcher.best_obj == 0.0
    assert searcher.best_config == analysis.best_config
