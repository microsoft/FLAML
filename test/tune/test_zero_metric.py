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
