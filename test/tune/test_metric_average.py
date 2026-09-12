import pytest

from flaml import tune


@pytest.mark.parametrize("sparse", [False, True])
def test_average_counts_reported_metric_values(sparse):
    def evaluate(config):
        for value in [1.0, 3.0, 5.0]:
            tune.report(score=value, secondary=value)
            if sparse:
                tune.report(score=value)

    analysis = tune.run(evaluate, config={}, metric="score", mode="min", num_samples=1, use_ray=False, verbose=0)
    stats = analysis.trials[0].metric_analysis
    assert stats["score"]["avg"] == pytest.approx(3.0)
    assert stats["secondary"]["avg"] == pytest.approx(3.0)
    assert analysis.get_best_trial(metric="secondary", mode="min", scope="avg") is analysis.trials[0]
