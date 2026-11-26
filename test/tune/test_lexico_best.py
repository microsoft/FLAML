import pytest

from flaml import tune
from flaml.tune.trial_runner import SimpleTrial
from flaml.tune.tune import ExperimentAnalysis


def _analysis(results, metrics, modes, tolerances):
    trials = []
    for i, result in enumerate(results):
        trial = SimpleTrial(config={"i": i})
        trial.update_last_result(result)
        trials.append(trial)
    lexico_objectives = {
        "metrics": metrics,
        "modes": modes,
        "tolerances": tolerances,
        "targets": {m: -float("inf") if mode == "min" else float("inf") for m, mode in zip(metrics, modes)},
    }
    return ExperimentAnalysis(trials, metric=metrics[0], mode=modes[0], lexico_objectives=lexico_objectives)


@pytest.mark.parametrize(
    "results, metrics, modes, tolerances, best_index",
    [
        # "max" objective: 10.5 is within 10% of 11, so the lower "time" decides.
        (
            [{"acc": 10.0, "time": 5.0}, {"acc": 11.0, "time": 4.0}, {"acc": 10.5, "time": 1.0}],
            ["acc", "time"],
            ["max", "min"],
            {"acc": "10%", "time": 0},
            2,
        ),
        # "min" objective with negative values: -10.5 is within 10% of -11.
        (
            [{"loss": -10.0, "time": 5.0}, {"loss": -11.0, "time": 4.0}, {"loss": -10.5, "time": 1.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            2,
        ),
        # "max" objective last: 5 is not within 10% of 11.
        (
            [{"time": 1.0, "acc": 11.0}, {"time": 1.0, "acc": 5.0}],
            ["time", "acc"],
            ["min", "max"],
            {"time": 0, "acc": "10%"},
            0,
        ),
        # Positive "min" values keep the same bound: 9.5 * 1.1 = 10.45 excludes 11.
        (
            [{"loss": 10.0, "time": 5.0}, {"loss": 11.0, "time": 1.0}, {"loss": 9.5, "time": 4.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            2,
        ),
        # 0% keeps only the best "max" value.
        (
            [{"acc": 10.0, "time": 1.0}, {"acc": 11.0, "time": 4.0}],
            ["acc", "time"],
            ["max", "min"],
            {"acc": "0%", "time": 0},
            1,
        ),
    ],
)
def test_lexico_best_percentage_tolerance(results, metrics, modes, tolerances, best_index):
    analysis = _analysis(results, metrics, modes, tolerances)
    assert analysis.best_trial is analysis.trials[best_index]
    assert analysis.best_result == results[best_index]


def test_tune_run_max_objective_percentage_tolerance():
    def evaluate(config):
        x = config["x"]
        return {"acc": 10 + x, "time": (x - 0.5) ** 2}

    analysis = tune.run(
        evaluate,
        config={"x": tune.uniform(0, 1)},
        lexico_objectives={
            "metrics": ["acc", "time"],
            "modes": ["max", "min"],
            "tolerances": {"acc": "10%", "time": 0},
            "targets": {},
        },
        num_samples=10,
        use_ray=False,
        verbose=0,
    )
    # Every "acc" is in [10, 11], within 10% of the best one, so the lowest "time" wins.
    assert analysis.best_result["time"] == min(trial.last_result["time"] for trial in analysis.trials)
