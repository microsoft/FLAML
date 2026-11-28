from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from flaml import tune
from flaml.tune.trial_runner import SimpleTrial
from flaml.tune.tune import ExperimentAnalysis


# The Decimal cases depend on the context precision and rounding, so the tests pin the default ones in case a
# caller or an earlier test changed them.
def _decimal_context():
    return localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN))


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
        # "max" objective on the exact boundary: 55 is within 45% of 100, 54 is not.
        (
            [{"score": 100.0, "time": 5.0}, {"score": 55.0, "time": 1.0}, {"score": 54.0, "time": 0.0}],
            ["score", "time"],
            ["max", "min"],
            {"score": "45%", "time": 0},
            1,
        ),
        # "max" objective on a boundary that is exact in binary: 0.025 == 0.1 / 4.
        (
            [{"score": 0.1, "time": 5.0}, {"score": 0.025, "time": 1.0}, {"score": 0.024, "time": 0.0}],
            ["score", "time"],
            ["max", "min"],
            {"score": "75%", "time": 0},
            1,
        ),
        # Positive "min" values on a decimal boundary: 0.05151 is within 1% of 0.051, 0.05152 is not.
        (
            [{"loss": 0.051, "time": 5.0}, {"loss": 0.05151, "time": 1.0}, {"loss": 0.05152, "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "1%", "time": 0},
            1,
        ),
        # Large "max" values must not overflow: 6e306 is within 45% of 1e307, 1e306 is not.
        (
            [{"score": 1e307, "time": 5.0}, {"score": 6e306, "time": 1.0}, {"score": 1e306, "time": 0.0}],
            ["score", "time"],
            ["max", "min"],
            {"score": "45%", "time": 0},
            1,
        ),
        # Large "min" values must not overflow: 3e307 is not within 100% of 1e307.
        (
            [{"loss": 1e307, "time": 5.0}, {"loss": 3e307, "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "100%", "time": 0},
            0,
        ),
        # float16 values must not overflow: 20000 is not within 10% of 10000.
        (
            [{"loss": np.float16(10000), "time": 5.0}, {"loss": np.float16(20000), "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            0,
        ),
        # int8 values must not overflow in abs(): -120 is within 10% of -128, -100 is not.
        (
            [
                {"loss": np.int8(-128), "time": 5.0},
                {"loss": np.int8(-120), "time": 1.0},
                {"loss": np.int8(-100), "time": 0.0},
            ],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Python integers beyond float precision: 2**70 + 2**60 is within 10% of 2**70, 2**71 is not.
        (
            [{"loss": 2**70, "time": 5.0}, {"loss": 2**70 + 2**60, "time": 1.0}, {"loss": 2**71, "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Integer best with fractional values: 21/20 is within 10% of 1, 6/5 is not.
        (
            [{"loss": 1, "time": 5.0}, {"loss": Fraction(21, 20), "time": 1.0}, {"loss": Fraction(6, 5), "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Integer best with decimal values: 1.05 is within 10% of 1, 1.2 is not.
        (
            [{"loss": 1, "time": 5.0}, {"loss": Decimal("1.05"), "time": 1.0}, {"loss": Decimal("1.2"), "time": 0.0}],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Fraction best on the exact boundary: 11/30 is within 10% of 1/3, 2/5 is not.
        (
            [
                {"loss": Fraction(1, 3), "time": 5.0},
                {"loss": Fraction(11, 30), "time": 1.0},
                {"loss": Fraction(2, 5), "time": 0.0},
            ],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Decimal best beyond float precision: -0.90000000000000000009 is within 10% of -1.0000000000000000001.
        (
            [
                {"loss": Decimal("-1.0000000000000000001"), "time": 5.0},
                {"loss": Decimal("-0.90000000000000000009"), "time": 1.0},
                {"loss": Decimal("-0.9"), "time": 0.0},
            ],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Fraction best with float values: 1.1 is within 10% of 1 as printed, a float or a fraction above it is not.
        (
            [
                {"loss": Fraction(1), "time": 5.0},
                {"loss": 1.1, "time": 2.0},
                {"loss": Fraction(11, 10) + Fraction(1, 10**30), "time": 1.0},
                {"loss": float(np.nextafter(1.1, 2)), "time": 0.0},
            ],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10%", "time": 0},
            1,
        ),
        # Decimal "max" values beyond the context precision: 4.5000000000000000000000000054 is within 10% of
        # 5.000000000000000000000000006, 4.5000000000000000000000000053 is not.
        (
            [
                {"acc": Decimal("5.000000000000000000000000006"), "time": 5.0},
                {"acc": Decimal("4.5000000000000000000000000054"), "time": 1.0},
                {"acc": Decimal("4.5000000000000000000000000053"), "time": 0.0},
            ],
            ["acc", "time"],
            ["max", "min"],
            {"acc": "10%", "time": 0},
            1,
        ),
        # Percentage with more digits than a float holds: 1.100000000000000001 is within 10.0000000000000001% of 1.
        (
            [
                {"loss": Decimal(1), "time": 5.0},
                {"loss": Decimal("1.100000000000000001"), "time": 1.0},
                {"loss": Decimal("1.100000000000000002"), "time": 0.0},
            ],
            ["loss", "time"],
            ["min", "min"],
            {"loss": "10.0000000000000001%", "time": 0},
            1,
        ),
        # Infinite best "max" value: only the infinite values are kept.
        (
            [{"acc": float("inf"), "time": 5.0}, {"acc": float("inf"), "time": 4.0}, {"acc": 5.0, "time": 1.0}],
            ["acc", "time"],
            ["max", "min"],
            {"acc": "10%", "time": 0},
            1,
        ),
    ],
)
def test_lexico_best_percentage_tolerance(results, metrics, modes, tolerances, best_index):
    with _decimal_context():
        analysis = _analysis(results, metrics, modes, tolerances)
        assert analysis.best_trial is analysis.trials[best_index]
        assert analysis.best_result == results[best_index]


def test_lexico_best_invalid_percentage_with_infinite_best():
    analysis = _analysis(
        [{"acc": float("inf"), "time": 1.0}], ["acc", "time"], ["max", "min"], {"acc": "ten%", "time": 0}
    )
    with pytest.raises(ValueError):
        analysis.best_trial


@pytest.mark.parametrize(
    "results, metrics, modes, tolerances, best_index",
    [
        # Decimal "max" value beyond the context precision with a zero tolerance: the best value is kept.
        (
            [{"acc": Decimal("4.99999999999999999999999999999"), "time": 5.0}, {"acc": Decimal(4), "time": 1.0}],
            ["acc", "time"],
            ["max", "min"],
            {"acc": 0, "time": 0},
            0,
        ),
        # Decimal "max" value with a tolerance below the context precision: the value 1E-40 below is kept,
        # the value 2E-40 below is not.
        (
            [
                {"acc": Decimal("4.99999999999999999999999999999"), "time": 5.0},
                {"acc": Decimal("4.9999999999999999999999999999899999999999"), "time": 1.0},
                {"acc": Decimal("4.9999999999999999999999999999899999999998"), "time": 0.0},
            ],
            ["acc", "time"],
            ["max", "min"],
            {"acc": Decimal("1E-40"), "time": 0},
            1,
        ),
        # Negative Decimal "min" value beyond the context precision with a zero tolerance, as the last objective.
        (
            [{"time": 1.0, "loss": Decimal("-4.99999999999999999999999999999")}, {"time": 1.0, "loss": Decimal(-4)}],
            ["time", "loss"],
            ["min", "min"],
            {"time": 0, "loss": 0},
            0,
        ),
    ],
)
def test_lexico_best_absolute_tolerance_decimal_digits(results, metrics, modes, tolerances, best_index):
    with _decimal_context():
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
