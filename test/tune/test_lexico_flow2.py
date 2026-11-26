import math
from decimal import Decimal
from fractions import Fraction

import pytest

from flaml import tune
from flaml.tune.searcher.flow2 import FLOW2


def _incumbent_values(mode, values, lexico_objectives=None):
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="m",
        mode=mode,
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives=lexico_objectives,
    )
    incumbents = []
    for i, value in enumerate(values):
        trial_id = str(i)
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"m": value})
        best_obj = searcher.best_obj["m"] if lexico_objectives else searcher.best_obj
        # best_obj is minimized internally, so it is negated for "max".
        incumbents.append(best_obj if mode == "min" else -best_obj)
    return incumbents


@pytest.mark.parametrize(
    "mode, values, expected",
    [
        ("min", [1.0, 5.0, 3.0, 0.5, 2.0], [1.0, 1.0, 1.0, 0.5, 0.5]),
        ("max", [1.0, 0.2, 0.6, 2.0, 1.5], [1.0, 1.0, 1.0, 2.0, 2.0]),
    ],
)
def test_lexico_flow2_compares_first_result_with_incumbent(mode, values, expected):
    lexico_objectives = {"metrics": ["m"], "modes": [mode], "tolerances": {}, "targets": {}}

    assert _incumbent_values(mode, values) == expected
    # With one objective, lexicographic FLOW2 must keep the same incumbent.
    assert _incumbent_values(mode, values, lexico_objectives) == expected


def test_lexico_flow2_breaks_primary_tie_with_secondary_objective():
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="a",
        mode="min",
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives={"metrics": ["a", "b"], "modes": ["min", "min"], "tolerances": {}, "targets": {}},
    )
    incumbents = []
    for i, (a, b) in enumerate([(1, 5), (1, 3), (1, 4), (1, 2), (2, 0), (0.5, 9)]):
        trial_id = str(i)
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"a": a, "b": b})
        incumbents.append((searcher.best_obj["a"], searcher.best_obj["b"]))
    # A tie on "a" is broken by "b"; a worse "a" is never accepted.
    assert incumbents == [(1, 5), (1, 3), (1, 3), (1, 2), (1, 2), (0.5, 9)]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([Fraction(2, 3), Fraction(1, 3), Fraction(1, 2)], [Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)]),
        ([Decimal("0.6"), Decimal("0.3"), Decimal("0.5")], [Decimal("0.6"), Decimal("0.3"), Decimal("0.3")]),
    ],
)
def test_lexico_flow2_accepts_non_float_objectives(values, expected):
    lexico_objectives = {"metrics": ["m"], "modes": ["min"], "tolerances": {}, "targets": {}}

    assert _incumbent_values("min", values, lexico_objectives) == expected


def test_lexico_flow2_ranks_nan_objective_last():
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="a",
        mode="min",
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives={"metrics": ["a", "b"], "modes": ["min", "min"], "tolerances": {}, "targets": {}},
    )
    nan = float("nan")
    incumbents = []
    for i, (a, b) in enumerate([(nan, 5), (1, 3), (nan, 1), (1, nan), (0.5, 9), (0.2, nan)]):
        trial_id = str(i)
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"a": a, "b": b})
        incumbents.append((searcher.best_obj["a"], searcher.best_obj["b"]))
    # A NaN objective ranks as that objective's worst value: a NaN "a" loses, and a NaN "b" does not
    # break a tie on "a", but a better "a" still wins when "b" is NaN.
    assert incumbents[1:-1] == [(1, 3), (1, 3), (1, 3), (0.5, 9)]
    assert incumbents[-1][0] == 0.2 and math.isnan(incumbents[-1][1])


def test_lexico_flow2_mixed_modes_use_negated_max_target():
    lexico_objectives = {
        "metrics": ["a", "b"],
        "modes": ["max", "min"],
        "tolerances": {"a": 1.0, "b": 10.0},
        "targets": {},
    }
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="a",
        mode="max",
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives=lexico_objectives,
    )
    incumbents = []
    for i, (a, b) in enumerate([(9.5, 55), (1.0, 0), (10.0, 58)]):
        trial_id = str(i)
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"a": a, "b": b})
        incumbents.append((-searcher.best_obj["a"], searcher.best_obj["b"]))
    # (1, 0) is outside the tolerance on "a", so its "b" must not tighten the bound on "b".
    # (10, 58) is within both tolerances and has a better "a", as ExperimentAnalysis.lexico_best picks.
    assert incumbents == [(9.5, 55), (9.5, 55), (10.0, 58)]


inf = float("inf")


@pytest.mark.parametrize(
    "modes, tolerance, results, expected",
    [
        (["max", "min"], "10%", [(10, 5), (11, 4), (10, 3), (9, 1)], [(10, 5), (11, 4), (10, 3), (10, 3)]),
        (["min", "min"], "10%", [(-10, 5), (-11, 4), (-10, 3), (-9, 1)], [(-10, 5), (-11, 4), (-10, 3), (-10, 3)]),
        (["max", "min"], "0%", [(10, 5), (10, 4), (9, 1)], [(10, 5), (10, 4), (10, 4)]),
        (["max", "min"], "10%", [(inf, 5), (inf, 4), (5, 1)], [(inf, 5), (inf, 4), (inf, 4)]),
    ],
)
def test_lexico_flow2_percentage_tolerance_on_negative_values(modes, tolerance, results, expected):
    lexico_objectives = {"metrics": ["a", "b"], "modes": modes, "tolerances": {"a": tolerance}, "targets": {}}
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="a",
        mode=modes[0],
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives=lexico_objectives,
    )
    incumbents = []
    for i, (a, b) in enumerate(results):
        trial_id = str(i)
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"a": a, "b": b})
        a_best = searcher.best_obj["a"]
        incumbents.append((-a_best if modes[0] == "max" else a_best, searcher.best_obj["b"]))
    # A maximized "a" is negated, so its best value is negative. 10% must still allow a slightly
    # worse "a", here 10 against 11, and must not leave every result outside the bound.
    assert incumbents == expected


def test_lexico_flow2_set_search_properties_resets_history():
    searcher = FLOW2(
        init_config={"x": 0.5},
        metric="a",
        mode="min",
        space={"x": tune.uniform(0.0, 1.0)},
        lexico_objectives={"metrics": ["a", "b"], "modes": ["min", "min"], "tolerances": {}, "targets": {}},
    )
    incumbents = []

    def complete(trial_id, a, b):
        assert searcher.suggest(trial_id) is not None
        searcher.on_trial_complete(trial_id, {"a": a, "b": b})
        incumbents.append((searcher.best_obj["a"], searcher.best_obj["b"]))

    complete("0", 100, 5)
    complete("1", 90, 6)
    # A new space resets the incumbent, so the history of the old incumbents must not be kept.
    searcher.set_search_properties(config={"x": tune.uniform(0.0, 1.0)})
    complete("2", 1, 10)
    complete("3", 2, 0)
    # (2, 0) has a worse "a" than the incumbent (1, 10) and zero tolerance, so it must be rejected.
    assert incumbents == [(100, 5), (90, 6), (1, 10), (1, 10)]
