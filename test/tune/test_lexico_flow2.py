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
