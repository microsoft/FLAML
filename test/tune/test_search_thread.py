"""Search-thread state and configuration contracts."""

import pytest

from flaml.tune.searcher.search_thread import _recursive_dict_update


def test_recursive_dict_update_simple():
    """Test simple non-nested dictionary update."""
    target = {"a": 1, "b": 2}
    source = {"c": 3}
    _recursive_dict_update(target, source)
    assert target == {"a": 1, "b": 2, "c": 3}


def test_recursive_dict_update_override():
    """Test that source values override target values for non-dict values."""
    target = {"a": 1, "b": 2}
    source = {"b": 3}
    _recursive_dict_update(target, source)
    assert target == {"a": 1, "b": 3}


def test_recursive_dict_update_nested():
    """Test nested dictionary merge (the main use case for XGBoost params)."""
    target = {
        "num_boost_round": 10,
        "params": {
            "max_depth": 12,
            "eta": 0.020168455186106736,
            "min_child_weight": 1.4504723523894132,
            "scale_pos_weight": 3.794258636185337,
            "gamma": 0.4985070123025904,
        },
    }
    source = {
        "params": {
            "verbosity": 3,
            "booster": "gbtree",
            "eval_metric": "auc",
            "tree_method": "hist",
            "objective": "binary:logistic",
        }
    }
    _recursive_dict_update(target, source)

    # Check that sampled params are preserved
    assert target["params"]["max_depth"] == 12
    assert target["params"]["eta"] == 0.020168455186106736
    assert target["params"]["min_child_weight"] == 1.4504723523894132
    assert target["params"]["scale_pos_weight"] == 3.794258636185337
    assert target["params"]["gamma"] == 0.4985070123025904

    # Check that const params are added
    assert target["params"]["verbosity"] == 3
    assert target["params"]["booster"] == "gbtree"
    assert target["params"]["eval_metric"] == "auc"
    assert target["params"]["tree_method"] == "hist"
    assert target["params"]["objective"] == "binary:logistic"

    # Check top-level param is preserved
    assert target["num_boost_round"] == 10


def test_recursive_dict_update_deeply_nested():
    """Test deeply nested dictionary merge."""
    target = {"a": {"b": {"c": 1, "d": 2}}}
    source = {"a": {"b": {"e": 3}}}
    _recursive_dict_update(target, source)
    assert target == {"a": {"b": {"c": 1, "d": 2, "e": 3}}}


def test_recursive_dict_update_mixed_types():
    """Test that non-dict values in source replace dict values in target."""
    target = {"a": {"b": 1}}
    source = {"a": 2}
    _recursive_dict_update(target, source)
    assert target == {"a": 2}


def test_recursive_dict_update_empty_dicts():
    """Test with empty dictionaries."""
    target = {}
    source = {"a": 1}
    _recursive_dict_update(target, source)
    assert target == {"a": 1}

    target = {"a": 1}
    source = {}
    _recursive_dict_update(target, source)
    assert target == {"a": 1}


def test_recursive_dict_update_none_values():
    """Test that None values are properly handled."""
    target = {"a": 1, "b": None}
    source = {"b": 2, "c": None}
    _recursive_dict_update(target, source)
    assert target == {"a": 1, "b": 2, "c": None}


@pytest.mark.parametrize("through_thread", [False, True])
def test_failed_intermediate_scores_are_not_completed_optuna_observations(through_thread):
    from optuna.trial import TrialState

    from flaml import tune
    from flaml.tune.searcher.search_thread import SearchThread
    from flaml.tune.searcher.suggestion import OptunaSearch

    space = {"x": tune.uniform(0, 1)}
    searcher = OptunaSearch(space, metric="loss", mode="min")
    searcher.space = space
    proposer = SearchThread(search_alg=searcher) if through_thread else searcher
    for trial_id, loss, error in (("failed", -10.0, True), ("successful", 1.0, False)):
        config = proposer.suggest(trial_id)
        result = {"loss": loss, "config": config, "training_iteration": 0, "time_total_s": 1.0}
        proposer.on_trial_result(trial_id, result)
        proposer.on_trial_complete(trial_id, result, error=error)

    assert [trial.state for trial in searcher._ot_study.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert searcher._ot_study.best_value == 1.0
    if through_thread:
        assert proposer.running == 0
        assert proposer.obj_best1 == 1.0
        assert proposer.best_result["loss"] == 1.0


@pytest.mark.parametrize("points", [[0, 1], [1, 0], [0, 2]])
def test_cfo_does_not_restart_from_a_failed_candidate(points):
    from flaml import tune
    from flaml.tune.searcher.blendsearch import CFO

    searcher = CFO(
        space={"x": tune.choice([0, 1, 2])},
        points_to_evaluate=[{"x": value} for value in points],
        metric="loss",
        mode="min",
    )

    def objective(config):
        loss = 1.0 if config["x"] == 1 else -10.0
        tune.report(loss=loss)
        return {"loss": loss} if config["x"] == 1 else {}

    result = tune.run(objective, search_alg=searcher, num_samples=2, mlflow_logging=False, verbose=0)
    assert [trial.status for trial in result.trials] == ["TERMINATED" if value == 1 else "ERROR" for value in points]
    local_threads = [thread for thread in searcher._search_thread_pool.values() if thread._is_ls]
    assert len(local_threads) == (1 if 1 in points else 0)
    assert all(thread.obj_best1 == 1.0 for thread in local_threads)
    assert all(not candidate or candidate["loss"] == 1.0 for candidate in searcher._candidate_start_points.values())


@pytest.mark.parametrize("lexicographic", [False, True])
@pytest.mark.parametrize("successful_sibling", [False, True])
@pytest.mark.parametrize("sibling_finishes_first", [False, True])
def test_failed_provisional_incumbent_rolls_back_without_losing_successes(
    lexicographic, successful_sibling, sibling_finishes_first
):
    from flaml import tune
    from flaml.tune.searcher.flow2 import FLOW2

    space = {"x": tune.uniform(0, 1)}
    objectives = (
        {
            "metrics": ["loss", "cost"],
            "modes": ["min", "min"],
            "tolerances": {"loss": 0.0, "cost": 0.0},
            "targets": {"loss": -float("inf"), "cost": -float("inf")},
        }
        if lexicographic
        else None
    )
    initial = {"x": 0.5}
    searcher = FLOW2(initial, metric="loss", mode="min", space=space, lexico_objectives=objectives)
    searcher = searcher.create(initial, {"loss": 10.0, "cost": 2.0} if lexicographic else 10.0, 1.0, space)
    failed = searcher.suggest("failed")
    successful = searcher.suggest("successful") if successful_sibling else None
    searcher.on_trial_result("failed", {"loss": 0.0, "cost": 2.0, "time_total_s": 1.0})
    assert searcher.best_config == failed

    if successful_sibling:
        searcher.on_trial_result("successful", {"loss": 5.0, "cost": 1.0, "time_total_s": 2.0})
        if sibling_finishes_first:
            searcher.on_trial_complete("successful", {"loss": 5.0, "cost": 1.0, "time_total_s": 2.0})
    searcher.on_trial_complete("failed", {"loss": 0.0, "cost": 2.0, "time_total_s": 3.0}, error=True)
    if successful_sibling and not sibling_finishes_first:
        searcher.on_trial_complete("successful", {"loss": 5.0, "cost": 1.0, "time_total_s": 2.0})

    assert searcher.best_config == (successful if successful_sibling else initial)
    assert (searcher.best_obj["loss"] if lexicographic else searcher.best_obj) == (5.0 if successful_sibling else 10.0)
    if lexicographic:
        assert 0.0 not in searcher._histories["loss"]
    assert searcher.suggest("next") is not None
    assert searcher._proposed_by["next"] == searcher.incumbent
