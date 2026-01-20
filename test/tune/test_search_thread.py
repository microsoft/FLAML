"""Tests for SearchThread nested dictionary update fix."""

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
