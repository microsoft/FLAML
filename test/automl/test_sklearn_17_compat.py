"""Test sklearn 1.7+ compatibility for estimator type detection.

This test ensures that FLAML estimators are properly recognized as
regressors or classifiers by sklearn's is_regressor() and is_classifier()
functions, which is required for sklearn 1.7+ ensemble methods.
"""

import pytest
from sklearn.base import is_classifier, is_regressor

from flaml.automl.model import (
    ExtraTreesEstimator,
    LGBMEstimator,
    RandomForestEstimator,
    XGBoostSklearnEstimator,
)


def test_extra_trees_regressor_type():
    """Test that ExtraTreesEstimator with regression task is recognized as regressor."""
    est = ExtraTreesEstimator(task="regression")
    assert is_regressor(est), "ExtraTreesEstimator(task='regression') should be recognized as a regressor"
    assert not is_classifier(est), "ExtraTreesEstimator(task='regression') should not be recognized as a classifier"


def test_extra_trees_classifier_type():
    """Test that ExtraTreesEstimator with classification task is recognized as classifier."""
    est = ExtraTreesEstimator(task="binary")
    assert is_classifier(est), "ExtraTreesEstimator(task='binary') should be recognized as a classifier"
    assert not is_regressor(est), "ExtraTreesEstimator(task='binary') should not be recognized as a regressor"

    est = ExtraTreesEstimator(task="multiclass")
    assert is_classifier(est), "ExtraTreesEstimator(task='multiclass') should be recognized as a classifier"
    assert not is_regressor(est), "ExtraTreesEstimator(task='multiclass') should not be recognized as a regressor"


def test_random_forest_regressor_type():
    """Test that RandomForestEstimator with regression task is recognized as regressor."""
    est = RandomForestEstimator(task="regression")
    assert is_regressor(est), "RandomForestEstimator(task='regression') should be recognized as a regressor"
    assert not is_classifier(est), "RandomForestEstimator(task='regression') should not be recognized as a classifier"


def test_random_forest_classifier_type():
    """Test that RandomForestEstimator with classification task is recognized as classifier."""
    est = RandomForestEstimator(task="binary")
    assert is_classifier(est), "RandomForestEstimator(task='binary') should be recognized as a classifier"
    assert not is_regressor(est), "RandomForestEstimator(task='binary') should not be recognized as a regressor"


def test_lgbm_regressor_type():
    """Test that LGBMEstimator with regression task is recognized as regressor."""
    est = LGBMEstimator(task="regression")
    assert is_regressor(est), "LGBMEstimator(task='regression') should be recognized as a regressor"
    assert not is_classifier(est), "LGBMEstimator(task='regression') should not be recognized as a classifier"


def test_lgbm_classifier_type():
    """Test that LGBMEstimator with classification task is recognized as classifier."""
    est = LGBMEstimator(task="binary")
    assert is_classifier(est), "LGBMEstimator(task='binary') should be recognized as a classifier"
    assert not is_regressor(est), "LGBMEstimator(task='binary') should not be recognized as a regressor"


def test_xgboost_regressor_type():
    """Test that XGBoostSklearnEstimator with regression task is recognized as regressor."""
    est = XGBoostSklearnEstimator(task="regression")
    assert is_regressor(est), "XGBoostSklearnEstimator(task='regression') should be recognized as a regressor"
    assert not is_classifier(est), "XGBoostSklearnEstimator(task='regression') should not be recognized as a classifier"


def test_xgboost_classifier_type():
    """Test that XGBoostSklearnEstimator with classification task is recognized as classifier."""
    est = XGBoostSklearnEstimator(task="binary")
    assert is_classifier(est), "XGBoostSklearnEstimator(task='binary') should be recognized as a classifier"
    assert not is_regressor(est), "XGBoostSklearnEstimator(task='binary') should not be recognized as a regressor"


if __name__ == "__main__":
    # Run all tests
    test_extra_trees_regressor_type()
    test_extra_trees_classifier_type()
    test_random_forest_regressor_type()
    test_random_forest_classifier_type()
    test_lgbm_regressor_type()
    test_lgbm_classifier_type()
    test_xgboost_regressor_type()
    test_xgboost_classifier_type()
    print("All sklearn 1.7+ compatibility tests passed!")
