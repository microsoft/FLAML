"""Test sklearn 1.7+ compatibility for estimator type detection.

This test ensures that FLAML estimators are properly recognized as
regressors or classifiers by sklearn's is_regressor() and is_classifier()
functions, which is required for sklearn 1.7+ ensemble methods.
"""

import pytest
from sklearn.base import is_classifier, is_regressor

from flaml.automl._lightgbm_compat import patch_lightgbm_sklearn_validation
from flaml.automl.model import (
    ExtraTreesEstimator,
    LGBMEstimator,
    RandomForestEstimator,
    XGBoostSklearnEstimator,
)


@pytest.mark.parametrize("alias", ["_LGBMCheckXY", "_LGBMCheckArray"])
def test_lightgbm_validation_keyword_compatibility(monkeypatch, alias):
    import lightgbm.sklearn as lightgbm_sklearn

    calls = []

    def validator(*args, ensure_all_finite=True, **kwargs):
        calls.append((args, ensure_all_finite, kwargs))
        return args

    monkeypatch.setattr(lightgbm_sklearn, alias, validator, raising=False)
    patch_lightgbm_sklearn_validation()
    wrapped = getattr(lightgbm_sklearn, alias)
    assert wrapped("X", force_all_finite=False, accept_sparse=True) == ("X",)
    assert calls == [(("X",), False, {"accept_sparse": True})]
    wrapped("X", force_all_finite=False, ensure_all_finite="allow-nan")
    assert calls[-1][1] == "allow-nan"
    patch_lightgbm_sklearn_validation()
    assert getattr(lightgbm_sklearn, alias) is wrapped


def test_lightgbm_compatible_and_uninspectable_aliases_are_unchanged(monkeypatch):
    import lightgbm.sklearn as lightgbm_sklearn

    class UninspectableValidator:
        __signature__ = 123

        def __call__(self, *args, **kwargs):
            return args

    def compatible_validator(X, *, force_all_finite=True):
        return X

    uninspectable = UninspectableValidator()
    monkeypatch.setattr(lightgbm_sklearn, "_LGBMCheckXY", uninspectable, raising=False)
    monkeypatch.setattr(lightgbm_sklearn, "_LGBMCheckArray", compatible_validator, raising=False)
    patch_lightgbm_sklearn_validation()
    assert lightgbm_sklearn._LGBMCheckXY is uninspectable
    assert lightgbm_sklearn._LGBMCheckArray is compatible_validator


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
