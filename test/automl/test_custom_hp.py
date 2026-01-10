import sys

import pytest

from flaml import AutoML, tune

try:
    import transformers

    _transformers_installed = True
except ImportError:
    _transformers_installed = False


@pytest.mark.skipif(
    sys.platform == "darwin" or not _transformers_installed, reason="do not run on mac os or transformers not installed"
)
def test_custom_hp_nlp():
    from test.nlp.utils import get_automl_settings, get_toy_data_seqclassification

    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    automl = AutoML()

    automl_settings = get_automl_settings()
    automl_settings["custom_hp"] = None
    automl_settings["custom_hp"] = {
        "transformer": {
            "model_path": {
                "domain": tune.choice(["google/electra-small-discriminator"]),
            },
            "num_train_epochs": {"domain": 3},
        }
    }
    automl_settings["fit_kwargs_by_estimator"] = {
        "transformer": {
            "output_dir": "test/data/output/",
            "fp16": False,
        }
    }
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)


def test_custom_hp():
    from sklearn.datasets import load_iris

    X_train, y_train = load_iris(return_X_y=True)
    automl = AutoML()
    custom_hp = {
        "xgboost": {
            "n_estimators": {
                "domain": tune.lograndint(lower=1, upper=100),
                "low_cost_init_value": 1,
            },
        },
        "rf": {
            "max_leaves": {
                "domain": None,  # disable search
            },
        },
        "lgbm": {
            "subsample": {
                "domain": tune.uniform(lower=0.1, upper=1.0),
                "init_value": 1.0,
            },
            "subsample_freq": {
                "domain": 1,  # subsample_freq must > 0 to enable subsample
            },
        },
    }
    automl.fit(X_train, y_train, custom_hp=custom_hp, time_budget=2)
    print(automl.best_config_per_estimator)


def test_lgbm_objective():
    """Test that objective parameter can be set via custom_hp for LGBMEstimator"""
    import numpy as np

    # Create a simple regression dataset
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100) * 100  # Scale to avoid division issues with MAPE

    automl = AutoML()
    settings = {
        "time_budget": 3,
        "metric": "mape",
        "task": "regression",
        "estimator_list": ["lgbm"],
        "verbose": 0,
        "custom_hp": {
            "lgbm": {
                "objective": {
                    "domain": "mape"  # Fixed value, not tuned
                }
            }
        },
    }

    automl.fit(X_train, y_train, **settings)

    # Verify that objective was set correctly
    assert "objective" in automl.best_config, "objective should be in best_config"
    assert automl.best_config["objective"] == "mape", "objective should be 'mape'"

    # Verify the model has the correct objective
    if hasattr(automl.model, "estimator") and hasattr(automl.model.estimator, "get_params"):
        model_params = automl.model.estimator.get_params()
        assert model_params.get("objective") == "mape", "Model should use 'mape' objective"

    print("Test passed: objective parameter works correctly with LGBMEstimator")


if __name__ == "__main__":
    test_custom_hp()
    test_lgbm_objective()
