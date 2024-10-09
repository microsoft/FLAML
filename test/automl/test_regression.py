import unittest
from test.conftest import evaluate_cv_folds_with_underlying_model

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import (
    fetch_california_housing,
    make_regression,
)

from flaml import AutoML
from flaml.automl.data import get_output_from_log
from flaml.automl.model import XGBoostEstimator


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


class MyXGB1(XGBoostEstimator):
    """XGBoostEstimator with logregobj as the objective function"""

    def __init__(self, **config):
        super().__init__(objective=logregobj, **config)


class MyXGB2(XGBoostEstimator):
    """XGBoostEstimator with 'reg:squarederror' as the objective function"""

    def __init__(self, **config):
        super().__init__(objective="reg:squarederror", **config)


class TestRegression(unittest.TestCase):
    def test_regression(self):
        automl = AutoML()
        automl_settings = {
            "time_budget": 2,
            "task": "regression",
            "log_file_name": "test/california.log",
            "log_training_metric": True,
            "n_jobs": 1,
            "model_history": True,
        }
        X_train, y_train = fetch_california_housing(return_X_y=True)
        n = int(len(y_train) * 9 // 10)
        automl.fit(X_train=X_train[:n], y_train=y_train[:n], X_val=X_train[n:], y_val=y_train[n:], **automl_settings)
        assert automl._state.eval_method == "holdout"
        y_pred = automl.predict(X_train)
        print(y_pred)
        print(automl.model.estimator)
        n_iter = automl.model.estimator.get_params("n_estimators")
        print(automl.config_history)
        print(automl.best_model_for_estimator("xgboost"))
        print(automl.best_iteration)
        print(automl.best_estimator)
        print(get_output_from_log(automl_settings["log_file_name"], 1))
        automl.retrain_from_log(
            task="regression",
            log_file_name=automl_settings["log_file_name"],
            X_train=X_train,
            y_train=y_train,
            train_full=True,
            time_budget=1,
        )
        automl.retrain_from_log(
            task="regression",
            log_file_name=automl_settings["log_file_name"],
            X_train=X_train,
            y_train=y_train,
            time_budget=0,
        )
        automl = AutoML()
        automl.retrain_from_log(
            task="regression",
            log_file_name=automl_settings["log_file_name"],
            X_train=X_train[:n],
            y_train=y_train[:n],
            train_full=True,
        )
        print(automl.model.estimator)
        y_pred2 = automl.predict(X_train)
        # In some rare case, the last config is early stopped and it's the best config. But the logged config's n_estimator is not reduced.
        assert n_iter != automl.model.estimator.get_params("n_estimator") or (y_pred == y_pred2).all()

    def test_sparse_matrix_regression(self):
        X_train = scipy.sparse.random(300, 900, density=0.0001)
        y_train = np.random.uniform(size=300)
        X_val = scipy.sparse.random(100, 900, density=0.0001)
        y_val = np.random.uniform(size=100)
        automl = AutoML()
        settings = {
            "time_budget": 2,
            "metric": "mae",
            "task": "regression",
            "log_file_name": "test/sparse_regression.log",
            "n_jobs": 1,
            "model_history": True,
            "keep_search_state": True,
            "verbose": 0,
            "early_stop": True,
        }
        automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)
        assert automl._state.X_val.shape == X_val.shape
        print(automl.predict(X_train))
        print(automl.model)
        print(automl.config_history)
        print(automl.best_model_for_estimator("rf"))
        print(automl.best_iteration)
        print(automl.best_estimator)
        print(automl.best_config)
        print(automl.best_loss)
        print(automl.best_config_train_time)

        settings.update(
            {
                "estimator_list": ["catboost"],
                "keep_search_state": False,
                "model_history": False,
                "use_best_model": False,
                "time_budget": None,
                "max_iter": 2,
                "custom_hp": {"catboost": {"n_estimators": {"domain": 100}}},
            }
        )
        automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)

    def test_parallel(self, hpo_method=None):
        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 10,
            "task": "regression",
            "log_file_name": "test/california.log",
            "log_type": "all",
            "n_jobs": 1,
            "n_concurrent_trials": 10,
            "hpo_method": hpo_method,
        }
        X_train, y_train = fetch_california_housing(return_X_y=True)
        try:
            automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
            print(automl_experiment.predict(X_train))
            print(automl_experiment.model)
            print(automl_experiment.config_history)
            print(automl_experiment.best_model_for_estimator("xgboost"))
            print(automl_experiment.best_iteration)
            print(automl_experiment.best_estimator)
        except ImportError:
            return

    def test_sparse_matrix_regression_holdout(self):
        X_train = scipy.sparse.random(8, 100)
        y_train = np.random.uniform(size=8)
        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 1,
            "eval_method": "holdout",
            "task": "regression",
            "log_file_name": "test/sparse_regression.log",
            "n_jobs": 1,
            "model_history": True,
            "metric": "mse",
            "sample_weight": np.ones(len(y_train)),
            "early_stop": True,
        }
        automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.best_model_for_estimator("rf"))
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)

    def test_regression_xgboost(self):
        X_train = scipy.sparse.random(300, 900, density=0.0001)
        y_train = np.random.uniform(size=300)
        X_val = scipy.sparse.random(100, 900, density=0.0001)
        y_val = np.random.uniform(size=100)
        automl_experiment = AutoML()
        automl_experiment.add_learner(learner_name="my_xgb1", learner_class=MyXGB1)
        automl_experiment.add_learner(learner_name="my_xgb2", learner_class=MyXGB2)
        automl_settings = {
            "time_budget": 2,
            "estimator_list": ["my_xgb1", "my_xgb2"],
            "task": "regression",
            "log_file_name": "test/regression_xgboost.log",
            "n_jobs": 1,
            "model_history": True,
            "keep_search_state": True,
            "early_stop": True,
        }
        automl_experiment.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
        assert automl_experiment._state.X_val.shape == X_val.shape
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.best_model_for_estimator("my_xgb2"))
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)
        print(automl_experiment.best_config)
        print(automl_experiment.best_loss)
        print(automl_experiment.best_config_train_time)


def test_multioutput():
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor, RegressorChain

    # create regression data
    X, y = make_regression(n_targets=3)

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # train the model
    model = MultiOutputRegressor(AutoML(task="regression", time_budget=1))
    model.fit(X_train, y_train)

    # predict
    print(model.predict(X_test))

    # train the model
    model = RegressorChain(AutoML(task="regression", time_budget=1))
    model.fit(X_train, y_train)

    # predict
    print(model.predict(X_test))


@pytest.mark.parametrize(
    "estimator",
    [
        # "catboost",
        "extra_tree",
        "histgb",
        "kneighbor",
        "lgbm",
        "rf",
        "xgboost",
        "xgb_limitdepth",
    ],
)
def test_reproducibility_of_regression_models(estimator: str):
    """FLAML finds the best model for a given dataset, which it then provides to users.

    However, there are reported issues where FLAML was providing an incorrect model - see here:
    https://github.com/microsoft/FLAML/issues/1317
    In this test we take the best regression model which FLAML provided us, and then retrain and test it on the
    same folds, to verify that the result is reproducible.
    """
    automl = AutoML()
    automl_settings = {
        "max_iter": 2,
        "time_budget": -1,
        "task": "regression",
        "n_jobs": 1,
        "estimator_list": [estimator],
        "eval_method": "cv",
        "n_splits": 3,
        "metric": "r2",
        "keep_search_state": True,
        "skip_transform": True,
    }
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    automl.fit(X_train=X, y_train=y, **automl_settings)
    best_model = automl.model
    assert best_model is not None
    config = best_model.get_params()
    val_loss_flaml = automl.best_result["val_loss"]

    # Take the best model, and see if we can reproduce the best result
    reproduced_val_loss, metric_for_logging, train_time, pred_time = automl._state.task.evaluate_model_CV(
        config=config,
        estimator=best_model,
        X_train_all=automl._state.X_train_all,
        y_train_all=automl._state.y_train_all,
        budget=None,
        kf=automl._state.kf,
        eval_metric="r2",
        best_val_loss=None,
        cv_score_agg_func=None,
        log_training_metric=False,
        fit_kwargs=None,
        free_mem_ratio=0,
    )
    assert pytest.approx(val_loss_flaml) == reproduced_val_loss


@pytest.mark.parametrize(
    "estimator",
    [
        # "catboost",
        "extra_tree",
        "histgb",
        "kneighbor",
        # "lgbm",
        "rf",
        "xgboost",
        "xgb_limitdepth",
    ],
)
def test_reproducibility_of_underlying_regression_models(estimator: str):
    """FLAML finds the best model for a given dataset, which it then provides to users.

    However, there are reported issues where FLAML was providing an incorrect model - see here:
    https://github.com/microsoft/FLAML/issues/1317
    FLAML defines FLAMLised models, which wrap around the underlying (SKLearn/XGBoost/CatBoost) model.
    Ideally, FLAMLised models should perform identically to the underlying model, when fitted
    to the same data, with no budget. This verifies that this is the case for regression models.
    In this test we take the best model which FLAML provided us, extract the underlying model,
     before retraining and testing it on the same folds - to verify that the result is reproducible.
    """
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "time_budget": -1,
        "task": "regression",
        "n_jobs": 1,
        "estimator_list": [estimator],
        "eval_method": "cv",
        "n_splits": 10,
        "metric": "r2",
        "keep_search_state": True,
        "skip_transform": True,
    }
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    automl.fit(X_train=X, y_train=y, **automl_settings)
    best_model = automl.model
    assert best_model is not None
    val_loss_flaml = automl.best_result["val_loss"]
    reproduced_val_loss_underlying_model = np.mean(
        evaluate_cv_folds_with_underlying_model(
            automl._state.X_train_all, automl._state.y_train_all, automl._state.kf, best_model.model, "regression"
        )
    )

    assert pytest.approx(val_loss_flaml) == reproduced_val_loss_underlying_model


if __name__ == "__main__":
    unittest.main()
