import os
import platform
import sys
import unittest
from tempfile import TemporaryDirectory

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from flaml import AutoML
from flaml.automl.training_log import training_log_reader


class TestTrainingLog(unittest.TestCase):
    def test_training_log(self, path="test_training_log.log", estimator_list="auto", use_ray=False):
        with TemporaryDirectory() as d:
            filename = os.path.join(d, path)

            # Run a simple job.
            automl = AutoML()
            automl_settings = {
                "time_budget": 1,
                "metric": "mse",
                "task": "regression",
                "log_file_name": filename,
                "log_training_metric": True,
                "mem_thres": 1024 * 1024,
                "n_jobs": 1,
                "model_history": True,
                "train_time_limit": 0.1,
                "verbose": 3,
                # "ensemble": True,
                "keep_search_state": True,
                "estimator_list": estimator_list,
                "seed": 42,
                "custom_hp": {"xgboost": {"random_state": {"domain": 42, "init_value": 42}}},
                # Keep model replay equivalence independent of feature-engineering choices.
                "featurization": "off",
            }
            X, y = fetch_california_housing(return_X_y=True, data_home="test")
            X_train, X_holdout, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            # Check if the training log file is populated.
            self.assertTrue(os.path.exists(filename))
            if automl.best_estimator:
                estimator, config = automl.best_estimator, automl.best_config
                model0 = automl.best_model_for_estimator(estimator)
                print(model0.params)
                if "n_estimators" in config:
                    assert model0.params["n_estimators"] == config["n_estimators"]

                # train on full data with no time limit
                automl._state.time_budget = -1
                model, _ = automl._state._train_with_config(estimator, config)

                # assuming estimator & config are saved and loaded as follows
                automl = AutoML()
                automl.fit(
                    X_train=X_train,
                    y_train=y_train,
                    max_iter=1,
                    task="regression",
                    metric="mse",
                    eval_method="holdout",
                    estimator_list=[estimator],
                    n_jobs=1,
                    starting_points={estimator: config},
                    use_ray=use_ray,
                    featurization="off",
                    custom_hp=automl_settings["custom_hp"],
                )
                print(automl.best_config)
                import numpy as np

                same_repr = str(model.estimator) == str(automl.model.estimator)
                if estimator == "xgboost":
                    y_range = float(np.ptp(y_train)) or 1.0
                    # Prior platform noise was about 0.14; check both partitions
                    # regardless of repr equality, with a substantially tighter bound.
                    tolerance = max(0.25, 0.05 * y_range)
                    for partition in (X_train, X_holdout):
                        preds_a = model.estimator.predict(partition)
                        preds_b = automl.model.estimator.predict(partition)
                        max_abs = float(np.max(np.abs(preds_a - preds_b)))
                        assert max_abs <= tolerance, f"xgboost predictions diverge by {max_abs}, limit {tolerance}"
                elif estimator == "catboost" and not same_repr:
                    assert str(model.estimator.get_all_params()) == str(automl.model.estimator.get_all_params())
                else:
                    assert same_repr
                automl.fit(
                    X_train=X_train,
                    y_train=y_train,
                    max_iter=1,
                    task="regression",
                    estimator_list=[estimator],
                    n_jobs=1,
                    starting_points={estimator: {}},
                )
                print(automl.best_config)

                with training_log_reader(filename) as reader:
                    count = 0
                    for record in reader.records():
                        print(record)
                        count += 1
                    self.assertGreater(count, 0)

            automl_settings["log_file_name"] = ""
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            if automl._selected:
                automl._selected.update(None, 0)
            automl = AutoML()
            automl.fit(X_train=X_train, y_train=y_train, max_iter=0, task="regression")

    def test_illfilename(self):
        try:
            self.test_training_log("/")
        except IsADirectoryError:
            print("IsADirectoryError happens as expected in linux.")
        except PermissionError:
            print("PermissionError happens as expected in windows.")
        except FileExistsError:
            print("FileExistsError happens as expected in MacOS.")

    def test_each_estimator(self):
        try:
            import ray

            ray.shutdown()
            ray.init()
            use_ray = True
        except ImportError:
            use_ray = False
        self.test_training_log(estimator_list=["xgboost"], use_ray=use_ray)
        if not (sys.platform == "win32" and platform.machine() == "ARM64"):
            self.test_training_log(estimator_list=["catboost"], use_ray=use_ray)
        self.test_training_log(estimator_list=["extra_tree"], use_ray=use_ray)
        self.test_training_log(estimator_list=["rf"], use_ray=use_ray)
        self.test_training_log(estimator_list=["lgbm"], use_ray=use_ray)
