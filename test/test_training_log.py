import os
import unittest
from tempfile import TemporaryDirectory

from sklearn.datasets import fetch_california_housing

from flaml import AutoML
from flaml.training_log import training_log_reader


class TestTrainingLog(unittest.TestCase):
    def test_training_log(self, path="test_training_log.log"):

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
                "train_time_limit": 0.01,
                "verbose": 3,
                "ensemble": True,
                "keep_search_state": True,
            }
            X_train, y_train = fetch_california_housing(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            automl._state._train_with_config(automl.best_estimator, automl.best_config)

            # Check if the training log file is populated.
            self.assertTrue(os.path.exists(filename))
            with training_log_reader(filename) as reader:
                count = 0
                for record in reader.records():
                    print(record)
                    count += 1
                self.assertGreater(count, 0)

            automl_settings["log_file_name"] = None
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
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
