import os
import unittest
import logging
from tempfile import TemporaryDirectory

from sklearn.datasets import load_boston

from flaml import AutoML


class TestLogging(unittest.TestCase):

    def test_logging_level(self):

        from flaml import logger, logger_formatter

        with TemporaryDirectory() as d:
            filename = os.path.join(d, 'test_logging_level.log')

            # Configure logging for the FLAML logger.
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(filename)
            fh.setFormatter(logger_formatter)
            logger.addHandler(fh)

            # Run a simple job.
            automl_experiment = AutoML()
            automl_settings = {
                "time_budget": 2,
                "metric": 'mse',
                "task": 'regression',
                "log_file_name": "test/boston.log",
                "log_training_metric": True,
                "model_history": True
            }
            X_train, y_train = load_boston(return_X_y=True)
            n = len(y_train)
            automl_experiment.fit(X_train=X_train[:n >> 1], y_train=y_train[:n >> 1],
                                  X_val=X_train[n >> 1:], y_val=y_train[n >> 1:],
                                  **automl_settings)

            # Release handler.
            fh.flush()
            fh.close()

            # Check if the log file is populated.
            self.assertTrue(os.path.exists(filename))
            with open(filename) as f:
                content = f.read()
                self.assertTrue(len(content) > 0)
