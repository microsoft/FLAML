"""Tests for the public preprocessor APIs."""
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from flaml import AutoML


class TestPreprocessAPI(unittest.TestCase):
    """Test cases for the public preprocess() API methods."""

    def test_automl_preprocess_before_fit(self):
        """Test that calling preprocess before fit raises an error."""
        automl = AutoML()
        X_test = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(AttributeError) as context:
            automl.preprocess(X_test)
        # Check that an error is raised about not being fitted
        self.assertIn("fit()", str(context.exception))

    def test_automl_preprocess_classification(self):
        """Test task-level preprocessing for classification."""
        # Load dataset
        X, y = load_breast_cancer(return_X_y=True)
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:450]

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "classification",
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Test task-level preprocessing
        X_preprocessed = automl.preprocess(X_test)

        # Verify the output is not None and has the right shape
        self.assertIsNotNone(X_preprocessed)
        self.assertEqual(X_preprocessed.shape[0], X_test.shape[0])

    def test_automl_preprocess_regression(self):
        """Test task-level preprocessing for regression."""
        # Load dataset
        X, y = load_diabetes(return_X_y=True)
        X_train, y_train = X[:300], y[:300]
        X_test = X[300:350]

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "regression",
            "metric": "r2",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Test task-level preprocessing
        X_preprocessed = automl.preprocess(X_test)

        # Verify the output
        self.assertIsNotNone(X_preprocessed)
        self.assertEqual(X_preprocessed.shape[0], X_test.shape[0])

    def test_automl_preprocess_with_dataframe(self):
        """Test task-level preprocessing with pandas DataFrame."""
        # Create a simple dataset
        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5] * 20,
                "feature2": [5, 4, 3, 2, 1] * 20,
                "category": ["a", "b", "a", "b", "a"] * 20,
            }
        )
        y_train = pd.Series([0, 1, 0, 1, 0] * 20)

        X_test = pd.DataFrame(
            {
                "feature1": [6, 7, 8],
                "feature2": [1, 2, 3],
                "category": ["a", "b", "a"],
            }
        )

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "classification",
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Test preprocessing
        X_preprocessed = automl.preprocess(X_test)

        # Verify the output - check the number of rows matches
        self.assertIsNotNone(X_preprocessed)
        preprocessed_len = len(X_preprocessed) if hasattr(X_preprocessed, "__len__") else X_preprocessed.shape[0]
        self.assertEqual(preprocessed_len, len(X_test))

    def test_estimator_preprocess(self):
        """Test estimator-level preprocessing."""
        # Load dataset
        X, y = load_breast_cancer(return_X_y=True)
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:450]

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "classification",
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Get the trained estimator
        estimator = automl.model
        self.assertIsNotNone(estimator)

        # First apply task-level preprocessing
        X_task_preprocessed = automl.preprocess(X_test)

        # Then apply estimator-level preprocessing
        X_estimator_preprocessed = estimator.preprocess(X_task_preprocessed)

        # Verify the output
        self.assertIsNotNone(X_estimator_preprocessed)
        self.assertEqual(X_estimator_preprocessed.shape[0], X_test.shape[0])

    def test_preprocess_pipeline(self):
        """Test the complete preprocessing pipeline (task-level then estimator-level)."""
        # Load dataset
        X, y = load_breast_cancer(return_X_y=True)
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:450]

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "classification",
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Apply the complete preprocessing pipeline
        X_task_preprocessed = automl.preprocess(X_test)
        X_final = automl.model.preprocess(X_task_preprocessed)

        # Verify predictions work with preprocessed data
        # The internal predict already does this preprocessing,
        # but we verify our manual preprocessing gives consistent results
        y_pred_manual = automl.model._model.predict(X_final)
        y_pred_auto = automl.predict(X_test)

        # Both should give the same predictions
        np.testing.assert_array_equal(y_pred_manual, y_pred_auto)

    def test_preprocess_with_mixed_types(self):
        """Test preprocessing with mixed data types."""
        # Create dataset with mixed types
        X_train = pd.DataFrame(
            {
                "numeric1": np.random.rand(100),
                "numeric2": np.random.randint(0, 100, 100),
                "categorical": np.random.choice(["cat", "dog", "bird"], 100),
                "boolean": np.random.choice([True, False], 100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        X_test = pd.DataFrame(
            {
                "numeric1": np.random.rand(10),
                "numeric2": np.random.randint(0, 100, 10),
                "categorical": np.random.choice(["cat", "dog", "bird"], 10),
                "boolean": np.random.choice([True, False], 10),
            }
        )

        # Train AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 5,
            "task": "classification",
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "verbose": 0,
        }
        automl.fit(X_train, y_train, **automl_settings)

        # Test preprocessing
        X_preprocessed = automl.preprocess(X_test)

        # Verify the output
        self.assertIsNotNone(X_preprocessed)

    def test_estimator_preprocess_without_automl(self):
        """Test that estimator.preprocess() can be used independently."""
        from flaml.automl.model import LGBMEstimator

        # Create a simple estimator
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        estimator = LGBMEstimator(task="classification")
        estimator.fit(X_train, y_train)

        # Test preprocessing
        X_test = np.random.rand(10, 5)
        X_preprocessed = estimator.preprocess(X_test)

        # Verify the output
        self.assertIsNotNone(X_preprocessed)
        self.assertEqual(X_preprocessed.shape, X_test.shape)


if __name__ == "__main__":
    unittest.main()
