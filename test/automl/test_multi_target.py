"""Tests for multi-target regression support in FLAML AutoML."""
import unittest

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from flaml import AutoML


class TestMultiTargetRegression(unittest.TestCase):
    """Test multi-target regression functionality."""

    def setUp(self):
        """Create multi-target regression datasets for testing."""
        # Create synthetic multi-target regression data
        self.X, self.y = make_regression(
            n_samples=200, n_features=10, n_targets=3, random_state=42, noise=0.1
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_multi_target_with_xgboost(self):
        """Test multi-target regression with XGBoost."""
        automl = AutoML()
        automl.fit(
            self.X_train,
            self.y_train,
            task="regression",
            time_budget=5,
            estimator_list=["xgboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)
        self.assertEqual(automl.best_estimator, "xgboost")

        # Check predictions shape
        y_pred = automl.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)
        self.assertEqual(y_pred.ndim, 2)

    def test_multi_target_with_catboost(self):
        """Test multi-target regression with CatBoost."""
        try:
            import catboost  # noqa: F401
        except ImportError:
            pytest.skip("CatBoost not installed")

        automl = AutoML()
        automl.fit(
            self.X_train,
            self.y_train,
            task="regression",
            time_budget=5,
            estimator_list=["catboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)
        self.assertEqual(automl.best_estimator, "catboost")

        # Check predictions shape
        y_pred = automl.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)
        self.assertEqual(y_pred.ndim, 2)

    def test_unsupported_estimator_filtered_out(self):
        """Test that unsupported estimators are filtered for multi-target."""
        automl = AutoML()
        with self.assertRaises(ValueError) as context:
            automl.fit(
                self.X_train,
                self.y_train,
                task="regression",
                time_budget=5,
                estimator_list=["lgbm"],
                verbose=0,
            )
        self.assertIn("Multi-target regression only supports", str(context.exception))

    def test_auto_estimator_list(self):
        """Test that auto estimator list works with multi-target."""
        automl = AutoML()
        automl.fit(
            self.X_train,
            self.y_train,
            task="regression",
            time_budget=10,
            verbose=0,
        )

        # Check that only supported estimators were used
        self.assertIn(automl.best_estimator, ["xgboost", "xgb_limitdepth", "catboost"])

        # Check predictions shape
        y_pred = automl.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_multi_target_with_validation_set(self):
        """Test multi-target regression with explicit validation set."""
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        automl = AutoML()
        automl.fit(
            X_train_sub,
            y_train_sub,
            X_val=X_val,
            y_val=y_val,
            task="regression",
            time_budget=5,
            estimator_list=["xgboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)

        # Check predictions shape
        y_pred = automl.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_multi_target_with_dataframe(self):
        """Test multi-target regression with pandas DataFrame."""
        X_df = pd.DataFrame(self.X_train, columns=[f"feature_{i}" for i in range(self.X_train.shape[1])])
        y_df = pd.DataFrame(self.y_train, columns=[f"target_{i}" for i in range(self.y_train.shape[1])])

        automl = AutoML()
        automl.fit(
            X_df,
            y_df,
            task="regression",
            time_budget=5,
            estimator_list=["xgboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)

        # Check predictions shape
        X_test_df = pd.DataFrame(self.X_test, columns=[f"feature_{i}" for i in range(self.X_test.shape[1])])
        y_pred = automl.predict(X_test_df)
        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_single_target_still_works(self):
        """Test that single-target regression still works correctly."""
        X, y = make_regression(n_samples=200, n_features=10, n_targets=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        automl = AutoML()
        automl.fit(
            X_train,
            y_train,
            task="regression",
            time_budget=5,
            estimator_list=["lgbm", "xgboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)

        # Check predictions shape (should be 1D or (n, 1))
        y_pred = automl.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))

    def test_multi_target_cv(self):
        """Test multi-target regression with cross-validation."""
        automl = AutoML()
        automl.fit(
            self.X_train,
            self.y_train,
            task="regression",
            time_budget=10,
            eval_method="cv",
            n_splits=3,
            estimator_list=["xgboost"],
            verbose=0,
        )

        # Check that the model was trained
        self.assertIsNotNone(automl.model)

        # Check predictions shape
        y_pred = automl.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)


if __name__ == "__main__":
    unittest.main()
