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
            "max_iter": 5,
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
            "max_iter": 5,
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
            "max_iter": 5,
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
            "max_iter": 5,
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
            "max_iter": 5,
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
            "max_iter": 5,
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


class TestCategoricalEncodingStability(unittest.TestCase):
    """Regression coverage for #1101 — DataTransformer must produce the same
    integer code for the same categorical value regardless of which values
    happen to be present in the predict-time DataFrame."""

    def _fit_simple(self):
        from flaml.automl.data import DataTransformer
        from flaml.automl.task.factory import task_factory

        rng = np.random.RandomState(0)
        n = 100
        fit_df = pd.DataFrame({"a": rng.randn(n), "gender": rng.choice(["M", "F"], n)})
        fit_y = pd.Series(rng.randn(n))

        transformer = DataTransformer()
        task = task_factory("regression", fit_df, fit_y)
        X_fit, _ = transformer.fit_transform(fit_df.copy(), fit_y, task)
        return transformer, X_fit

    def test_codes_stable_when_predict_uses_only_a_subset(self):
        transformer, X_fit = self._fit_simple()
        fit_code_for_M = int(X_fit["gender"].cat.codes[X_fit["gender"] == "M"].iloc[0])

        # Predict-time DataFrame contains only "M" rows.
        predict_df = pd.DataFrame({"a": np.zeros(20), "gender": ["M"] * 20})
        X_pred = transformer.transform(predict_df.copy())
        pred_code_for_M = int(X_pred["gender"].cat.codes[X_pred["gender"] == "M"].iloc[0])

        self.assertEqual(
            fit_code_for_M,
            pred_code_for_M,
            "categorical code for 'M' drifted between fit and predict — see #1101",
        )

    def test_unseen_categories_emit_warning_and_map_to_sentinel(self):
        import warnings

        transformer, _ = self._fit_simple()
        predict_df = pd.DataFrame({"a": np.zeros(5), "gender": ["M", "F", "X", "M", "Y"]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X_pred = transformer.transform(predict_df.copy())

        unseen_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "unseen at fit time" in str(w.message)
        ]
        self.assertEqual(len(unseen_warnings), 1)
        message = str(unseen_warnings[0].message)
        self.assertIn("gender", message)
        self.assertIn("X", message)
        self.assertIn("Y", message)

        # Unseen "X" and "Y" rows must be encoded as the "__NAN__" sentinel slot
        # and seen "F" / "M" codes must still match fit-time codes.
        nan_code = list(X_pred["gender"].cat.categories).index("__NAN__")
        unseen_rows = X_pred["gender"].cat.codes[predict_df["gender"].isin(["X", "Y"]).values]
        self.assertTrue((unseen_rows == nan_code).all())


class TestOrdinalEncoderBackedTransform(unittest.TestCase):
    """Coverage for the categorical-encoding refactor in #1564: DataTransformer
    uses sklearn's OrdinalEncoder as the source of truth for the per-column
    category list at fit time, and the three-tier backward-compat fallback
    (`_ordinal_encoder` → `_cat_categories` → legacy) at transform time so that
    pickles from any recent FLAML version continue to load."""

    def _fit_simple(self):
        from flaml.automl.data import DataTransformer
        from flaml.automl.task.factory import task_factory

        rng = np.random.RandomState(0)
        n = 100
        fit_df = pd.DataFrame({"a": rng.randn(n), "gender": rng.choice(["M", "F"], n)})
        fit_y = pd.Series(rng.randn(n))
        transformer = DataTransformer()
        task = task_factory("regression", fit_df, fit_y)
        X_fit, _ = transformer.fit_transform(fit_df.copy(), fit_y, task)
        return transformer, X_fit

    def test_fit_transform_installs_ordinal_encoder(self):
        from sklearn.preprocessing import OrdinalEncoder

        transformer, _ = self._fit_simple()
        self.assertTrue(hasattr(transformer, "_ordinal_encoder"))
        self.assertIsInstance(transformer._ordinal_encoder, OrdinalEncoder)
        self.assertIn("gender", list(transformer._ordinal_encoder.feature_names_in_))

    def test_ordinal_encoder_path_matches_1561_semantics(self):
        """Refactored path preserves the observable behavior from #1561:
        - known-category codes are stable across fit/predict distributions,
        - unseen values are remapped to the "__NAN__" sentinel code,
        - a `UserWarning` is emitted on unseen values."""
        import warnings

        transformer, X_fit = self._fit_simple()
        fit_M_code = int(X_fit["gender"].cat.codes[X_fit["gender"] == "M"].iloc[0])

        # (a) Stability under a smaller predict-time value set
        predict_df = pd.DataFrame({"a": np.zeros(20), "gender": ["M"] * 20})
        X_pred = transformer.transform(predict_df.copy())
        pred_M_code = int(X_pred["gender"].cat.codes[X_pred["gender"] == "M"].iloc[0])
        self.assertEqual(fit_M_code, pred_M_code)

        # (b) Unseen-category warning + (c) sentinel remap
        predict_df2 = pd.DataFrame({"a": np.zeros(5), "gender": ["M", "F", "X", "M", "Y"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X_pred2 = transformer.transform(predict_df2.copy())
        unseen_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "unseen at fit time" in str(w.message)
        ]
        self.assertEqual(len(unseen_warnings), 1)
        nan_code = list(X_pred2["gender"].cat.categories).index("__NAN__")
        unseen_rows = X_pred2["gender"].cat.codes[predict_df2["gender"].isin(["X", "Y"]).values]
        self.assertTrue((unseen_rows == nan_code).all())

    def test_transform_falls_back_to_cat_categories_when_encoder_missing(self):
        """Pickles produced between #1561 and #1564 only have `_cat_categories`.
        `transform()` must still work correctly for them."""
        transformer, X_fit = self._fit_simple()
        # Simulate a #1561-era pickle by removing the encoder and installing the
        # ad-hoc dict the older code produced.
        del transformer._ordinal_encoder
        transformer._cat_categories = {
            "gender": list(X_fit["gender"].cat.categories) + ["__NAN__"],
        }

        fit_M_code = int(X_fit["gender"].cat.codes[X_fit["gender"] == "M"].iloc[0])
        predict_df = pd.DataFrame({"a": np.zeros(20), "gender": ["M"] * 20})
        X_pred = transformer.transform(predict_df.copy())
        pred_M_code = int(X_pred["gender"].cat.codes[X_pred["gender"] == "M"].iloc[0])
        self.assertEqual(fit_M_code, pred_M_code)

    def test_transform_legacy_pickle_without_either_attribute(self):
        """Pickles from before #1561 have neither `_ordinal_encoder` nor
        `_cat_categories`. `transform()` must not raise; it falls through to the
        legacy `astype("category")` path. Drift is possible on those pickles
        (that is the bug that #1561 and #1564 fix), but load-and-predict must
        continue to work without upgrading users' pickle files."""
        transformer, _ = self._fit_simple()
        del transformer._ordinal_encoder  # no `_cat_categories` was ever set

        predict_df = pd.DataFrame({"a": np.zeros(20), "gender": ["M"] * 20})
        # Legacy path just does astype("category") — no exception, no warning.
        X_pred = transformer.transform(predict_df.copy())
        self.assertEqual(str(X_pred["gender"].dtype), "category")


if __name__ == "__main__":
    unittest.main()
