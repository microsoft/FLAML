"""Test NumPy 2.0+ compatibility.

This test ensures FLAML works correctly with both NumPy 1.x and NumPy 2.0+.
NumPy 2.0 removed several deprecated attributes including np.NaN (use np.nan instead).
"""

import unittest

import numpy as np


class TestNumPyCompatibility(unittest.TestCase):
    """Test compatibility with NumPy 1.x and 2.0+."""

    def test_numpy_version(self):
        """Test that numpy is available and has a version."""
        self.assertTrue(hasattr(np, "__version__"))
        self.assertTrue(len(np.__version__) > 0)
        print(f"Testing with NumPy version: {np.__version__}")

    def test_numpy_nan_lowercase(self):
        """Test that np.nan (lowercase) works in all NumPy versions."""
        # np.nan should work in both NumPy 1.x and 2.0+
        self.assertTrue(np.isnan(np.nan))

    def test_numpy_inf_lowercase(self):
        """Test that np.inf (lowercase) works in all NumPy versions."""
        # np.inf should work in both NumPy 1.x and 2.0+
        self.assertTrue(np.isinf(np.inf))

    def test_numpy2_deprecated_attributes(self):
        """Test handling of NumPy 2.0 removed attributes."""
        numpy_major = int(np.__version__.split(".")[0])

        if numpy_major >= 2:
            # In NumPy 2.0+, np.NaN should not exist
            with self.assertRaises(AttributeError):
                _ = np.NaN
            print("✓ Confirmed np.NaN is not available in NumPy 2.0+ (as expected)")
        else:
            # In NumPy 1.x, np.NaN might exist but we shouldn't use it
            # We just verify our code uses np.nan instead
            print(f"✓ Running with NumPy {np.__version__} (< 2.0)")

    def test_flaml_imports_with_numpy(self):
        """Test that FLAML imports successfully with current NumPy version."""
        try:
            import flaml

            self.assertTrue(hasattr(flaml, "__version__"))
            print(f"✓ FLAML {flaml.__version__} imports successfully with NumPy {np.__version__}")
        except ImportError as e:
            self.fail(f"Failed to import FLAML: {e}")

    def test_automl_basic_functionality(self):
        """Test basic AutoML functionality with current NumPy version."""
        try:
            from flaml import AutoML
            from sklearn.datasets import make_classification

            # Create a small dataset
            X, y = make_classification(n_samples=50, n_features=4, random_state=42)

            # Test AutoML basic operations
            automl = AutoML()
            self.assertIsNotNone(automl)

            # Test fit and predict with minimal time budget
            automl.fit(X, y, task="classification", time_budget=1, verbose=0)
            predictions = automl.predict(X[:5])

            self.assertEqual(len(predictions), 5)
            print(f"✓ AutoML fit and predict work correctly with NumPy {np.__version__}")

        except ImportError:
            # AutoML might not be installed (requires flaml[automl])
            self.skipTest("AutoML not available (install flaml[automl] to test)")
        except Exception as e:
            self.fail(f"AutoML test failed with NumPy {np.__version__}: {e}")


if __name__ == "__main__":
    unittest.main()
