"""
Demo script showing multi-target regression support in FLAML AutoML.

This script demonstrates:
1. Creating a multi-target regression dataset
2. Training an AutoML model with multi-target support
3. Making predictions with multi-target output
4. Comparing with single-target approach using MultiOutputRegressor wrapper
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from flaml import AutoML

# Create synthetic multi-target regression data
print("=" * 60)
print("Creating Multi-Target Regression Dataset")
print("=" * 60)

X, y = make_regression(
    n_samples=500,
    n_features=20,
    n_targets=3,  # 3 target variables
    random_state=42,
    noise=0.1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
print(f"Test set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
print()

# Train AutoML with multi-target support
print("=" * 60)
print("Training AutoML with Multi-Target Support")
print("=" * 60)

automl = AutoML()
automl.fit(
    X_train,
    y_train,
    task="regression",
    time_budget=30,  # 30 seconds
    verbose=0,
)

print(f"Best estimator: {automl.best_estimator}")
print(f"Best loss: {automl.best_loss:.4f}")
print()

# Make predictions
print("=" * 60)
print("Making Predictions")
print("=" * 60)

y_pred = automl.predict(X_test)
print(f"Predictions shape: {y_pred.shape}")
print(f"First 3 predictions:\n{y_pred[:3]}")
print()

# Evaluate performance
print("=" * 60)
print("Performance Metrics")
print("=" * 60)

# Overall metrics (averaged across all targets)
mse_overall = mean_squared_error(y_test, y_pred)
r2_overall = r2_score(y_test, y_pred)

print(f"Overall MSE: {mse_overall:.4f}")
print(f"Overall R²: {r2_overall:.4f}")
print()

# Per-target metrics
print("Per-Target Metrics:")
for i in range(y_test.shape[1]):
    mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"  Target {i}: MSE = {mse_i:.4f}, R² = {r2_i:.4f}")
print()

# Compare with pandas DataFrame input
print("=" * 60)
print("Testing with Pandas DataFrame")
print("=" * 60)

X_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
y_df = pd.DataFrame(y_train, columns=[f"target_{i}" for i in range(y_train.shape[1])])

automl_df = AutoML()
automl_df.fit(
    X_df,
    y_df,
    task="regression",
    time_budget=30,
    verbose=0,
)

print(f"Best estimator (DataFrame): {automl_df.best_estimator}")
print(f"Best loss (DataFrame): {automl_df.best_loss:.4f}")
print()

# Demonstrate filtering of unsupported estimators
print("=" * 60)
print("Demonstrating Estimator Filtering")
print("=" * 60)

print("Attempting to use LightGBM (unsupported for multi-target)...")
try:
    automl_lgbm = AutoML()
    automl_lgbm.fit(
        X_train,
        y_train,
        task="regression",
        time_budget=5,
        estimator_list=["lgbm"],  # LightGBM doesn't support multi-target
        verbose=0,
    )
    print("ERROR: LightGBM should not work with multi-target!")
except ValueError as e:
    print(f"✓ Expected error: {e}")
print()

# Compare supported estimators
print("=" * 60)
print("Comparing Supported Estimators")
print("=" * 60)

for estimator in ["xgboost", "catboost"]:
    try:
        print(f"\nTesting {estimator}...")
        automl_est = AutoML()
        automl_est.fit(
            X_train[:200],  # Use subset for speed
            y_train[:200],
            task="regression",
            time_budget=10,
            estimator_list=[estimator],
            verbose=0,
        )
        y_pred_est = automl_est.predict(X_test)
        mse_est = mean_squared_error(y_test, y_pred_est)
        print(f"  ✓ {estimator}: MSE = {mse_est:.4f}")
    except ImportError:
        print(f"  ⊗ {estimator}: Not installed")
    except Exception as e:
        print(f"  ✗ {estimator}: Error - {e}")

print()
print("=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("1. FLAML now supports multi-target regression natively")
print("2. Only XGBoost and CatBoost are supported for multi-target")
print("3. Works with both numpy arrays and pandas DataFrames")
print("4. Predictions maintain the (n_samples, n_targets) shape")
print("5. Sklearn metrics automatically average across targets")
