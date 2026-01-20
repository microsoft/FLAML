"""
Example demonstrating the use of FLAML's preprocess() API.

This script shows how to use both task-level and estimator-level preprocessing
APIs exposed by FLAML AutoML.
"""

from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load and split data
print("Loading breast cancer dataset...")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train AutoML model
print("\nTraining AutoML model...")
automl = AutoML()
automl_settings = {
    "time_budget": 10,  # 10 seconds
    "task": "classification",
    "metric": "accuracy",
    "estimator_list": ["lgbm", "xgboost"],
    "verbose": 0,
}
automl.fit(X_train, y_train, **automl_settings)

print(f"Best estimator: {automl.best_estimator}")
print(f"Best accuracy: {1 - automl.best_loss:.4f}")

# Example 1: Using task-level preprocessing
print("\n" + "=" * 60)
print("Example 1: Task-level preprocessing")
print("=" * 60)
X_test_task = automl.preprocess(X_test)
print(f"Original test data shape: {X_test.shape}")
print(f"After task preprocessing: {X_test_task.shape}")

# Example 2: Using estimator-level preprocessing
print("\n" + "=" * 60)
print("Example 2: Estimator-level preprocessing")
print("=" * 60)
estimator = automl.model
X_test_estimator = estimator.preprocess(X_test_task)
print(f"After estimator preprocessing: {X_test_estimator.shape}")

# Example 3: Complete preprocessing pipeline
print("\n" + "=" * 60)
print("Example 3: Complete preprocessing pipeline")
print("=" * 60)
# Apply both levels of preprocessing
X_preprocessed = automl.preprocess(X_test)
X_final = automl.model.preprocess(X_preprocessed)

# Manual prediction using fully preprocessed data
y_pred_manual = automl.model._model.predict(X_final)

# Compare with AutoML's predict method (which does preprocessing internally)
y_pred_auto = automl.predict(X_test)

print(f"Predictions match: {np.array_equal(y_pred_manual, y_pred_auto)}")
print(f"Manual prediction sample: {y_pred_manual[:5]}")
print(f"Auto prediction sample: {y_pred_auto[:5]}")

# Example 4: Using preprocessing for custom inference
print("\n" + "=" * 60)
print("Example 4: Custom inference with preprocessing")
print("=" * 60)
# You might want to apply preprocessing separately for:
# - Debugging
# - Custom inference pipelines
# - Integration with other tools

# Get preprocessed features
X_features = automl.preprocess(X_test)
X_features = automl.model.preprocess(X_features)

# Now you can use these features with the underlying model or for analysis
print(f"Preprocessed features ready for custom use: {X_features.shape}")
print(f"Feature statistics - Mean: {np.mean(X_features):.4f}, Std: {np.std(X_features):.4f}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("The preprocess() API allows you to:")
print("1. Apply task-level preprocessing with automl.preprocess()")
print("2. Apply estimator-level preprocessing with estimator.preprocess()")
print("3. Chain both for complete preprocessing pipeline")
print("4. Use preprocessed data for custom inference or analysis")
print("\nNote: Task-level preprocessing should always be applied before")
print("      estimator-level preprocessing.")
