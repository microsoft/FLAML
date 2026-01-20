"""Test to ensure no training and test set overlap for classification tasks"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from flaml import AutoML


def test_no_overlap_holdout_classification():
    """Test that training and validation sets don't overlap in holdout classification"""
    # Load iris dataset
    dic_data = load_iris(as_frame=True)
    iris_data = dic_data["frame"]
    
    # Prepare data
    x_train = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].to_numpy()
    y_train = iris_data['target']
    
    # Train with holdout strategy
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "metric": 'accuracy',
        "task": 'classification',
        "estimator_list": ['lgbm'],
        "eval_method": "holdout",
        "split_type": "stratified",
        "keep_search_state": True,
        "retrain_full": False,  # Keep train/val split for testing
        "auto_augment": False,
        "verbose": 0,
    }
    automl.fit(x_train, y_train, **automl_settings)
    
    # Check that there's no overlap
    input_size = len(x_train)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)
    
    # The sum should not exceed the input size
    assert train_size + val_size == input_size, \
        f"Overlap detected! Input: {input_size}, Train: {train_size}, Val: {val_size}, Total: {train_size + val_size}"
    
    # Verify all classes are represented in the combined train+val
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))
    all_labels = set(np.unique(y_train))
    
    # Check that all labels are covered by the union of train and val sets
    combined_labels = train_labels.union(val_labels)
    assert combined_labels == all_labels, \
        f"Not all labels present. All: {all_labels}, Train: {train_labels}, Val: {val_labels}"
    
    print(f"✓ Test passed: No overlap detected. Input: {input_size}, Train: {train_size}, Val: {val_size}")


def test_no_overlap_uniform_split():
    """Test that training and validation sets don't overlap with uniform split"""
    # Load iris dataset
    dic_data = load_iris(as_frame=True)
    iris_data = dic_data["frame"]
    
    # Prepare data
    x_train = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].to_numpy()
    y_train = iris_data['target']
    
    # Train with uniform split
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "metric": 'accuracy',
        "task": 'classification',
        "estimator_list": ['lgbm'],
        "eval_method": "holdout",
        "split_type": "uniform",
        "keep_search_state": True,
        "retrain_full": False,
        "auto_augment": False,
        "verbose": 0,
    }
    automl.fit(x_train, y_train, **automl_settings)
    
    # Check that there's no overlap
    input_size = len(x_train)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)
    
    # The sum should not exceed the input size
    assert train_size + val_size == input_size, \
        f"Overlap detected! Input: {input_size}, Train: {train_size}, Val: {val_size}, Total: {train_size + val_size}"
    
    print(f"✓ Test passed: No overlap detected with uniform split. Input: {input_size}, Train: {train_size}, Val: {val_size}")


def test_all_labels_present_when_needed():
    """Test that missing labels are added only when necessary"""
    # Create a dataset where some classes are very rare
    np.random.seed(42)
    
    # Create imbalanced dataset
    X = np.random.randn(100, 4)
    # Class 0: 80 samples, Class 1: 15 samples, Class 2: 5 samples
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    automl = AutoML()
    automl_settings = {
        "max_iter": 3,
        "metric": 'accuracy',
        "task": 'classification',
        "estimator_list": ['lgbm'],
        "eval_method": "holdout",
        "split_type": "stratified",
        "keep_search_state": True,
        "retrain_full": False,
        "auto_augment": False,
        "verbose": 0,
    }
    automl.fit(X, y, **automl_settings)
    
    # Check that there's no overlap
    input_size = len(X)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)
    
    # The sum should not exceed the input size
    assert train_size + val_size == input_size, \
        f"Overlap detected! Input: {input_size}, Train: {train_size}, Val: {val_size}, Total: {train_size + val_size}"
    
    # Verify all classes are represented in the combined train+val
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))
    all_labels = set(np.unique(y))
    
    combined_labels = train_labels.union(val_labels)
    assert combined_labels == all_labels, \
        f"Not all labels present. All: {all_labels}, Train: {train_labels}, Val: {val_labels}"
    
    print(f"✓ Test passed: All labels present without overlap. Input: {input_size}, Train: {train_size}, Val: {val_size}")


if __name__ == "__main__":
    test_no_overlap_holdout_classification()
    test_no_overlap_uniform_split()
    test_all_labels_present_when_needed()
    print("\n✓ All tests passed!")
