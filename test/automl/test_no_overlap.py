"""Test to ensure correct label overlap handling for classification tasks"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification

from flaml import AutoML


def test_allow_label_overlap_true():
    """Test with allow_label_overlap=True (fast mode, default)"""
    # Load iris dataset
    dic_data = load_iris(as_frame=True)
    iris_data = dic_data["frame"]

    # Prepare data
    x_train = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].to_numpy()
    y_train = iris_data["target"]

    # Train with fast mode (default)
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm"],
        "eval_method": "holdout",
        "split_type": "stratified",
        "keep_search_state": True,
        "retrain_full": False,
        "auto_augment": False,
        "verbose": 0,
        "allow_label_overlap": True,  # Fast mode
    }
    automl.fit(x_train, y_train, **automl_settings)

    # Check results
    input_size = len(x_train)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)

    # With stratified split on balanced data, fast mode may have no overlap
    assert (
        train_size + val_size >= input_size
    ), f"Inconsistent sizes. Input: {input_size}, Train: {train_size}, Val: {val_size}"

    # Verify all classes are represented in both sets
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))
    all_labels = set(np.unique(y_train))

    assert train_labels == all_labels, f"Not all labels in train. All: {all_labels}, Train: {train_labels}"
    assert val_labels == all_labels, f"Not all labels in val. All: {all_labels}, Val: {val_labels}"

    print(
        f"✓ Test passed (fast mode): Input: {input_size}, Train: {train_size}, Val: {val_size}, "
        f"Overlap: {train_size + val_size - input_size}"
    )


def test_allow_label_overlap_false():
    """Test with allow_label_overlap=False (precise mode)"""
    # Load iris dataset
    dic_data = load_iris(as_frame=True)
    iris_data = dic_data["frame"]

    # Prepare data
    x_train = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].to_numpy()
    y_train = iris_data["target"]

    # Train with precise mode
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm"],
        "eval_method": "holdout",
        "split_type": "stratified",
        "keep_search_state": True,
        "retrain_full": False,
        "auto_augment": False,
        "verbose": 0,
        "allow_label_overlap": False,  # Precise mode
    }
    automl.fit(x_train, y_train, **automl_settings)

    # Check that there's no overlap (or minimal overlap for single-instance classes)
    input_size = len(x_train)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)

    # Verify all classes are represented
    all_labels = set(np.unique(y_train))

    # Should have no overlap or minimal overlap
    overlap = train_size + val_size - input_size
    assert overlap <= len(all_labels), f"Excessive overlap: {overlap}"

    # Verify all classes are represented
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))

    combined_labels = train_labels.union(val_labels)
    assert combined_labels == all_labels, f"Not all labels present. All: {all_labels}, Combined: {combined_labels}"

    print(
        f"✓ Test passed (precise mode): Input: {input_size}, Train: {train_size}, Val: {val_size}, "
        f"Overlap: {overlap}"
    )


def test_uniform_split_with_overlap_control():
    """Test with uniform split and both overlap modes"""
    # Load iris dataset
    dic_data = load_iris(as_frame=True)
    iris_data = dic_data["frame"]

    # Prepare data
    x_train = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].to_numpy()
    y_train = iris_data["target"]

    # Test precise mode with uniform split
    automl = AutoML()
    automl_settings = {
        "max_iter": 5,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm"],
        "eval_method": "holdout",
        "split_type": "uniform",
        "keep_search_state": True,
        "retrain_full": False,
        "auto_augment": False,
        "verbose": 0,
        "allow_label_overlap": False,  # Precise mode
    }
    automl.fit(x_train, y_train, **automl_settings)

    input_size = len(x_train)
    train_size = len(automl._state.X_train)
    val_size = len(automl._state.X_val)

    # Verify all classes are represented
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))
    all_labels = set(np.unique(y_train))

    combined_labels = train_labels.union(val_labels)
    assert combined_labels == all_labels, "Not all labels present with uniform split"

    print(f"✓ Test passed (uniform split): Input: {input_size}, Train: {train_size}, Val: {val_size}")


def test_with_sample_weights():
    """Test label overlap handling with sample weights"""
    # Create a simple dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Create sample weights (giving more weight to some samples)
    sample_weight = np.random.uniform(0.5, 2.0, size=len(y))

    # Test fast mode with sample weights
    automl_fast = AutoML()
    automl_fast.fit(
        X,
        y,
        task="classification",
        metric="accuracy",
        estimator_list=["lgbm"],
        eval_method="holdout",
        split_type="stratified",
        max_iter=3,
        keep_search_state=True,
        retrain_full=False,
        auto_augment=False,
        verbose=0,
        allow_label_overlap=True,  # Fast mode
        sample_weight=sample_weight,
    )

    # Verify all labels present
    train_labels_fast = set(np.unique(automl_fast._state.y_train))
    val_labels_fast = set(np.unique(automl_fast._state.y_val))
    all_labels = set(np.unique(y))

    assert train_labels_fast == all_labels, "Not all labels in train (fast mode with weights)"
    assert val_labels_fast == all_labels, "Not all labels in val (fast mode with weights)"

    # Test precise mode with sample weights
    automl_precise = AutoML()
    automl_precise.fit(
        X,
        y,
        task="classification",
        metric="accuracy",
        estimator_list=["lgbm"],
        eval_method="holdout",
        split_type="stratified",
        max_iter=3,
        keep_search_state=True,
        retrain_full=False,
        auto_augment=False,
        verbose=0,
        allow_label_overlap=False,  # Precise mode
        sample_weight=sample_weight,
    )

    # Verify all labels present
    train_labels_precise = set(np.unique(automl_precise._state.y_train))
    val_labels_precise = set(np.unique(automl_precise._state.y_val))

    combined_labels = train_labels_precise.union(val_labels_precise)
    assert combined_labels == all_labels, "Not all labels present (precise mode with weights)"

    print("✓ Test passed with sample weights (fast and precise modes)")


def test_single_instance_class():
    """Test handling of single-instance classes"""
    # Create imbalanced dataset where one class has only 1 instance
    X = np.random.randn(50, 4)
    y = np.array([0] * 40 + [1] * 9 + [2] * 1)  # Class 2 has only 1 instance

    # Test precise mode - should add single instance to both sets
    automl = AutoML()
    automl.fit(
        X,
        y,
        task="classification",
        metric="accuracy",
        estimator_list=["lgbm"],
        eval_method="holdout",
        split_type="uniform",
        max_iter=3,
        keep_search_state=True,
        retrain_full=False,
        auto_augment=False,
        verbose=0,
        allow_label_overlap=False,  # Precise mode
    )

    # Verify all labels present
    train_labels = set(np.unique(automl._state.y_train))
    val_labels = set(np.unique(automl._state.y_val))
    all_labels = set(np.unique(y))

    # Single-instance class should be in both sets
    combined_labels = train_labels.union(val_labels)
    assert combined_labels == all_labels, "Not all labels present with single-instance class"

    # Check that single-instance class (label 2) is in both sets
    assert 2 in train_labels, "Single-instance class not in train"
    assert 2 in val_labels, "Single-instance class not in val"

    print("✓ Test passed with single-instance class")


if __name__ == "__main__":
    test_allow_label_overlap_true()
    test_allow_label_overlap_false()
    test_uniform_split_with_overlap_control()
    test_with_sample_weights()
    test_single_instance_class()
    print("\n✓ All tests passed!")
