"""Tests for flaml.automl.task.generic_task.GenericTask to improve coverage."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from flaml.automl.task.generic_task import GenericTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides):
    """Create a minimal state-like object."""
    defaults = dict(
        fit_kwargs={},
        fit_kwargs_by_estimator={},
        X_val=None,
        y_val=None,
        groups=None,
        groups_all=None,
        groups_val=None,
        weight_val=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_automl():
    """Create a minimal AutoML-like object."""
    obj = SimpleNamespace(
        _df=False,
        _nrow=0,
        _ndim=0,
        _skip_transform=True,
        _transformer=False,
        _label_transformer=False,
        _X_train_all=None,
        _y_train_all=None,
        _sample_weight_full=None,
        _feature_names_in_=None,
        data_size_full=0,
    )
    return obj


class TestEstimatorsProperty:
    def test_estimators_loaded(self):
        """Lines 46-101: estimators property lazy-loads estimator classes."""
        task = GenericTask("binary")
        estimators = task.estimators
        assert isinstance(estimators, dict)
        assert "lgbm" in estimators
        assert "xgboost" in estimators
        # Second access returns cached
        assert task.estimators is estimators


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------


class TestValidateData:
    def test_1d_numpy_array_reshape(self):
        """Line 125-126: 1D X_train_all gets reshaped to 2D."""
        task = GenericTask("regression")
        automl = _make_automl()
        state = _make_state()
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([0, 1, 0])
        task.validate_data(automl, state, X, y, dataframe=None, label=None)
        assert automl._X_train_all.shape == (3, 1)

    def test_dataframe_label_path(self):
        """Lines 140-156: dataframe+label path."""
        task = GenericTask("regression")
        automl = _make_automl()
        state = _make_state()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
        task.validate_data(automl, state, X_train_all=None, y_train_all=None, dataframe=df, label="target")
        assert automl._df is True
        assert automl._nrow == 3
        assert automl._ndim == 2

    def test_missing_inputs_raises(self):
        """Line 158: neither X+y nor dataframe+label raises ValueError."""
        task = GenericTask("regression")
        automl = _make_automl()
        state = _make_state()
        with pytest.raises(ValueError, match="either X_train"):
            task.validate_data(automl, state, None, None, None, None)

    def test_ts_forecast_dataframe_path(self):
        """Lines 136-138, 151-152: ts_forecast with numpy X converts to DataFrame."""
        task = GenericTask("ts_forecast")
        automl = _make_automl()
        state = _make_state()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        y = np.array([100.0, 200.0, 300.0])
        # Mock _validate_ts_data to avoid full TS validation
        task._validate_ts_data = lambda *a: (pd.DataFrame(a[0]) if len(a) > 1 else a[0], a[1] if len(a) > 1 else a[0])
        task.validate_data(automl, state, X, y, dataframe=None, label=None)
        # numpy input → _df is False initially, but ts path converts X to DataFrame
        assert isinstance(automl._X_train_all, pd.DataFrame)

    def test_ts_forecast_dataframe_label_path(self):
        """Line 151-152: ts_forecast via dataframe+label path."""
        task = GenericTask("ts_forecast")
        automl = _make_automl()
        state = _make_state()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0.1, 0.2, 0.3]})
        task._validate_ts_data = lambda d: d
        task.validate_data(automl, state, X_train_all=None, y_train_all=None, dataframe=df, label="target")
        assert automl._df is True

    def test_sparse_input_skip_transform(self):
        """Line 196-197: sparse X skips transform."""
        from scipy.sparse import csr_matrix

        task = GenericTask("regression")
        automl = _make_automl()
        automl._skip_transform = False
        state = _make_state()
        X = csr_matrix(np.array([[1, 0], [0, 1], [1, 1]]))
        y = np.array([0, 1, 2])
        task.validate_data(automl, state, X, y, dataframe=None, label=None)
        assert automl._transformer is False
        assert automl._label_transformer is False


# ---------------------------------------------------------------------------
# _train_test_split
# ---------------------------------------------------------------------------


class TestTrainTestSplit:
    def test_split_with_sample_weight(self):
        """Lines 312-338: split with sample_weight in fit_kwargs (non-spark)."""
        task = GenericTask("binary")
        state = _make_state(fit_kwargs={"sample_weight": np.ones(100)})
        X = np.random.rand(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        X_train, X_val, y_train, y_val = task._train_test_split(state, X, y, split_ratio=0.2, stratify=y)
        assert len(X_train) + len(X_val) == 100
        assert "sample_weight" in state.fit_kwargs
        assert len(state.fit_kwargs["sample_weight"]) == len(X_train)

    def test_split_with_first_and_sample_weight(self):
        """Lines 332-335: split with first != None and sample_weight."""
        task = GenericTask("multiclass")
        weights = np.arange(100, dtype=float)
        state = _make_state(fit_kwargs={"sample_weight": weights})
        X = np.random.rand(100, 3)
        y = np.array([0] * 40 + [1] * 40 + [2] * 20)
        first = np.array([0])  # first index of a rare label
        X_train, X_val, y_train, y_val = task._train_test_split(state, X, y, first=first, split_ratio=0.2, stratify=y)
        assert len(X_train) > 0
        assert len(X_val) > 0

    def test_split_no_weight_no_spark(self):
        """Lines 339-346: no weight, no spark."""
        task = GenericTask("binary")
        state = _make_state()
        X = np.random.rand(50, 2)
        y = np.array([0] * 25 + [1] * 25)
        X_train, X_val, y_train, y_val = task._train_test_split(state, X, y, split_ratio=0.2, stratify=y)
        assert len(X_train) == 40
        assert len(X_val) == 10


# ---------------------------------------------------------------------------
# _handle_missing_labels_fast
# ---------------------------------------------------------------------------


class TestHandleMissingLabelsFast:
    def test_missing_in_train_numpy(self):
        """Lines 413-436: label missing from train added from full set (numpy)."""
        task = GenericTask("multiclass")
        state = _make_state()
        # Full data has labels 0,1,2 but train only has 0,1
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[1], [2], [4], [5]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3]])
        y_val = np.array([2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
        )
        assert 2 in y_train2

    def test_missing_in_val_numpy(self):
        """Lines 454-477: label missing from val added from full set (numpy)."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[3]])
        y_train = np.array([2])
        X_val = np.array([[1], [2], [4], [5]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
        )
        assert 2 in y_val2

    def test_missing_in_both_numpy(self):
        """Covers both train and val missing branches."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = np.array([[1], [2], [3], [4], [5], [6]])
        y_all = np.array([0, 1, 2, 3, 0, 1])
        # Train has 0,1 – missing 2,3
        X_train = np.array([[1], [2], [5], [6]])
        y_train = np.array([0, 1, 0, 1])
        # Val has 2,3 – missing 0,1
        X_val = np.array([[3], [4]])
        y_val = np.array([2, 3])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
        )
        for label in [0, 1, 2, 3]:
            assert label in y_train2
            assert label in y_val2

    def test_missing_labels_with_sample_weight(self):
        """Lines 439-451, 480-496: weight handling in missing-label branches."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([1.0, 2.0, 4.0, 5.0])},
            weight_val=np.array([3.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[1], [2], [4], [5]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3]])
        y_val = np.array([2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
        )
        assert 2 in y_train2
        assert len(state.fit_kwargs["sample_weight"]) == len(y_train2)

    def test_missing_in_val_with_weight(self):
        """Lines 480-496: weight_val updated when label missing from val."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([3.0])},
            weight_val=np.array([1.0, 2.0, 4.0, 5.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[3]])
        y_train = np.array([2])
        X_val = np.array([[1], [2], [4], [5]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
        )
        assert 2 in y_val2
        assert len(state.weight_val) == len(y_val2)

    def test_missing_labels_df_path(self):
        """DataFrame path for missing labels (data_is_df=True)."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
        y_all = pd.Series([0, 1, 2, 0, 1])
        X_train = pd.DataFrame({"f": [1, 2, 4, 5]})
        y_train = pd.Series([0, 1, 0, 1])
        X_val = pd.DataFrame({"f": [3]})
        y_val = pd.Series([2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_fast(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=True,
        )
        assert 2 in y_train2.values


# ---------------------------------------------------------------------------
# _handle_missing_labels_no_overlap
# ---------------------------------------------------------------------------


class TestHandleMissingLabelsNoOverlap:
    def test_single_instance_class_missing_in_train(self):
        """Lines 560-577: single-instance class added to train (unavoidable overlap)."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        # Label 2 has 1 instance, all in val but not train
        X_train = np.array([[1], [2], [4], [5]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3]])
        y_val = np.array([2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_train2

    def test_multi_instance_class_missing_in_train(self):
        """Lines 593-650: multi-instance class re-split to avoid overlap."""
        task = GenericTask("multiclass")
        state = _make_state()
        # Label 2 has 3 instances, all in val
        X_all = np.array([[1], [2], [3], [4], [5], [6], [7]])
        y_all = np.array([0, 1, 2, 2, 2, 0, 1])
        X_train = np.array([[1], [2], [6], [7]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3], [4], [5]])
        y_val = np.array([2, 2, 2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_train2
        # val should still have some label-2 instances (remaining)
        assert 2 in y_val2

    def test_single_instance_class_missing_in_val(self):
        """Lines 699-732: single-instance class added to val (unavoidable overlap)."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[3]])
        y_train = np.array([2])
        X_val = np.array([[1], [2], [4], [5]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_val2

    def test_multi_instance_class_missing_in_val(self):
        """Lines 733-790: multi-instance class re-split to avoid overlap."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = np.array([[1], [2], [3], [4], [5], [6], [7]])
        y_all = np.array([0, 1, 2, 2, 2, 0, 1])
        X_train = np.array([[3], [4], [5]])
        y_train = np.array([2, 2, 2])
        X_val = np.array([[1], [2], [6], [7]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_val2
        assert 2 in y_train2

    def test_no_overlap_with_sample_weight_train(self):
        """Lines 580-592, 653-686: weight handling for missing-in-train."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([1.0, 2.0, 6.0, 7.0])},
            weight_val=np.array([3.0, 4.0, 5.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5], [6], [7]])
        y_all = np.array([0, 1, 2, 2, 2, 0, 1])
        X_train = np.array([[1], [2], [6], [7]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3], [4], [5]])
        y_val = np.array([2, 2, 2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_train2

    def test_no_overlap_with_sample_weight_val(self):
        """Lines 793-821: weight handling for missing-in-val."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([3.0, 4.0, 5.0])},
            weight_val=np.array([1.0, 2.0, 6.0, 7.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5], [6], [7]])
        y_all = np.array([0, 1, 2, 2, 2, 0, 1])
        X_train = np.array([[3], [4], [5]])
        y_train = np.array([2, 2, 2])
        X_val = np.array([[1], [2], [6], [7]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_val2

    def test_no_overlap_single_instance_with_weight_in_train(self):
        """Single-instance class missing in train with sample weights."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([1.0, 2.0, 4.0, 5.0])},
            weight_val=np.array([3.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[1], [2], [4], [5]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[3]])
        y_val = np.array([2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_train2

    def test_no_overlap_single_instance_with_weight_in_val(self):
        """Single-instance class missing in val with sample weights."""
        task = GenericTask("multiclass")
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(
            fit_kwargs={"sample_weight": np.array([3.0])},
            weight_val=np.array([1.0, 2.0, 4.0, 5.0]),
            sample_weight_all=weights,
        )
        X_all = np.array([[1], [2], [3], [4], [5]])
        y_all = np.array([0, 1, 2, 0, 1])
        X_train = np.array([[3]])
        y_train = np.array([2])
        X_val = np.array([[1], [2], [4], [5]])
        y_val = np.array([0, 1, 0, 1])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=False,
            split_ratio=0.2,
        )
        assert 2 in y_val2

    def test_no_overlap_dataframe_path(self):
        """DataFrame path for no-overlap (data_is_df=True)."""
        task = GenericTask("multiclass")
        state = _make_state()
        X_all = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7]})
        y_all = pd.Series([0, 1, 2, 2, 2, 0, 1])
        X_train = pd.DataFrame({"f": [1, 2, 6, 7]})
        y_train = pd.Series([0, 1, 0, 1])
        X_val = pd.DataFrame({"f": [3, 4, 5]})
        y_val = pd.Series([2, 2, 2])

        X_train2, X_val2, y_train2, y_val2 = task._handle_missing_labels_no_overlap(
            state,
            X_train,
            X_val,
            y_train,
            y_val,
            X_all,
            y_all,
            is_spark_dataframe=False,
            data_is_df=True,
            split_ratio=0.2,
        )
        assert 2 in y_train2.values


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_prepare_data_classification_holdout(self):
        """Lines 965-1007: classification holdout with label handling."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(100, 3)
        y = np.array([0] * 40 + [1] * 40 + [2] * 20)
        result = task.prepare_data(
            state,
            X,
            y,
            auto_augment=True,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert result is None  # holdout returns None
        assert state.X_train is not None
        assert state.X_val is not None

    def test_prepare_data_with_sample_weight_shuffle(self):
        """Lines 887-900: shuffle with sample_weight_full."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(50, 2)
        y = np.array([0] * 20 + [1] * 20 + [2] * 10)
        weights = np.ones(50)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=weights,
        )
        assert hasattr(state, "sample_weight_all")

    def test_prepare_data_with_sample_weight_series(self):
        """Line 900: pd.Series sample_weight reset_index path."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = pd.DataFrame({"a": range(50), "b": range(50)})
        y = pd.Series([0] * 20 + [1] * 20 + [2] * 10)
        weights = pd.Series(np.ones(50))
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=True,
            sample_weight_full=weights,
        )
        assert isinstance(state.sample_weight_all, pd.Series)

    def test_prepare_data_regression_holdout(self):
        """Lines 1009-1012: regression holdout path."""
        task = GenericTask("regression")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="uniform",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert state.X_val is not None

    def test_prepare_data_cv_stratified(self):
        """Lines 1030-1039: stratified CV creates RepeatedStratifiedKFold."""
        from sklearn.model_selection import RepeatedStratifiedKFold

        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(100, 3)
        y = np.array([0] * 40 + [1] * 40 + [2] * 20)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert isinstance(state.kf, RepeatedStratifiedKFold)

    def test_prepare_data_cv_uniform(self):
        """Lines 1061-1063: uniform CV creates RepeatedKFold."""
        from sklearn.model_selection import RepeatedKFold

        task = GenericTask("regression")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type="uniform",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert isinstance(state.kf, RepeatedKFold)

    def test_prepare_data_time_split_no_forecast(self):
        """Lines 1058-1059: time split for non-forecast uses plain TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit

        task = GenericTask("regression")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type="time",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert isinstance(state.kf, TimeSeriesSplit)

    def test_prepare_data_ts_forecast_cv(self):
        """Lines 1042-1053: ts_forecast with period adjustment."""
        from sklearn.model_selection import TimeSeriesSplit

        task = GenericTask("ts_forecast")
        state = _make_state(
            X_val=None,
            y_val=None,
            fit_kwargs={"period": 5},
        )
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type="time",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert isinstance(state.kf, TimeSeriesSplit)

    def test_prepare_data_ts_forecast_cv_small_data(self):
        """Lines 1046-1052: period adjustment reduces n_splits when data is small."""
        from sklearn.model_selection import TimeSeriesSplit

        task = GenericTask("ts_forecast")
        state = _make_state(
            X_val=None,
            y_val=None,
            fit_kwargs={"period": 10},
        )
        X = np.random.rand(35, 3)
        y = np.random.rand(35)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type="time",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert isinstance(state.kf, TimeSeriesSplit)

    def test_prepare_data_y_train_dataframe_to_series(self):
        """Lines 1002-1007: y_train DataFrame converted to Series after split."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = pd.DataFrame({"a": range(100), "b": range(100)})
        y = pd.Series([0] * 40 + [1] * 40 + [2] * 20, name="target")
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=True,
            sample_weight_full=None,
        )
        assert isinstance(state.y_train, pd.Series)
        assert isinstance(state.y_val, pd.Series)

    def test_prepare_data_no_overlap(self):
        """Lines 987-1000: allow_label_overlap=False uses no-overlap handler."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(100, 3)
        y = np.array([0] * 40 + [1] * 40 + [2] * 20)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
            allow_label_overlap=False,
        )
        assert state.X_train is not None

    def test_prepare_data_auto_augment_rare_class(self):
        """Lines 848-881: auto_augment augments rare classes."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(50, 3)
        # class 2 has only 5 instances (rare < 20)
        y = np.array([0] * 20 + [1] * 25 + [2] * 5)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=True,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        # After augmentation, y_train_all should have more instances of class 2
        assert len(state.y_train_all) > 50

    def test_prepare_data_auto_augment_rare_class_df(self):
        """Line 873, 877: auto_augment with DataFrame/Series (rare class augmentation)."""
        task = GenericTask("multiclass")
        state = _make_state(X_val=None, y_val=None)
        X = pd.DataFrame({"a": range(50), "b": range(50, 100)})
        # class 2 has only 5 instances (rare < 20)
        y = pd.Series([0] * 20 + [1] * 25 + [2] * 5)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=True,
            eval_method="holdout",
            split_type="stratified",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=True,
            sample_weight_full=None,
        )
        assert len(state.y_train_all) > 50

    def test_prepare_data_holdout_time_split_no_weight(self):
        """Lines 934-939: time split holdout without sample_weight (regression)."""
        task = GenericTask("regression")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="uniform",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert state.X_val is not None

    def test_prepare_data_holdout_with_sample_weight_full_shuffle(self):
        """Lines 887-900: shuffle path with sample_weight_full triggers weight shuffle."""
        task = GenericTask("regression")
        weights = np.arange(60, dtype=float)
        state = _make_state(X_val=None, y_val=None, fit_kwargs={})
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="holdout",
            split_type="uniform",
            split_ratio=0.2,
            n_splits=5,
            data_is_df=False,
            sample_weight_full=weights,
        )
        assert hasattr(state, "sample_weight_all")
        assert state.X_val is not None

    def test_prepare_data_custom_splitter(self):
        """Lines 1064-1066: custom splitter object."""
        from sklearn.model_selection import KFold

        task = GenericTask("regression")
        state = _make_state(X_val=None, y_val=None)
        X = np.random.rand(60, 3)
        y = np.random.rand(60)
        custom_kf = KFold(n_splits=3)
        task.prepare_data(
            state,
            X,
            y,
            auto_augment=False,
            eval_method="cv",
            split_type=custom_kf,
            split_ratio=0.2,
            n_splits=3,
            data_is_df=False,
            sample_weight_full=None,
        )
        assert state.kf is custom_kf


# ---------------------------------------------------------------------------
# decide_split_type
# ---------------------------------------------------------------------------


class TestDecideSplitType:
    def test_classification_auto(self):
        """Line 1092: auto → stratified for classification."""
        task = GenericTask("classification")
        y = np.array([0] * 30 + [1] * 30)
        result = task.decide_split_type("auto", y, {})
        assert result == "stratified"
        assert task.name == "binary"

    def test_classification_auto_with_groups(self):
        """Line 1092: auto with groups → group."""
        task = GenericTask("classification")
        y = np.array([0] * 30 + [1] * 30)
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = task.decide_split_type("auto", y, {}, groups=groups)
        assert result == "group"

    def test_classification_explicit(self):
        """Line 1092: explicit split_type returned as-is."""
        task = GenericTask("binary")
        y = np.array([0] * 30 + [1] * 30)
        assert task.decide_split_type("uniform", y, {}) == "uniform"
        assert task.decide_split_type("time", y, {}) == "time"

    def test_regression_auto(self):
        """Line 1096: regression auto → uniform."""
        task = GenericTask("regression")
        y = np.random.rand(60)
        assert task.decide_split_type("auto", y, {}) == "uniform"

    def test_regression_time(self):
        """Line 1096: regression explicit time."""
        task = GenericTask("regression")
        y = np.random.rand(60)
        assert task.decide_split_type("time", y, {}) == "time"

    def test_rank(self):
        """Lines 1098-1101: rank task → group."""
        task = GenericTask("rank")
        y = np.random.rand(60)
        groups = np.arange(60)
        assert task.decide_split_type("auto", y, {}, groups=groups) == "group"

    def test_rank_no_groups_raises(self):
        """Line 1099: rank without groups raises."""
        task = GenericTask("rank")
        y = np.random.rand(60)
        with pytest.raises(AssertionError, match="groups must be specified"):
            task.decide_split_type("auto", y, {})

    def test_nlg(self):
        """Lines 1103-1105: summarization → uniform."""
        task = GenericTask("summarization")
        y = np.array(["a"] * 60)
        assert task.decide_split_type("auto", y, {}) == "uniform"

    def test_nlg_explicit_time(self):
        """Line 1104: summarization explicit time."""
        task = GenericTask("summarization")
        y = np.array(["a"] * 60)
        assert task.decide_split_type("time", y, {}) == "time"

    def test_custom_splitter_object(self):
        """Lines 1081-1088: custom splitter object returned."""
        from sklearn.model_selection import KFold

        task = GenericTask("binary")
        y = np.array([0] * 30 + [1] * 30)
        kf = KFold(n_splits=3)
        result = task.decide_split_type(kf, y, {})
        assert result is kf

    def test_multiclass_classification_name_update(self):
        """Line 1080: classification with >2 labels becomes multiclass."""
        task = GenericTask("classification")
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        task.decide_split_type("auto", y, {})
        assert task.name == "multiclass"


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_list_input_single_row(self):
        """Lines 1108-1123: List input converted to DataFrame."""
        task = GenericTask("seq-classification")
        transformer = SimpleNamespace(_str_columns=["col0", "col1"], transform=lambda x: x)
        result = task.preprocess(["hello", "world"], transformer=transformer)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col0", "col1"]

    def test_list_of_lists_input(self):
        """Lines 1110-1111: list-of-lists transposed."""
        task = GenericTask("seq-classification")
        transformer = SimpleNamespace(_str_columns=["col0", "col1"], transform=lambda x: x)
        result = task.preprocess([["a", "b"], ["c", "d"]], transformer=transformer)
        assert isinstance(result, pd.DataFrame)

    def test_int_input(self):
        """Lines 1124-1125: int input returned as-is."""
        task = GenericTask("regression")
        assert task.preprocess(42) == 42

    def test_sparse_input(self):
        """Lines 1128-1129: sparse input converted to csr."""
        from scipy.sparse import csc_matrix

        task = GenericTask("regression")
        X = csc_matrix(np.array([[1, 0], [0, 1]]))
        result = task.preprocess(X)
        assert result.format == "csr"

    def test_ndarray_no_transform(self):
        """Numpy array passed through without transformer."""
        task = GenericTask("regression")
        X = np.array([[1, 2], [3, 4]])
        result = task.preprocess(X)
        np.testing.assert_array_equal(result, X)

    def test_list_input_index_error(self):
        """Line 1122-1123: IndexError when test data has more columns."""
        task = GenericTask("seq-classification")
        transformer = SimpleNamespace(_str_columns=["col0"])
        with pytest.raises(IndexError, match="more columns"):
            task.preprocess(["a", "b", "c"], transformer=transformer)


# ---------------------------------------------------------------------------
# default_estimator_list
# ---------------------------------------------------------------------------


class TestDefaultEstimatorList:
    def test_auto_classification(self):
        """Default estimator list for binary classification."""
        task = GenericTask("binary")
        result = task.default_estimator_list("auto")
        assert "lgbm" in result
        assert "lrl1" in result
        assert all(not e.endswith("_spark") for e in result)

    def test_auto_regression(self):
        """Default estimator list for regression."""
        task = GenericTask("regression")
        result = task.default_estimator_list("auto")
        assert "lgbm" in result
        assert "lrl1" not in result

    def test_auto_rank(self):
        """Lines 1299-1300: rank task estimators."""
        task = GenericTask("rank")
        result = task.default_estimator_list("auto")
        assert "lgbm" in result
        assert "xgboost" in result

    def test_auto_nlp(self):
        """Lines 1301-1302: NLP task returns transformer."""
        task = GenericTask("seq-classification")
        result = task.default_estimator_list("auto")
        assert result == ["transformer"]

    def test_custom_list_filters_spark(self):
        """Lines 1286-1297: non-spark filters out _spark estimators."""
        task = GenericTask("binary")
        result = task.default_estimator_list(["lgbm", "lgbm_spark", "rf"])
        assert "lgbm_spark" not in result
        assert "lgbm" in result

    def test_custom_list_all_spark_raises(self):
        """Lines 1288-1292: all-spark list raises for non-spark."""
        task = GenericTask("binary")
        with pytest.raises(ValueError, match="Non-spark"):
            task.default_estimator_list(["lgbm_spark", "rf_spark"])

    def test_custom_list_warns_on_filter(self):
        """Lines 1293-1297: warning when some estimators filtered."""
        task = GenericTask("binary")
        result = task.default_estimator_list(["lgbm", "lgbm_spark"])
        assert result == ["lgbm"]

    def test_spark_dataframe_filters_non_spark(self):
        """Lines 1272-1284: spark dataframe filters non-spark estimators."""
        task = GenericTask("binary")
        result = task.default_estimator_list(["lgbm", "lgbm_spark"], is_spark_dataframe=True)
        assert result == ["lgbm_spark"]

    def test_spark_dataframe_all_non_spark_raises(self):
        """Lines 1275-1279: all non-spark list with spark raises."""
        task = GenericTask("binary")
        with pytest.raises(ValueError, match="Spark dataframes"):
            task.default_estimator_list(["lgbm", "rf"], is_spark_dataframe=True)

    def test_spark_dataframe_warns_on_filter(self):
        """Lines 1280-1284: warning when some estimators filtered for spark."""
        task = GenericTask("binary")
        result = task.default_estimator_list(["lgbm", "lgbm_spark", "rf_spark"], is_spark_dataframe=True)
        assert "lgbm" not in result

    def test_auto_with_catboost(self):
        """Lines 1316-1321: catboost added if available."""
        task = GenericTask("binary")
        result = task.default_estimator_list("auto")
        # catboost may or may not be installed
        try:
            import catboost

            assert "catboost" in result
        except ImportError:
            assert "catboost" not in result


# ---------------------------------------------------------------------------
# default_metric
# ---------------------------------------------------------------------------


class TestDefaultMetric:
    def test_explicit_metric(self):
        """Line 1350-1351: explicit metric returned as-is."""
        task = GenericTask("binary")
        assert task.default_metric("f1") == "f1"

    def test_binary(self):
        """Line 1357-1358: binary → roc_auc."""
        task = GenericTask("binary")
        assert task.default_metric("auto") == "roc_auc"

    def test_multiclass(self):
        """Lines 1359-1360: multiclass → log_loss."""
        task = GenericTask("multiclass")
        assert task.default_metric("auto") == "log_loss"

    def test_ts_forecast(self):
        """Lines 1361-1362: ts_forecast → mape."""
        task = GenericTask("ts_forecast")
        assert task.default_metric("auto") == "mape"

    def test_rank(self):
        """Lines 1363-1364: rank → ndcg."""
        task = GenericTask("rank")
        assert task.default_metric("auto") == "ndcg"

    def test_regression(self):
        """Lines 1365-1366: regression → r2."""
        task = GenericTask("regression")
        assert task.default_metric("auto") == "r2"

    def test_nlp_seq_classification(self):
        """Lines 1353-1356: NLP task uses HF default metric."""
        task = GenericTask("seq-classification")
        result = task.default_metric("auto")
        assert isinstance(result, str)

    def test_nlp_summarization(self):
        """Lines 1353-1356: NLP summarization default metric."""
        task = GenericTask("summarization")
        result = task.default_metric("auto")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# evaluate_model_CV (basic non-spark path)
# ---------------------------------------------------------------------------


class TestEvaluateModelCV:
    def test_basic_cv_classification(self):
        """Lines 1136-1267: basic CV evaluation for classification."""
        from sklearn.model_selection import RepeatedStratifiedKFold

        task = GenericTask("binary")
        X = np.random.rand(40, 3)
        y = np.array([0] * 20 + [1] * 20)
        kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.5, {"log_loss": 0.5}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="log_loss",
                best_val_loss=1.0,
            )
        assert val_loss is not None
        assert mock_estimator.cleanup.call_count == 2

    def test_cv_with_sample_weight(self):
        """Lines 1168-1170, 1221-1225: CV with sample weights."""
        from sklearn.model_selection import RepeatedKFold

        task = GenericTask("regression")
        X = np.random.rand(40, 3)
        y = np.random.rand(40)
        kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=42)

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()
        weights = np.ones(40)

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.3, {"r2": 0.7}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="r2",
                best_val_loss=1.0,
                fit_kwargs={"sample_weight": weights},
            )
        assert val_loss is not None

    def test_cv_with_dataframe(self):
        """Lines 1214-1216: CV with DataFrame input."""
        from sklearn.model_selection import RepeatedKFold

        task = GenericTask("regression")
        X = pd.DataFrame({"a": range(40), "b": range(40)})
        y = np.random.rand(40)
        kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=42)

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.3, {"r2": 0.7}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="r2",
                best_val_loss=1.0,
            )
        assert val_loss is not None

    def test_cv_time_series_split(self):
        """Lines 1193-1194: TimeSeriesSplit path."""
        from sklearn.model_selection import TimeSeriesSplit

        task = GenericTask("regression")
        X = np.random.rand(40, 3)
        y = np.random.rand(40)
        kf = TimeSeriesSplit(n_splits=2)

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.3, {"r2": 0.7}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="r2",
                best_val_loss=1.0,
            )
        assert val_loss is not None

    def test_cv_group_kfold(self):
        """Lines 1189-1192: GroupKFold path."""
        from sklearn.model_selection import GroupKFold

        task = GenericTask("regression")
        X = np.random.rand(40, 3)
        y = np.random.rand(40)
        kf = GroupKFold(n_splits=2)
        groups = np.array([0] * 20 + [1] * 20)
        kf.groups = groups

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.3, {"r2": 0.7}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="r2",
                best_val_loss=1.0,
            )
        assert val_loss is not None

    def test_cv_custom_splitter(self):
        """Lines 1196-1199: custom splitter fallback."""
        from sklearn.model_selection import KFold

        task = GenericTask("regression")
        X = np.random.rand(40, 3)
        y = np.random.rand(40)
        kf = KFold(n_splits=2)

        mock_estimator = MagicMock()
        mock_estimator.cleanup = MagicMock()

        with patch("flaml.automl.task.generic_task.get_val_loss") as mock_val:
            mock_val.return_value = (0.3, {"r2": 0.7}, 1.0, 0.1)
            val_loss, metric, train_time, pred_time = task.evaluate_model_CV(
                config={},
                estimator=mock_estimator,
                X_train_all=X,
                y_train_all=y,
                budget=10,
                kf=kf,
                eval_metric="r2",
                best_val_loss=1.0,
            )
        assert val_loss is not None
