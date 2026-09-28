"""Tests to improve coverage for flaml/automl/ml.py."""

import numpy as np
import pandas as pd
import pytest

from flaml.automl.ml import (
    default_cv_score_agg_func,
    get_y_pred,
    is_in_sklearn_metric_name_set,
    is_min_metric,
    metric_loss_score,
    sklearn_metric_loss_score,
    to_numpy,
)


class TestImportFallbacks:
    """Cover import fallback branches (lines 31-37)."""

    def test_sklearn_metrics_available(self):
        """Lines 31-32: sklearn metrics imported."""
        from sklearn.metrics import accuracy_score

        assert callable(accuracy_score)

    def test_featurization_import(self):
        """Lines 34-37: Featurization import attempted."""
        from flaml.automl import ml as _mod

        # Featurization may or may not be available
        assert hasattr(_mod, "Featurization")


class TestIsInSklearnMetricNameSet:
    def test_standard_metrics(self):
        """Line 192: is_in_sklearn_metric_name_set."""
        assert is_in_sklearn_metric_name_set("accuracy")
        assert is_in_sklearn_metric_name_set("r2")
        assert is_in_sklearn_metric_name_set("ndcg")
        assert is_in_sklearn_metric_name_set("ndcg@5")
        assert not is_in_sklearn_metric_name_set("unknown_metric")


class TestIsMinMetric:
    def test_min_metrics(self):
        assert is_min_metric("rmse")
        assert is_min_metric("mae")
        assert is_min_metric("log_loss")
        assert not is_min_metric("accuracy")


class TestSklearnMetricLossScore:
    """Cover various branches in sklearn_metric_loss_score (lines 282-296)."""

    def test_ndcg_with_k(self):
        """Cover lines 282-294: ndcg@k with groups."""
        y_true = [3, 2, 1, 0, 3, 2]
        y_pred = [3.1, 2.1, 0.9, 0.1, 2.9, 2.0]
        groups = [0, 0, 0, 1, 1, 1]
        score = sklearn_metric_loss_score("ndcg@3", y_true, y_pred, groups=groups)
        assert isinstance(score, float)

    def test_ndcg_without_k(self):
        """Cover line 296: ndcg without @k."""
        y_true = [3, 2, 1, 0]
        y_pred = [3.0, 2.0, 1.0, 0.0]
        score = sklearn_metric_loss_score("ndcg", y_true, y_pred)
        assert isinstance(score, float)

    def test_accuracy(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 1, 0]
        score = sklearn_metric_loss_score("accuracy", y_true, y_pred)
        assert score == 0.0  # perfect => 1 - 1 = 0

    def test_r2(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        score = sklearn_metric_loss_score("r2", y_true, y_pred)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_rmse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        score = sklearn_metric_loss_score("rmse", y_true, y_pred)
        assert score == 0.0

    def test_mae(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        score = sklearn_metric_loss_score("mae", y_true, y_pred)
        assert score == pytest.approx(0.5)

    def test_log_loss(self):
        y_true = [0, 1, 0, 1]
        y_pred = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
        score = sklearn_metric_loss_score("log_loss", y_pred, y_true, labels=[0, 1])
        assert score > 0

    def test_f1(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        score = sklearn_metric_loss_score("f1", y_true, y_pred)
        assert 0 <= score <= 1

    def test_micro_f1(self):
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 2, 0]
        score = sklearn_metric_loss_score("micro_f1", y_true, y_pred)
        assert score == 0.0

    def test_macro_f1(self):
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 2, 0]
        score = sklearn_metric_loss_score("macro_f1", y_true, y_pred)
        assert score == 0.0

    def test_mape(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.2, 3.3]
        score = sklearn_metric_loss_score("mape", y_true, y_pred)
        assert score > 0

    def test_mse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        score = sklearn_metric_loss_score("mse", y_true, y_pred)
        assert score == 0.0

    def test_ap(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0.1, 0.9, 0.8, 0.2]
        score = sklearn_metric_loss_score("ap", y_pred, y_true)
        assert 0 <= score <= 1

    def test_roc_auc_ovr(self):
        """Cover roc_auc_ovr branch."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        score = sklearn_metric_loss_score("roc_auc_ovr", y_pred, y_true)
        assert isinstance(score, float)

    def test_roc_auc_ovo(self):
        """Cover roc_auc_ovo branch."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        score = sklearn_metric_loss_score("roc_auc_ovo", y_pred, y_true)
        assert isinstance(score, float)

    def test_roc_auc_value_error(self):
        """Cover roc_auc branch (line 242): roc_auc with valid input."""
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.9, 0.2, 0.8]
        score = sklearn_metric_loss_score("roc_auc", y_pred, y_true)
        assert 0 <= score <= 1


class TestMetricLossScore:
    """Cover metric_loss_score for huggingface fallback (lines 141-188)."""

    def test_unknown_metric_raises(self):
        """Cover lines 164-172: ImportError/ValueError for unknown metric."""
        y_true = [0, 1]
        y_pred = [0, 1]
        with pytest.raises((ValueError, Exception)):
            metric_loss_score("totally_fake_metric_xyz_12345", y_true, y_pred)


class TestToNumpy:
    """Cover to_numpy (lines 325-331)."""

    def test_series_input(self):
        """Lines 326-327: Series branch."""
        s = pd.Series([1, 2, 3])
        result = to_numpy(s)
        assert result.shape == (3, 1)

    def test_dataframe_input(self):
        """Lines 326-327: DataFrame branch."""
        df = pd.DataFrame({"a": [1, 2]})
        result = to_numpy(df)
        assert result.shape == (2, 1)


class TestDefaultCvScoreAggFunc:
    """Cover default_cv_score_agg_func (lines 600-615)."""

    def test_dict_metrics(self):
        """Lines 606-607: dict aggregation."""
        val_losses = [0.2, 0.3, 0.5]
        log_metrics = [{"acc": 0.8}, {"acc": 0.7}, {"acc": 0.5}]
        loss, metrics = default_cv_score_agg_func(val_losses, log_metrics)
        assert loss == pytest.approx(1.0 / 3.0)
        assert "acc" in metrics

    def test_scalar_metrics(self):
        """Lines 608-609: scalar aggregation."""
        val_losses = [0.2, 0.4]
        log_metrics = [0.8, 0.6]
        loss, metrics = default_cv_score_agg_func(val_losses, log_metrics)
        assert loss == pytest.approx(0.3)
        assert metrics == pytest.approx(0.7)

    def test_none_metrics(self):
        """Lines 604-605: first fold initializes."""
        val_losses = [0.5]
        log_metrics = [{"acc": 0.5}]
        loss, metrics = default_cv_score_agg_func(val_losses, log_metrics)
        assert loss == 0.5

    def test_empty_metric_returns_none(self):
        """Line 610: metrics_to_log is falsy."""
        val_losses = [0.5]
        log_metrics = [None]
        loss, metrics = default_cv_score_agg_func(val_losses, log_metrics)
        assert metrics is None


class TestGetYPred:
    """Cover get_y_pred (lines 300-322)."""

    def test_predict_branch(self):
        """Lines 316-317: default predict branch."""
        from unittest.mock import MagicMock

        from flaml.automl.task.factory import task_factory

        estimator = MagicMock()
        estimator.predict.return_value = np.array([0, 1, 0])
        task = task_factory("classification")
        result = get_y_pred(estimator, np.zeros((3, 2)), "accuracy", task)
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_roc_auc_binary_branch(self):
        """Lines 301-306: roc_auc binary predict_proba."""
        from unittest.mock import MagicMock

        from flaml.automl.task.factory import task_factory

        estimator = MagicMock()
        estimator.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
        task = task_factory("binary")
        result = get_y_pred(estimator, np.zeros((3, 2)), "roc_auc", task)
        np.testing.assert_array_almost_equal(result, [0.8, 0.4, 0.7])

    def test_log_loss_branch(self):
        """Lines 307-315: log_loss predict_proba."""
        from unittest.mock import MagicMock

        from flaml.automl.task.factory import task_factory

        estimator = MagicMock()
        proba = np.array([[0.2, 0.8], [0.6, 0.4]])
        estimator.predict_proba.return_value = proba
        task = task_factory("multiclass")
        result = get_y_pred(estimator, np.zeros((2, 2)), "log_loss", task)
        np.testing.assert_array_equal(result, proba)

    def test_predict_returns_series(self):
        """Lines 319-320: result is pandas Series."""
        from unittest.mock import MagicMock

        from flaml.automl.task.factory import task_factory

        estimator = MagicMock()
        estimator.predict.return_value = pd.Series([0, 1, 0])
        task = task_factory("classification")
        result = get_y_pred(estimator, np.zeros((3, 2)), "accuracy", task)
        assert isinstance(result, np.ndarray)
