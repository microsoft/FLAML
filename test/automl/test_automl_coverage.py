"""Tests to improve coverage for flaml/automl/automl.py."""

import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from flaml import AutoML


class TestAutoMLImportFallbacks:
    """Cover import fallback branches at module top (lines 52-90)."""

    def test_mlflow_import_is_attempted(self):
        """Lines 56-59: mlflow import try/except."""
        # mlflow may or may not be installed; just ensure the module loads
        from flaml.automl import automl as _mod

        assert hasattr(_mod, "mlflow")

    def test_ray_available_flag(self):
        """Lines 86-92: ray_available flag."""
        from flaml.automl import automl as _mod

        assert isinstance(_mod.ray_available, bool)


class TestAutoMLValidateMetric:
    """Cover _validate_metric_parameter (line 553+)."""

    def test_auto_is_valid(self):
        AutoML._validate_metric_parameter("auto", allow_auto=True)

    def test_callable_is_valid(self):
        AutoML._validate_metric_parameter(lambda: None, allow_auto=True)

    def test_string_metric_is_valid(self):
        AutoML._validate_metric_parameter("accuracy", allow_auto=False)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="must be either a string or a callable"):
            AutoML._validate_metric_parameter(123, allow_auto=False)


class TestAutoMLGetSetState:
    """Cover __getstate__ / __setstate__ (lines 501-551)."""

    def test_pickle_roundtrip_no_mlflow_integration(self):
        """Lines 540-551: __setstate__ restores mlflow_integration attrs."""
        automl = AutoML(task="classification")
        data = pickle.dumps(automl)
        restored = pickle.loads(data)
        assert hasattr(restored, "_settings")

    def test_getstate_strips_mlflow_integration(self):
        """Lines 510-531: __getstate__ cleans mlflow_integration."""
        automl = AutoML(task="classification")

        class FakeMLflowIntegration:
            futures = {"some_future": True}
            futures_log_model = {"some_future": True}

            def train_func():
                return None

            mlflow_client = "client"

        automl.mlflow_integration = FakeMLflowIntegration()
        state = automl.__getstate__()
        mi = state["mlflow_integration"]
        assert mi.futures == {}
        assert mi.futures_log_model == {}
        assert mi.train_func is None
        assert mi.mlflow_client is None

    def test_setstate_with_mlflow_integration(self):
        """Lines 537-551: __setstate__ with mlflow_integration present."""
        automl = AutoML(task="classification")

        class FakeMLflowIntegration:
            pass

        automl.mlflow_integration = FakeMLflowIntegration()
        state = automl.__getstate__()
        new_automl = AutoML.__new__(AutoML)
        new_automl.__setstate__(state)
        mi = new_automl.mlflow_integration
        assert mi.futures == {}
        assert mi.futures_log_model == {}
        assert mi.train_func is None


class TestAutoMLProperties:
    """Cover property accessors (lines 803, 819, etc.)."""

    def test_supported_metrics(self):
        """Line 803: supported_metrics returns 3-tuple (it's a property)."""
        automl = AutoML(task="classification")
        result = automl.supported_metrics
        assert len(result) == 3

    def test_label_transformer_before_fit(self):
        """Line 819: label_transformer returns None before fit."""
        automl = AutoML(task="classification")
        assert automl.label_transformer is None

    def test_feature_transformer_before_fit(self):
        """Lines 806-814: feature_transformer before fit."""
        automl = AutoML(task="classification")
        result = automl.feature_transformer
        assert result is None

    def test_model_before_fit(self):
        """Line 591: model returns None before fit."""
        automl = AutoML(task="classification")
        assert automl.model is None

    def test_classes_before_fit(self):
        """Lines 822-830: classes_ returns None before fit."""
        automl = AutoML(task="classification")
        assert automl.classes_ is None


class TestAutoMLPreprocess:
    """Cover preprocess method (lines 984-989)."""

    def test_preprocess_before_fit_raises(self):
        """Line 985: preprocess raises if not fitted."""
        automl = AutoML.__new__(AutoML)
        with pytest.raises(AttributeError, match="not been fitted"):
            automl.preprocess(pd.DataFrame({"a": [1, 2]}))


class TestAutoMLPredictBeforeFit:
    """Cover predict/predict_proba with no trained estimator (lines 938-939)."""

    def test_predict_proba_no_estimator(self):
        """Lines 938-939: predict_proba returns None when no estimator."""
        automl = AutoML(task="classification")
        result = automl.predict_proba(pd.DataFrame({"a": [1, 2]}))
        assert result is None


class TestAutoMLPickle:
    """Cover pickle/load_pickle methods."""

    def test_pickle_and_load(self):
        """Cover basic pickle flow."""
        automl = AutoML(task="classification")
        # Set minimal attributes needed by pickle()
        automl.estimator_list = []
        automl._search_states = {}
        automl.mlflow_integration = None
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
        try:
            automl.pickle(fname)
            loaded = AutoML.load_pickle(fname, load_spark_models=False)
            assert hasattr(loaded, "_settings")
        finally:
            import os

            os.unlink(fname)


class TestAutoMLInit:
    """Cover __init__ settings branches."""

    def test_use_ray_and_spark_both_true_raises(self):
        """Line 489: ValueError when both use_ray and use_spark are True."""
        with pytest.raises(ValueError, match="use_ray and use_spark cannot be both True"):
            AutoML(task="classification", use_ray=True, use_spark=True)


class TestAutoMLSizeFunction:
    """Cover the module-level size function (lines 95-104)."""

    def test_size_function(self):
        from flaml.automl.automl import size

        class FakeLearner:
            @staticmethod
            def size(config):
                return 42

        result = size({"my_learner": FakeLearner}, {"learner": "my_learner"})
        assert result == 42

    def test_size_function_with_ml_key(self):
        from flaml.automl.automl import size

        class FakeLearner:
            @staticmethod
            def size(config):
                return 99

        result = size({"my_learner": FakeLearner}, {"ml": {"learner": "my_learner"}})
        assert result == 99
