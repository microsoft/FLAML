"""Test log_training_metric with time series models."""
import pytest
import numpy as np
import pandas as pd

from flaml import AutoML


def _prepare_test_data():
    """Helper function to prepare test data."""
    import statsmodels.api as sm
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    return data[:100]


def test_log_training_metric_with_arima():
    """Test that ARIMA works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    data = _prepare_test_data()
    
    automl = AutoML()
    settings = {
        "time_budget": 10,
        "metric": "mape",
        "task": "ts_forecast",
        "eval_method": "holdout",
        "log_training_metric": True,  # This should not cause errors
        "estimator_list": ["arima"],
        "verbose": 0,
    }
    
    # This should not raise an IndexError
    automl.fit(dataframe=data, label="y", period=12, **settings)
    
    assert automl.best_estimator == "arima"


def test_log_training_metric_with_sarimax():
    """Test that SARIMAX works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    data = _prepare_test_data()
    
    automl = AutoML()
    settings = {
        "time_budget": 10,
        "metric": "mape",
        "task": "ts_forecast",
        "eval_method": "holdout",
        "log_training_metric": True,  # This should not cause errors
        "estimator_list": ["sarimax"],
        "verbose": 0,
    }
    
    # This should not raise an IndexError
    automl.fit(dataframe=data, label="y", period=12, **settings)
    
    assert automl.best_estimator == "sarimax"


def test_log_training_metric_with_holt_winters():
    """Test that Holt-Winters works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    data = _prepare_test_data()
    
    automl = AutoML()
    settings = {
        "time_budget": 10,
        "metric": "mape",
        "task": "ts_forecast",
        "eval_method": "holdout",
        "log_training_metric": True,  # This should not cause errors
        "estimator_list": ["holt-winters"],
        "verbose": 0,
    }
    
    # This should not raise an IndexError
    automl.fit(dataframe=data, label="y", period=12, **settings)
    
    assert automl.best_estimator == "holt-winters"


def test_log_training_metric_with_all_ts_models():
    """Test that all statistical TS models work with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    data = _prepare_test_data()
    
    automl = AutoML()
    settings = {
        "time_budget": 15,
        "metric": "mape",
        "task": "ts_forecast",
        "eval_method": "holdout",
        "log_training_metric": True,  # This should not cause errors
        "estimator_list": ["arima", "sarimax", "holt-winters"],
        "verbose": 0,
    }
    
    # This should not raise an IndexError
    automl.fit(dataframe=data, label="y", period=12, **settings)
    
    assert automl.best_estimator in ["arima", "sarimax", "holt-winters"]


def test_log_training_metric_with_ml_models():
    """Test that ML-based TS models still work with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    data = _prepare_test_data()
    
    automl = AutoML()
    settings = {
        "time_budget": 10,
        "metric": "mape",
        "task": "ts_forecast",
        "eval_method": "holdout",
        "log_training_metric": True,  # This should work for ML models
        "estimator_list": ["lgbm"],
        "verbose": 0,
    }
    
    # This should work fine
    automl.fit(dataframe=data, label="y", period=12, **settings)
    
    assert automl.best_estimator == "lgbm"
