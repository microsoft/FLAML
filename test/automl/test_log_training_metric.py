"""Test log_training_metric with time series forecasting models."""

import numpy as np
import pandas as pd
import pytest


def prepare_airline_data():
    """Prepare a simple time series dataset."""
    # Create simple time series data similar to airline data
    # Use fixed seed for reproducible tests
    np.random.seed(42)
    dates = pd.date_range(start="1949-01-01", periods=50, freq="MS")
    values = np.arange(50, dtype=np.float64) + np.random.randn(50) * 5
    
    return pd.DataFrame({
        "ds": dates,
        "y": values,
    })


def test_log_training_metric_with_arima():
    """Test that ARIMA works with log_training_metric=True."""
    from flaml import AutoML
    
    train_df = prepare_airline_data()
    
    config = {
        "task": "ts_forecast",
        "time_budget": 5,
        "metric": "mape",
        "eval_method": "holdout",
        "seed": 42,
        "verbose": 0,
        "estimator_list": ["arima"],
        "log_training_metric": True,  # This should work without errors
    }
    
    automl = AutoML()
    automl.fit(dataframe=train_df, label="y", period=1, **config)
    
    assert automl.best_estimator == "arima"


def test_log_training_metric_with_sarimax():
    """Test that SARIMAX works with log_training_metric=True."""
    from flaml import AutoML
    
    train_df = prepare_airline_data()
    
    config = {
        "task": "ts_forecast",
        "time_budget": 5,
        "metric": "mape",
        "eval_method": "holdout",
        "seed": 42,
        "verbose": 0,
        "estimator_list": ["sarimax"],
        "log_training_metric": True,  # This should work without errors
    }
    
    automl = AutoML()
    automl.fit(dataframe=train_df, label="y", period=1, **config)
    
    assert automl.best_estimator == "sarimax"


def test_log_training_metric_with_holt_winters():
    """Test that Holt-Winters works with log_training_metric=True."""
    from flaml import AutoML
    
    train_df = prepare_airline_data()
    
    config = {
        "task": "ts_forecast",
        "time_budget": 5,
        "metric": "mape",
        "eval_method": "holdout",
        "seed": 42,
        "verbose": 0,
        "estimator_list": ["holt-winters"],
        "log_training_metric": True,  # This should work without errors
    }
    
    automl = AutoML()
    automl.fit(dataframe=train_df, label="y", period=1, **config)
    
    assert automl.best_estimator == "holt-winters"


def test_log_training_metric_with_all_ts_estimators():
    """Test that all TS estimators work with log_training_metric=True."""
    from flaml import AutoML
    
    train_df = prepare_airline_data()
    
    config = {
        "task": "ts_forecast",
        "time_budget": 10,
        "metric": "mape",
        "eval_method": "holdout",
        "seed": 42,
        "verbose": 0,
        "estimator_list": ["arima", "sarimax", "holt-winters"],
        "log_training_metric": True,  # This should work without errors
    }
    
    automl = AutoML()
    automl.fit(dataframe=train_df, label="y", period=1, **config)
    
    # Should complete successfully
    assert automl.best_estimator in ["arima", "sarimax", "holt-winters"]


if __name__ == "__main__":
    test_log_training_metric_with_arima()
    test_log_training_metric_with_sarimax()
    test_log_training_metric_with_holt_winters()
    test_log_training_metric_with_all_ts_estimators()
    print("All tests passed!")
