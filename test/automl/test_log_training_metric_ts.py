"""Test log_training_metric with time series models."""
import pytest
import numpy as np
import pandas as pd

from flaml import AutoML


def test_log_training_metric_with_arima():
    """Test that ARIMA works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    data = data[:100]
    
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
    print(f"✅ ARIMA with log_training_metric=True: SUCCESS")


def test_log_training_metric_with_sarimax():
    """Test that SARIMAX works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    data = data[:100]
    
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
    print(f"✅ SARIMAX with log_training_metric=True: SUCCESS")


def test_log_training_metric_with_holt_winters():
    """Test that Holt-Winters works with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    data = data[:100]
    
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
    print(f"✅ Holt-Winters with log_training_metric=True: SUCCESS")


def test_log_training_metric_with_all_ts_models():
    """Test that all statistical TS models work with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    data = data[:100]
    
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
    print(f"✅ All TS models with log_training_metric=True: SUCCESS")


def test_log_training_metric_with_ml_models():
    """Test that ML-based TS models still work with log_training_metric=True."""
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")
    
    # Load sample data
    data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={"index": "ds", "co2": "y"})
    
    # Use a small subset for fast testing
    data = data[:100]
    
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
    print(f"✅ ML models with log_training_metric=True: SUCCESS")


if __name__ == "__main__":
    test_log_training_metric_with_arima()
    test_log_training_metric_with_sarimax()
    test_log_training_metric_with_holt_winters()
    test_log_training_metric_with_all_ts_models()
    test_log_training_metric_with_ml_models()
    print("\n✅ All tests passed!")
