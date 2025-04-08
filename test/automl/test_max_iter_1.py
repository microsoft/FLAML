import mlflow
import numpy as np
import pandas as pd

from flaml import AutoML


def test_max_iter_1():
    date_rng = pd.date_range(start="2024-01-01", periods=100, freq="H")
    X = pd.DataFrame({"ds": date_rng})
    y_train_24h = np.random.rand(len(X)) * 100

    # AutoML
    settings = {
        "max_iter": 1,
        "estimator_list": ["xgboost", "lgbm"],
        "starting_points": {"xgboost": {}, "lgbm": {}},
        "task": "ts_forecast",
        "log_file_name": "test_max_iter_1.log",
        "seed": 41,
        "mlflow_exp_name": "TestExp-max_iter-1",
        "use_spark": False,
        "n_concurrent_trials": 1,
        "verbose": 1,
        "featurization": "off",
        "metric": "rmse",
        "mlflow_logging": True,
    }

    automl = AutoML(**settings)

    with mlflow.start_run(run_name="AutoMLModel-XGBoost-and-LGBM-max_iter_1"):
        automl.fit(
            X_train=X,
            y_train=y_train_24h,
            period=24,
            X_val=X,
            y_val=y_train_24h,
            split_ratio=0,
            force_cancel=False,
        )

    assert automl.model is not None, "AutoML failed to return a model"
    assert automl.best_run_id is not None, "Best run ID should not be None with mlflow logging"

    print("Best model:", automl.model)
    print("Best run ID:", automl.best_run_id)


if __name__ == "__main__":
    test_max_iter_1()
