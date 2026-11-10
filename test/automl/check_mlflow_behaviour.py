import mlflow
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import flaml
from flaml import AutoML

mlflow.set_experiment("mlflow_behaviour")


def _automl_example():
    automl = AutoML()
    automl_settings = {
        "time_budget": 2,  # in seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": "iris.log",
        "mlflow_exp_name": "mlflow_behaviour",
    }
    X_train, y_train = load_iris(return_X_y=True)
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)


def _tune_example():
    def _sklearn_tune(config):
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
        rf = RandomForestRegressor(**config)
        rf.fit(train_x, train_y)
        pred = rf.predict(test_x)
        r2 = r2_score(test_y, pred)
        return {"r2": r2}

    params = {
        "n_estimators": flaml.tune.randint(100, 1000),
        "min_samples_leaf": flaml.tune.randint(1, 10),
    }
    n_child_runs = 3
    flaml.tune.run(
        _sklearn_tune,
        params,
        metric="r2",
        mode="max",
        num_samples=n_child_runs,
        verbose=5,
        mlflow_exp_name="mlflow_behaviour",
    )


def test_automl_autolog_run():
    mlflow.autolog()
    with mlflow.start_run(run_name="automl_branch_autolog_run"):
        _automl_example()
    mlflow.autolog(disable=True)


def test_automl_autolog_norun():
    mlflow.autolog()
    _automl_example()
    mlflow.autolog(disable=True)


def test_automl_noauto_run():
    with mlflow.start_run(run_name="automl_branch_noauto_run"):
        _automl_example()


def test_automl_noauto_norun():
    _automl_example()


def test_tune_autolog_run():
    mlflow.autolog()
    with mlflow.start_run(run_name="tune_branch_autolog_run"):
        _tune_example()
    mlflow.autolog(disable=True)


def test_tune_autolog_norun():
    mlflow.autolog()
    _tune_example()
    mlflow.autolog(disable=True)


def test_tune_noauto_run():
    with mlflow.start_run(run_name="tune_branch_noauto_run"):
        _tune_example()


def test_tune_noauto_norun():
    _tune_example()
