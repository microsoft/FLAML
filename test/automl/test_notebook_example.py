import sys
from test._artifact_paths import artifact_path
from urllib.error import URLError

import pytest

try:
    from minio.error import ServerError
except ImportError:

    class ServerError(Exception):
        pass


try:
    from openml.exceptions import OpenMLServerException
except ImportError:

    class OpenMLServerException(Exception):
        pass


from requests.exceptions import RequestException


def test_automl(tmp_path, budget=10, dataset_format="dataframe", hpo_method=None):
    import urllib3

    from flaml.automl.data import load_openml_dataset

    performance_check_budget = 600
    if (
        sys.platform == "darwin"
        and budget < performance_check_budget
        and dataset_format == "dataframe"
        and "3.9" in sys.version
    ):
        budget = performance_check_budget  # revise the buget on macos
    if budget == performance_check_budget:
        budget = None
        max_iter = 60
    else:
        max_iter = None if budget is None or budget < 0 else 4
    try:
        X_train, X_test, y_train, y_test = load_openml_dataset(
            dataset_id=1169, data_dir=artifact_path(__file__, "datasets"), dataset_format=dataset_format
        )
    except (
        OpenMLServerException,
        RequestException,
        urllib3.exceptions.ReadTimeoutError,
        URLError,
        ServerError,
    ) as e:
        pytest.skip(f"OpenML dataset is unavailable ({type(e).__name__}).")
    if budget is not None and budget > 0:
        X_train, y_train = X_train[:8192], y_train[:8192]
        X_test, y_test = X_test[:2048], y_test[:2048]
    """ import AutoML class from flaml package """
    from flaml import AutoML

    automl = AutoML()
    settings = {
        "time_budget": budget,  # total running time in seconds
        "max_iter": max_iter,  # maximum number of iterations
        "metric": "accuracy",  # primary metrics can be chosen from: ['accuracy','roc_auc','roc_auc_ovr','roc_auc_ovo','f1','log_loss','mae','mse','r2']
        "task": "classification",  # task type
        "log_file_name": str(tmp_path / "airlines_experiment.log"),  # flaml log file
        "seed": 7654321,  # random seed
        "hpo_method": hpo_method,
        "log_type": "all",
        "model_history": False,
        "n_jobs": 1,
        "estimator_list": [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "rf",
            "extra_tree",
        ],  # list of ML learners
        "eval_method": "holdout",
    }
    """The main flaml automl API"""
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    """ retrieve best config and best learner """
    print("Best ML leaner:", automl.best_estimator)
    assert automl.best_estimator, "The bounded notebook example did not train a model."
    print("Best hyperparmeter config:", automl.best_config)
    print(f"Best accuracy on validation data: {1 - automl.best_loss:.4g}")
    print(f"Training duration of best run: {automl.best_config_train_time:.4g} s")
    print(automl.model.estimator)
    print(automl.best_config_per_estimator)
    print("time taken to find best model:", automl.time_to_find_best_model)
    """ pickle and save the automl object """
    import pickle

    with open(tmp_path / "automl.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    """ compute predictions of testing dataset """
    y_pred = automl.predict(X_test)
    print("Predicted labels", y_pred)
    print("True labels", y_test)
    y_pred_proba = automl.predict_proba(X_test)[:, 1]
    """ compute different metric values on testing dataset """
    from flaml.automl.ml import sklearn_metric_loss_score

    accuracy = 1 - sklearn_metric_loss_score("accuracy", y_pred, y_test)
    print("accuracy", "=", accuracy)
    print("roc_auc", "=", 1 - sklearn_metric_loss_score("roc_auc", y_pred_proba, y_test))
    print("log_loss", "=", sklearn_metric_loss_score("log_loss", y_pred_proba, y_test))
    if budget is None:
        assert accuracy >= 0.669, "the accuracy of flaml should be larger than 0.67"
    from flaml.automl.data import get_output_from_log

    (
        time_history,
        best_valid_loss_history,
        valid_loss_history,
        config_history,
        metric_history,
    ) = get_output_from_log(filename=settings["log_file_name"], time_budget=6)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)
    print(automl.feature_names_in_)
    print(automl.feature_importances_)
    if budget is not None:
        automl.fit(X_train=X_train, y_train=y_train, ensemble=True, **settings)


@pytest.mark.skipif(
    sys.platform in ["win32"] and sys.version.startswith("3.9"),
    reason="do not run if windows and python 3.9",
)
def test_automl_array(tmp_path):
    test_automl(tmp_path, 5, "array", "bs")


def test_notebook_cache_corruption_is_not_silently_accepted(tmp_path, monkeypatch):
    import pickle

    from flaml.automl import data

    def corrupted(**kwargs):
        raise pickle.UnpicklingError("invalid local cache")

    monkeypatch.setattr(data, "load_openml_dataset", corrupted)
    with pytest.raises(pickle.UnpicklingError, match="invalid local cache"):
        test_automl(tmp_path)


def test_notebook_network_unavailability_is_explicitly_skipped(tmp_path, monkeypatch):
    from requests.exceptions import ConnectionError

    from flaml.automl import data

    def unavailable(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "load_openml_dataset", unavailable)
    with pytest.raises(pytest.skip.Exception):
        test_automl(tmp_path)


def test_notebook_smoke_fits_and_reloads_without_network(tmp_path, monkeypatch):
    import pickle

    from sklearn.datasets import load_breast_cancer

    from flaml.automl import data

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X["hospital"] = ["a" if index % 2 else "b" for index in range(len(X))]
    monkeypatch.setattr(
        data, "load_openml_dataset", lambda **kwargs: (X.iloc[:400], X.iloc[400:], y.iloc[:400], y.iloc[400:])
    )
    test_automl(tmp_path)
    with open(tmp_path / "automl.pkl", "rb") as stream:
        fitted = pickle.load(stream)
    assert len(fitted.predict(X.iloc[400:])) == len(y.iloc[400:])
    assert set(fitted.classes_) == {0, 1}


def _test_nobudget(tmp_path):
    # needs large RAM to run this test
    test_automl(tmp_path, -1)


def test_mlflow():
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow"])
    import mlflow

    from flaml.automl.data import load_openml_task

    try:
        X_train, X_test, y_train, y_test = load_openml_task(task_id=7592, data_dir=artifact_path(__file__, "datasets"))
    except (OpenMLServerException, RequestException, URLError, ServerError) as e:
        pytest.skip(f"OpenML task is unavailable ({type(e).__name__}).")
    """ import AutoML class from flaml package """
    from flaml import AutoML

    automl = AutoML()
    settings = {
        "time_budget": 5,  # total running time in seconds
        "metric": "accuracy",  # primary metrics can be chosen from: ['accuracy','roc_auc','roc_auc_ovr','roc_auc_ovo','f1','log_loss','mae','mse','r2']
        "estimator_list": ["lgbm", "rf", "xgboost"],  # list of ML learners
        "task": "classification",  # task type
        "sample": False,  # whether to subsample training data
        "log_file_name": artifact_path(__file__, "adult.log"),
        "learner_selector": "roundrobin",
    }
    mlflow.set_experiment("flaml")
    with mlflow.start_run() as run:
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        mlflow.sklearn.log_model(automl, "automl")
    loaded_model = mlflow.pyfunc.load_model(f"{run.info.artifact_uri}/automl")
    print(loaded_model.predict(X_test))
    automl._mem_thres = 0
    print(automl.trainable(automl.points_to_evaluate[0]))

    settings["use_ray"] = True
    try:
        with mlflow.start_run() as run:
            automl.fit(X_train=X_train, y_train=y_train, **settings)
            mlflow.sklearn.log_model(automl, "automl")
        automl = mlflow.sklearn.load_model(f"{run.info.artifact_uri}/automl")
        print(automl.predict_proba(X_test))
    except ImportError:
        pass


def test_mlflow_iris():
    import mlflow
    from sklearn.datasets import load_iris

    from flaml import AutoML

    with mlflow.start_run():
        automl = AutoML()
        automl_settings = {
            "time_budget": 2,  # in seconds
            "metric": "accuracy",
            "task": "classification",
            "log_file_name": artifact_path(__file__, "iris.log"),
        }
        X_train, y_train = load_iris(return_X_y=True)
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    # subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "mlflow"])


if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as output_dir:
        test_automl(Path(output_dir), 600)
