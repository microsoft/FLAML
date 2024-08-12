import pickle

import mlflow
import mlflow.entities
import pytest
from pandas import DataFrame
from sklearn.datasets import load_iris

from flaml import AutoML


class TestMLFlowLoggingParam:
    def test_should_start_new_run_by_default(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML()
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            try:
                self._check_mlflow_parameters(automl, parent_run.info)
            except FileNotFoundError:
                print("[WARNING]: No file found")

        children = self._get_child_runs(parent)
        assert len(children) >= 1, f"Expected at least 1 child run, got {len(children)}"

    def test_should_not_start_new_run_when_mlflow_logging_set_to_false_in_init(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML(mlflow_logging=False)
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            try:
                self._check_mlflow_parameters(automl, parent_run.info)
            except FileNotFoundError:
                print("[WARNING]: No file found")

        children = self._get_child_runs(parent)
        assert len(children) == 0, f"Expected 0 child runs, got {len(children)}"

    def test_should_not_start_new_run_when_mlflow_logging_set_to_false_in_fit(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML()
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, mlflow_logging=False, **automl_settings)
            try:
                self._check_mlflow_parameters(automl, parent_run.info)
            except FileNotFoundError:
                print("[WARNING]: No file found")

        children = self._get_child_runs(parent)
        assert len(children) == 0, f"Expected 0 child runs, got {len(children)}"

    def test_should_start_new_run_when_mlflow_logging_set_to_true_in_fit(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML(mlflow_logging=False)
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, mlflow_logging=True, **automl_settings)
            try:
                self._check_mlflow_parameters(automl, parent_run.info)
            except FileNotFoundError:
                print("[WARNING]: No file found")

        children = self._get_child_runs(parent)
        assert len(children) >= 1, f"Expected at least 1 child run, got {len(children)}"

    @staticmethod
    def _get_child_runs(parent_run: mlflow.entities.Run) -> DataFrame:
        experiment_id = parent_run.info.experiment_id
        return mlflow.search_runs(
            [experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
        )

    @staticmethod
    def _check_mlflow_parameters(automl: AutoML, run_info: mlflow.entities.RunInfo):
        with open(
            f"./mlruns/{run_info.experiment_id}/{run_info.run_id}/artifacts/automl_pipeline/model.pkl", "rb"
        ) as f:
            t = pickle.load(f)
            if __name__ == "__main__":
                print(t)
            for param in automl.model._model._get_param_names():
                assert eval("t._final_estimator._model" + f".{param}") == eval(
                    "automl.model._model" + f".{param}"
                ), "The mlflow logging not consistent with automl model"
                if __name__ == "__main__":
                    print(param, "\t", eval("automl.model._model" + f".{param}"))
        print("[INFO]: Successfully Logged")

    @pytest.fixture(scope="class")
    def automl_settings(self):
        return {
            "time_budget": 5,  # in seconds
            "metric": "accuracy",
            "task": "classification",
            "log_file_name": "iris.log",
        }


if __name__ == "__main__":
    s = TestMLFlowLoggingParam()
    automl_settings = {
        "time_budget": 5,  # in seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": "iris.log",
    }
    s.test_should_start_new_run_by_default(automl_settings)
    s.test_should_start_new_run_when_mlflow_logging_set_to_true_in_fit(automl_settings)
