from test._artifact_paths import artifact_path

import mlflow
import mlflow.entities
import numpy as np
import pytest
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from flaml import AutoML


def test_autologged_model_round_trip():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    automl = AutoML()
    with mlflow.start_run() as run:
        automl.fit(
            X_train,
            y_train,
            task="classification",
            estimator_list=["rf"],
            max_iter=1,
            verbose=0,
        )

    loaded = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    check_is_fitted(loaded)
    assert np.array_equal(automl.predict(X_test), loaded.predict(X_test))


class TestMLFlowLoggingParam:
    def test_update_and_install_requirements(self):
        import mlflow
        from sklearn import tree

        from flaml.fabric.mlflow import update_and_install_requirements

        with mlflow.start_run(run_name="test") as run:
            sk_model = tree.DecisionTreeClassifier()
            mlflow.sklearn.log_model(sk_model, "model", registered_model_name="test")

        update_and_install_requirements(run_id=run.info.run_id)

    def test_should_start_new_run_by_default(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML()
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            self._check_mlflow_parameters(automl, parent_run.info)

        children = self._get_child_runs(parent)
        assert len(children) >= 1, f"Expected at least 1 child run, got {len(children)}"

    def test_should_not_start_new_run_when_mlflow_logging_set_to_false_in_init(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML(mlflow_logging=False)
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            assert mlflow.tracking.MlflowClient().list_artifacts(parent_run.info.run_id) == []

        children = self._get_child_runs(parent)
        assert len(children) == 0, f"Expected 0 child runs, got {len(children)}"

    def test_should_not_start_new_run_when_mlflow_logging_set_to_false_in_fit(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML()
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, mlflow_logging=False, **automl_settings)
            assert mlflow.tracking.MlflowClient().list_artifacts(parent_run.info.run_id) == []

        children = self._get_child_runs(parent)
        assert len(children) == 0, f"Expected 0 child runs, got {len(children)}"

    def test_should_start_new_run_when_mlflow_logging_set_to_true_in_fit(self, automl_settings):
        with mlflow.start_run() as parent_run:
            parent = mlflow.last_active_run()
            automl = AutoML(mlflow_logging=False)
            X_train, y_train = load_iris(return_X_y=True)
            automl.fit(X_train=X_train, y_train=y_train, mlflow_logging=True, **automl_settings)
            self._check_mlflow_parameters(automl, parent_run.info)

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
        restored = mlflow.sklearn.load_model(f"runs:/{run_info.run_id}/model")
        check_is_fitted(restored)
        assert restored.named_steps["estimator"].estimator.get_params() == automl.model.estimator.get_params()
        X, _ = load_iris(return_X_y=True)
        np.testing.assert_array_equal(restored.predict(X), automl.predict(X))

    @pytest.fixture(scope="class")
    def automl_settings(self):
        return {
            "max_iter": 2,
            "estimator_list": ["rf"],
            "n_jobs": 1,
            "custom_hp": {"rf": {"n_estimators": {"domain": 4, "init_value": 4}}},
            "metric": "accuracy",
            "task": "classification",
            "log_file_name": artifact_path(__file__, "iris.log"),
        }


if __name__ == "__main__":
    s = TestMLFlowLoggingParam()
    automl_settings = {
        "time_budget": 5,  # in seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": artifact_path(__file__, "iris.log"),
    }
    s.test_should_start_new_run_by_default(automl_settings)
    s.test_should_start_new_run_when_mlflow_logging_set_to_true_in_fit(automl_settings)
