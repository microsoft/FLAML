import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from flaml import AutoML
from flaml.automl.ml import compute_estimator
from flaml.automl.model import LGBMEstimator, RandomForestEstimator
from flaml.automl.task.factory import task_factory
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.autofe import Featurization, parse_autofe_config


class CustomForest(RandomForestEstimator):
    pass


def test_custom_learner_does_not_inherit_last_builtin_name(monkeypatch):
    task = task_factory("classification")
    monkeypatch.setattr(type(task), "estimators", property(lambda self: {"unrelated_spark": LGBMEstimator}))
    space = parse_autofe_config("auto", np.ones((10, 2)), task, CustomForest)
    assert "fe.numerical" in space


def test_custom_spark_learner_name_retains_spark_exclusion():
    space = parse_autofe_config(
        "auto", np.ones((10, 2)), task_factory("classification"), CustomForest, estimator_name="custom_spark"
    )
    assert space == {}


def test_registered_learner_name_reaches_featurization(monkeypatch):
    seen = []

    def capture_config(*args, **kwargs):
        seen.append(kwargs.get("estimator_name"))
        return {}

    monkeypatch.setattr("flaml.automl.state.parse_autofe_config", capture_config)
    automl = AutoML(estimator_list=["custom_rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    automl.add_learner("custom_rf", CustomForest)
    X = np.arange(80).reshape(40, 2)
    automl.fit(X, np.tile([0, 1], 20), featurization="auto")
    assert seen == ["custom_rf"]


def test_autofe_and_resampler_only_receive_each_training_fold():
    seen = []

    class RecordingResampler(BaseEstimator):
        def fit_resample(self, X, y):
            seen.append(set(X.index))
            np.testing.assert_allclose(X.mean(), 0, atol=1e-12)
            return pd.concat([X, X]), np.concatenate([y, y])

    task = task_factory("classification")
    task._resampler = RecordingResampler()
    X = pd.DataFrame({"a": np.arange(18, dtype=float), "b": np.arange(18, dtype=float) ** 2})
    y = np.array([0] * 12 + [1] * 6)
    folds = StratifiedKFold(n_splits=3)
    compute_estimator(
        X,
        y,
        None,
        None,
        weight_val=None,
        groups_val=None,
        budget=None,
        kf=folds,
        config_dic={"n_estimators": 2, "fe.numerical": "scaler_standard"},
        task=task,
        estimator_name="rf",
        eval_method="cv",
        eval_metric="accuracy",
        n_jobs=1,
    )
    assert seen == [set(train) for train, _ in folds.split(X, y)]


def test_cardinality_selector_preserves_feature_names():
    data = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]})
    autofe = Featurization(params={"fe.selection": "cardinality"})
    transformed = autofe.fit_transform(data)
    pd.testing.assert_frame_equal(transformed, data)


def test_time_series_featurization_excludes_test_partition():
    train = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=4), "y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    test = pd.DataFrame({"ds": pd.date_range("2025-01-05", periods=2), "y": [100.0, 101.0], "x": [100.0, 101.0]})
    dataset = TimeSeriesDataset(train_data=train, time_col="ds", target_names=["y"], test_data=test)
    autofe = Featurization(params={"fe.numerical": "scaler_standard"})
    transformed = autofe.fit_transform(dataset)
    assert transformed.train_data["x"].mean() == pytest.approx(0.0)
    assert transformed.test_data["x"].min() > 10.0
    pd.testing.assert_series_equal(transformed.test_data["y"], test["y"])


def test_refit_without_transformers_clears_previous_pipeline():
    autofe = Featurization(params={"fe.numerical": "scaler_standard"})
    autofe.fit(pd.DataFrame({"a": [1.0, 2.0]}))
    assert autofe.pipeline is not None

    categorical_data = pd.DataFrame({"b": pd.Categorical(["x", "y"])})
    autofe.fit(categorical_data)
    assert autofe.pipeline is None
    assert autofe.detail_config == []
    check_is_fitted(autofe)
    pd.testing.assert_frame_equal(autofe.transform(categorical_data), categorical_data)


def test_failed_refit_is_not_fitted():
    autofe = Featurization(params={"fe.numerical": "scaler_standard"})
    autofe.fit(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        autofe.fit(pd.DataFrame({"a": [1.0, np.inf]}))
    assert autofe.pipeline is None
    assert autofe.detail_config == []
    with pytest.raises(NotFittedError):
        check_is_fitted(autofe)


@pytest.mark.parametrize("evaluation", ["holdout", "cv"])
def test_featurization_only_fits_training_partition(monkeypatch, evaluation):
    fit_partitions = []
    original_fit = Featurization.fit

    def record_fit(self, X, y=None):
        fit_partitions.append((X.index.tolist(), np.asarray(y).tolist()))
        return original_fit(self, X, y)

    monkeypatch.setattr(Featurization, "fit", record_fit)
    X = pd.DataFrame({"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) ** 2})
    y = np.arange(12, dtype=float)
    X_val = pd.DataFrame({"a": [100.0, 101.0], "b": [10000.0, 10201.0]}, index=[100, 101])
    y_val = np.array([100.0, 101.0])
    kf = KFold(n_splits=3)
    compute_estimator(
        X,
        y,
        X_val if evaluation == "holdout" else None,
        y_val if evaluation == "holdout" else None,
        weight_val=None,
        groups_val=None,
        budget=None,
        kf=kf,
        config_dic={"n_estimators": 2, "fe.numerical": "scaler_standard"},
        task=task_factory("regression"),
        estimator_name="rf",
        eval_method=evaluation,
        eval_metric="mse",
        n_jobs=1,
    )
    if evaluation == "holdout":
        assert fit_partitions == [(X.index.tolist(), y.tolist())]
    else:
        expected = [(set(train), set(y[train])) for train, _ in kf.split(X)]
        assert [(set(rows), set(labels)) for rows, labels in fit_partitions] == expected
