import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from flaml.automl.ml import compute_estimator
from flaml.automl.task.factory import task_factory
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.autofe import Featurization


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
