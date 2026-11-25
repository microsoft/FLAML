import numpy as np
import pandas as pd
import pytest

from flaml import AutoML
from flaml.automl.time_series.ts_data import TimeSeriesDataset


@pytest.mark.parametrize("name", [None, "other_name", "sales"])
@pytest.mark.parametrize("index_kind", ["default", "offset", "datetime"])
def test_series_targets_use_requested_name_without_mutating_inputs(name, index_kind):
    features = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=4)})
    if index_kind == "offset":
        features.index += 100
    elif index_kind == "datetime":
        features.index = features.ds
    target = pd.Series([1.0, 2.0, 3.0, 4.0], name=name, index=[10, 11, 12, 13])
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)

    combined = TimeSeriesDataset.to_dataframe(features, target, ["sales"], "ds")

    expected = features.copy()
    expected["sales"] = target.to_numpy()
    pd.testing.assert_frame_equal(combined, expected)
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_series_equal(target, original_target)
    dataset = TimeSeriesDataset(combined, "ds", "sales")
    assert dataset.y_train.tolist() == target.tolist()
    assert dataset.X_train.columns.tolist() == ["ds"]


def _fit_forecast(target):
    pytest.importorskip("statsmodels")
    train = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=20), "sales": np.arange(20, dtype=float)})
    features = pd.DataFrame({"ds": pd.date_range("2024-01-21", periods=3)})
    automl = AutoML()
    automl.fit(
        dataframe=train,
        label="sales",
        X_val=features,
        y_val=target,
        task="ts_forecast",
        period=3,
        estimator_list=["naive"],
        metric="mae",
        max_iter=1,
        time_budget=10,
        retrain_full=False,
        mlflow_logging=False,
        verbose=0,
    )

    return automl.predict(features), automl.best_loss


@pytest.fixture(scope="module")
def forecast_reference():
    return _fit_forecast(np.array([20.0, 21.0, 22.0]))


@pytest.mark.parametrize("name", [None, "other_name", "sales"])
def test_automl_forecast_accepts_validation_series_name(name, forecast_reference):
    target = pd.Series([20.0, 21.0, 22.0], name=name)
    original_target = target.copy(deep=True)
    predictions, loss = _fit_forecast(target)

    np.testing.assert_allclose(predictions, forecast_reference[0])
    assert loss == pytest.approx(forecast_reference[1])
    pd.testing.assert_series_equal(target, original_target)
