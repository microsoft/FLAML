import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from flaml.automl.time_series import SeasonalNaive, TimeSeriesDataset


@pytest.fixture
def seasonal_forecast():
    train = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=8, freq="D"), "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]}
    )
    future = pd.DataFrame({"ds": pd.date_range("2024-01-09", periods=4, freq="D")})
    estimator = SeasonalNaive(season=2)
    estimator.fit(TimeSeriesDataset(train, time_col="ds", target_names="y"))
    np.testing.assert_allclose(estimator.predict(future), [7.0, 9.0, 7.0, 9.0])
    return estimator, train, future


@pytest.mark.parametrize("references", [[6.0, 10.0, 8.0, 12.0], [8.0, 8.0, 8.0, 8.0], [7.0, 9.0, 7.0, 9.0]])
@pytest.mark.parametrize("use_dataset", [False, True])
def test_time_series_default_score_uses_reference_variance(seasonal_forecast, references, use_dataset):
    estimator, train, future = seasonal_forecast
    expected = r2_score(references, estimator.predict(future))
    if use_dataset:
        dataset = TimeSeriesDataset(train, time_col="ds", target_names="y", test_data=future.assign(y=references))
        actual = estimator.score(dataset, None)
    else:
        actual = estimator.score(future, pd.Series(references))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["r2", "mae"])
def test_time_series_explicit_metric_keeps_loss_convention(seasonal_forecast, metric):
    estimator, _, future = seasonal_forecast
    references = pd.Series([6.0, 10.0, 8.0, 12.0])
    predictions = estimator.predict(future)
    expected = (
        1.0 - r2_score(references, predictions)
        if metric == "r2"
        else np.abs(references - predictions.to_numpy()).mean()
    )
    assert estimator.score(future, references, metric=metric) == pytest.approx(expected)
