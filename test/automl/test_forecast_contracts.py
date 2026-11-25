import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from flaml import AutoML
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.automl.time_series.ts_model import HoltWinters, TimeSeriesEstimator


@pytest.mark.parametrize("task_name", ["ts_forecast_panel", "ts_forecast"])
def test_panel_auto_holdout_preserves_composite_keys_covariates_and_absolute_index(task_name):
    seen = []

    class PanelEstimator(TimeSeriesEstimator):
        @classmethod
        def search_space(cls, **kwargs):
            return {}

        def fit(self, X_train, y_train=None, **kwargs):
            super().fit(X_train, y_train, **kwargs)
            frame = X_train.train_data
            assert {"country", "store", "static", "time_idx"} <= set(frame)
            assert frame["time_idx"].min() == 100
            assert frame["time_idx"].dtype.kind == "i"
            for _, group in frame.groupby(["country", "store"], observed=True):
                assert group["time_idx"].is_monotonic_increasing
            seen.append(frame.copy())
            self._model = DummyRegressor().fit(np.zeros((len(frame), 1)), frame["target"])
            return 0.0

        def predict(self, X, **kwargs):
            frame = X.test_data if isinstance(X, TimeSeriesDataset) else X
            assert frame["time_idx"].min() >= 100
            return self._model.predict(np.zeros((len(frame), 1)))

    frame = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-01-01", periods=10)) * 2,
            "country": ["same"] * 20,
            "store": ["a"] * 10 + ["b"] * 10,
            "static": [42.0] * 20,
            "time_idx": list(range(100, 110)) * 2,
            "target": np.arange(20.0),
        }
    ).sample(frac=1, random_state=4)
    original = frame.copy(deep=True)
    automl = AutoML(max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    automl.add_learner("panel_contract", PanelEstimator)
    automl.fit(
        dataframe=frame,
        label="target",
        task=task_name,
        period=2,
        group_ids=["country", "store"],
        estimator_list=["panel_contract"],
        metric="mae",
    )
    assert len(seen) >= 2
    assert len(seen[0]) == 16 and len(seen[-1]) == 20
    assert not automl._state.task._preprocess_for_autofe
    assert len(automl.predict(frame.drop(columns="target"))) == len(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("train_full", [False, True])
def test_forecast_log_replay_keeps_the_recorded_training_boundary(tmp_path, train_full):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "target": np.arange(30.0)})
    log = str(tmp_path / "training.log")
    automl = AutoML(
        estimator_list=["naive"],
        max_iter=2,
        n_jobs=1,
        mlflow_logging=False,
        verbose=0,
        eval_method="holdout",
        retrain_full=False,
    )
    automl.fit(dataframe=frame, label="target", task="ts_forecast", period=3, log_file_name=log)
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    restored.retrain_from_log(
        log,
        dataframe=frame,
        label="target",
        task="ts_forecast",
        period=3,
        eval_method="holdout",
        train_full=train_full,
    )
    boundary = frame["date"].iloc[-1 if train_full else -4]
    assert restored.model.train_end_date == boundary
    assert restored.model.X_train.test_data.empty
    assert restored.predict(3).index[0] == boundary + pd.Timedelta(days=1)


def test_all_rejected_log_raises_without_replacing_a_previous_model(tmp_path):
    X = np.arange(120).reshape(60, 2)
    y = np.tile([0, 1], 30)
    automl = AutoML(estimator_list=["rf"], max_iter=2, n_jobs=1, mlflow_logging=False, verbose=0, log_type="all")
    log = tmp_path / "source.log"
    automl.fit(X, y, log_file_name=str(log))
    previous_model, previous_task = automl.model, automl._state.task
    predictions = automl.predict(X)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("validation_loss" in record for record in records)
    for record in records:
        if "validation_loss" in record:
            record["validation_loss"] = float("inf")
    log.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(ValueError, match="finite"):
        automl.retrain_from_log(str(log), X_train=X, y_train=y, task="regression")
    assert automl.model is previous_model
    assert automl._state.task is previous_task
    np.testing.assert_array_equal(automl.predict(X), predictions)


def test_forecast_autofe_log_recipe_keeps_the_forecasting_wrapper(tmp_path):
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=30), "feature": np.arange(30.0), "target": np.arange(30.0)}
    )
    log = str(tmp_path / "forecast.log")
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
    )
    automl.fit(dataframe=frame, label="target", task="ts_forecast", period=2, log_file_name=log)
    recipe = automl.get_estimator_from_log(log, record_id=0, task="ts_forecast")
    recipe.fit(TimeSeriesDataset(frame, "date", "target"), estimator__period=2)
    future = pd.DataFrame({"date": pd.date_range("2026-01-31", periods=2), "feature": [30.0, 31.0]})
    assert np.isfinite(recipe.predict(future)).all()


def test_legacy_regular_statsmodels_forecasts_use_the_fitted_index_boundary():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=24), "target": np.arange(24.0) + 10})
    estimator = HoltWinters(task="ts_forecast", seasonal_periods=4, seasonal=None, trend="add", damped_trend=False)
    estimator.fit(TimeSeriesDataset(frame, "date", "target"), None)
    expected = estimator.predict(3)
    del estimator.train_end_date
    pd.testing.assert_series_equal(estimator.predict(3), expected)
    future = pd.DataFrame({"date": expected.index})
    pd.testing.assert_series_equal(estimator.predict(future), expected)


def test_holt_winters_uses_the_seasonal_period_it_searches():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=24), "target": 10 + np.tile([0.0, 1.0, 2.0, 1.0], 6)}
    )
    estimator = HoltWinters(task="ts_forecast", seasonal_periods=4, seasonal="add", trend="add", damped_trend=False)
    with patch("statsmodels.tsa.holtwinters.ExponentialSmoothing", wraps=ExponentialSmoothing) as constructor:
        estimator.fit(TimeSeriesDataset(frame, "date", "target"), None)
    assert constructor.call_args.kwargs["seasonal_periods"] == 4
    assert estimator.model.model.seasonal_periods == 4


@pytest.mark.parametrize("rejected_loss", [float("inf"), float("nan")])
def test_log_replay_skips_rejected_trials_and_breaks_ties_by_training_size(tmp_path, rejected_loss):
    X = np.arange(120).reshape(60, 2)
    y = np.tile([0, 1], 30)
    source = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        mlflow_logging=False,
        verbose=0,
        featurization="auto",
        log_type="all",
    )
    log = tmp_path / "source.log"
    source.fit(X, y, log_file_name=str(log))
    template = next(json.loads(line) for line in log.read_text().splitlines() if "validation_loss" in line)
    records = []
    for index, (loss, size, trees) in enumerate(((rejected_loss, 60, 4), (0.1, 30, 8), (0.1, 60, 16))):
        record = {
            **template,
            "record_id": index,
            "validation_loss": loss,
            "sample_size": size,
            "wall_clock_time": index + 1,
        }
        record["config"] = {**template["config"], "n_estimators": trees}
        records.append(record)
    log.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    restored.retrain_from_log(str(log), X_train=X, y_train=y, train_full=True)
    assert restored.model.estimator.n_estimators == 16
    assert restored.automl_pipeline is not None
    with pytest.raises(ValueError, match="finite"):
        restored.retrain_from_log(str(log), X_train=X, y_train=y, time_budget=1.5)
