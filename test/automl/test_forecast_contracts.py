import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from flaml import AutoML
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.automl.time_series.ts_model import HoltWinters, TimeSeriesEstimator


@pytest.mark.parametrize("task_name", ["ts_forecast_panel", "ts_forecast"])
@pytest.mark.parametrize("group_values", [("a", "b"), (2**53, 2**53 + 1)])
@pytest.mark.parametrize("time_idx", ["time_idx", "step"])
def test_panel_auto_holdout_preserves_composite_keys_covariates_and_absolute_index(task_name, group_values, time_idx):
    seen = []

    class PanelEstimator(TimeSeriesEstimator):
        @classmethod
        def search_space(cls, **kwargs):
            return {}

        def fit(self, X_train, y_train=None, **kwargs):
            super().fit(X_train, y_train, **kwargs)
            frame = X_train.train_data
            assert {"country", "store", "static", time_idx} <= set(frame)
            assert frame[time_idx].min() == 100
            assert frame[time_idx].dtype.kind == "i"
            assert set(frame["store"].unique()) == set(group_values)
            for _, group in frame.groupby(["country", "store"], observed=True):
                assert group[time_idx].is_monotonic_increasing
            seen.append(frame.copy())
            self._model = DummyRegressor().fit(np.zeros((len(frame), 1)), frame["target"])
            return 0.0

        def predict(self, X, **kwargs):
            frame = X.test_data if isinstance(X, TimeSeriesDataset) else X
            assert frame[time_idx].min() >= 100
            return self._model.predict(np.zeros((len(frame), 1)))

    frame = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-01-01", periods=10)) * 2,
            "country": ["same"] * 20,
            "store": [group_values[0]] * 10 + [group_values[1]] * 10,
            "static": [42.0] * 20,
            time_idx: list(range(100, 110)) * 2,
            "target": np.arange(20.0),
        }
    ).sample(frac=1, random_state=4)
    original = frame.copy(deep=True)
    automl = AutoML(max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    automl.add_learner("panel_contract", PanelEstimator)
    data = (
        {"dataframe": frame, "label": "target"}
        if time_idx == "time_idx"
        else {"X_train": TimeSeriesDataset(frame, "date", "target", time_idx=time_idx)}
    )
    automl.fit(
        **data,
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
@pytest.mark.parametrize("eval_method", ["auto", "holdout"])
def test_forecast_log_replay_keeps_the_recorded_training_boundary(tmp_path, train_full, eval_method):
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
        eval_method=eval_method,
        train_full=train_full,
    )
    boundary = frame["date"].iloc[-1 if train_full else -4]
    assert restored.model.train_end_date == boundary
    assert restored.model.X_train.test_data.empty
    assert restored.predict(3).index[0] == boundary + pd.Timedelta(days=1)


def test_full_forecast_replay_does_not_require_an_unused_cv_split(tmp_path):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "target": np.arange(10.0)})
    log = str(tmp_path / "training.log")
    source = AutoML(
        estimator_list=["naive"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        eval_method="holdout",
        retrain_full=False,
    )
    source.fit(dataframe=frame, label="target", task="ts_forecast", period=3, log_file_name=log)
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    restored.retrain_from_log(log, dataframe=frame, label="target", task="ts_forecast", period=3, train_full=True)
    assert restored.model.train_end_date == frame["date"].iloc[-1]
    assert len(restored.model.X_train.train_data) == len(frame)


@pytest.mark.parametrize("featurization", ["off", "auto"])
def test_forecast_explicit_validation_normalizes_timestamp_strings(featurization):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=36),
            "feature": np.arange(36.0),
            "target": np.arange(36.0),
        }
    )
    predictions = []
    for strings in (False, True):
        data = frame.copy()
        if strings:
            data["date"] = data["date"].astype(str)
        original = data.copy(deep=True)
        automl = AutoML(
            estimator_list=["rf"],
            max_iter=1,
            n_jobs=1,
            verbose=0,
            mlflow_logging=False,
            featurization=featurization,
            starting_points={"rf": {"n_estimators": 4, "fe.extraction": "null"}},
        )
        automl.fit(
            X_train=data.iloc[:30].drop(columns="target"),
            y_train=data["target"].iloc[:30],
            X_val=data.iloc[30:].drop(columns="target"),
            y_val=data["target"].iloc[30:],
            task="ts_forecast",
            period=6,
        )
        predictions.append(automl.predict(data.iloc[30:].drop(columns="target")))
        pd.testing.assert_frame_equal(data, original)
    np.testing.assert_allclose(predictions[0], predictions[1])


@pytest.mark.parametrize("period", [0, -1])
def test_nonpositive_forecast_period_is_rejected_before_splitting(period):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=12), "target": np.arange(12.0)})
    automl = AutoML(estimator_list=["naive"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    with pytest.raises(ValueError, match="period.*positive"):
        automl.fit(dataframe=frame, label="target", task="ts_forecast", period=period)


def test_empty_forecast_validation_does_not_predict_or_divide_by_zero():
    from flaml.automl.ml import _eval_estimator
    from flaml.automl.task.factory import task_factory

    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4), "target": range(4)})
    dataset = TimeSeriesDataset(frame, "date", "target")
    estimator = Mock()
    estimator.predict.side_effect = AssertionError("An empty validation partition must not be predicted.")
    loss, metrics, latency = _eval_estimator(
        {}, estimator, dataset, None, dataset, None, None, None, "mae", task_factory("ts_forecast")
    )
    assert np.isinf(loss)
    assert metrics["pred_time"] == latency == 0
    estimator.predict.assert_not_called()


@pytest.fixture
def tft_decoder():
    from flaml.automl.time_series.tft import TemporalFusionTransformerEstimator

    def make(time_idx="time_idx"):
        estimator = TemporalFusionTransformerEstimator(task="ts_forecast_panel")
        estimator.time_col, estimator.target_names = "date", ["target"]
        estimator.time_idx, estimator.group_ids = time_idx, ["group"]
        estimator.max_encoder_length, estimator.max_prediction_length = 4, 2
        estimator.data = pd.DataFrame(
            {
                "date": list(pd.date_range("2026-01-01", periods=10)) * 2,
                "group": ["a"] * 10 + ["b"] * 10,
                time_idx: list(range(100, 110)) * 2,
                "target": np.arange(20.0) + 1,
            }
        )

        def predict(frame, **kwargs):
            values = np.array([[20.0, 21.0], [10.0, 11.0]])
            if kwargs.get("return_index"):
                return SimpleNamespace(output=values, index=pd.DataFrame({"group": ["b", "a"], time_idx: [110, 110]}))
            return SimpleNamespace(numpy=lambda: values)

        estimator._model = SimpleNamespace(predict=Mock(side_effect=predict))
        return estimator

    return make


@pytest.mark.parametrize("time_idx", ["time_idx", "step"])
def test_tft_prediction_retains_absolute_coordinates_and_request_order(tft_decoder, time_idx):
    estimator = tft_decoder(time_idx)
    request = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-12", "2026-01-11", "2026-01-11", "2026-01-12"]),
            "group": ["b", "a", "b", "a"],
            time_idx: [111, 110, 110, 111],
        },
        index=[40, 10, 30, 20],
    )
    original = request.copy(deep=True)
    prediction = estimator.predict(request)
    np.testing.assert_array_equal(prediction, [21.0, 10.0, 20.0, 11.0])
    assert prediction.index.tolist() == list(zip(request["group"], request["date"]))
    pd.testing.assert_frame_equal(request, original)
    sent = estimator.model.predict.call_args.args[0]
    assert sent[time_idx].max() == 111


def test_tft_rejects_a_delayed_horizon_instead_of_relabeling_predictions(tft_decoder):
    estimator = tft_decoder()
    request = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-01-16", periods=2)) * 2,
            "group": ["a", "a", "b", "b"],
            "time_idx": [115, 116, 115, 116],
        }
    )
    original = request.copy(deep=True)
    with pytest.raises(ValueError, match="consecutive.*training"):
        estimator.predict(request)
    estimator.model.predict.assert_not_called()
    pd.testing.assert_frame_equal(request, original)


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
