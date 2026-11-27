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
    from sklearn.base import clone

    from flaml.automl.pipeline import make_automl_feature_transformer, make_automl_pipeline

    features = original.drop(columns="target")
    pipeline = clone(make_automl_pipeline(automl, automl.model))
    pipeline.fit(features, original["target"], estimator__period=2, estimator__group_ids=["country", "store"])
    assert len(pipeline.predict(features)) == len(features)
    transformed = clone(make_automl_feature_transformer(automl, automl.model)).fit_transform(
        features, original["target"]
    )
    pd.testing.assert_series_equal(transformed["store"], features["store"])
    pd.testing.assert_series_equal(transformed["country"], features["country"])
    pd.testing.assert_series_equal(transformed[time_idx], features[time_idx])
    assert "static" in transformed


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


@pytest.mark.parametrize("timezone", [None, "UTC"])
@pytest.mark.parametrize("change", ["shift", "shorten"])
def test_forecast_replay_rejects_unreconstructable_date_boundaries(tmp_path, timezone, change):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30, tz=timezone), "target": np.arange(30.0)})
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
    changed = frame.copy()
    if change == "shift":
        changed["date"] += pd.Timedelta(days=1)
    else:
        changed = changed.iloc[:26]
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    with pytest.raises(ValueError, match="recorded training boundary"):
        restored.retrain_from_log(log, dataframe=changed, label="target", task="ts_forecast", period=3)


def test_legacy_forecast_replay_honors_embedded_validation(tmp_path):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "target": np.arange(30.0)})
    dataset = TimeSeriesDataset(frame.iloc[:27], "date", "target", test_data=frame.iloc[27:])
    log = tmp_path / "legacy.log"
    source = AutoML(
        estimator_list=["naive"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        eval_method="holdout",
        retrain_full=False,
    )
    source.fit(X_train=dataset, task="ts_forecast", period=3, log_file_name=str(log))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    for record in records:
        record.pop("training_info", None)
    log.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    restored.retrain_from_log(str(log), X_train=dataset, task="ts_forecast", period=3)
    assert restored.model.train_end_date == frame["date"].iloc[26]
    assert len(restored.model.X_train.train_data) == 27


@pytest.mark.parametrize("time_idx", ["time_idx", "step"])
def test_panel_replay_preserves_the_exact_sampled_group_rows(tmp_path, time_idx):
    class PanelRows(TimeSeriesEstimator):
        @classmethod
        def search_space(cls, **kwargs):
            return {}

        def fit(self, X_train, y_train=None, **kwargs):
            super().fit(X_train, y_train, **kwargs)
            self.training_keys = list(X_train.train_data[["store", time_idx]].itertuples(index=False, name=None))
            self._model = DummyRegressor().fit(np.zeros((len(self.training_keys), 1)), X_train.train_data["target"])
            return 0.0

        def predict(self, X, **kwargs):
            frame = X.test_data if isinstance(X, TimeSeriesDataset) else X
            return self._model.predict(np.zeros((len(frame), 1)))

    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2026-01-01", periods=20), 2),
            "store": ["b", "a"] * 20,
            time_idx: np.repeat(np.arange(100, 120), 2),
            "target": np.arange(40.0),
        }
    )
    dataset = TimeSeriesDataset(frame, "date", "target", time_idx=time_idx)
    log = tmp_path / "panel.log"
    source = AutoML(max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False, retrain_full=False, keep_search_state=True)
    source.add_learner("panel_rows", PanelRows)
    source.fit(
        X_train=dataset,
        task="ts_forecast_panel",
        period=2,
        group_ids=["store"],
        estimator_list=["panel_rows"],
        metric="mae",
        log_file_name=str(log),
    )
    record = next(json.loads(line) for line in log.read_text().splitlines() if "validation_loss" in line)
    sampled = source._state._train_with_config("panel_rows", record["config"], sample_size=5)[0]
    expected = sampled.training_keys
    record["sample_size"] = 5
    log.write_text(json.dumps(record) + "\n")
    restored = AutoML(n_jobs=1, mlflow_logging=False, verbose=0)
    restored.add_learner("panel_rows", PanelRows)
    restored.retrain_from_log(str(log), X_train=dataset, task="ts_forecast_panel", period=2, group_ids=["store"])
    assert len(expected) == 5
    assert restored.model.training_keys == expected


@pytest.mark.parametrize("location", ["top", "estimator"])
@pytest.mark.parametrize("group_ids", ["store", ("store",), np.array(["store", "country"])])
def test_panel_group_types_are_validated_before_copying(location, group_ids):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "time_idx": range(10),
            "store": ["a"] * 10,
            "country": ["same"] * 10,
            "target": np.arange(10.0),
        }
    )
    kwargs = (
        {"group_ids": group_ids}
        if location == "top"
        else {"fit_kwargs_by_estimator": {"tft": {"group_ids": group_ids}}}
    )
    automl = AutoML(estimator_list=["tft"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    with pytest.raises(ValueError, match="nonempty list.*group_ids"):
        automl.fit(dataframe=frame, label="target", task="ts_forecast_panel", period=2, **kwargs)


@pytest.mark.parametrize("invalid", ["missing", "null"])
def test_panel_group_keys_must_exist_and_be_nonmissing(invalid):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "time_idx": range(10),
            "store": ["a"] * 10,
            "target": np.arange(10.0),
        }
    )
    if invalid == "missing":
        frame = frame.drop(columns="store")
    else:
        frame.loc[0, "store"] = None
    automl = AutoML(estimator_list=["tft"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    with pytest.raises(ValueError, match="nonmissing group keys"):
        automl.fit(dataframe=frame, label="target", task="ts_forecast_panel", period=2, group_ids=["store"])


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
        evaluated = []

        def validation_metric(X_val, y_val, estimator, *args):
            assert isinstance(X_val, TimeSeriesDataset)
            pd.testing.assert_series_equal(
                X_val.test_data["date"].reset_index(drop=True),
                frame["date"].iloc[30:].reset_index(drop=True),
            )
            predicted = np.asarray(estimator.predict(X_val)).reshape(-1)
            expected = np.asarray(X_val.y_val).reshape(-1)
            evaluated.append(len(predicted))
            return float(np.mean(np.abs(predicted - expected))), {}

        data = frame.copy()
        if strings:
            data["date"] = data["date"].astype(str)
        original = data.copy(deep=True)
        automl = AutoML(
            estimator_list=["rf"],
            max_iter=2,
            n_jobs=1,
            verbose=0,
            mlflow_logging=False,
            featurization=featurization,
            metric=validation_metric,
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
        assert evaluated and set(evaluated) == {6}
        predictions.append(automl.predict(data.iloc[30:].drop(columns="target")))
        pd.testing.assert_frame_equal(data, original)
    np.testing.assert_allclose(predictions[0], predictions[1])


@pytest.mark.parametrize("period", [0, -1])
def test_nonpositive_forecast_period_is_rejected_before_splitting(period):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=12), "target": np.arange(12.0)})
    automl = AutoML(estimator_list=["naive"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    with pytest.raises(ValueError, match="period.*positive"):
        automl.fit(dataframe=frame, label="target", task="ts_forecast", period=period)


@pytest.mark.parametrize("missing_partition", [False, True])
def test_empty_forecast_validation_does_not_predict_or_divide_by_zero(missing_partition):
    from flaml.automl.ml import get_val_loss
    from flaml.automl.task.factory import task_factory

    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4), "target": range(4)})
    dataset = TimeSeriesDataset(frame, "date", "target")
    if missing_partition:
        dataset.test_data = None
    estimator = Mock(spec=["fit", "predict"])
    estimator.predict.side_effect = AssertionError("An empty validation partition must not be predicted.")
    loss, metrics, _, latency = get_val_loss(
        {}, estimator, dataset, None, dataset, None, None, None, "mae", task_factory("ts_forecast")
    )
    assert np.isinf(loss)
    assert metrics["pred_time"] == latency == 0
    estimator.predict.assert_not_called()


def test_validation_latency_does_not_recount_the_validation_frame(monkeypatch):
    from flaml.automl import ml
    from flaml.automl.task.factory import task_factory

    class ValidationFrame:
        reads = 0

        @property
        def shape(self):
            self.reads += 1
            return (2, 1)

    validation = ValidationFrame()
    monkeypatch.setattr(ml, "get_y_pred", lambda *args: np.array([1.0, 2.0]))
    loss, _, _, _ = ml._eval_estimator(
        {}, object(), None, None, validation, np.array([1.0, 2.0]), None, None, "mae", task_factory("regression")
    )
    assert loss == 0
    assert validation.reads == 1


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

        def predict(data, **kwargs):
            frame = data.frame if isinstance(data, SimpleNamespace) else data
            horizon = data.horizon if isinstance(data, SimpleNamespace) else estimator.max_prediction_length
            groups = [group for group in ("b", "a") if group in set(frame["group"])]
            values = np.array(
                [[(20.0 if group == "b" else 10.0) + step for step in range(horizon)] for group in groups]
            )
            if kwargs.get("return_index"):
                starts = [int(frame.loc[frame["group"] == group, time_idx].max()) - horizon + 1 for group in groups]
                return SimpleNamespace(output=values, index=pd.DataFrame({"group": groups, time_idx: starts}))
            return SimpleNamespace(numpy=lambda: values)

        estimator._model = SimpleNamespace(predict=Mock(side_effect=predict), dataset_parameters={})
        estimator._prediction_dataset = Mock(
            side_effect=lambda frame, horizon: SimpleNamespace(frame=frame, horizon=horizon)
        )
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
    assert getattr(sent, "frame", sent)[time_idx].max() == 111


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


def test_tft_short_and_unequal_horizons_keep_the_latest_encoder_history(tft_decoder):
    estimator = tft_decoder()
    request = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-12", "2026-01-11", "2026-01-11"]),
            "group": ["b", "a", "b"],
            "time_idx": [111, 110, 110],
        }
    )
    prediction = estimator.predict(request)
    np.testing.assert_array_equal(prediction, [21.0, 10.0, 20.0])
    assert estimator._prediction_dataset.call_count == 2
    for call in estimator._prediction_dataset.call_args_list:
        frame, horizon = call.args
        assert set(frame["time_idx"]) == set(range(106, 110 + horizon))


def test_tft_groups_with_different_origins_get_separate_prediction_boundaries(tft_decoder):
    estimator = tft_decoder()
    estimator.data.loc[estimator.data["group"] == "b", "time_idx"] += 7
    request = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-11", "2026-01-11"]), "group": ["a", "b"], "time_idx": [110, 117]}
    )
    np.testing.assert_array_equal(estimator.predict(request), [10.0, 20.0])
    assert estimator._prediction_dataset.call_count == 2
    for call in estimator._prediction_dataset.call_args_list:
        frame, horizon = call.args
        origins = frame.groupby("group")["time_idx"].max() - horizon + 1
        assert origins.nunique() == 1


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-01-09", "2026-01-10"],
        ["2026-01-13", "2026-01-12"],
        ["2026-01-12", "2026-01-12"],
        ["NaT", "2026-01-13"],
    ],
)
def test_tft_irregular_calendar_rejects_invalid_future_timestamps(tft_decoder, dates):
    estimator = tft_decoder()
    estimator.data.loc[(estimator.data["group"] == "a") & (estimator.data["time_idx"] == 109), "date"] = pd.Timestamp(
        "2026-01-11"
    )
    request = pd.DataFrame({"date": dates, "group": ["a", "a"], "time_idx": [110, 111]})
    with pytest.raises(ValueError, match="timestamps"):
        estimator.predict(request)
    estimator.model.predict.assert_not_called()


@pytest.mark.parametrize("irregular", [False, True])
def test_tft_calendar_index_inference_requires_regular_history(tft_decoder, irregular):
    estimator = tft_decoder()
    request = pd.DataFrame({"date": pd.date_range("2026-01-11", periods=2), "group": ["a", "a"]})
    if irregular:
        estimator.data.loc[
            (estimator.data["group"] == "a") & (estimator.data["time_idx"] == 109), "date"
        ] = pd.Timestamp("2026-01-11")
        request["date"] += pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="explicit.*indexes"):
            estimator.predict(request)
        estimator.model.predict.assert_not_called()
    else:
        np.testing.assert_array_equal(estimator.predict(request), [10.0, 11.0])


@pytest.mark.parametrize("invalid", ["unknown_group", "missing_group", "null_group", "float_index", "too_long"])
def test_tft_rejects_invalid_coordinates_before_model_prediction(tft_decoder, invalid):
    estimator = tft_decoder()
    request = pd.DataFrame(
        {"date": pd.date_range("2026-01-11", periods=2), "group": ["a", "a"], "time_idx": [110, 111]}
    )
    if invalid == "unknown_group":
        request["group"] = "unknown"
    elif invalid == "missing_group":
        request = request.drop(columns="group")
    elif invalid == "null_group":
        request.loc[0, "group"] = None
    elif invalid == "float_index":
        request["time_idx"] = request["time_idx"].astype(float)
    else:
        request.loc[2] = [pd.Timestamp("2026-01-13"), "a", 112]
    with pytest.raises(ValueError):
        estimator.predict(request)
    estimator.model.predict.assert_not_called()


@pytest.mark.parametrize("missing_partition", [False, True])
def test_tft_empty_validation_returns_no_forecasts(tft_decoder, missing_partition):
    estimator = tft_decoder()
    dataset = TimeSeriesDataset(estimator.data.copy(), "date", "target")
    if missing_partition:
        dataset.test_data = None
    assert estimator.predict(dataset).empty
    estimator.model.predict.assert_not_called()


def test_tft_training_metric_does_not_abort_future_validation(tft_decoder, caplog):
    from flaml.automl.ml import _eval_estimator
    from flaml.automl.task.factory import task_factory

    estimator = tft_decoder()
    future = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-01-11", periods=2)) * 2,
            "group": ["a", "a", "b", "b"],
            "time_idx": [110, 111, 110, 111],
            "target": [10.0, 11.0, 20.0, 21.0],
        }
    )
    dataset = TimeSeriesDataset(estimator.data.copy(), "date", "target", test_data=future)
    loss, metrics, _, _ = _eval_estimator(
        {},
        estimator,
        dataset,
        None,
        dataset,
        None,
        None,
        None,
        "mae",
        task_factory("ts_forecast_panel"),
        log_training_metric=True,
    )
    assert loss == 0
    assert "train_loss" not in metrics
    assert "training metric" in caplog.text.lower()
    assert estimator.model.predict.call_count == 1


@pytest.mark.parametrize("minimum_encoder", [4, 8])
def test_real_tft_short_mixed_horizons_use_the_correct_forecast_origin(tmp_path, monkeypatch, minimum_encoder):
    pytest.importorskip("pytorch_forecasting")
    import torch

    from flaml.automl.time_series.tft import TemporalFusionTransformerEstimator

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        frame = pd.DataFrame(
            {
                "date": list(pd.date_range("2026-01-01", periods=16)) * 2,
                "country": ["same"] * 32,
                "store": ["a"] * 16 + ["b"] * 16,
                "time_idx": list(range(100, 116)) * 2,
                "target": np.arange(32.0) + 1,
            }
        )
        estimator = TemporalFusionTransformerEstimator(
            task="ts_forecast_panel",
            hidden_size=4,
            hidden_continuous_size=2,
            attention_head_size=1,
            dropout=0.1,
            learning_rate=0.01,
        )
        estimator.fit(
            TimeSeriesDataset(frame, "date", "target"),
            None,
            period=3,
            group_ids=["country", "store"],
            static_categoricals=["country", "store"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            lags={"target": [2]},
            max_encoder_length=8,
            min_encoder_length=minimum_encoder,
            max_epochs=1,
            batch_size=4,
            log_dir=str(tmp_path),
        )
        observed = []
        predict = estimator.model.predict

        def inspect_origin(data, **kwargs):
            assert data.reals == estimator.model.hparams.x_reals
            result = predict(data, **kwargs)
            assert set(result.index["time_idx"]) == {116}
            observed.extend(result.index["store"].tolist())
            return result

        monkeypatch.setattr(estimator.model, "predict", inspect_origin)
        request = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-18", "2026-01-17", "2026-01-17"]),
                "country": ["same"] * 3,
                "store": ["b", "a", "b"],
                "time_idx": [117, 116, 116],
            },
            index=[9, 2, 5],
        )
        prediction = estimator.predict(request)
        assert np.isfinite(prediction.to_numpy()).all()
        assert prediction.index.tolist() == list(
            request[["country", "store", "date"]].itertuples(index=False, name=None)
        )
        assert sorted(observed) == ["a", "b"]
    finally:
        torch.set_num_threads(previous_threads)


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
