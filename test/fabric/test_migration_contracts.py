import json
import pickle
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from flaml import AutoML


@pytest.mark.parametrize(
    "blocked,imports",
    [
        (["sklearn", "pandas", "xgboost"], "from flaml.automl import *\nassert callable(register_automl_pipeline)"),
        (
            ["lightgbm", "xgboost"],
            "from flaml import AutoML\nimport numpy as np\n"
            "try:\n"
            "    AutoML(max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False).fit("
            "np.arange(80).reshape(40, 2), np.tile([0, 1], 20))\n"
            "except ImportError as error:\n"
            "    assert 'flaml[automl]' in str(error)\n"
            "else:\n"
            "    raise AssertionError('Missing default learner dependency was not reported')",
        ),
        (
            ["sklearn", "pandas", "xgboost"],
            "from flaml.automl import logger_formatter; from flaml.automl.training_log import training_log_reader",
        ),
        (
            ["xgboost"],
            "from flaml.default import RandomForestClassifier; from flaml.visualization import plot_slice\n"
            "import numpy as np\n"
            "X = np.arange(80).reshape(40, 2)\n"
            "model = RandomForestClassifier(n_estimators=4, max_depth=2, n_jobs=1).fit(X, np.tile([0, 1], 20))\n"
            "assert len(model.predict(X[:3])) == 3",
        ),
    ],
)
def test_lightweight_imports_do_not_require_full_automl(blocked, imports):
    code = f"import sys\nfor name in {blocked!r}: sys.modules[name] = None\n{imports}\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_history_only_pickle_does_not_initialize_tracking(monkeypatch):
    import mlflow

    from flaml.fabric.mlflow import MLflowIntegration

    automl = AutoML()
    automl.mlflow_integration = MLflowIntegration(only_history=True)
    payload = pickle.dumps(automl)
    client = Mock(side_effect=AssertionError("History loading must remain local"))
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", client)
    restored = pickle.loads(payload)
    assert restored.mlflow_integration.mlflow_client is None
    client.assert_not_called()


def test_old_automl_state_can_be_refitted():
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, mlflow_logging=False, verbose=0)
    state = automl.__getstate__()
    for key in ("featurization", "fit_kwargs_by_estimator", "custom_hp"):
        state["_settings"].pop(key, None)
    state.pop("_setting_keys", None)
    restored = AutoML.__new__(AutoML)
    restored.__setstate__(state)
    X = np.arange(80).reshape(40, 2)
    restored.fit(X, np.tile([0, 1], 20))
    assert len(restored.predict(X[:3])) == 3
    assert restored._featurization == "off"


def test_clone_preserves_lowcode_entrypoint():
    automl = AutoML(entrypoint="low-code", automl_mode="Quick", max_iter=1)
    copied = clone(automl)
    assert copied.get_params()["entrypoint"] == "low-code"
    assert copied.get_params()["automl_mode"] == "Quick"


def test_replay_requires_recorded_autofe(monkeypatch):
    from flaml.automl import ml
    from flaml.automl.task.factory import task_factory

    monkeypatch.setattr(ml, "Featurization", None)
    with pytest.raises(ImportError, match="autofe"):
        ml.train_estimator(
            X_train=None,
            y_train=None,
            config_dic={"n_estimators": 4, "fe.extraction": "PCA"},
            task=task_factory("classification"),
            estimator_name="rf",
        )


def test_augmentation_preserves_nonconsecutive_row_labels():
    from flaml.automl.task.generic_task import _TrainingPartitionAugmenter

    X = pd.DataFrame({"value": range(12)}, index=np.arange(12) * 10)
    y = pd.Series([1, 1] + [0] * 10, index=X.index)
    features, labels = _TrainingPartitionAugmenter().fit_resample(X, y)
    assert len(features) == len(labels)
    assert set(features.loc[np.asarray(labels) == 1, "value"]) == {0, 1}
    assert (labels == 1).sum() >= 20


def test_automl_history_preserves_explicit_numeric_none():
    from flaml.fabric.visualization import get_hp_df

    automl = AutoML()
    automl._best_estimator = "rf"
    automl.mlflow_integration = SimpleNamespace(
        infos=[
            {"metrics": {"validation_loss": 0.2}, "params": {"learner": "rf", "choice": None}},
            {"metrics": {"validation_loss": 0.1}, "params": {"learner": "rf", "choice": 1.0}},
        ]
    )
    parameters, scores = get_hp_df(automl)
    assert parameters["choice"].iloc[0] is None
    assert parameters["choice"].iloc[1] == 1.0
    np.testing.assert_allclose(scores, [0.8, 0.9])


def test_forecast_autofe_accepts_unlabeled_future_partition():
    from flaml.automl.time_series.ts_data import TimeSeriesDataset
    from flaml.fabric.autofe import Featurization

    frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=8), "value": range(8), "y": range(8)})
    history = TimeSeriesDataset(frame.iloc[:6], "ds", ["y"])
    featurizer = Featurization(params={"fe.numerical": "scaler_standard"}).fit(history)
    future = TimeSeriesDataset(frame.iloc[:6], "ds", ["y"], test_data=frame.iloc[6:].drop(columns="y"))
    transformed = featurizer.transform(future)
    assert len(transformed.test_data) == 2
    assert "y" not in transformed.test_data
    pd.testing.assert_series_equal(transformed.test_data["ds"], future.test_data["ds"])


def test_selection_recipe_survives_a_fit_that_drops_no_columns():
    from flaml.automl.task.factory import task_factory
    from flaml.fabric.autofe import Featurization

    task = task_factory("regression")
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 3.0, 4.0]})
    source = Featurization(params={"fe.selection": "variance"}, task=task).fit(X)
    config = json.loads(json.dumps(source.config))
    restored = Featurization(config=config, task=task)
    output = restored.fit_transform(pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [4.0, 4.0, 4.0]}))
    assert list(output.columns) == ["a"]


@pytest.mark.parametrize("explicit_validation", [False, True])
def test_forecast_search_never_fits_validation_rows(monkeypatch, explicit_validation):
    from flaml.automl.time_series.ts_data import TimeSeriesDataset
    from flaml.fabric.autofe import Featurization

    seen = []
    fit = Featurization.fit

    def capture(self, X, y=None):
        if isinstance(X, TimeSeriesDataset):
            seen.append(set(X.train_data["ds"]))
        return fit(self, X, y)

    monkeypatch.setattr(Featurization, "fit", capture)
    frame = pd.DataFrame(
        {"ds": pd.date_range("2026-01-01", periods=30), "feature": np.arange(30.0), "target": np.arange(30.0)}
    )
    train, validation = frame.iloc[:27], frame.iloc[27:]
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1 if explicit_validation else 2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.numerical": "scaler_standard"}},
    )
    options = (
        {"X_val": validation.drop(columns="target"), "y_val": validation["target"]}
        if explicit_validation
        else {"retrain_full": False}
    )
    automl.fit(
        dataframe=train if explicit_validation else frame,
        label="target",
        task="ts_forecast",
        metric="mse",
        period=3,
        eval_method="holdout",
        **options,
    )
    assert seen
    assert all(rows <= set(train["ds"]) for rows in seen)
    assert all(rows.isdisjoint(validation["ds"]) for rows in seen)


def test_forecast_autofe_rejects_unseen_validation_labels():
    training = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=24), "feature": np.arange(24.0)})
    validation = pd.DataFrame({"ds": pd.date_range("2026-01-25", periods=3), "feature": np.arange(24.0, 27.0)})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        retrain_full=False,
    )
    with pytest.raises(ValueError, match="unseen"):
        automl.fit(
            training,
            pd.Series(["a", "b"] * 12, name="label"),
            X_val=validation,
            y_val=pd.Series(["validation-only"] * 3, name="label"),
            period=3,
            task="ts_forecast_classification",
            metric="accuracy",
        )


def test_forecast_fit_preserves_caller_timestamp_strings():
    frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=12).strftime("%Y-%m-%d"), "y": range(12)})
    original = frame.copy(deep=True)
    automl = AutoML(estimator_list=["naive"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False)
    automl.fit(dataframe=frame, label="y", task="ts_forecast", period=2)
    pd.testing.assert_frame_equal(frame, original)


def test_forecast_preprocessor_refit_replaces_fitted_column_names():
    from flaml.automl.pipeline import _TaskPreprocessor
    from flaml.automl.task.factory import task_factory

    task = task_factory("ts_forecast")
    task.time_col, task.target_names = "old_date", ["old_target"]
    preprocessor = _TaskPreprocessor(task)
    X = pd.DataFrame({"new_date": pd.date_range("2026-01-01", periods=5), "feature": range(5)})
    y = pd.Series(range(5), name="new_target")
    result = preprocessor.fit_transform(X, y)
    assert result.time_col == "new_date"
    assert result.target_names == ["new_target"]
    assert task.time_col == "old_date"
    assert task.target_names == ["old_target"]


def test_exported_forecast_pipeline_refits_embedded_string_targets():
    from flaml.automl.time_series.ts_data import TimeSeriesDataset

    frame = pd.DataFrame(
        {"ds": pd.date_range("2026-01-01", periods=30), "feature": np.arange(30.0), "label": np.tile(["a", "b"], 15)}
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.numerical": "scaler_standard"}},
    )
    automl.fit(dataframe=frame, label="label", task="ts_forecast_classification", period=2, eval_method="holdout")
    pipeline = automl.automl_pipeline
    data = TimeSeriesDataset(frame, "ds", "label")
    pipeline.fit(data, estimator__period=2)
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=2), "feature": [30.0, 31.0]})
    assert set(pipeline.predict(future)) <= {"a", "b"}
    pd.testing.assert_frame_equal(data.train_data, frame)


def test_forecast_duplicates_preserve_panel_rows_and_reject_conflicts():
    from flaml.automl.task.time_series_task import remove_ts_duplicates

    frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3), "y": range(3)})
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    pd.testing.assert_frame_equal(remove_ts_duplicates(duplicated, "ds"), frame)
    conflict = frame.iloc[[0]].assign(y=100)
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        remove_ts_duplicates(pd.concat([frame, conflict]), "ds")
    panel = pd.concat([frame.assign(group="a"), frame.assign(group="b")])
    pd.testing.assert_frame_equal(remove_ts_duplicates(panel, "ds", allow_repeated_timestamps=True), panel)


@pytest.mark.parametrize("task_name", ["ts_forecast", "ts_forecast_panel"])
def test_panel_cv_fails_with_actionable_holdout_guidance(task_name):
    frame = pd.DataFrame(
        {
            "ds": list(pd.date_range("2026-01-01", periods=10)) * 2,
            "group": ["a"] * 10 + ["b"] * 10,
            "time_idx": list(range(10)) * 2,
            "target": np.arange(20.0),
        }
    )
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, mlflow_logging=False, verbose=0)
    with pytest.raises(ValueError, match="use eval_method='holdout'"):
        automl.fit(dataframe=frame, label="target", task=task_name, period=2, group_ids=["group"], eval_method="cv")


@pytest.mark.parametrize("extra", ["label_transformer", "resampler"])
def test_fit_predict_cannot_bypass_target_or_sampling_contracts(extra):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder

    from flaml.automl.pipeline import AutoMLPipeline
    from flaml.automl.task.generic_task import _TrainingPartitionAugmenter

    option = LabelEncoder() if extra == "label_transformer" else _TrainingPartitionAugmenter()
    pipeline = AutoMLPipeline([("model", KMeans(n_clusters=2, n_init=1))], **{extra: option})
    with pytest.raises(NotImplementedError, match="use fit"):
        pipeline.fit_predict(np.arange(8).reshape(4, 2), ["a", "a", "b", "b"])


@pytest.mark.parametrize("starting_points", [{"rf": {"fe.extraction": "null"}}, {"rf": [{"fe.extraction": "null"}]}])
def test_force_mode_rejects_noop_warm_start(starting_points):
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="force",
        starting_points=starting_points,
    )
    with pytest.raises(ValueError, match="AutoFE starting"):
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))


def test_json_feature_recipe_retains_reordered_numeric_column():
    from flaml.automl.task.factory import task_factory
    from flaml.fabric.autofe import Featurization

    frame = pd.DataFrame({42: [1.0, 2.0, 3.0], "group": pd.Categorical(["a", "b", "a"])})
    task = task_factory("regression")
    source = Featurization(params={"fe.numerical": "scaler_standard"}, task=task)
    expected = source.fit_transform(frame)
    restored = Featurization(config=json.loads(json.dumps(source.config)), task=task)
    actual = restored.fit_transform(frame[["group", 42]])
    pd.testing.assert_frame_equal(actual, expected)


def test_numeric_categories_with_missing_values_remain_encodable():
    from flaml.automl.data import DataTransformer
    from flaml.automl.pipeline import _TaskPreprocessor
    from flaml.fabric.autofe import Featurization

    frame = pd.DataFrame({"category": pd.Categorical([1, 2, None, 1])})
    preprocessor = _TaskPreprocessor("classification", DataTransformer(), preserve_features=True)
    normalized = preprocessor.fit_transform(frame, np.array([0, 1, 0, 1]))
    featurizer = Featurization(params={"fe.categorical": "ordinal"})
    transformed = featurizer.fit_transform(normalized)
    assert np.isfinite(transformed.to_numpy()).all()
    future = pd.DataFrame({"category": pd.Categorical([3, None, 1])})
    prediction_input = featurizer.transform(preprocessor.transform(future))
    assert np.isfinite(prediction_input.to_numpy()).all()
    assert prediction_input.iloc[2, 0] == transformed.iloc[0, 0]


@pytest.mark.parametrize("fail", [False, True])
def test_local_tune_preserves_autolog_options_and_parent_run(tmp_path, fail):
    code = f"""
import copy
import mlflow
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
from flaml import tune
mlflow.set_tracking_uri({(tmp_path / 'tracking').as_uri()!r})
mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)
before = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
def objective(config):
    assert AUTOLOGGING_INTEGRATIONS['sklearn']['log_models'] is False
    if {fail!r}:
        raise ValueError('objective failed')
    return {{'loss': 1.0}}
with mlflow.start_run() as parent:
    try:
        tune.run(objective, config={{}}, metric='loss', mode='min', num_samples=1, use_ray=False, verbose=0)
    except ValueError as exc:
        assert {fail!r} and str(exc) == 'objective failed'
    else:
        assert not {fail!r}
    assert mlflow.active_run().info.run_id == parent.info.run_id
for name, options in before.items():
    assert AUTOLOGGING_INTEGRATIONS[name] == options, (name, before, AUTOLOGGING_INTEGRATIONS)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("pipeline", [False, True])
def test_wrong_run_never_receives_model_artifacts(monkeypatch, pipeline):
    import mlflow
    import mlflow.sklearn

    import flaml.fabric.mlflow as integration_module

    integration = integration_module.MLflowIntegration(only_history=True)
    integration.parent_run_id = "parent"
    monkeypatch.setattr(mlflow, "active_run", lambda: SimpleNamespace(info=SimpleNamespace(run_id="unrelated")))
    write = Mock()
    executor = Mock()
    monkeypatch.setattr(mlflow.sklearn, "log_model", write)
    monkeypatch.setattr(integration_module, "executor", executor)
    if pipeline:
        result = integration._log_pipeline(object(), "sklearn", "model", None, "target", "rf")
    else:
        result = integration.log_model(object(), "rf", run_id="target")
    assert "Error" in result
    write.assert_not_called()
    executor.submit.assert_not_called()


def test_nonwinning_trial_logs_its_own_model(monkeypatch):
    import mlflow

    from flaml.fabric.mlflow import MLflowIntegration

    captured = []

    def capture(self, automl, model, estimator, signature=None, run_id=None):
        captured.append((run_id, model.estimator.n_estimators))

    monkeypatch.setattr(MLflowIntegration, "pickle_and_log_automl_artifacts", capture)

    def metric(X_val, y_val, estimator, *args, **kwargs):
        return (0.1 if estimator.params["n_estimators"] == 4 else 0.9), {}

    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        model_history=True,
        log_type="all",
        retrain_full=False,
        eval_method="holdout",
        metric=metric,
        starting_points={"rf": [{"n_estimators": 4}, {"n_estimators": 8}]},
    )
    with mlflow.start_run() as parent:
        automl.fit(np.arange(120).reshape(60, 2), np.tile([0, 1], 30))
    assert [trees for run_id, trees in captured if run_id != parent.info.run_id] == [4, 8]


def test_wait_futures_drains_writes_submitted_by_model_tasks(monkeypatch):
    import importlib
    from concurrent.futures import Future

    module = importlib.import_module("flaml.automl.automl")
    model, requirements = Future(), Future()
    integration = SimpleNamespace(futures={}, futures_log_model={model: "model"})

    def completed(futures):
        if model in futures:
            integration.futures[requirements] = "requirements"
            model.set_result(None)
            return iter([model])
        if requirements in futures:
            requirements.set_result(None)
            return iter([requirements])
        return iter([])

    monkeypatch.setattr(module, "as_completed", completed)
    automl = AutoML()
    automl.mlflow_integration = integration
    automl.wait_futures()
    assert requirements.done()


def test_mlflow_adoption_ignores_unowned_runs(monkeypatch):
    import mlflow

    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration(only_history=True, extra_tag={"extra_tag.sid": "caller-shared"})
    integration.autolog = True
    integration.best_iteration = 0
    integration.experiment_id = "experiment"
    integration.mlflow_client = Mock()
    foreign = SimpleNamespace(
        info=SimpleNamespace(run_id="foreign", start_time=(integration.start_time + 1) * 1000),
        data=SimpleNamespace(params={}, metrics={}, tags={"flaml.owner_id": "someone-else"}),
    )
    search = Mock(return_value=[foreign, foreign])
    monkeypatch.setattr(mlflow, "search_runs", search)
    integration.adopt_children()
    assert integration.extra_tag["flaml.owner_id"] in search.call_args.kwargs["filter_string"]
    integration.mlflow_client.delete_run.assert_not_called()
    integration.mlflow_client.set_tag.assert_not_called()


def test_fit_argument_values_are_not_reporting_metadata(monkeypatch):
    from flaml.fabric.mlflow import MLflowIntegration

    marker = "synthetic-private-fit-value"
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        automl_mode="Quick",
        entrypoint="low-code",
    )

    class ReachedSearch(Exception):
        pass

    monkeypatch.setattr(automl, "_search", Mock(side_effect=ReachedSearch))
    with pytest.raises(ReachedSearch):
        automl.fit(
            np.arange(80).reshape(40, 2),
            np.tile([0, 1], 20),
            fit_kwargs_by_estimator={"rf": {"hub_token": marker, "opaque_payload": {"value": marker}}},
        )
    assert marker not in json.dumps(automl._automl_user_configurations)
    assert automl._automl_user_configurations["automl_mode"] == "Quick"
    assert automl._automl_user_configurations["entrypoint"] == "low-code"
    automl._iter_per_learner = {"rf": 1}
    automl._state.time_from_start = 1.0
    state = SimpleNamespace(
        val_loss=0.1,
        best_loss=0.1,
        config={},
        trial_time=0.1,
        sample_size=40,
        learner_class=AutoML,
        metric_for_logging=None,
    )
    integration = MLflowIntegration(only_history=True)
    integration.record_state(automl, state, "rf")
    assert marker not in json.dumps(integration.infos)
    integration.mlflow_client = Mock()
    integration._log_automl_configurations("run")
    assert marker not in str(integration.mlflow_client.log_text.call_args_list)


@pytest.mark.parametrize("pipeline", [False, True])
def test_failed_model_write_closes_only_its_owned_run(monkeypatch, pipeline):
    import mlflow
    import mlflow.sklearn

    from flaml.fabric.mlflow import MLflowIntegration

    integration = MLflowIntegration(only_history=True)
    with mlflow.start_run() as child:
        target = child.info.run_id
    monkeypatch.setattr(mlflow.sklearn, "log_model", Mock(side_effect=RuntimeError("artifact failed")))
    with mlflow.start_run() as parent:
        integration.parent_run_id = parent.info.run_id
        with pytest.raises(RuntimeError, match="artifact failed"):
            if pipeline:
                integration._log_pipeline(object(), "sklearn", "model", None, target, "rf")
            else:
                integration.log_model(object(), "rf", run_id=target)
        assert mlflow.active_run().info.run_id == parent.info.run_id


@pytest.mark.parametrize(
    "values",
    [
        ["9007199254740992", "9007199254740993", "1.5"],
        [Decimal("9007199254740992.5"), Decimal("1"), Decimal("2")],
        [Decimal("1e1000"), Decimal("1.5"), Decimal("2.5")],
    ],
)
def test_numeric_inference_does_not_destroy_exact_large_values(values):
    from flaml.automl.data import auto_convert_dtypes_pandas

    converted, schema = auto_convert_dtypes_pandas(pd.DataFrame({"value": values}), sample_ratio=1)
    assert schema["value"] == "string"
    assert converted["value"].tolist() == [str(value) for value in values]


def test_datetime_hint_does_not_stringify_opaque_objects():
    from flaml.automl.data import auto_convert_dtypes_pandas

    class Opaque:
        def __str__(self):
            raise RuntimeError("opaque object")

    value = Opaque()
    frame = pd.DataFrame({"opaque": [value, value], "number": ["1", "2"]})
    converted, schema = auto_convert_dtypes_pandas(frame)
    assert converted["opaque"].iloc[0] is value
    assert converted["number"].tolist() == [1, 2]


@pytest.mark.parametrize("seq2seq", [False, True])
def test_trainer_log_forwards_new_transformers_arguments(monkeypatch, seq2seq):
    import importlib.util
    from pathlib import Path

    calls = []

    class BaseTrainer:
        def log(self, logs, *args, **kwargs):
            calls.append(("base", logs, args, kwargs))

    class SequenceTrainer(BaseTrainer):
        def log(self, logs, *args, **kwargs):
            calls.append(("sequence", logs, args, kwargs))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Seq2SeqTrainer=SequenceTrainer))
    path = Path(__file__).resolve().parents[2] / "flaml" / "automl" / "nlp" / "huggingface" / "trainer.py"
    spec = importlib.util.spec_from_file_location("trainer_log_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = module.TrainerForAuto()
    trainer._is_seq2seq = seq2seq
    logs = {"epoch": 1.0, "loss": 0.5}
    trainer.log(logs, 17, start_time=123.0)
    assert calls == [("sequence" if seq2seq else "base", logs, (17,), {"start_time": 123.0})]
    assert trainer.intermediate_results[1.0] == logs


def test_numpy_feature_recipe_roundtrip_preserves_scaling():
    from flaml.automl.task.factory import task_factory
    from flaml.fabric.autofe import Featurization

    values = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    task = task_factory("regression")
    source = Featurization(params={"fe.numerical": "scaler_standard"}, task=task)
    expected = source.fit_transform(values)
    restored = Featurization(config=json.loads(json.dumps(source.config)), task=task)
    pd.testing.assert_frame_equal(restored.fit_transform(values), expected)
