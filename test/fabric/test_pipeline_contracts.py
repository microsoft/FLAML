import pickle
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from flaml import AutoML
from flaml.automl.model import RandomForestEstimator
from flaml.automl.time_series import TimeSeriesDataset
from flaml.fabric.autofe import Featurization


def fit_classifier(labels=(0, 1), skip_transform=False, log_file_name="", max_iter=1):
    X = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 3)), columns=["a", "b", "c"])
    y = np.asarray(labels)[(X["a"] > 0).astype(int)]
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=max_iter,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        skip_transform=skip_transform,
        log_file_name=log_file_name,
        starting_points={
            "rf": {
                "n_estimators": 4,
                "max_features": 1.0,
                "max_leaves": 4,
                "fe.selection": "null",
                "fe.categorical": "ordinal",
                "fe.extraction": "LDA",
            }
        },
    )
    automl.fit(X, y)
    assert automl.model.autofe is not None
    return automl, X, y


@pytest.mark.parametrize("labels", [("approve", "deny"), (10, 20), (0, 1)])
def test_exported_pipeline_preserves_target_labels(labels):
    automl, X, y = fit_classifier(labels)
    pipeline = automl.automl_pipeline
    np.testing.assert_array_equal(pipeline.predict(X), automl.predict(X))
    np.testing.assert_array_equal(pipeline.classes_, automl.classes_)
    np.testing.assert_allclose(pipeline.predict_proba(X), automl.predict_proba(X))
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(restored.predict(X), automl.predict(X))


@pytest.mark.parametrize("labels", [("approve", "deny"), (10, 20)])
def test_registered_pipeline_retains_original_labels(tmp_path, monkeypatch, labels):
    mlflow = pytest.importorskip("mlflow")
    from flaml.automl import register_automl_pipeline

    automl, X, _ = fit_classifier(labels)
    original_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    try:
        with mlflow.start_run():
            version = register_automl_pipeline(automl, model_name="original-labels")
        restored = mlflow.sklearn.load_model(f"models:/original-labels/{version.version}")
        np.testing.assert_array_equal(restored.predict(X), automl.predict(X))
        np.testing.assert_array_equal(restored.classes_, automl.classes_)
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(original_uri)


def test_pipeline_omits_disabled_task_preprocessor():
    automl, X, _ = fit_classifier(skip_transform=True)
    assert automl._transformer is False
    np.testing.assert_array_equal(automl.automl_pipeline.predict(X), automl.predict(X))
    assert automl.feature_transformer.transform(X).shape == (len(X), 1)


def test_trial_pipeline_uses_its_own_featurizer(monkeypatch):
    from flaml.fabric.mlflow import MLflowIntegration

    automl, X, y = fit_classifier()
    X_processed = automl._transformer.transform(X)
    trial_model = RandomForestEstimator(task="binary", n_estimators=2, n_jobs=1)
    trial_model.autofe = None
    trial_model.fit(X_processed, y)
    integration = MLflowIntegration.__new__(MLflowIntegration)
    integration._do_log_model = True
    integration.resume_params = {}
    log_pipeline = Mock()
    monkeypatch.setattr(integration, "_log_pipeline", log_pipeline)
    integration.pickle_and_log_automl_artifacts(automl, trial_model, "rf", run_id="trial")
    saved = log_pipeline.call_args.args[0]
    np.testing.assert_array_equal(saved.predict(X), trial_model.predict(X_processed))
    assert len(automl.feature_transformer.steps) == 2


@pytest.mark.parametrize("fitted", [False, True])
def test_featurization_clone_keeps_initialization_not_learned_state(fitted):
    source = Featurization(params={"fe.numerical": "scaler_standard"})
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    if fitted:
        source.fit(data)
    copied = clone(source)
    assert copied.params == source.params
    assert not copied.__sklearn_is_fitted__()
    transformed = copied.fit_transform(data)
    np.testing.assert_allclose(transformed["a"], [-1.22474487, 0.0, 1.22474487])
    copied.set_params(params={"fe.numerical": "scaler_minmax"})
    np.testing.assert_allclose(copied.fit_transform(data)["a"], [0.0, 0.5, 1.0])


def test_reconstructed_autofe_pipeline_supports_cross_validation():
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) ** 2})
    pipeline = Pipeline(
        [
            ("autofe", Featurization(params={"fe.numerical": "scaler_standard"})),
            ("model", RandomForestRegressor(n_estimators=2, random_state=7)),
        ]
    )
    assert np.isfinite(cross_val_score(pipeline, X, X["a"], cv=2)).all()


def test_unfitted_and_failed_refit_transform_raise():
    featurization = Featurization(params={"fe.numerical": "scaler_standard"})
    data = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(NotFittedError):
        featurization.transform(data)
    featurization.fit(data)
    with pytest.raises(ValueError):
        featurization.fit(pd.DataFrame({"a": [1.0, np.inf]}))
    with pytest.raises(NotFittedError):
        featurization.transform(data)


def test_successful_noop_display_and_transform():
    data = pd.DataFrame({"a": [1.0, 2.0]})
    featurization = Featurization(params={}).fit(data)
    assert isinstance(repr(featurization), str)
    assert "text/plain" in featurization._repr_mimebundle_()
    pd.testing.assert_frame_equal(featurization.transform(data), data)


def test_time_series_featurization_retains_timestamp_context():
    train = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=10), "y": np.arange(10), "x": np.arange(10, dtype=float)}
    )
    future = pd.DataFrame({"ds": pd.date_range("2025-01-11", periods=2), "x": [10.0, 11.0]})
    featurization = Featurization(params={"fe.extraction": "PCA"}).fit(TimeSeriesDataset(train, "ds", "y"))
    expected = featurization.transform(future, time_col="ds")
    actual = Pipeline([("features", featurization)]).transform(future)
    pd.testing.assert_frame_equal(actual, expected)


def test_forecast_score_and_export_share_feature_context():
    data = pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-01", periods=40),
            "x": np.sin(np.arange(40)),
            "y": np.arange(40, dtype=float),
        }
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "max_features": 1.0, "max_leaves": 4, "fe.extraction": "PCA"}},
    )
    automl.fit(dataframe=data.iloc[:-4], label="y", task="ts_forecast", period=4)
    future = data.iloc[-4:].drop(columns="y")
    expected = automl.predict(future)
    np.testing.assert_allclose(automl.automl_pipeline.predict(future), expected)
    transformed = automl.model.autofe.transform(automl.preprocess(future), time_col="ds")
    expected_score = automl.model.score(transformed, data["y"].iloc[-4:])
    assert automl.score(future, data["y"].iloc[-4:]) == pytest.approx(expected_score)
    refitted = clone(automl.automl_pipeline).fit(
        data.iloc[:-4].drop(columns="y"), data["y"].iloc[:-4], estimator__period=4
    )
    np.testing.assert_allclose(refitted.predict(future), expected)


def test_exported_classifier_supports_refit_and_cross_validation():
    automl, X, y = fit_classifier(("approve", "deny"))
    pipeline = clone(automl.automl_pipeline)
    pipeline.fit(X, y)
    np.testing.assert_array_equal(pipeline.classes_, automl.classes_)
    assert set(pipeline.predict(X)) == set(y)
    scores = cross_val_score(automl.automl_pipeline, X, y, cv=2, scoring="accuracy")
    assert np.isfinite(scores).all()
    assert np.min(scores) > 0.6


def test_export_is_a_detached_fitted_snapshot():
    automl, X, y = fit_classifier(("approve", "deny"))
    expected = automl.predict(X)
    exported = automl.automl_pipeline
    assert exported[-1] is not automl.model
    assert exported[0].task is not automl._state.task
    np.testing.assert_array_equal(exported.predict(X), expected)
    exported.fit(X, np.where(y == "approve", "deny", "approve"))
    np.testing.assert_array_equal(automl.predict(X), expected)
    assert not np.array_equal(exported.predict(X), expected)


def test_feature_pipeline_is_detached_and_refittable():
    automl, X, y = fit_classifier(("approve", "deny"))
    expected = automl.predict(X)
    features = automl.feature_transformer
    assert list(features.named_steps) == ["feature_transformer", "autofe"]
    assert features["autofe"] is not automl.model.autofe
    features.fit(X, np.where(y == "approve", "deny", "approve"))
    clone(features).fit(X, y)
    np.testing.assert_array_equal(automl.predict(X), expected)


def test_failed_pipeline_validation_keeps_previous_label_mapping():
    from test.automl.test_resampler import DuplicateMinorityResampler

    automl, X, y = fit_classifier(("approve", "deny"))
    pipeline = automl.automl_pipeline
    pipeline.resampler = DuplicateMinorityResampler()
    expected = pipeline.predict(X)
    with pytest.raises(ValueError, match="Cannot combine 'resampler' with 'sample_weight'"):
        pipeline.fit(X, np.where(y == "approve", "cat", "dog"), estimator__sample_weight=np.ones(len(y)))
    np.testing.assert_array_equal(pipeline.predict(X), expected)
    np.testing.assert_array_equal(pipeline.classes_, ["approve", "deny"])


@pytest.mark.parametrize("invalid", ["fraction", "nan", "multiple-columns", "wrong-row-count"])
def test_pipeline_rejects_invalid_classifier_outputs(monkeypatch, invalid):
    automl, X, _ = fit_classifier(("approve", "deny"))
    pipeline = automl.automl_pipeline
    predictions = {
        "fraction": np.full(len(X), 0.2),
        "nan": np.full(len(X), np.nan),
        "multiple-columns": np.zeros((len(X), 2)),
        "wrong-row-count": np.zeros(len(X) - 1),
    }
    monkeypatch.setattr(pipeline[-1], "predict", lambda *args, **kwargs: predictions[invalid])
    with pytest.raises(ValueError, match="Classifier predictions"):
        pipeline.predict(X)


def test_pipeline_accepts_integral_single_column_classifier_outputs(monkeypatch):
    automl, X, _ = fit_classifier(("approve", "deny"))
    pipeline = automl.automl_pipeline
    monkeypatch.setattr(pipeline[-1], "predict", lambda *args, **kwargs: np.zeros((len(X), 1)))
    assert pipeline.predict(X).tolist() == ["approve"] * len(X)


def test_unfitted_task_preprocessor_has_a_defined_error():
    from flaml.automl.data import DataTransformer
    from flaml.automl.pipeline import _TaskPreprocessor

    preprocessor = _TaskPreprocessor("regression", DataTransformer())
    with pytest.raises(NotFittedError):
        preprocessor.transform(pd.DataFrame({"feature": [1.0]}))


@pytest.mark.parametrize("as_numpy", [False, True])
def test_fold_preprocessing_is_retained_for_raw_prediction_and_export(as_numpy):
    frame = pd.DataFrame(np.random.RandomState(7).normal(size=(40, 3)), columns=["a", "b", "c"])
    target = np.where(frame["b"] > 0, "yes", "no")
    frame.loc[::5, "a"] = np.nan
    X = frame.to_numpy() if as_numpy else frame
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    automl.fit(X, target)
    assert automl._transformer._skip_feature_transform
    assert automl.model._flaml_preprocessor.__sklearn_is_fitted__()
    assert not np.isnan(np.asarray(automl.preprocess(X), dtype=float)).any()
    expected = automl.predict(X)
    np.testing.assert_array_equal(automl.automl_pipeline.predict(X), expected)
    restored = clone(automl.automl_pipeline).fit(X, target)
    np.testing.assert_array_equal(restored.classes_, automl.classes_)


def test_feature_off_retains_existing_global_preprocessing():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0] * 10, "b": np.arange(40.0)})
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, featurization="off")
    automl.fit(frame, np.tile([0, 1], 20))
    assert not automl._transformer._skip_feature_transform
    assert getattr(automl.model, "_flaml_preprocessor", None) is None


def test_unsupervised_reconstruction_and_feature_clone_fit_without_targets():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    features = Featurization(
        config=[{"stage": "numerical", "method": "scaler_standard", "columns": ["a", "b"]}],
        task="classification",
    )
    transformed = features.fit_transform(frame)
    np.testing.assert_allclose(transformed.mean(), 0.0, atol=1e-12)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.selection": "null", "fe.extraction": "PCA"}},
    )
    automl.fit(pd.concat([frame] * 10, ignore_index=True), np.tile([0, 1], 20))
    cloned = clone(automl.feature_transformer).fit(frame)
    assert len(cloned.transform(frame)) == len(frame)


def test_supervised_feature_stage_requires_targets():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="LDA requires target labels"):
        Featurization(params={"fe.extraction": "LDA"}).fit(frame)


@pytest.mark.parametrize("columns", [["0", 0], [10, 20], [0, 1], ["flaml_column_1", 0]])
def test_mixed_column_labels_work_through_automl_and_export(columns):
    frame = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 2)), columns=columns)
    target = (frame.iloc[:, 0] > 0).astype(int)
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    automl.fit(frame, target)
    expected = automl.predict(frame)
    reversed_frame = frame.iloc[:, ::-1]
    np.testing.assert_array_equal(automl.predict(reversed_frame), expected)
    np.testing.assert_array_equal(automl.automl_pipeline.predict(reversed_frame), expected)
    refitted = clone(automl.automl_pipeline).fit(reversed_frame, target)
    assert len(refitted.predict(frame)) == len(frame)


def test_notebook_feature_examples_use_unlabeled_feature_inputs(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebook" / "trident" / "featurization.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    standalone = next(source for source in sources if "autofe = automl.model.autofe" in source)
    reconstructed = next(source for source in sources if "re_autofe = Featurization(" in source)
    automl, X, y = fit_classifier(("approve", "deny"))
    train = X.copy()
    train["target"] = y
    test = train.iloc[:6].copy()
    (tmp_path / "autofe_config.json").write_text(json.dumps(automl.model.autofe.config), encoding="utf-8")
    displayed = []
    namespace = {
        "automl": automl,
        "train": train,
        "test": test,
        "json": json,
        "automl_settings": {"task": "classification"},
        "display": displayed.append,
    }
    monkeypatch.chdir(tmp_path)
    exec(compile(standalone, "standalone-feature-cell", "exec"), namespace)
    exec(compile(reconstructed, "reconstructed-feature-cell", "exec"), namespace)
    frames = [item for item in displayed if isinstance(item, pd.DataFrame)]
    assert len(frames) == 2
    assert all(len(frame) == len(test) and "target" not in frame for frame in frames)


def test_exported_refit_resamples_only_training_rows():
    from test.automl.test_resampler import DuplicateMinorityResampler

    X = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 3)), columns=["a", "b", "c"])
    y = np.where(np.arange(60) % 10 == 0, "minority", "majority")
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        resampler=DuplicateMinorityResampler(),
    )
    automl.fit(X, y)
    pipeline = clone(automl.automl_pipeline)
    DuplicateMinorityResampler.reset_counters()
    pipeline.fit(X, y)
    assert DuplicateMinorityResampler.n_calls == 1
    assert DuplicateMinorityResampler.seen_input_sizes == [60]
    assert DuplicateMinorityResampler.seen_output_sizes == [108]
    pipeline.predict(X)
    assert DuplicateMinorityResampler.n_calls == 1
    DuplicateMinorityResampler.reset_counters()
    assert np.isfinite(cross_val_score(pipeline, X, y, cv=2, scoring="accuracy")).all()
    assert DuplicateMinorityResampler.n_calls == 2
    assert DuplicateMinorityResampler.seen_input_sizes == [30, 30]
    for metadata in ("sample_weight", "groups"):
        with pytest.raises(ValueError, match=f"Cannot combine 'resampler' with '{metadata}'"):
            pipeline.fit(X, y, **{f"estimator__{metadata}": np.ones(len(y))})


@pytest.mark.parametrize("previous_fit", [False, True])
def test_log_retrained_model_exports_actual_featurizer(tmp_path, previous_fit):
    log_file = str(tmp_path / "training.log")
    source, X, y = fit_classifier(("approve", "deny"), log_file_name=log_file, max_iter=2)
    retrained = AutoML(max_iter=1, estimator_list=["rf"], n_jobs=1, mlflow_logging=False, verbose=0)
    if previous_fit:
        retrained.fit(X, y, featurization="off")
    retrained.retrain_from_log(log_file, X_train=X, y_train=y, train_full=True, record_id=0)
    assert retrained.model.autofe is not None
    exported = retrained.automl_pipeline
    np.testing.assert_array_equal(exported.predict(X), retrained.predict(X))
    np.testing.assert_array_equal(exported.classes_, source.classes_)


def test_log_retrained_pipeline_registration_uses_default_name(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from flaml.automl import register_automl_pipeline

    log_file = str(tmp_path / "training.log")
    _, X, y = fit_classifier(("approve", "deny"), log_file_name=log_file, max_iter=2)
    retrained = AutoML(n_jobs=1, mlflow_logging=False)
    retrained.retrain_from_log(log_file, X_train=X, y_train=y, train_full=True, record_id=0)
    original_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    mlflow.set_tracking_uri((tmp_path / "tracking").as_uri())
    try:
        with mlflow.start_run():
            version = register_automl_pipeline(retrained)
        assert version.name == "flaml_pipeline"
        restored = mlflow.sklearn.load_model(f"models:/flaml_pipeline/{version.version}")
        np.testing.assert_array_equal(restored.predict(X), retrained.predict(X))
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(original_uri)


def test_force_featurization_applies_to_single_trial_without_starting_points():
    X = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    automl = AutoML(max_iter=1, estimator_list=["rf"], n_jobs=1, featurization="force", mlflow_logging=False, verbose=0)
    automl.fit(X, y)
    assert automl.model.autofe is not None
    assert all(value != "null" for value in automl.model.autofe.params.values())
    assert automl.model.autofe.pipeline is not None


@pytest.mark.parametrize("columns", [[0, 1], [10, 20], ["0", 0], ["flaml_column_1", 0]])
@pytest.mark.parametrize("method", ["cardinality", "scaler_standard"])
def test_featurization_preserves_nonstring_column_identity(columns, method):
    data = pd.DataFrame({"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) ** 2})
    data.columns = columns
    original = data.copy(deep=True)
    stage = "selection" if method == "cardinality" else "numerical"
    featurization = Featurization(params={f"fe.{stage}": method})
    actual = featurization.fit_transform(data)
    expected = featurization.transform(data[columns[::-1]])
    np.testing.assert_allclose(actual, expected)
    assert actual.columns.is_unique
    pd.testing.assert_frame_equal(data, original)


def test_exported_forecast_accepts_horizons_without_mutating_frames():
    data = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=32), "unused": 1, "y": np.arange(32, dtype=float)})
    automl = AutoML(
        max_iter=1, estimator_list=["naive"], n_jobs=1, featurization="auto", mlflow_logging=False, verbose=0
    )
    automl.fit(dataframe=data.iloc[:-4], label="y", task="ts_forecast", period=4)
    pipeline = automl.automl_pipeline
    future = data.iloc[-4:].drop(columns="y")
    original = future.copy(deep=True)
    for _ in range(2):
        np.testing.assert_allclose(pipeline.predict(future), automl.predict(future))
        pd.testing.assert_frame_equal(future, original)
    np.testing.assert_allclose(pipeline.predict(4), automl.predict(4))
    refitted = clone(pipeline).fit(data.iloc[:-4].drop(columns="y"), data["y"].iloc[:-4], estimator__period=4)
    np.testing.assert_allclose(refitted.predict(future), automl.predict(future))
