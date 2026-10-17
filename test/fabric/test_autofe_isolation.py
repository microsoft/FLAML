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


@pytest.mark.parametrize("starting_points", [{"rf": [{"n_estimators": 4}]}, "data"])
def test_force_featurization_completes_list_and_zero_shot_starting_points(starting_points):
    X = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 3)), columns=["a", "b", "c"])
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="force",
        starting_points=starting_points,
    )
    automl.fit(X, (X["a"] > 0).astype(int))
    assert automl.model.autofe is not None
    assert all(choice != "null" for choice in automl.model.autofe.params.values())


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("max_iter", [1, 3])
def test_warm_start_can_disable_previous_featurization(as_list, max_iter, caplog):
    config = {"n_estimators": 4, "max_features": 1.0, "max_leaves": 4, "fe.extraction": "PCA"}
    original = config.copy()
    starting_points = {"rf": [config] if as_list else config}
    X = pd.DataFrame(np.random.RandomState(7).normal(size=(60, 3)), columns=["a", "b", "c"])
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=max_iter,
        n_jobs=1,
        verbose=2,
        mlflow_logging=False,
        featurization="off",
        starting_points=starting_points,
    )
    automl.fit(X, (X["a"] > 0).astype(int))
    assert automl.model.autofe is None
    assert config == original
    assert any("AutoFE" in record.message and "off" in record.message for record in caplog.records)


@pytest.mark.parametrize("numerical", [False, True])
def test_generated_prefix_cleanup_preserves_real_double_underscores(numerical):
    data = pd.DataFrame({"a__b": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]})
    params = {"fe.selection": "cardinality"}
    if numerical:
        params["fe.numerical"] = "scaler_standard"
    transformed = Featurization(params=params).fit_transform(data)
    assert transformed.columns.tolist() == data.columns.tolist()
    assert transformed.columns.is_unique


@pytest.mark.parametrize("scaled", [False, True])
def test_cardinality_uses_categorical_provenance_not_user_spelling(scaled):
    frame = pd.DataFrame({"categorical__signal": np.arange(8.0), "other": np.arange(8.0) ** 2})
    params = {"fe.selection": "cardinality"}
    if scaled:
        params["fe.numerical"] = "scaler_standard"
    transformed = Featurization(params=params).fit_transform(frame)
    assert transformed.columns.tolist() == frame.columns.tolist()


def test_pca_configuration_is_json_serializable_and_reconstructable():
    import json

    data = pd.DataFrame({"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) ** 2})
    source = Featurization(params={"fe.extraction": "PCA"}).fit(data)
    config = json.loads(json.dumps(source.config))
    restored = Featurization(config=config, task="regression").fit(data, data["a"])
    np.testing.assert_allclose(restored.transform(data), source.transform(data))


def test_selection_metadata_uses_actual_stage_input_columns():
    data = pd.DataFrame({"number": np.arange(8, dtype=float), "category": pd.Categorical(list("abcdefgh"))})
    source = Featurization(params={"fe.categorical": "ordinal", "fe.selection": "cardinality"}).fit(data)
    selection = next(stage for stage in source.config if stage["stage"] == "selection")
    assert selection["columns"] == ["category"]
    assert source.transform(data).columns.tolist() == ["number"]


def test_forecast_configuration_reconstruction_uses_embedded_targets():
    import json

    data = TimeSeriesDataset(
        pd.DataFrame({"date": pd.date_range("2025-01-01", periods=8), "x": np.arange(8.0), "target": np.arange(8.0)}),
        "date",
        "target",
        time_idx="step",
    )
    source = Featurization(params={"fe.numerical": "scaler_standard"}).fit(data)
    restored = Featurization(config=json.loads(json.dumps(source.config)), task="ts_forecast").fit(data)
    assert restored.flaml_transformer.transformer_.time_idx == "step"
    pd.testing.assert_frame_equal(restored.transform(data).train_data, source.transform(data).train_data)
    future = pd.DataFrame({"date": pd.date_range("2025-01-09", periods=2), "x": [8.0, 9.0]})
    pd.testing.assert_frame_equal(restored.transform(future), source.transform(future))


def test_parameter_change_invalidates_fitted_featurization():
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    features = Featurization(params={"fe.numerical": "scaler_standard"}).fit(frame)
    features.set_params(params={"fe.numerical": "scaler_minmax"})
    assert not features.__sklearn_is_fitted__()
    assert features.time_col_ is None
    with pytest.raises(NotFittedError):
        features.transform(frame)
    np.testing.assert_allclose(features.fit_transform(frame)["feature"], [0.0, 0.5, 1.0])


def test_forecast_dataframe_reconstruction_retains_time_context():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=8), "x": np.arange(8.0)})
    target = pd.Series(np.arange(8.0), name="target")
    config = [{"stage": "numerical", "method": "scaler_standard", "columns": ["x"]}]
    features = Featurization(config=config, task="ts_forecast")
    transformed = features.fit_transform(frame, target)
    assert features.time_col_ == "date"
    pd.testing.assert_series_equal(transformed["date"], frame["date"])
    future = pd.DataFrame({"date": pd.date_range("2025-01-09", periods=2), "x": [8.0, 9.0]})
    assert list(features.transform(future).columns) == ["date", "x"]


def test_expected_autofe_candidate_failure_does_not_abort_search(caplog):
    frame = pd.DataFrame({"feature": [0.0] * 10 + [1.0] * 10})
    target = np.tile([0, 1], 10)
    settings = {"n_estimators": 4, "max_features": 1.0, "max_leaves": 4, "fe.extraction": "null"}
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=2,
        mlflow_logging=False,
        featurization="auto",
        auto_augment=False,
        eval_method="cv",
        split_type=KFold(2),
        starting_points={
            "rf": [
                {**settings, "fe.selection": "variance"},
                {**settings, "fe.selection": "null"},
            ]
        },
    )
    automl.fit(frame, target)
    assert np.isfinite(automl.best_loss)
    assert automl.best_config["fe.selection"] == "null"
    assert any("AutoFE candidate rejected" in record.message for record in caplog.records)


@pytest.mark.parametrize("extraction", ["PCA", "null"])
def test_empty_cardinality_candidate_is_rejected_before_learner(extraction, caplog):
    frame = pd.DataFrame({"category": pd.Categorical([f"value-{i}" for i in range(20)] * 2)})
    settings = {"n_estimators": 4, "fe.categorical": "ordinal"}
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=2,
        mlflow_logging=False,
        featurization="auto",
        auto_augment=False,
        eval_method="cv",
        split_type=KFold(2),
        starting_points={
            "rf": [
                {**settings, "fe.selection": "cardinality", "fe.extraction": extraction},
                {**settings, "fe.selection": "null", "fe.extraction": "null"},
            ]
        },
    )
    automl.fit(frame, np.tile([0, 1], 20))
    assert np.isfinite(automl.best_loss)
    assert automl.best_config["fe.selection"] == "null"
    assert any("AutoFE candidate rejected" in record.message for record in caplog.records)


def test_automl_cv_imputation_is_fitted_only_on_each_training_partition(monkeypatch):
    captured = []
    fit = Featurization.fit

    def inspect_training(self, X, y=None):
        if isinstance(X, pd.DataFrame) and set(X.index) == {0, 1, 2}:
            captured.append(X.sort_index()["feature"].to_numpy().copy())
        return fit(self, X, y)

    monkeypatch.setattr(Featurization, "fit", inspect_training)
    for scale in (50.0, 500.0):
        frame = pd.DataFrame({"feature": [0.0, np.nan, 2.0, scale, scale * 2, scale * 3]})
        automl = AutoML(
            estimator_list=["rf"],
            max_iter=2,
            n_jobs=1,
            verbose=0,
            mlflow_logging=False,
            featurization="auto",
            eval_method="cv",
            split_type=KFold(2),
            starting_points={
                "rf": [
                    {"n_estimators": 4, "fe.extraction": "PCA"},
                    {"n_estimators": 5, "fe.extraction": "PCA"},
                ]
            },
        )
        automl.fit(frame, np.arange(6.0), task="regression")
    assert captured
    for transformed in captured:
        np.testing.assert_allclose(transformed, [0.0, 1.0, 2.0])


def test_forecast_autofe_preprocessing_excludes_explicit_validation_rows(monkeypatch):
    captured = []
    fit = Featurization.fit

    def inspect_training(self, X, y=None):
        if isinstance(X, TimeSeriesDataset) and len(X.train_data) == 14:
            captured.append(X.train_data["feature"].to_numpy().copy())
        return fit(self, X, y)

    monkeypatch.setattr(Featurization, "fit", inspect_training)
    training = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=14), "feature": np.arange(14.0)})
    training.loc[1, "feature"] = np.nan
    validation = pd.DataFrame({"ds": pd.date_range("2026-01-15", periods=2), "feature": [1000.0, 2000.0]})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={
            "rf": [{"n_estimators": 4, "fe.extraction": "PCA"}, {"n_estimators": 5, "fe.extraction": "PCA"}]
        },
    )
    automl.fit(
        training,
        pd.Series(np.arange(14.0), name="target"),
        X_val=validation,
        y_val=pd.Series([14.0, 15.0], name="target"),
        task="ts_forecast",
        period=2,
    )
    assert captured
    for transformed in captured:
        assert transformed[1] == pytest.approx(training["feature"].median())


def test_categorical_noop_preserves_mixed_value_identity():
    frame = pd.DataFrame({"category": pd.Categorical([1, "1", 1, "1"])})
    transformed = Featurization(params={}).fit_transform(frame)
    pd.testing.assert_frame_equal(transformed, frame)


def test_ordinal_encoding_rejects_ambiguous_mixed_category_types():
    frame = pd.DataFrame({"category": pd.Categorical([1, "1", 1, "1"])})
    with pytest.raises(ValueError, match="mixed categorical"):
        Featurization(params={"fe.categorical": "ordinal"}).fit(frame)


def test_autofe_rejects_singleton_classes_before_cross_validation():
    frame = pd.DataFrame({"row_id": np.arange(21.0), "feature": np.arange(21.0) ** 2})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        eval_method="cv",
        n_splits=2,
        starting_points={"rf": {"n_estimators": 4, "fe.extraction": "LDA"}},
    )
    with pytest.raises(ValueError, match="AutoFE requires disjoint"):
        automl.fit(frame, np.array([0] * 20 + [1]))


def test_feature_off_preserves_legacy_singleton_augmentation():
    frame = pd.DataFrame({"feature": np.arange(21.0)})
    automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, featurization="off")
    automl.fit(frame, np.array([0] * 20 + [1]))
    assert len(automl.predict(frame)) == len(frame)
    assert getattr(automl._state.task, "_resampler", None) is None


def test_default_autofe_augmentation_is_training_partition_only(monkeypatch):
    sizes = []
    original_fit = Featurization.fit

    def capture_fit(self, X, y=None):
        sizes.append(len(X))
        return original_fit(self, X, y)

    monkeypatch.setattr(Featurization, "fit", capture_fit)
    observed = []

    def isolated_metric(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
        assert set(X_train["row_id"]).isdisjoint(set(X_val["row_id"]))
        assert np.unique(y_train, return_counts=True)[1].min() >= 20
        observed.append((len(X_train), len(X_val)))
        return 0.5, {}

    frame = pd.DataFrame({"row_id": np.arange(22.0), "feature": np.arange(22.0) ** 2})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        eval_method="cv",
        n_splits=2,
        metric=isolated_metric,
        starting_points={
            "rf": [
                {"n_estimators": 4, "fe.selection": "null", "fe.extraction": "null"},
                {"n_estimators": 5, "fe.selection": "null", "fe.extraction": "null"},
            ]
        },
    )
    automl.fit(frame, np.array([0] * 20 + [1] * 2))
    assert observed
    assert all(validation_size == 11 for _, validation_size in observed)
    assert 11 in sizes
    assert set(sizes) <= {11, 22}


def test_unexpected_autofe_programming_error_still_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unexpected implementation failure")

    monkeypatch.setattr(Featurization, "fit_transform", fail)
    automl = AutoML(estimator_list=["rf"], max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    with pytest.raises(ValueError, match="unexpected implementation failure"):
        automl.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))


@pytest.mark.parametrize("kind", ["sparse", "nlp"])
@pytest.mark.parametrize("mode", ["off", "auto", "force"])
def test_unsupported_feature_warnings_require_explicit_optin(kind, mode, caplog):
    from scipy.sparse import csr_matrix

    from flaml.fabric.autofe import parse_autofe_config

    task = task_factory("seq-classification" if kind == "nlp" else "classification")
    data = pd.DataFrame({"text": ["a", "b"]}) if kind == "nlp" else csr_matrix(np.eye(2))
    assert parse_autofe_config(mode, data, task, RandomForestEstimator) == {}
    warnings = [record for record in caplog.records if "Auto featurization is not supported" in record.message]
    assert bool(warnings) is (mode != "off")


@pytest.mark.parametrize("evaluation", ["holdout", "cv"])
@pytest.mark.parametrize("enabled", [False, True])
def test_prediction_latency_includes_only_serving_transform_cost(monkeypatch, evaluation, enabled):
    import flaml.automl.ml as module

    clock = [0.0]

    class TimedTransformer:
        def __init__(self, duration):
            self.duration = duration

        def fit_transform(self, X, y=None):
            clock[0] += 100.0
            return X

        def transform(self, X):
            clock[0] += self.duration
            return X

    class FastEstimator:
        def __init__(self, **kwargs):
            self.params = kwargs

        def fit(self, X, y, **kwargs):
            clock[0] += 200.0

        def predict(self, X):
            return np.zeros(len(X))

        def cleanup(self):
            pass

    def attach(estimator, task):
        if enabled:
            estimator._flaml_preprocessor = TimedTransformer(0.2)

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(module, "Featurization", lambda **kwargs: TimedTransformer(0.4))
    monkeypatch.setattr(module, "_attach_task_preprocessor", attach)
    X = pd.DataFrame({"x": np.arange(12.0)})
    y = np.zeros(12)
    _, _, metrics, _, latency = module.compute_estimator(
        X if evaluation == "cv" else X.iloc[:8],
        y if evaluation == "cv" else y[:8],
        X.iloc[8:],
        y[8:],
        None,
        None,
        None,
        KFold(3),
        {"fe.selection": "null"} if enabled else {},
        task_factory("regression"),
        "fast",
        evaluation,
        "mse",
        estimator_class=FastEstimator,
    )
    assert latency == pytest.approx(0.15 if enabled else 0.0, abs=0.01)
    assert metrics["pred_time"] == pytest.approx(latency)
    assert (latency > 0.1) is enabled


def test_feature_registry_retains_legacy_alias():
    from flaml.fabric.autofe import available_methods, avaliable_methods

    assert available_methods is avaliable_methods


def test_zero_within_class_rank_lda_candidate_is_rejected_and_search_continues():
    X = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0], "id": pd.Categorical(["a", "b", "c", "d"])})
    validation = pd.DataFrame({"x": [0.0, 1.0], "id": pd.Categorical(["a", "c"])})
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={
            "rf": [
                {"n_estimators": 4, "fe.categorical": "ordinal", "fe.selection": "cardinality", "fe.extraction": "LDA"},
                {"n_estimators": 5, "fe.categorical": "ordinal", "fe.selection": "null", "fe.extraction": "null"},
            ]
        },
    )
    automl.fit(X, np.array([0, 0, 1, 1]), X_val=validation, y_val=np.array([0, 1]))
    assert np.isfinite(automl.best_loss)
    assert automl.best_config["fe.extraction"] == "null"


def test_explicit_forecast_validation_rows_never_become_training(monkeypatch):
    class ReachedSearch(Exception):
        pass

    automl = AutoML(estimator_list=["naive"], max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False)

    def inspect_partitions():
        data = automl._state.X_train
        assert data.train_data["y"].tolist() == [1.0, 3.0, 5.0, 7.0]
        assert data.test_data["y"].tolist() == [1002.0, 1004.0]
        raise ReachedSearch

    monkeypatch.setattr(automl, "_search", inspect_partitions)
    X = pd.DataFrame({"ds": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-07"])})
    validation = pd.DataFrame({"ds": pd.to_datetime(["2025-01-02", "2025-01-04"])})
    with pytest.raises(ReachedSearch):
        automl.fit(
            X,
            pd.Series([1.0, 3.0, 5.0, 7.0], name="y"),
            X_val=validation,
            y_val=pd.Series([1002.0, 1004.0], name="y"),
            task="ts_forecast",
            period=2,
        )


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
