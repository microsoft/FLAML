import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from flaml import AutoML, tune
from flaml.automl.state import SearchState
from flaml.automl.task.factory import task_factory
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.autofe import Featurization


def _forecast(extraction="null"):
    frame = pd.DataFrame(
        {
            "y": np.arange(36.0),
            "date": pd.date_range("2026-01-01", periods=36),
            "target": np.tile(["cold", "warm"], 18),
        }
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1 if extraction != "null" else 2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        task="ts_forecast_classification",
        eval_method="holdout",
        metric="accuracy",
        starting_points={"rf": {"n_estimators": 4, "fe.selection": "null", "fe.extraction": extraction}},
    )
    automl.fit(TimeSeriesDataset(frame, "date", "target"), period=2)
    return automl, frame.drop(columns="target"), frame["target"]


def test_feature_only_forecast_refit_is_target_free_and_returns_a_feature_frame():
    automl, X, y = _forecast()
    X = X.iloc[::-1].copy()
    feature_pipeline = clone(automl.feature_transformer)
    expected = feature_pipeline.fit(X).transform(X)
    transformed = clone(automl.feature_transformer).fit_transform(X)
    assert isinstance(transformed, pd.DataFrame)
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_index_equal(transformed.index, X.index)
    assert "target" not in transformed
    pd.testing.assert_series_equal(transformed["date"], X["date"])
    with_labels = clone(automl.feature_transformer).fit_transform(X, y.loc[X.index])
    pd.testing.assert_frame_equal(with_labels, expected)


@pytest.mark.parametrize("nonleading_time", [False, True])
@pytest.mark.parametrize("initialization", ["recipe", "params"])
def test_forecast_recipe_supports_unlabeled_scaling(nonleading_time, initialization):
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=6), "feature": np.arange(6.0), "target": range(6)}
    )
    source = Featurization(params={"fe.numerical": "scaler_standard"}).fit(TimeSeriesDataset(frame, "date", "target"))
    recipe = (
        Featurization(config=source.config, task="ts_forecast")
        if initialization == "recipe"
        else Featurization(params={"fe.numerical": "scaler_standard"}, task="ts_forecast")
    )
    X = frame.drop(columns="target")
    if nonleading_time:
        X = X[["feature", "date"]]
    transformed = recipe.fit_transform(X)
    np.testing.assert_allclose(transformed["feature"], (X["feature"] - X["feature"].mean()) / X["feature"].std(ddof=0))
    pd.testing.assert_series_equal(transformed["date"], X["date"])


@pytest.mark.parametrize("with_labels", [False, True])
def test_feature_only_forecast_transform_keeps_features_named_like_an_old_target(with_labels):
    automl, X, _ = _forecast()
    features = X.rename(columns={"y": "target"})
    labels = np.arange(len(features)) if with_labels else None
    pipeline = clone(automl.feature_transformer).fit(features, labels)
    expected = pipeline.transform(features)
    dataset = TimeSeriesDataset(features.assign(actual_target=np.arange(len(features))), "date", "actual_target")
    actual = pipeline.transform(dataset)
    pd.testing.assert_frame_equal(actual, expected)
    assert "target" in actual


def test_full_forecast_refit_keeps_an_old_target_name_used_as_a_regressor():
    automl, X, y = _forecast()
    features = X.rename(columns={"y": "target"})
    pipeline = clone(automl.automl_pipeline).fit(features, y.to_numpy(), estimator__period=2)
    assert "target" in pipeline.steps[-1][1].user_regressors
    assert "target" not in pipeline.named_steps["feature_transformer"].task_.target_names
    assert set(pipeline.predict(features.iloc[-2:])) <= {"cold", "warm"}


def test_forecast_preprocessing_without_a_fitted_trial_has_an_actionable_error(monkeypatch):
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "target": np.arange(30.0)})
    automl = AutoML(estimator_list=["rf"], max_iter=2, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    monkeypatch.setattr(automl, "_search_sequential", lambda: None)
    automl.fit(dataframe=frame, label="target", task="ts_forecast", period=2)
    assert automl.model is None
    with pytest.raises(AttributeError, match="AutoFE preprocessing requires a fitted model"):
        automl.preprocess(frame.drop(columns="target"))


def test_forecast_pipeline_preserves_nonleading_time_and_named_encoded_target():
    automl, X, y = _forecast()
    pipeline = clone(automl.automl_pipeline).fit(X, y, estimator__period=2)
    preprocessor = pipeline.named_steps["feature_transformer"]
    assert preprocessor.task_.time_col == "date"
    assert preprocessor.task_.target_names == ["target"]
    assert "y" in pipeline.steps[-1][1].user_regressors
    future = pd.DataFrame({"y": [36.0, 37.0], "date": pd.date_range("2026-02-06", periods=2)})
    assert set(pipeline.predict(future)) <= {"cold", "warm"}
    validation = future.assign(target=["cold", "warm"])
    dataset = TimeSeriesDataset(X.assign(target=y), "date", "target", test_data=validation)
    assert np.isfinite(pipeline.score(dataset, metric="accuracy"))
    assert dataset.test_data["target"].tolist() == ["cold", "warm"]


def test_forecast_ordinal_encoding_preserves_numeric_category_missing_values():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=36),
            "category": pd.Categorical([1, 2, None] * 12),
            "target": np.arange(36.0),
        }
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.categorical": "ordinal"}},
    )
    automl.fit(dataframe=frame, label="target", task="ts_forecast", period=2)
    future = pd.DataFrame({"date": pd.date_range("2026-02-06", periods=2), "category": pd.Categorical([2, None])})
    np.testing.assert_allclose(automl.automl_pipeline.predict(future), automl.predict(future))
    assert frame["category"].isna().sum() == 12


def test_custom_feature_domain_cannot_reenable_excluded_panel_autofe():
    from flaml.automl.time_series import RF_TS

    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "target": range(30)})
    state = SearchState(
        RF_TS,
        TimeSeriesDataset(frame, "date", "target"),
        task_factory("ts_forecast_panel"),
        period=2,
        max_iter=1,
        budget=-1,
        featurization="auto",
        estimator_name="rf",
        custom_hp={"fe.extraction": {"domain": tune.choice(["PCA"])}},
    )
    assert not any(name.startswith("fe.") for name in state.search_space)


@pytest.mark.parametrize("starting_point", [{"fe.extraction": "null"}, [{"fe.extraction": "null"}]])
def test_force_mode_rejects_custom_noop_domain(starting_point):
    from flaml.automl.model import RandomForestEstimator

    with pytest.raises(ValueError, match="force"):
        SearchState(
            RandomForestEstimator,
            np.arange(120).reshape(40, 3),
            task_factory("classification"),
            max_iter=1,
            budget=-1,
            featurization="force",
            estimator_name="rf",
            starting_point=starting_point,
            custom_hp={"fe.extraction": {"domain": tune.choice(["null", "PCA"])}},
        )


@pytest.mark.parametrize("label_index_offset", [0, 1000])
def test_first_rare_class_repeats_positionally_without_reusing_index_masks(label_index_offset):
    from flaml.automl.task.generic_task import _TrainingPartitionAugmenter

    X = pd.DataFrame({"feature": range(22)}, index=np.arange(22) * 10)
    y = pd.Series([0, 0] + [1] * 20, index=X.index + label_index_offset)
    augmented, labels = _TrainingPartitionAugmenter().fit_resample(X, y)
    assert len(augmented) == len(labels) == 40
    assert set(augmented.loc[np.asarray(labels) == 0, "feature"]) == {0, 1}
    assert (labels == 0).sum() == 20


def test_forecast_feature_refit_uses_embedded_targets_for_lda():
    automl, X, y = _forecast("LDA")
    features = clone(automl.feature_transformer).fit_transform(TimeSeriesDataset(X.assign(target=y), "date", "target"))
    assert isinstance(features, pd.DataFrame)
    assert features.shape == (len(X), 2)
    pd.testing.assert_series_equal(features["date"], X["date"])


@pytest.mark.parametrize("as_dataset", [False, True])
def test_forecast_feature_refit_propagates_a_renamed_nonleading_timestamp(as_dataset):
    automl, X, y = _forecast()
    X = X.rename(columns={"date": "new_date"})
    data = TimeSeriesDataset(X.assign(target=y), "new_date", "target") if as_dataset else X
    features = clone(automl.feature_transformer).fit_transform(data, None if as_dataset else y)
    assert "y" in features
    pd.testing.assert_series_equal(features["y"], X["y"])
    pd.testing.assert_series_equal(features["new_date"], X["new_date"])


def test_forecast_pipeline_refit_learns_new_string_target_labels():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=36),
            "feature": np.arange(36.0),
            "target": np.tile([0, 1], 18),
        }
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        task="ts_forecast_classification",
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.extraction": "null"}},
    )
    automl.fit(TimeSeriesDataset(frame, "date", "target"), period=2)
    frame["target"] = np.tile(["cold", "warm"], 18)
    pipeline = clone(automl.automl_pipeline).fit(TimeSeriesDataset(frame, "date", "target"), estimator__period=2)
    future = pd.DataFrame({"date": pd.date_range("2026-02-06", periods=2), "feature": [36.0, 37.0]})
    assert set(pipeline.classes_) == {"cold", "warm"}
    assert set(pipeline.predict(future)) <= {"cold", "warm"}


@pytest.mark.parametrize("mode", ["auto", "force"])
def test_tft_cannot_receive_generic_autofe_for_a_nonpanel_task(mode):
    from flaml.automl.time_series import RF_TS

    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "target": range(30)})
    state = SearchState(
        RF_TS,
        TimeSeriesDataset(frame, "date", "target"),
        task_factory("ts_forecast"),
        period=2,
        max_iter=1,
        budget=-1,
        featurization=mode,
        estimator_name="tft",
    )
    assert not any(name.startswith("fe.") for name in state.search_space)


def test_tabular_autofe_rejects_a_time_holdout_class_absent_from_training():
    X = pd.DataFrame({"feature": np.arange(40.0)})
    y = pd.Series(["A", "C"] * 15 + ["B"] * 10)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        split_type="time",
        split_ratio=0.25,
        eval_method="holdout",
        retrain_full=False,
        metric="accuracy",
    )
    with pytest.raises(ValueError, match="every class.*training"):
        automl.fit(X, y)


def test_tabular_autofe_rejects_a_class_deficient_trial_sample():
    X = pd.DataFrame({"feature": np.arange(120.0)})
    y = pd.Series(["A", "C"] * 35 + ["B"] * 20 + ["A", "B", "C"] * 10)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        split_type="time",
        split_ratio=0.25,
        eval_method="holdout",
        retrain_full=False,
        auto_augment=False,
        sample=True,
        min_sample_size=12,
        metric="accuracy",
        starting_points={
            "rf": {"n_estimators": 4, "fe.selection": "null", "fe.extraction": "null", "FLAML_sample_size": 12}
        },
    )
    with pytest.raises(ValueError, match="every class.*sampled training"):
        automl.fit(X, y)


def test_tabular_autofe_rejects_a_group_holdout_with_an_unseen_class():
    X = pd.DataFrame({"feature": np.arange(30.0)})
    y = pd.Series(np.repeat(["A", "B", "C"], 10))
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        split_type="group",
        split_ratio=1 / 3,
        eval_method="holdout",
        retrain_full=False,
        auto_augment=False,
        metric="accuracy",
    )
    with pytest.raises(ValueError, match="every class.*training"):
        automl.fit(X, y, groups=np.repeat(np.arange(3), 10))


@pytest.mark.parametrize("stage", ["holdout", "cv", "export"])
def test_autofe_validates_the_effective_class_vocabulary_at_each_fit(stage):
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import TimeSeriesSplit

    class DropMiddleClass(BaseEstimator):
        def fit_resample(self, X, y):
            keep = np.asarray(y) != 1
            return X.loc[keep], np.asarray(y)[keep]

    X = pd.DataFrame({"feature": np.arange(90.0)})
    y = np.tile(["A", "B", "C"], 30)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={"rf": {"n_estimators": 4, "fe.extraction": "null", "fe.selection": "null"}},
        eval_method="cv" if stage == "cv" else "holdout",
        split_type=TimeSeriesSplit(n_splits=2) if stage == "cv" else "stratified",
        retrain_full=False,
        metric="accuracy",
        auto_augment=False,
    )
    if stage == "cv":
        y = np.repeat(["A", "B", "C"], 30)
        with pytest.raises(ValueError, match="every class.*training"):
            automl.fit(X, y)
    elif stage == "holdout":
        with pytest.raises(ValueError, match="every class.*training"):
            automl.fit(X, y, resampler=DropMiddleClass())
    else:
        automl.fit(X, y)
        pipeline = clone(automl.automl_pipeline).set_params(resampler=DropMiddleClass())
        with pytest.raises(ValueError, match="every class.*training"):
            pipeline.fit(X, y)


def test_unused_tft_structural_kwargs_do_not_change_the_forecast_task():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=30),
            "series": ["a"] * 30,
            "feature": np.arange(30.0),
            "target": np.arange(30.0),
        }
    )
    automl = AutoML(max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, featurization="auto")
    automl.fit(
        dataframe=frame,
        label="target",
        task="ts_forecast",
        period=2,
        estimator_list=["rf"],
        fit_kwargs_by_estimator={"tft": {"group_ids": ["series"]}},
    )
    assert not automl._state.task.is_ts_forecastpanel()
    assert automl._state.task._preprocess_for_autofe
