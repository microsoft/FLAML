import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from flaml import AutoML, tune
from flaml.automl.state import SearchState
from flaml.automl.task.factory import task_factory
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.autofe import Featurization


def _forecast():
    frame = pd.DataFrame(
        {
            "y": np.arange(36.0),
            "date": pd.date_range("2026-01-01", periods=36),
            "target": np.tile(["cold", "warm"], 18),
        }
    )
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        task="ts_forecast_classification",
        eval_method="holdout",
        metric="accuracy",
        starting_points={"rf": {"n_estimators": 4, "fe.selection": "null", "fe.extraction": "null"}},
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


def test_forecast_recipe_supports_unlabeled_scaling():
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=6), "feature": np.arange(6.0), "target": range(6)}
    )
    source = Featurization(params={"fe.numerical": "scaler_standard"}).fit(TimeSeriesDataset(frame, "date", "target"))
    recipe = Featurization(config=source.config, task="ts_forecast")
    X = frame.drop(columns="target")
    transformed = recipe.fit_transform(X)
    np.testing.assert_allclose(transformed["feature"], (X["feature"] - X["feature"].mean()) / X["feature"].std(ddof=0))
    pd.testing.assert_series_equal(transformed["date"], X["date"])


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
