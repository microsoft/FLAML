from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from flaml import AutoML, tune
from flaml.fabric.visualization import get_param_importance, plot_optimization_history, plot_slice


def history_result():
    automl = AutoML()
    automl._best_estimator = "arima"
    automl.mlflow_integration = SimpleNamespace(
        infos=[
            {"metrics": {"validation_loss": loss}, "params": {"learner": "arima", "p": p}}
            for p, loss in enumerate([0.5, 0.1, 0.3])
        ]
    )
    return automl


def test_optimization_history_tracks_best_score():
    figure = plot_optimization_history(history_result())
    np.testing.assert_allclose(figure.data[0].y, [0.5, 0.9, 0.7])
    np.testing.assert_allclose(figure.data[1].y, [0.5, 0.9, 0.9])
    assert "score" in figure.layout.yaxis.title.text.lower()
    assert figure.layout.yaxis.range is None


def test_forecast_importance_uses_resolved_search_space():
    automl = history_result()
    automl._search_states = {"arima": SimpleNamespace(search_space={"p": tune.randint(0, 4)})}
    automl._state.learner_classes = {
        "arima": SimpleNamespace(search_space=Mock(side_effect=AssertionError("wrong API")))
    }
    importance = get_param_importance(automl)
    assert importance == {"p": 1.0}


def test_automl_slice_does_not_clip_unbounded_scores():
    result = history_result()
    for info, loss in zip(result.mlflow_integration.infos, [3.0, 0.5, -2.0]):
        info["metrics"]["validation_loss"] = loss
    figure = plot_slice(result, params=["p"])
    np.testing.assert_allclose(figure.data[0].y, [-2.0, 0.5, 3.0])
    assert figure.layout.yaxis.range is None


def test_tune_slice_labels_the_actual_metric():
    result = tune.run(
        lambda config: {"cost": config["x"] * 10.0},
        config={"x": tune.choice([-1, 2])},
        points_to_evaluate=[{"x": -1}, {"x": 2}],
        metric="cost",
        mode="min",
        num_samples=2,
        use_ray=False,
        mlflow_logging=False,
        verbose=0,
    )
    figure = plot_slice(result, params=["x"])
    assert list(figure.data[0].y) == [-10.0, 20.0]
    assert figure.layout.yaxis.range is None
    assert figure.layout.yaxis.title.text == "cost"


def test_visualization_facade_only_exports_supported_functions():
    import flaml.fabric.visualization as implementation
    import flaml.visualization as public

    assert "plot" in public.__all__
    assert "get_param_importance" in public.__all__
    assert set(public.__all__) == set(implementation.__all__)
    for name in public.__all__:
        assert getattr(public, name) is getattr(implementation, name)
        assert callable(getattr(public, name))
    assert not hasattr(public, "np")
    assert not hasattr(public, "AutoML")


def test_missing_plotly_has_an_actionable_dependency_error():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockPlotly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "plotly":
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, BlockPlotly())
        try:
            import flaml.visualization
        except ImportError as error:
            assert "flaml[fabric_python]" in str(error), error
        else:
            raise AssertionError("Plotly import unexpectedly succeeded")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["parallel_coordinate", "contour"])
def test_automl_color_scales_agree_with_maximized_scores(kind):
    import plotly.colors

    from flaml.fabric.visualization import plot

    result = history_result()
    for number, info in enumerate(result.mlflow_integration.infos):
        info["params"]["q"] = number % 2
    figure = plot(result, kind, params=["p", "q"])
    actual = figure.layout.coloraxis.colorscale if kind == "parallel_coordinate" else figure.data[0].colorscale
    expected = plotly.colors.make_colorscale(plotly.colors.sequential.Emrld)
    assert [color for _, color in actual] == [color for _, color in expected]
    np.testing.assert_allclose(
        [position for position, _ in actual], [position for position, _ in expected], rtol=0, atol=1e-15
    )


def test_missing_public_trial_history_has_a_defined_error(monkeypatch):
    from flaml.fabric.visualization import plot

    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", "false")
    result = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0, mlflow_logging=False, model_history=True)
    result.fit(np.arange(80).reshape(40, 2), np.tile([0, 1], 20))
    with pytest.raises(ValueError, match="trial history"):
        plot(result, "optimization_history")


def test_visualization_uses_native_domain_conversion():
    from flaml.fabric.visualization import optuna_space

    result = history_result()
    result._search_states = {
        "arima": SimpleNamespace(
            search_space={
                "integer": tune.randint(1, 5),
                "logarithmic": tune.loguniform(0.001, 1.0),
                "quantized": tune.quniform(0.0, 1.0, 0.25),
            }
        )
    }
    space = optuna_space(result)
    assert set(space) == {"integer", "logarithmic", "quantized"}
    assert (space["integer"].low, space["integer"].high) == (1, 4)
    assert space["logarithmic"].log
    assert space["quantized"].step == 0.25


def test_visualization_preserves_ray_numeric_domains():
    ray_tune = pytest.importorskip("ray.tune")
    from flaml.fabric.visualization import optuna_space

    result = history_result()
    result._search_states = {
        "arima": SimpleNamespace(search_space={"integer": ray_tune.randint(1, 5), "real": ray_tune.uniform(0.0, 1.0)})
    }
    converted = optuna_space(result)
    assert set(converted) == {"integer", "real"}
    assert converted["integer"].high == 4


@pytest.mark.parametrize("conditional", [False, True])
def test_slice_handles_null_and_absent_categorical_parameters(monkeypatch, conditional):
    import pandas as pd

    from flaml.fabric import visualization

    result = history_result()
    values = [None, "balanced", None] if not conditional else [np.nan, "balanced", np.nan]
    monkeypatch.setattr(
        visualization,
        "get_hp_df",
        lambda *args, **kwargs: (pd.DataFrame({"class_weight": values}), pd.Series([0.1, 0.2, 0.3])),
    )
    figure = visualization.plot_slice(result)
    assert len(figure.data[0].x) == len(figure.data[0].y) == 3
    assert all(np.isfinite(figure.data[0].marker.color))


def test_nested_tune_results_share_distribution_names():
    from flaml.fabric.visualization import get_hp_df

    analysis = tune.run(
        lambda config: {"loss": config["model"]["x"]},
        config={"model": {"x": tune.uniform(0.0, 1.0)}},
        points_to_evaluate=[{"model": {"x": 0.1}}, {"model": {"x": 0.8}}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=False,
        mlflow_logging=False,
        verbose=0,
    )
    frame, scores = get_hp_df(analysis, params=["model/x"])
    assert frame.columns.tolist() == ["model/x"]
    np.testing.assert_allclose(np.asarray(scores, dtype=float), [0.1, 0.8])
    assert get_param_importance(analysis, params=["model/x"]) == {"model/x": 1.0}


def test_auxiliary_score_does_not_replace_selected_objective():
    from flaml.fabric.visualization import get_hp_df

    analysis = tune.run(
        lambda config: {"loss": config["x"], "score": 100 + config["x"]},
        config={"x": tune.uniform(0.0, 1.0)},
        points_to_evaluate=[{"x": 0.1}, {"x": 0.8}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=False,
        mlflow_logging=False,
        verbose=0,
    )
    _, scores = get_hp_df(analysis)
    assert scores.ndim == 1
    np.testing.assert_allclose(np.asarray(scores, dtype=float), [0.1, 0.8])
    assert get_param_importance(analysis) == {"x": 1.0}


def test_report_frame_preserves_none_and_absent_parameter_distinction(monkeypatch):
    import pandas as pd

    from flaml.fabric import visualization
    from flaml.tune.tune import ExperimentAnalysis

    analysis = ExperimentAnalysis([], metric="loss", mode="min")
    monkeypatch.setattr(
        visualization,
        "_get_tune_df",
        lambda result: pd.DataFrame({"config": [{"choice": None}, {}, {"choice": 1.0}], "score": [0.0, 2.0, 1.0]}),
    )
    frame, _ = visualization.get_hp_df(analysis)
    assert frame.loc[0, "choice"] is None
    assert frame.loc[1, "choice"] is pd.NA


def test_numeric_none_category_survives_real_tune_reporting():
    analysis = tune.run(
        lambda config: {"loss": 0.0 if config["choice"] is None else 1.0},
        config={"choice": tune.choice([None, 1.0])},
        points_to_evaluate=[{"choice": None}, {"choice": 1.0}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=False,
        mlflow_logging=False,
        verbose=0,
    )
    assert get_param_importance(analysis) == {"choice": 1.0}


def test_multidimensional_contour_retains_evaluated_trial_markers():
    from flaml.fabric.visualization import plot_contour

    result = history_result()
    for index, info in enumerate(result.mlflow_integration.infos):
        info["params"].update(q=index % 2, r=index + 1)
    figure = plot_contour(result, params=["p", "q", "r"])
    contours = [trace for trace in figure.data if trace.type == "contour"]
    markers = [trace for trace in figure.data if trace.type == "scatter" and trace.mode == "markers"]
    assert len(contours) == len(markers) == 6


@pytest.mark.parametrize("mode,expected", [("min", [np.nan, 0.5, 0.5, 0.2]), ("max", [np.nan, 0.5, 0.5, 0.5])])
def test_running_best_ignores_invalid_trials(mode, expected):
    from flaml.tune.tune import ExperimentAnalysis

    trials = [
        SimpleNamespace(
            trial_id=str(index),
            last_result={"loss": score, "config": {"x": index}, "time_total_s": 1.0},
        )
        for index, score in enumerate([np.nan, 0.5, np.nan, 0.2])
    ]
    analysis = ExperimentAnalysis(trials, metric="loss", mode=mode)
    figure = plot_optimization_history(analysis)
    np.testing.assert_allclose(figure.data[1].y, expected, equal_nan=True)


def test_automl_running_best_ignores_invalid_loss():
    result = history_result()
    for info, loss in zip(result.mlflow_integration.infos, [np.nan, 0.5, 0.2]):
        info["metrics"]["validation_loss"] = loss
    figure = plot_optimization_history(result)
    np.testing.assert_allclose(figure.data[1].y, [np.nan, 0.5, 0.8], equal_nan=True)


def test_parallel_coordinates_preserve_a_score_named_parameter():
    from flaml.fabric.visualization import plot_parallel_coordinate

    analysis = tune.run(
        lambda config: {"loss": 10.0 - config["score"]},
        config={"score": tune.randint(1, 4)},
        points_to_evaluate=[{"score": 1}, {"score": 3}],
        metric="loss",
        mode="min",
        num_samples=2,
        use_ray=False,
        mlflow_logging=False,
        verbose=0,
    )
    figure = plot_parallel_coordinate(analysis, params=["score"])
    dimension = next(item for item in figure.data[0].dimensions if item.label == "score")
    np.testing.assert_array_equal(dimension.values, [1, 3])
    np.testing.assert_array_equal(figure.data[0].line.color, [9, 7])


def test_centralized_plot_diagnostics_do_not_include_caller_payloads(monkeypatch):
    from flaml.fabric import visualization

    logger = Mock()
    monkeypatch.setattr(visualization, "kusto_logger", logger)
    with pytest.raises(TypeError):
        visualization.plot(history_result(), "optimization_history", private_payload="private-caller-value")
    assert "private-caller-value" not in str(logger.mock_calls)


def test_edf_and_timeline_preserve_scores_and_timing():
    from flaml.fabric.visualization import plot_edf, plot_timeline

    automl = history_result()
    for info, duration, end in zip(automl.mlflow_integration.infos, [1.0, 2.0, 3.0], [4.0, 7.0, 11.0]):
        info["metrics"].update(trial_time=duration, wall_clock_time=end)
    edf = plot_edf(automl)
    assert len(edf.data) == 1
    np.testing.assert_allclose(edf.data[0].x, [0.5, 0.7, 0.9])
    np.testing.assert_allclose(edf.data[0].y, [1 / 3, 2 / 3, 1])
    assert edf.layout.xaxis.title.text == "Score"
    assert edf.layout.yaxis.title.text == "Probability"
    timeline = plot_timeline(automl)
    np.testing.assert_allclose(timeline.data[0].x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(timeline.data[0].base, dtype=float), [3.0, 5.0, 8.0])
    assert timeline.layout.xaxis.title.text == "Time (s)"
    assert timeline.layout.yaxis.title.text == "Trial"


@pytest.mark.parametrize("container", [list, dict])
def test_multi_study_edf_keeps_each_tune_series(container):
    from flaml.fabric.visualization import plot_edf, plot_timeline

    studies = [
        tune.run(
            lambda config: {"loss": float(config["x"])},
            config={"x": tune.randint(1, 5)},
            points_to_evaluate=[{"x": 1}, {"x": 3}],
            metric="loss",
            mode="min",
            num_samples=2,
            use_ray=False,
            mlflow_logging=False,
            verbose=0,
        )
        for _ in range(2)
    ]
    results = studies if container is list else {"first": studies[0], "second": studies[1]}
    figure = plot_edf(results)
    assert {trace.name for trace in figure.data} == ({"0", "1"} if container is list else {"first", "second"})
    for trace in figure.data:
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), [1, 3])
        np.testing.assert_allclose(trace.y, [0.5, 1.0])
    timeline = plot_timeline(studies[0])
    assert len(timeline.data) == 1
    starts = np.asarray([trial.start_time for trial in studies[0].trials])
    np.testing.assert_allclose(np.asarray(timeline.data[0].base, dtype=float), starts - starts.min())


def test_tune_timeline_preserves_overlap_and_scheduling_gaps():
    from flaml.fabric.visualization import plot_timeline
    from flaml.tune.tune import ExperimentAnalysis

    trials = [
        SimpleNamespace(
            trial_id=str(index),
            start_time=start,
            last_update_time=start + duration,
            last_result={"loss": 0.1, "config": {"x": index}, "time_total_s": duration},
        )
        for index, (start, duration) in enumerate([(100.0, 5.0), (100.0, 4.0), (112.0, 3.0)])
    ]
    figure = plot_timeline(ExperimentAnalysis(trials, metric="loss", mode="min"))
    np.testing.assert_allclose(np.asarray(figure.data[0].base, dtype=float), [0.0, 0.0, 12.0])
    np.testing.assert_allclose(np.asarray(figure.data[0].x, dtype=float), [5.0, 4.0, 3.0])
    assert figure.layout.meta["timing"] == "recorded"


def test_tune_timeline_labels_missing_time_metadata_as_approximate():
    from flaml.fabric.visualization import plot_timeline
    from flaml.tune.tune import ExperimentAnalysis

    trials = [
        SimpleNamespace(trial_id=str(index), last_result={"loss": 0.1, "config": {}, "time_total_s": duration})
        for index, duration in enumerate([2.0, 3.0])
    ]
    figure = plot_timeline(ExperimentAnalysis(trials, metric="loss", mode="min"))
    assert figure.layout.meta["timing"] == "approximate"
    assert "Estimated" in figure.data[0].name
    np.testing.assert_allclose(np.asarray(figure.data[0].base, dtype=float), [0.0, 2.0])


@pytest.mark.parametrize("learner_name", ["catboost", "nameless"])
def test_feature_importance_uses_the_fitted_reduced_schema(learner_name):
    from flaml.automl.model import RandomForestEstimator
    from flaml.fabric.visualization import plot_feature_importance

    class NamelessForest(RandomForestEstimator):
        def _fit(self, X_train, y_train, **kwargs):
            return super()._fit(np.asarray(X_train), y_train, **kwargs)

    if learner_name == "catboost":
        pytest.importorskip("catboost")
    X = np.random.RandomState(7).normal(size=(60, 3))
    frame = pd.DataFrame(X, columns=["raw_a", "raw_b", "raw_c"])
    automl = AutoML(
        estimator_list=[learner_name],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        starting_points={learner_name: {"n_estimators": 4, "fe.extraction": "LDA", "fe.selection": "null"}},
    )
    if learner_name == "nameless":
        automl.add_learner("nameless", NamelessForest)
    automl.fit(frame, (X[:, 0] > 0).astype(int))
    figure = plot_feature_importance(automl)
    assert len(figure.data[0].x) == len(figure.data[0].y) == 1
    assert figure.data[0].y[0] not in frame.columns
