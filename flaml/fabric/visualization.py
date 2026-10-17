import itertools
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:
    if exc.name is not None and exc.name.split(".")[0] == "plotly":
        raise ImportError("Visualization requires Plotly. Install flaml[fabric_python] to enable plotting.") from exc
    raise

from flaml.automl import AutoML
from flaml.automl.logger import logger
from flaml.automl.training_log import training_log_reader
from flaml.fabric.logger import init_kusto_logger
from flaml.tune.trial import flatten_dict
from flaml.tune.tune import EA as ExperimentAnalysis

kusto_logger = init_kusto_logger("flaml.visualization")
_AUTOML_SCORE_MODE = "max"

__all__ = [
    "get_hp_df",
    "optuna_space",
    "get_param_importance",
    "plot",
    "plot_optimization_history",
    "plot_feature_importance",
    "plot_parallel_coordinate",
    "plot_contour",
    "plot_edf",
    "plot_timeline",
    "plot_slice",
    "plot_param_importance",
]


def _get_tune_df(analysis: ExperimentAnalysis):
    metric = analysis.default_metric
    df = pd.DataFrame(analysis.results).T
    conf_cols = [col for col in df.columns if col.startswith("config/")]
    if metric != "score":
        df["score"] = df.pop(metric)
    starts = {trial.trial_id: getattr(trial, "start_time", None) for trial in analysis.trials}
    timing = [starts.get(trial_id) for trial_id in df.index]
    df = df.reset_index().drop(columns=conf_cols + ["index"])
    df.attrs["trial_start_times"] = timing
    return df


def _get_aml_df(aml: AutoML):
    integration = getattr(aml, "mlflow_integration", None)
    history = getattr(integration, "infos", None)
    if not history:
        raise ValueError(
            "No trial history was collected for this AutoML result. Enable MLflow logging with an active run "
            "or autologging, or use Fabric history collection, before requesting a history-based plot."
        )
    metrics_history = [h["metrics"] for h in history]
    params_history = [h["params"] for h in history]
    metrics_df = pd.DataFrame(metrics_history)
    params_df = pd.DataFrame(params_history)
    metrics_df["config"] = params_df.to_dict("records")
    metrics_df["score"] = 1 - metrics_df.pop("validation_loss")
    metrics_df["learner"] = params_df["learner"]
    return metrics_df


def get_hp_df(result, learner=None, params=None):
    if isinstance(result, AutoML):
        big_df = _get_aml_df(result)
        if learner is None:
            learner = result.best_estimator
        df = big_df[big_df["learner"] == learner]

    elif isinstance(result, ExperimentAnalysis):
        df = _get_tune_df(result)
    else:
        raise TypeError(f"`result` must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")

    configurations = []
    for config in df["config"]:
        if isinstance(result, AutoML) and isinstance(config.get("ml"), dict):
            config = config["ml"]
        configurations.append(flatten_dict(config, prevent_delimiter=True))
    hp_df = pd.DataFrame(configurations, index=df.index)
    for name in hp_df.columns:
        if any(name in config and config[name] is None for config in configurations):
            hp_df[name] = pd.Series(
                [config.get(name, pd.NA) for config in configurations], index=df.index, dtype=object
            )

    if params is not None:
        hp_df = hp_df[params]

    return hp_df, df["score"]


def optuna_space(result, learner=None, params=None):
    from flaml.tune.searcher.suggestion import OptunaSearch

    if isinstance(result, AutoML):
        if learner is None:
            learner = result.best_estimator
        src_space = result._search_states[learner].search_space
    elif isinstance(result, ExperimentAnalysis):
        src_space = result.search_space
    else:
        raise TypeError(f"`result` must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")
    converted = OptunaSearch.convert_search_space(src_space)
    return {name: domain for name, domain in converted.items() if params is None or name in params}


def get_param_importance(result, learner=None, params=None):
    hp_df, scores = get_hp_df(result, learner, params)
    search_space = optuna_space(result, learner, params)

    # f-ANOVA requires some variance in the objective and enough samples.
    # In small/short runs (common in tests) scores can be constant.
    try:
        if scores is None or len(scores) < 2 or getattr(scores, "nunique", lambda: 0)() < 2:
            warnings.warn(
                "Not enough score variance to compute f-ANOVA hyperparameter importance. "
                "Try increasing time_budget/max_iter."
            )
            return {}
        if hp_df is None or hp_df.shape[0] < 2 or not search_space:
            warnings.warn(
                "Not enough trials/search space information to compute f-ANOVA hyperparameter importance. "
                "Try increasing time_budget/max_iter."
            )
            return {}
    except Exception:
        # Best-effort; proceed to evaluator.
        pass

    from flaml.fabric.fanova import FanovaImportanceEvaluator

    evaluator = FanovaImportanceEvaluator()

    try:
        importance = evaluator.evaluate(hp_df, scores, search_space)
    except RuntimeError as e:
        # Optuna's f-ANOVA evaluator raises when total variance is zero.
        warnings.warn(
            f"Failed to compute f-ANOVA hyperparameter importance: {e}. "
            "Try increasing time_budget/max_iter so scores vary across trials."
        )
        return {}
    return importance


def plot(
    result,
    fig_type,
    **kwargs,
) -> go.Figure:
    """
    An unified entry point for plotting.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
        fig_type: The type of figure you want to plot. Available options are:
            - "optimization_history": Plot optimization history of all trials in the experiment.
            - "feature_importance": Plot importance for each feature in the dataset.
            - "parallel_coordinate": Plot the high-dimensional parameter relationships in the experiment.
                Extra arguments for this figure:
                    - learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
                    - params: The parameters you want to plot. Plot all parameters if not specified.

            - "contour": Plot the parameter relationship as contour plot in the experiment.
                Extra arguments for this figure:
                    - learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
                    - params: The parameters you want to plot. Plot all parameters if not specified.

            - "edf": Plot the objective value EDF (empirical distribution function) of the experiment.
            - "timeline": Plot the timeline of the experiment.
            - "slice": Plot the parameter relationship as slice plot in a study.
                Extra arguments for this figure:
                    - learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
                    - params: The parameters you want to plot. Plot all parameters if not specified.
            - "param_importance": Plot the hyperparameter importance of the experiment.
                This plot use f-ANOVA to evaluate hyperparameter importance.
                Extra arguments for this figure:
                    - learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
                    - params: The parameters you want to plot. Plot all parameters if not specified.

        **kwargs: Extra arguments for the plot function.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    plot_func_map = {
        "optimization_history": plot_optimization_history,
        "feature_importance": plot_feature_importance,
        "parallel_coordinate": plot_parallel_coordinate,
        "contour": plot_contour,
        "edf": plot_edf,
        "timeline": plot_timeline,
        "slice": plot_slice,
        "param_importance": plot_param_importance,
    }
    plot_kind = fig_type if fig_type in plot_func_map else "unsupported"
    kusto_logger.info(f"Plot requested: type={plot_kind}, option_count={len(kwargs)}")
    try:
        plot_func = plot_func_map[fig_type]
    except KeyError:
        raise ValueError(f"Invalid figure type: {fig_type}")
    fig = plot_func(result, **kwargs)
    return fig


def plot_optimization_history(result) -> go.Figure:
    """
    Plot optimization history of all trials in the experiment.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info(f"plot_optimization_history: result={type(result)}")
    if isinstance(result, AutoML):
        df = _get_aml_df(result)
        comp = max if _AUTOML_SCORE_MODE == "max" else min
        y_title = "Score (1 - validation loss)"
    elif isinstance(result, ExperimentAnalysis):
        df = _get_tune_df(result)
        comp = min if result.default_mode == "min" else max
        y_title = result.default_metric
    else:
        raise TypeError(f"result must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")

    scores = df["score"]
    valid_scores = pd.Series(scores, dtype=float)
    valid_scores = valid_scores.where(np.isfinite(valid_scores))
    best_scores = (valid_scores.cummax() if comp is max else valid_scores.cummin()).ffill()

    loss_fig = go.Scatter(x=df.index, y=df["score"], name="score", mode="markers")
    best_loss_fig = go.Scatter(x=df.index, y=best_scores, name="best_score", mode="lines")
    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": y_title},
    )
    fig = go.Figure(data=[loss_fig, best_loss_fig], layout=layout)
    return fig


def plot_feature_importance(result) -> go.Figure:
    """
    Plot importance for each feature in the dataset.

    Args:
        result: Your experiment result from AutoML. Not available for Hyperparameter Tuning.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info(f"plot_feature_importance: result={type(result)}")
    if not isinstance(result, AutoML):
        raise NotImplementedError(f"Feature importance plot is only available for AutoML, get {type(result)}")
    feat_importance = result.feature_importances_
    if feat_importance is None:
        logger.warning(
            "Feature importances are None. Possible reasons: "
            "1) the estimator is not tree-based, "
            "2) the model has not been fitted, or "
            "3) the estimator is an ensemble wrapper (e.g., GridSearchCV, Pipeline)."
        )
        fig = go.Figure()
        layout = go.Layout(
            title="Feature Importance",
            xaxis={"title": "Feature Importance"},
            yaxis={"title": "Feature Name"},
        )
        fig.add_annotation(
            text="Feature Importance are None.<br>Try using a tree-based estimator or increasing the time_budget/max_iter.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig
    feat_importance = np.asarray(feat_importance)
    if len(feat_importance.shape) == 2:
        feat_importance = np.sqrt(np.sum(feat_importance**2, axis=0))
    feature_names = result.feature_names_in_
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(len(feat_importance))]
    if len(feature_names) != len(feat_importance):
        raise ValueError("Feature importance values do not match the fitted feature schema.")
    fig = px.bar(x=feat_importance, y=feature_names, orientation="h")
    layout = go.Layout(
        title="Feature Importance",
        xaxis={"title": "Feature Importance"},
        yaxis={"title": "Feature Name"},
    )
    fig.update_layout(layout)
    return fig


def plot_parallel_coordinate(
    result,
    learner=None,
    params=None,
) -> go.Figure:
    """
    Plot the high-dimensional parameter relationships in the experiment.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
        learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
        params: The parameters you want to plot. Plot all parameters if not specified.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info("Parallel-coordinate plot requested")
    if isinstance(result, AutoML):
        better = _AUTOML_SCORE_MODE
    elif isinstance(result, ExperimentAnalysis):
        learner = "Tuning"
        better = result.default_mode
    else:
        raise TypeError(f"result must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")

    hp_df, score = get_hp_df(result, learner, params)
    objective_column = "Objective"
    while objective_column in hp_df.columns:
        objective_column = "_" + objective_column
    hp_df[objective_column] = np.asarray(score, dtype=float)
    if better == "max":
        cspace = plotly.colors.sequential.Emrld
    else:
        cspace = plotly.colors.sequential.Emrld_r

    if learner is None:
        learner = result.best_estimator

    fig = px.parallel_coordinates(hp_df, color=objective_column, color_continuous_scale=cspace)
    layout = go.Layout(
        title={
            "text": f"Parallel Coordinate Plot for {learner}",
            "y": 0.1,
            "automargin": True,
        }
    )
    fig.update_layout(layout)
    return fig


def plot_contour(
    result,
    learner=None,
    params=None,
) -> go.Figure:
    """
    Plot the parameter relationship as contour plot in the experiment.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
        learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
        params: The parameters you want to plot. Plot all parameters if not specified.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info("Contour plot requested")
    if isinstance(result, AutoML):
        better = _AUTOML_SCORE_MODE
    elif isinstance(result, ExperimentAnalysis):
        learner = "Tuning"
        better = result.default_mode
    else:
        raise TypeError(f"result must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")

    if better == "max":
        cspace = plotly.colors.sequential.Emrld
    else:
        cspace = plotly.colors.sequential.Emrld_r

    hp_df, scores = get_hp_df(result, learner, params)

    params = list(hp_df.columns)
    param_loc = {param: i + 1 for i, param in enumerate(params)}
    n_params = len(params)

    def _get_contour_scatter(params):
        contour = go.Contour(
            z=scores,
            x=hp_df[params[0]],
            y=hp_df[params[1]],
            colorscale=cspace,
        )
        scatter = go.Scatter(
            x=hp_df[params[0]],
            y=hp_df[params[1]],
            marker={"line": {"width": 2.0, "color": "Grey"}, "color": "black"},
            mode="markers",
            showlegend=False,
        )
        return contour, scatter

    if n_params == 2:
        fig = go.Figure()
        contour, scatter = _get_contour_scatter(params)
        layout = go.Layout(
            xaxis={"title": params[0]},
            yaxis={"title": params[1]},
        )
        fig.add_trace(contour)
        fig.add_trace(scatter)
        fig.update_layout(layout)
    else:
        combos = list(itertools.combinations(params, 2))
        combos += [(p1, p2) for p2, p1 in combos]
        combos += [(p, p) for p in params]
        fig = make_subplots(rows=n_params, cols=n_params)
        for combo in combos:
            xloc, yloc = param_loc[combo[0]], param_loc[combo[1]]
            if xloc == yloc:
                fig.add_trace(go.Scatter(), row=xloc, col=yloc)
            else:
                contour, scatter = _get_contour_scatter(combo)
                fig.add_trace(contour, col=param_loc[combo[0]], row=param_loc[combo[1]])
                fig.add_trace(scatter, col=param_loc[combo[0]], row=param_loc[combo[1]])
            if yloc != 1:
                fig.update_yaxes(showticklabels=False, row=xloc, col=yloc)
            else:
                fig.update_yaxes(
                    title={
                        "text": combo[0],
                        "standoff": 0,
                    },
                    row=xloc,
                    col=yloc,
                )
            if xloc != n_params:
                fig.update_xaxes(showticklabels=False, row=xloc, col=yloc)
            else:
                fig.update_xaxes(title_text=combo[1], row=xloc, col=yloc)

    layout = go.Layout(
        margin_autoexpand=True,
    )
    fig.update_layout(layout)

    return fig


def plot_edf(result) -> go.Figure:
    """
    Plot the objective value EDF (empirical distribution function) of the experiment.
    EDF is useful to analyze and improve search spaces.
    For instance, you can see a practical use case of EDF in the paper
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
            For AutoML experiments, each learner's trials form an optimization series.
            For hyperparameter tuning, you can provide either a single result or
                multiple results (in a list or dictionary) to this function.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info(f"plot_edf: result={type(result)}")
    if isinstance(result, AutoML):
        df = _get_aml_df(result)
        fig = px.ecdf(df, x="score", color="learner")
    else:
        if isinstance(result, ExperimentAnalysis):
            result = [result]
        df = pd.DataFrame()
        if isinstance(result, Dict):
            iters = result.keys()
        else:
            iters = range(len(result))
        for i in iters:
            sub_df = _get_tune_df(result[i])
            sub_df["tune_id"] = f"{i}"
            df = pd.concat([df, sub_df])
        fig = px.ecdf(df, x="score", color="tune_id")

    layout = go.Layout(
        title="Empirical Distribution Function",
        xaxis={"title": "Score"},
        yaxis={"title": "Probability"},
    )
    fig.update_layout(layout)
    return fig


def plot_timeline(result) -> go.Figure:
    """
    Plot the timeline of the experiment.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info(f"plot_timeline: result={type(result)}")
    fig = go.Figure()
    if isinstance(result, AutoML):
        df = _get_aml_df(result)
        for learner, sub_df in df.groupby("learner"):
            trace = go.Bar(
                x=sub_df["trial_time"],
                y=sub_df.index,
                base=sub_df["wall_clock_time"] - sub_df["trial_time"],
                orientation="h",
                name=learner,
            )
            fig.add_trace(trace)
    elif isinstance(result, ExperimentAnalysis):
        df = _get_tune_df(result)
        starts = df.attrs["trial_start_times"]
        recorded = all(isinstance(value, (int, float, np.number)) and np.isfinite(value) for value in starts)
        if recorded and starts:
            starts = np.asarray(starts, dtype=float)
            bases = starts - starts.min()
            timing = "recorded"
        else:
            bases = df["time_total_s"].cumsum() - df["time_total_s"]
            timing = "approximate"
        trace = go.Bar(
            x=df["time_total_s"],
            y=df.index,
            base=bases,
            orientation="h",
            name="Trials" if timing == "recorded" else "Estimated sequence (timing unavailable)",
        )
        fig.add_trace(trace)
        fig.update_layout(meta={"timing": timing})
    else:
        raise TypeError(f"result must be an instance of AutoML or ExperimentAnalysis, get {type(result)}")

    fig.update_layout(
        go.Layout(
            title="Timeline Plot",
            xaxis={"title": "Time (s)"},
            yaxis={"title": "Trial"},
        )
    )
    return fig


def plot_slice(result, learner=None, params=None) -> go.Figure:
    """
    Plot the parameter relationship as slice plot in a study.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
        learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
        params: The parameters you want to plot. Plot all parameters if not specified.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info("Slice plot requested")
    hp_df, scores = get_hp_df(result, learner, params)

    params = hp_df.columns.tolist()
    fig = make_subplots(cols=len(params))

    for i, param in enumerate(params):
        hp = hp_df[param]
        if hp.isna().any():
            missing_label = "<missing>"
            present_labels = {str(value) for value in hp.dropna()}
            while missing_label in present_labels:
                missing_label = "_" + missing_label
            hp = hp.astype(object).where(hp.notna(), missing_label)
        counts = hp.value_counts(dropna=False)
        color_count = hp.map(counts)
        trace = go.Scatter(
            x=hp,
            y=scores,
            mode="markers",
            marker={
                "line": {"width": 0.5, "color": "Grey"},
                "color": color_count,
                "colorscale": "Emrld",
                "colorbar": {
                    "title": "Trial Counts",
                    "showticklabels": False,
                },
            },
        )
        if i != 0:
            fig.update_yaxes(showticklabels=False, row=1, col=i + 1)
        else:
            y_title = "Score (1 - validation loss)" if isinstance(result, AutoML) else result.default_metric
            fig.update_yaxes(title_text=y_title, row=1, col=i + 1)
        fig.update_xaxes(title_text=param, row=1, col=i + 1)
        fig.add_trace(trace, col=i + 1, row=1)

    layout = go.Layout(
        showlegend=False,
    )
    fig.update_layout(layout)
    return fig


def plot_param_importance(result, learner=None, params=None) -> go.Figure:
    """
    Plot the hyperparameter importance of the experiment.
    This plot uses Optuna's f-ANOVA evaluator to assess hyperparameter importance.

    Args:
        result: The experiment result of AutoML.fit() or Tune.run(). Must be an instance of AutoML or ExperimentAnalysis.
        learner: The learner you want to plot. Only available for AutoML. Plot best learner if not specified.
        params: The parameters you want to plot. Plot all parameters if not specified.

    Returns:
        A `plotly.graph_objs.Figure()` object.
    """
    kusto_logger.info("Parameter-importance plot requested")
    importance = get_param_importance(result, learner, params)

    if not importance:
        logger.warning(
            "Hyperparameter importance is empty. Possible reasons: "
            "1) not enough variance in the objective, "
            "2) not enough trials, or "
            "3) no hyperparameters in the search space."
        )
        fig = go.Figure()
        fig.update_layout(
            title="Hyperparameter Importance",
            xaxis={"title": "Importance"},
            yaxis={"title": "Hyperparameter"},
        )
        fig.add_annotation(
            text="Not enough variance/trials to compute importance.<br>Try increasing time_budget/max_iter.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    if isinstance(result, AutoML):
        learner = result.best_estimator if learner is None else learner
    elif isinstance(result, ExperimentAnalysis):
        learner = "Tuning"
    fig = px.bar(x=importance.values(), y=importance.keys(), orientation="h")
    layout = go.Layout(
        title=f"Hyperparameter Importance for {learner}",
        xaxis={"title": "Importance"},
        yaxis={"title": "Hyperparameter"},
    )
    fig.update_layout(layout)
    return fig
