import json
import os
import pickle
import random
import sys
import tempfile
import time
from typing import MutableMapping

import mlflow
import pandas as pd
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, autologging_is_disabled
from scipy.sparse import issparse
from sklearn import tree

try:
    from pyspark.ml import Pipeline as SparkPipeline
except ImportError:

    class SparkPipeline:
        pass


# from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from sklearn.pipeline import Pipeline

from flaml.automl.logger import logger
from flaml.automl.spark import DataFrame, Series, psDataFrame, psSeries
from flaml.version import __version__

SEARCH_MAX_RESULTS = 5000  # Each train should not have more than 5000 trials
IS_RENAME_CHILD_RUN = os.environ.get("FLAML_IS_RENAME_CHILD_RUN", "false").lower() == "true"


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    if len(d) == 0:
        return d
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    keys = list(flat_dict.keys())
    for key in keys:
        if not isinstance(flat_dict[key], (int, float)):
            flat_dict.pop(key)
    return flat_dict


def is_autolog_enabled():
    return not all(autologging_is_disabled(k) for k in AUTOLOGGING_INTEGRATIONS.keys())


def get_mlflow_log_latency(model_history=False):
    st = time.time()
    with mlflow.start_run(nested=True, run_name="get_mlflow_log_latency") as run:
        if model_history:
            sk_model = tree.DecisionTreeClassifier()
            mlflow.sklearn.log_model(sk_model, "sk_models")
            mlflow.sklearn.log_model(Pipeline([("estimator", sk_model)]), "sk_pipeline")
            with tempfile.TemporaryDirectory() as tmpdir:
                pickle_fpath = os.path.join(tmpdir, f"tmp_{int(time.time()*1000)}")
                with open(pickle_fpath, "wb") as f:
                    pickle.dump(sk_model, f)
                mlflow.log_artifact(pickle_fpath, "sk_model1")
                mlflow.log_artifact(pickle_fpath, "sk_model2")
        mlflow.set_tag("synapseml.ui.visible", "false")  # not shown inline in fabric
    mlflow.delete_run(run.info.run_id)
    et = time.time()
    return et - st


def infer_signature(X_train=None, y_train=None, dataframe=None, label=None):
    if X_train is not None:
        if issparse(X_train):
            X_train = X_train.tocsr()
        elif isinstance(X_train, psDataFrame):
            X_train = X_train.to_spark(index_col="tmp_index_col")
            y_train = None
        try:
            signature = mlflow.models.infer_signature(X_train, y_train)
            return signature
        except (TypeError, MlflowException, Exception) as e:
            logger.debug(
                f"Failed to infer signature from X_train {type(X_train)} and y_train {type(y_train)}, error: {e}"
            )
    else:
        if dataframe is not None and label is not None:
            X = dataframe.drop(columns=label)
            y = dataframe[label]
            if isinstance(dataframe, psDataFrame):
                X = X.to_spark(index_col="tmp_index_col")
                y = None
            try:
                signature = mlflow.models.infer_signature(X, y)
                return signature
            except (TypeError, MlflowException, Exception) as e:
                logger.debug(
                    f"Failed to infer signature from dataframe {type(dataframe)} and label {label}, error: {e}"
                )


def _mlflow_wrapper(evaluation_func, mlflow_exp_id, mlflow_config=None, extra_tags=None, autolog=False):
    def wrapped(*args, **kwargs):
        if mlflow_config is not None:
            from synapse.ml.mlflow import set_mlflow_env_config

            set_mlflow_env_config(mlflow_config)
        import mlflow

        if mlflow_exp_id is not None:
            mlflow.set_experiment(experiment_id=mlflow_exp_id)
        if autolog:
            if mlflow.__version__ > "2.5.0" and extra_tags is not None:
                mlflow.autolog(silent=True, extra_tags=extra_tags)
            else:
                mlflow.autolog(silent=True)
            logger.debug("activated mlflow autologging on executor")
        else:
            mlflow.autolog(disable=True, silent=True)
        # with mlflow.start_run(nested=True):
        result = evaluation_func(*args, **kwargs)
        return result

    return wrapped


def _get_notebook_name():
    return None


def safe_json_dumps(obj):
    def default(o):
        return str(o)

    return json.dumps(obj, default=default)


class MLflowIntegration:
    def __init__(self, experiment_type="automl", mlflow_exp_name=None, extra_tag=None):
        try:
            from synapse.ml.mlflow import get_mlflow_env_config

            self.driver_mlflow_env_config = get_mlflow_env_config()
            self._on_internal = True
            self._notebook_name = _get_notebook_name()
        except ModuleNotFoundError:
            self.driver_mlflow_env_config = None
            self._on_internal = False
            self._notebook_name = None

        self.autolog = False
        self.manual_log = False
        self.parent_run_id = None
        self.parent_run_name = None
        self.log_type = "null"
        self.resume_params = {}
        self.train_func = None
        self.best_iteration = None
        self.best_run_id = None
        self.child_counter = 0
        self.infos = []
        self.manual_run_ids = []
        self.has_summary = False
        self.has_model = False
        self.only_history = False
        self._do_log_model = True

        self.extra_tag = (
            extra_tag
            if extra_tag is not None
            else {"extra_tag.sid": f"flaml_{__version__}_{int(time.time())}_{random.randint(1001, 9999)}"}
        )
        self.start_time = time.time()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        parent_run_info = mlflow.active_run().info if mlflow.active_run() is not None else None
        if parent_run_info:
            self.experiment_id = parent_run_info.experiment_id
            self.parent_run_id = parent_run_info.run_id
            # attribute run_name is not available before mlflow 2.0.1
            self.parent_run_name = parent_run_info.run_name if hasattr(parent_run_info, "run_name") else "flaml_run"
            if self.parent_run_name == "":
                self.parent_run_name = mlflow.active_run().data.tags["mlflow.runName"]
        else:
            if mlflow_exp_name is None:
                if mlflow.tracking.fluent._active_experiment_id is None:
                    mlflow_exp_name = self._notebook_name if self._notebook_name else "flaml_default_experiment"
                    mlflow.set_experiment(experiment_name=mlflow_exp_name)
            else:
                mlflow.set_experiment(experiment_name=mlflow_exp_name)
            self.experiment_id = mlflow.tracking.fluent._active_experiment_id
        self.experiment_name = mlflow.get_experiment(self.experiment_id).name
        self.experiment_type = experiment_type
        self.update_autolog_state()

        if self.autolog:
            # only end user created parent run in autolog scenario
            mlflow.end_run()

    def set_mlflow_config(self):
        if self.driver_mlflow_env_config is not None:
            from synapse.ml.mlflow import set_mlflow_env_config

            set_mlflow_env_config(self.driver_mlflow_env_config)

    def wrap_evaluation_function(self, evaluation_function):
        wrapped_evaluation_function = _mlflow_wrapper(
            evaluation_function, self.experiment_id, self.driver_mlflow_env_config, self.extra_tag, self.autolog
        )
        return wrapped_evaluation_function

    def set_best_iter(self, result):
        # result: AutoML or ExperimentAnalysis
        try:
            self.best_iteration = result.best_iteration
        except AttributeError:
            self.best_iteration = None

    def update_autolog_state(
        self,
    ):
        # Currently we disable autologging for better control in AutoML
        _autolog = is_autolog_enabled()
        self._do_log_model = AUTOLOGGING_INTEGRATIONS["mlflow"].get("log_models", True)
        if self.experiment_type == "automl":
            self.autolog = False
            self.manual_log = mlflow.active_run() is not None or _autolog
            self.log_type = "manual"
            if _autolog:
                logger.debug("Disabling autologging")
                self.resume_params = AUTOLOGGING_INTEGRATIONS["mlflow"].copy()
                mlflow.autolog(disable=True, silent=True, log_models=self._do_log_model)
                self.log_type = "r_autolog"  # 'r' for replace autolog with manual log

        elif self.experiment_type == "tune":
            self.autolog = _autolog
            self.manual_log = not self.autolog and mlflow.active_run() is not None

            if self.autolog:
                self.log_type = "autolog"

            if self.manual_log:
                self.log_type = "manual"
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

    def copy_mlflow_run(self, src_id, target_id, components=["param", "metric", "tag"]):
        src_run = self.mlflow_client.get_run(src_id)
        if "param" in components:
            for param_name, param_value in src_run.data.params.items():
                try:
                    self.mlflow_client.log_param(target_id, param_name, param_value)
                except mlflow.exceptions.MlflowException:
                    pass

        timestamp = int(time.time() * 1000)

        if "metric" in components:
            _metrics = [Metric(key, value, timestamp, 0) for key, value in src_run.data.metrics.items()]
        else:
            _metrics = []

        if "tag" in components:
            _tags = [
                RunTag(key, str(value))
                for key, value in src_run.data.tags.items()
                if key.startswith("flaml") or key.startswith("synapseml")
            ]
        else:
            _tags = []
        self.mlflow_client.log_batch(run_id=target_id, metrics=_metrics, params=[], tags=_tags)

    def record_trial(self, result, trial, metric):
        if isinstance(result, dict):
            metrics = flatten_dict(result)
            metric_name = str(list(metrics.keys()))
        else:
            metrics = {metric: result}
            metric_name = metric

        if "ml" in trial.config.keys():
            params = trial.config["ml"]
        else:
            params = trial.config

        info = {
            "metrics": metrics,
            "params": params,
            "tags": {
                "flaml.best_run": False,
                "flaml.iteration_number": self.child_counter,
                "flaml.version": __version__,
                "flaml.meric": metric_name,
                "flaml.run_source": "flaml-tune",
                "flaml.log_type": self.log_type,
            },
            "submetrics": {
                "values": [],
            },
        }

        self.infos.append(info)

        if not self.autolog and not self.manual_log:
            return

        if self.manual_log:
            with mlflow.start_run(
                nested=True, run_name=f"{self.parent_run_name}_child_{self.child_counter}"
            ) as child_run:
                self._log_info_to_run(info, child_run.info.run_id, log_params=True)
                self.manual_run_ids.append(child_run.info.run_id)
        self.child_counter += 1

    def log_tune(self, analysis, metric):
        self.set_best_iter(analysis)
        if self.autolog:
            if self.parent_run_id is not None:
                mlflow.start_run(run_id=self.parent_run_id, experiment_id=self.experiment_id)
                mlflow.log_metric("num_child_runs", len(self.infos))
            self.adopt_children(analysis)

        if self.manual_log:
            if "ml" in analysis.best_config.keys():
                mlflow.log_params(analysis.best_config["ml"])
            else:
                mlflow.log_params(analysis.best_config)
            mlflow.log_metric("best_" + metric, analysis.best_result[metric])
            best_mlflow_run_id = self.manual_run_ids[analysis.best_iteration]
            best_mlflow_run_name = self.mlflow_client.get_run(best_mlflow_run_id).info.run_name
            analysis.best_run_id = best_mlflow_run_id
            analysis.best_run_name = best_mlflow_run_name
            self.mlflow_client.set_tag(best_mlflow_run_id, "flaml.best_run", True)
            self.best_run_id = best_mlflow_run_id
            if not self.has_summary:
                self.copy_mlflow_run(best_mlflow_run_id, self.parent_run_id)
                self.has_summary = True

    def log_model(self, model, estimator, signature=None):
        if not self._do_log_model:
            return
        logger.debug(f"logging model {estimator}")
        if estimator.endswith("_spark"):
            mlflow.spark.log_model(model, estimator, signature=signature)
            mlflow.spark.log_model(model, "model", signature=signature)
        elif estimator in ["lgbm"]:
            mlflow.lightgbm.log_model(model, estimator, signature=signature)
        elif estimator in ["transformer", "transformer_ms"]:
            mlflow.transformers.log_model(model, estimator, signature=signature)
        elif estimator in ["arima", "sarimax", "holt-winters", "snaive", "naive", "savg", "avg", "ets"]:
            mlflow.statsmodels.log_model(model, estimator, signature=signature)
        elif estimator in ["tcn", "tft"]:
            mlflow.pytorch.log_model(model, estimator, signature=signature)
        elif estimator in ["prophet"]:
            mlflow.prophet.log_model(model, estimator, signature=signature)
        elif estimator in ["orbit"]:
            pass
        else:
            mlflow.sklearn.log_model(model, estimator, signature=signature)

    def _pickle_and_log_artifact(self, obj, artifact_name, pickle_fname="temp_.pkl"):
        if not self._do_log_model:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_fpath = os.path.join(tmpdir, pickle_fname)
            try:
                with open(pickle_fpath, "wb") as f:
                    pickle.dump(obj, f)
                mlflow.log_artifact(pickle_fpath, artifact_name)
            except Exception as e:
                logger.debug(f"Failed to pickle and log artifact {artifact_name}, error: {e}")

    def pickle_and_log_automl_artifacts(self, automl, model, estimator, signature=None):
        """log automl artifacts to mlflow
        load back with `automl = mlflow.pyfunc.load_model(model_run_id_or_uri)`, then do prediction with `automl.predict(X)`
        """
        logger.debug(f"logging automl artifacts {estimator}")
        self._pickle_and_log_artifact(automl.feature_transformer, "feature_transformer", "feature_transformer.pkl")
        self._pickle_and_log_artifact(automl.label_transformer, "label_transformer", "label_transformer.pkl")
        # Test test_mlflow 1 and 4 will get error: TypeError: cannot pickle '_io.TextIOWrapper' object
        # try:
        #     self._pickle_and_log_artifact(automl, "automl", "automl.pkl")
        # except TypeError:
        #     pass
        if estimator.endswith("_spark"):
            # spark pipeline is not supported yet
            return
        feature_transformer = automl.feature_transformer
        if isinstance(feature_transformer, Pipeline):
            pipeline = feature_transformer
            pipeline.steps.append(("estimator", model))
        elif isinstance(feature_transformer, SparkPipeline):
            pipeline = feature_transformer
            pipeline.stages.append(model)
        elif not estimator.endswith("_spark"):
            steps = [("feature_transformer", feature_transformer)]
            steps.append(("estimator", model))
            pipeline = Pipeline(steps)
        else:
            stages = [feature_transformer]
            stages.append(model)
            pipeline = SparkPipeline(stages=stages)
        if isinstance(pipeline, SparkPipeline):
            logger.debug(f"logging spark pipeline {estimator}")
            mlflow.spark.log_model(pipeline, "automl_pipeline", signature=signature)
        else:
            # Add a log named "model" to fit default settings
            logger.debug(f"logging sklearn pipeline {estimator}")
            mlflow.sklearn.log_model(pipeline, "automl_pipeline", signature=signature)
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)

    def record_state(self, automl, search_state, estimator):
        _st = time.time()
        automl_metric_name = (
            automl._state.metric if isinstance(automl._state.metric, str) else automl._state.error_metric
        )

        if automl._state.error_metric.startswith("1-"):
            automl_metric_value = 1 - search_state.val_loss
        elif automl._state.error_metric.startswith("-"):
            automl_metric_value = -search_state.val_loss
        else:
            automl_metric_value = search_state.val_loss

        if "ml" in search_state.config:
            config = search_state.config["ml"]
        else:
            config = search_state.config

        info = {
            "metrics": {
                "iter_counter": automl._track_iter,
                "trial_time": search_state.trial_time,
                "wall_clock_time": automl._state.time_from_start,
                "validation_loss": search_state.val_loss,
                "best_validation_loss": search_state.best_loss,
                automl_metric_name: automl_metric_value,
            },
            "tags": {
                "flaml.best_run": False,
                "flaml.estimator_name": estimator,
                "flaml.estimator_class": search_state.learner_class.__name__,
                "flaml.iteration_number": automl._track_iter,
                "flaml.version": __version__,
                "flaml.learner": estimator,
                "flaml.sample_size": search_state.sample_size,
                "flaml.meric": automl_metric_name,
                "flaml.run_source": "flaml-automl",
                "flaml.log_type": self.log_type,
                "flaml.automl_user_configurations": safe_json_dumps(automl._automl_user_configurations),
            },
            "params": {
                "sample_size": search_state.sample_size,
                "learner": estimator,
                **config,
            },
            "submetrics": {
                "iter_counter": automl._iter_per_learner[estimator],
                "values": [],
            },
        }

        if (search_state.metric_for_logging is not None) and (
            "intermediate_results" in search_state.metric_for_logging
        ):
            info["submetrics"]["values"] = search_state.metric_for_logging["intermediate_results"]

        self.infos.append(info)

        if not self.autolog and not self.manual_log:
            return
        if self.manual_log:
            if self.parent_run_name is not None:
                run_name = f"{self.parent_run_name}_child_{self.child_counter}"
            else:
                run_name = None
            with mlflow.start_run(nested=True, run_name=run_name) as child_run:
                self._log_info_to_run(info, child_run.info.run_id, log_params=True)
                if automl._state.model_history:
                    self.log_model(
                        search_state.trained_estimator._model, estimator, signature=automl.estimator_signature
                    )
                    self.pickle_and_log_automl_artifacts(
                        automl, search_state.trained_estimator, estimator, signature=automl.pipeline_signature
                    )
                self.manual_run_ids.append(child_run.info.run_id)
            self.child_counter += 1

    def log_automl(self, automl):
        self.set_best_iter(automl)
        if self.autolog:
            if self.parent_run_id is not None:
                mlflow.start_run(run_id=self.parent_run_id, experiment_id=self.experiment_id)
                mlflow.log_metric("best_validation_loss", automl._state.best_loss)
                mlflow.log_metric("best_iteration", automl._best_iteration)
                mlflow.log_metric("num_child_runs", len(self.infos))
                if automl._trained_estimator is not None and not self.has_model:
                    self.log_model(
                        automl._trained_estimator._model, automl.best_estimator, signature=automl.estimator_signature
                    )
                    self.pickle_and_log_automl_artifacts(
                        automl, automl.model, automl.best_estimator, signature=automl.pipeline_signature
                    )
                    self.has_model = True

            self.adopt_children(automl)

        if self.manual_log:
            best_mlflow_run_id = self.manual_run_ids[automl._best_iteration]
            best_run_name = self.mlflow_client.get_run(best_mlflow_run_id).info.run_name
            automl.best_run_id = best_mlflow_run_id
            automl.best_run_name = best_run_name
            self.mlflow_client.set_tag(best_mlflow_run_id, "flaml.best_run", True)
            self.best_run_id = best_mlflow_run_id
            if self.parent_run_id is not None:
                conf = automl._config_history[automl._best_iteration][1].copy()
                if "ml" in conf.keys():
                    conf = conf["ml"]

                mlflow.log_params(conf)
                mlflow.log_param("best_learner", automl._best_estimator)
                if not self.has_summary:
                    logger.info(f"logging best model {automl.best_estimator}")
                    self.copy_mlflow_run(best_mlflow_run_id, self.parent_run_id)
                    self.has_summary = True
                    if automl._trained_estimator is not None and not self.has_model:
                        self.log_model(
                            automl._trained_estimator._model,
                            automl.best_estimator,
                            signature=automl.estimator_signature,
                        )
                        self.pickle_and_log_automl_artifacts(
                            automl, automl.model, automl.best_estimator, signature=automl.pipeline_signature
                        )
                        self.has_model = True

    def resume_mlflow(self):
        if len(self.resume_params) > 0:
            mlflow.autolog(**self.resume_params)

    def _log_info_to_run(self, info, run_id, log_params=False):
        _metrics = [Metric(key, value, int(time.time() * 1000), 0) for key, value in info["metrics"].items()]
        _tags = [RunTag(key, str(value)) for key, value in info["tags"].items()]
        _params = [
            Param(key, str(value))
            for key, value in info["params"].items()
            if log_params or key in ["sample_size", "learner"]
        ]
        self.mlflow_client.log_batch(run_id=run_id, metrics=_metrics, params=_params, tags=_tags)

        if len(info["submetrics"]["values"]) > 0:
            for each_entry in info["submetrics"]["values"]:
                with mlflow.start_run(nested=True) as run:
                    each_entry.update({"iter_counter": info["submetrics"]["iter_counter"]})
                    _metrics = [Metric(key, value, int(time.time() * 1000), 0) for key, value in each_entry.items()]
                    _tags = [RunTag("mlflow.parentRunId", run_id)]
                    self.mlflow_client.log_batch(run_id=run.info.run_id, metrics=_metrics, params=[], tags=_tags)
            del info["submetrics"]["values"]

    def adopt_children(self, result=None):
        """
        Set autologging child runs to nested by fetching them after all child runs are completed.
        Note that this may cause disorder when concurrently starting multiple AutoML processes
        with the same experiment name if the MLflow version is less than or equal to "2.5.0".
        """
        if self.autolog:
            best_iteration = self.best_iteration
            if best_iteration is None:
                logger.warning("best_iteration is None, cannot identify best run")
            raw_autolog_child_runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=SEARCH_MAX_RESULTS,
                output_format="list",
                filter_string=(
                    f"tags.extra_tag.sid = '{self.extra_tag['extra_tag.sid']}'" if mlflow.__version__ > "2.5.0" else ""
                ),
            )
            self.child_counter = 0

            # From latest to earliest, remove duplicate cross-validation runs
            _exist_child_run_params = []  # for deduplication of cross-validation child runs
            _to_keep_autolog_child_runs = []
            for autolog_child_run in raw_autolog_child_runs:
                child_start_time = autolog_child_run.info.start_time / 1000

                if child_start_time < self.start_time:
                    continue

                _current_child_run_params = autolog_child_run.data.params
                # remove n_estimators as some models will train with small n_estimators to estimate time budget
                if self.experiment_type == "automl":
                    _current_child_run_params.pop("n_estimators", None)
                if _current_child_run_params in _exist_child_run_params:
                    # remove duplicate cross-validation run
                    self.mlflow_client.delete_run(autolog_child_run.info.run_id)
                    continue
                else:
                    _exist_child_run_params.append(_current_child_run_params)
                    _to_keep_autolog_child_runs.append(autolog_child_run)

            # From earliest to latest, set tags and child_counter
            autolog_child_runs = _to_keep_autolog_child_runs[::-1]
            for autolog_child_run in autolog_child_runs:
                child_run_id = autolog_child_run.info.run_id
                child_run_parent_id = autolog_child_run.data.tags.get("mlflow.parentRunId", None)
                child_start_time = autolog_child_run.info.start_time / 1000

                if child_start_time < self.start_time:
                    continue

                if all(
                    [
                        len(autolog_child_run.data.params) == 0,
                        len(autolog_child_run.data.metrics) == 0,
                        child_run_id != self.parent_run_id,
                    ]
                ):
                    # remove empty run
                    # empty run could be created by mlflow autologging
                    self.mlflow_client.delete_run(autolog_child_run.info.run_id)
                    continue

                if all(
                    [
                        child_run_id != self.parent_run_id,
                        child_run_parent_id is None or child_run_parent_id == self.parent_run_id,
                    ]
                ):
                    if self.parent_run_id is not None:
                        self.mlflow_client.set_tag(
                            child_run_id,
                            "mlflow.parentRunId",
                            self.parent_run_id,
                        )
                        if IS_RENAME_CHILD_RUN:
                            self.mlflow_client.set_tag(
                                child_run_id,
                                "mlflow.runName",
                                f"{self.parent_run_name}_child_{self.child_counter}",
                            )
                        self.mlflow_client.set_tag(child_run_id, "flaml.child_counter", self.child_counter)

                    # merge autolog child run and corresponding manual run
                    flaml_info = self.infos[self.child_counter]
                    child_run = self.mlflow_client.get_run(child_run_id)
                    self._log_info_to_run(flaml_info, child_run_id, log_params=False)

                    if self.experiment_type == "automl":
                        if "learner" not in child_run.data.params:
                            self.mlflow_client.log_param(child_run_id, "learner", flaml_info["params"]["learner"])
                        if "sample_size" not in child_run.data.params:
                            self.mlflow_client.log_param(
                                child_run_id, "sample_size", flaml_info["params"]["sample_size"]
                            )

                    if self.child_counter == best_iteration:
                        self.mlflow_client.set_tag(child_run_id, "flaml.best_run", True)
                        if result is not None:
                            result.best_run_id = child_run_id
                            result.best_run_name = child_run.info.run_name
                            self.best_run_id = child_run_id
                        if self.parent_run_id is not None and not self.has_summary:
                            self.copy_mlflow_run(child_run_id, self.parent_run_id)
                            self.has_summary = True
                    self.child_counter += 1

    def retrain(self, train_func, config):
        """retrain with given config, added for logging the best config and model to parent run.
        No more needed after v2.0.2post2 as we no longer log best config and model to parent run.
        """
        if self.autolog:
            self.set_mlflow_config()
            self.has_summary = True
            with mlflow.start_run(run_id=self.parent_run_id):
                train_func(config)

    def __del__(self):
        # mlflow.end_run()  # this will end the parent run when re-fit an AutoML instance. Bug 2922020: Inconsistent Run Creation Output
        self.resume_mlflow()


def register_automl_pipeline(automl, model_name=None, signature=None):
    pipeline = automl.automl_pipeline
    if pipeline is None:
        logger.warning("pipeline not found, cannot register it")
        return
    if model_name is None:
        model_name = automl._mlflow_exp_name + "_pipeline"
    if automl.best_run_id is None:
        mlflow.sklearn.log_model(
            pipeline,
            "automl_pipeline",
            registered_model_name=model_name,
            signature=automl.pipeline_signature if signature is None else signature,
        )
        mvs = mlflow.search_model_versions(
            filter_string=f"name='{model_name}'", order_by=["attribute.version_number ASC"], max_results=1
        )
        return mvs[0]
    else:
        best_run = mlflow.get_run(automl.best_run_id)
        model_uri = f"runs:/{best_run.info.run_id}/automl_pipeline"
        return mlflow.register_model(model_uri, model_name)
