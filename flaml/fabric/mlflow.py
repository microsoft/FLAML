import atexit
import functools
import json
import logging
import os
import pickle
import random
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from typing import MutableMapping

import mlflow
import pandas as pd
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, autologging_is_disabled
from packaging.requirements import Requirement
from scipy.sparse import issparse
from sklearn import tree

try:
    from pyspark.ml import PipelineModel as SparkPipelineModel
except ImportError:

    class SparkPipelineModel:
        pass


# from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from sklearn.pipeline import Pipeline

from flaml.automl.logger import logger
from flaml.automl.spark import DataFrame, Series, psDataFrame, psSeries
from flaml.version import __version__

SEARCH_MAX_RESULTS = 5000  # Each train should not have more than 5000 trials
IS_RENAME_CHILD_RUN = os.environ.get("FLAML_IS_RENAME_CHILD_RUN", "false").lower() == "true"
REMOVE_REQUIREMENT_LIST = [
    "synapseml-cognitive",
    "synapseml-core",
    "synapseml-deep-learning",
    "synapseml-internal",
    "synapseml-mlflow",
    "synapseml-opencv",
    "synapseml-vw",
    "synapseml-lightgbm",
    "synapseml-utils",
    "nni",
    "optuna",
]
OPTIONAL_REMOVE_REQUIREMENT_LIST = ["pytorch-lightning", "transformers"]

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = os.environ.get("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

MLFLOW_NUM_WORKERS = int(os.environ.get("FLAML_MLFLOW_NUM_WORKERS", os.cpu_count() * 4 if os.cpu_count() else 2))
executor = ThreadPoolExecutor(max_workers=MLFLOW_NUM_WORKERS)
atexit.register(lambda: executor.shutdown(wait=True))

IS_CLEAN_LOGS = os.environ.get("FLAML_IS_CLEAN_LOGS", "1")
if IS_CLEAN_LOGS == "1":
    logging.getLogger("synapse.ml").setLevel(logging.CRITICAL)
    logging.getLogger("mlflow.utils").setLevel(logging.CRITICAL)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.CRITICAL)
    logging.getLogger("mlflow.models.model").setLevel(logging.CRITICAL)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)


def convert_requirement(requirement_list: list[str]):
    ret = (
        [Requirement(s.strip().lower()) for s in requirement_list]
        if mlflow.__version__ <= "2.17.0"
        else requirement_list
    )
    return ret


def time_it(func_or_code=None):
    """
    Decorator or function that measures execution time.

    Can be used in three ways:
    1. As a decorator with no arguments: @time_it
    2. As a decorator with arguments: @time_it()
    3. As a function call with a string of code to execute and time: time_it("some_code()")

    Args:
        func_or_code (callable or str, optional): Either a function to decorate or
            a string of code to execute and time.

    Returns:
        callable or None: Returns a decorated function if used as a decorator,
            or None if used to execute a string of code.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"Execution of {func.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return wrapper

    if callable(func_or_code):
        return decorator(func_or_code)
    elif func_or_code is None:
        return decorator
    else:
        start_time = time.time()
        exec(func_or_code)
        end_time = time.time()
        logger.debug(f"Execution\n```\n{func_or_code}\n```\ntook {end_time - start_time:.4f} seconds")


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


def get_mlflow_log_latency(model_history=False, delete_run=True):
    try:
        FLAML_MLFLOW_LOG_LATENCY = float(os.getenv("FLAML_MLFLOW_LOG_LATENCY", 0))
    except ValueError:
        FLAML_MLFLOW_LOG_LATENCY = 0
    if FLAML_MLFLOW_LOG_LATENCY >= 0.1:
        return FLAML_MLFLOW_LOG_LATENCY
    st = time.time()
    with mlflow.start_run(nested=True, run_name="get_mlflow_log_latency") as run:
        if model_history:
            sk_model = tree.DecisionTreeClassifier()
            mlflow.sklearn.log_model(sk_model, "model")
            with tempfile.TemporaryDirectory() as tmpdir:
                pickle_fpath = os.path.join(tmpdir, f"tmp_{int(time.time() * 1000)}")
                with open(pickle_fpath, "wb") as f:
                    pickle.dump(sk_model, f)
                mlflow.log_artifact(pickle_fpath, "sk_model")
        mlflow.set_tag("synapseml.ui.visible", "false")  # not shown inline in fabric
    if delete_run:
        mlflow.delete_run(run.info.run_id)
    et = time.time()
    return 3 * (et - st)


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


def update_and_install_requirements(
    run_id=None,
    model_name=None,
    model_version=None,
    remove_list=None,
    artifact_path="model",
    dst_path=None,
    install_with_ipython=False,
):
    if not (run_id or (model_name and model_version)):
        raise ValueError(
            "Please provide `run_id` or both `model_name` and `model_version`. If all three are provided, `run_id` will be used."
        )

    if install_with_ipython:
        from IPython import get_ipython

    if not remove_list:
        remove_list = [
            "synapseml-cognitive",
            "synapseml-core",
            "synapseml-deep-learning",
            "synapseml-internal",
            "synapseml-mlflow",
            "synapseml-opencv",
            "synapseml-vw",
            "synapseml-lightgbm",
            "synapseml-utils",
            "flaml",  # flaml is needed for AutoML models, should be pre-installed in the runtime
            "pyspark",  # fabric internal pyspark should be pre-installed in the runtime
        ]

    # Download model artifacts
    client = mlflow.MlflowClient()
    if not run_id:
        run_id = client.get_model_version(model_name, model_version).run_id
    if not dst_path:
        dst_path = os.path.join(tempfile.gettempdir(), "model_artifacts")
    os.makedirs(dst_path, exist_ok=True)
    client.download_artifacts(run_id, artifact_path, dst_path)
    requirements_path = os.path.join(dst_path, artifact_path, "requirements.txt")
    with open(requirements_path) as f:
        reqs = f.read().splitlines()
        old_reqs = [Requirement(req) for req in reqs if req]
        old_reqs_dict = {req.name: str(req) for req in old_reqs}
        for req in remove_list:
            req = Requirement(req)
            if req.name in old_reqs_dict:
                old_reqs_dict.pop(req.name, None)
        new_reqs_list = list(old_reqs_dict.values())

    with open(requirements_path, "w") as f:
        f.write("\n".join(new_reqs_list))

    if install_with_ipython:
        get_ipython().run_line_magic("pip", f"install -r {requirements_path} -q")
    else:
        logger.info(f"You can run `pip install -r {requirements_path}` to install dependencies.")
    return requirements_path


def _mlflow_wrapper(evaluation_func, mlflow_exp_id, mlflow_config=None, extra_tags=None, autolog=False):
    def wrapped(*args, **kwargs):
        if mlflow_config is not None:
            try:
                from synapse.ml.mlflow import set_mlflow_env_config

                set_mlflow_env_config(mlflow_config)
            except Exception:
                pass
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
    try:
        import re

        from synapse.ml.mlflow import get_mlflow_env_config
        from synapse.ml.mlflow.shared_platform_utils import get_artifact

        notebook_id = get_mlflow_env_config(False).artifact_id
        current_notebook = get_artifact(notebook_id)
        notebook_name = re.sub("\\W+", "-", current_notebook.displayName).strip()

        return notebook_name
    except Exception as e:
        logger.debug(f"Failed to get notebook name: {e}")
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
        self.futures = {}
        self.futures_log_model = {}

        self.extra_tag = (
            extra_tag
            if extra_tag is not None
            else {"extra_tag.sid": f"flaml_{__version__}_{int(time.time())}_{random.randint(1001, 9999)}"}
        )
        self.start_time = time.time()
        self.experiment_type = experiment_type
        self.update_autolog_state()

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

        if self.autolog:
            # only end user created parent run in autolog scenario
            mlflow.end_run()

    def set_mlflow_config(self):
        if self.driver_mlflow_env_config is not None:
            try:
                from synapse.ml.mlflow import set_mlflow_env_config

                set_mlflow_env_config(self.driver_mlflow_env_config)
            except Exception:
                pass

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
        return f"Successfully copy_mlflow_run run_id {src_id} to run_id {target_id}"

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

    def log_model(self, model, estimator, signature=None, run_id=None):
        if not self._do_log_model:
            return
        logger.debug(f"logging model {estimator}")
        ret_message = f"Successfully log_model {estimator} to run_id {run_id}"
        optional_remove_list = (
            [] if estimator in ["transformer", "transformer_ms", "tcn", "tft"] else OPTIONAL_REMOVE_REQUIREMENT_LIST
        )
        run = mlflow.active_run()
        if run and run.info.run_id == self.parent_run_id:
            logger.debug(
                f"Current active run_id {run.info.run_id} == parent_run_id {self.parent_run_id}, Starting run_id {run_id}"
            )
            mlflow.start_run(run_id=run_id, nested=True)
        elif run and run.info.run_id != run_id:
            ret_message = (
                f"Error: Should log_model {estimator} to run_id {run_id}, but logged to run_id {run.info.run_id}"
            )
            logger.error(ret_message)
        else:
            logger.debug(f"No active run, start run_id {run_id}")
            mlflow.start_run(run_id=run_id)
        logger.debug(f"logged model {estimator} to run_id {mlflow.active_run().info.run_id}")
        if estimator.endswith("_spark"):
            # mlflow.spark.log_model(model, estimator, signature=signature)
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
            logger.warning(f"Unsupported model: {estimator}. No model logged.")
        else:
            mlflow.sklearn.log_model(model, estimator, signature=signature)
        future = executor.submit(
            lambda: mlflow.models.model.update_model_requirements(
                model_uri=f"runs:/{run_id}/{'model' if estimator.endswith('_spark') else estimator}",
                operation="remove",
                requirement_list=convert_requirement(REMOVE_REQUIREMENT_LIST + optional_remove_list),
            )
        )
        self.futures[future] = f"run_{run_id}_requirements_updated"
        if not run or run.info.run_id == self.parent_run_id:
            logger.debug(f"Ending current run_id {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        return ret_message

    def _pickle_and_log_artifact(self, obj, artifact_name, pickle_fname="temp_.pkl", run_id=None):
        if not self._do_log_model:
            return True
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_fpath = os.path.join(tmpdir, pickle_fname)
            try:
                with open(pickle_fpath, "wb") as f:
                    pickle.dump(obj, f)
                mlflow.log_artifact(pickle_fpath, artifact_name, run_id)
                return True
            except Exception as e:
                logger.debug(f"Failed to pickle and log {artifact_name}, error: {e}")
                return False

    def _log_pipeline(self, pipeline, flavor_name, pipeline_name, signature, run_id, estimator=None):
        logger.debug(f"logging pipeline {flavor_name}:{pipeline_name}:{estimator}")
        ret_message = f"Successfully _log_pipeline {flavor_name}:{pipeline_name}:{estimator} to run_id {run_id}"
        optional_remove_list = (
            [] if estimator in ["transformer", "transformer_ms", "tcn", "tft"] else OPTIONAL_REMOVE_REQUIREMENT_LIST
        )
        run = mlflow.active_run()
        if run and run.info.run_id == self.parent_run_id:
            logger.debug(
                f"Current active run_id {run.info.run_id} == parent_run_id {self.parent_run_id}, Starting run_id {run_id}"
            )
            mlflow.start_run(run_id=run_id, nested=True)
        elif run and run.info.run_id != run_id:
            ret_message = f"Error: Should _log_pipeline {flavor_name}:{pipeline_name}:{estimator} model to run_id {run_id}, but logged to run_id {run.info.run_id}"
            logger.error(ret_message)
        else:
            logger.debug(f"No active run, start run_id {run_id}")
            mlflow.start_run(run_id=run_id)
        logger.debug(
            f"logging pipeline {flavor_name}:{pipeline_name}:{estimator} to run_id {mlflow.active_run().info.run_id}"
        )
        if flavor_name == "sklearn":
            mlflow.sklearn.log_model(pipeline, pipeline_name, signature=signature)
        elif flavor_name == "spark":
            mlflow.spark.log_model(pipeline, pipeline_name, signature=signature)
        else:
            logger.warning(f"Unsupported pipeline flavor: {flavor_name}. No model logged.")
        future = executor.submit(
            lambda: mlflow.models.model.update_model_requirements(
                model_uri=f"runs:/{run_id}/{pipeline_name}",
                operation="remove",
                requirement_list=convert_requirement(REMOVE_REQUIREMENT_LIST + optional_remove_list),
            )
        )
        self.futures[future] = f"run_{run_id}_requirements_updated"
        if not run or run.info.run_id == self.parent_run_id:
            logger.debug(f"Ending current run_id {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        return ret_message

    def pickle_and_log_automl_artifacts(self, automl, model, estimator, signature=None, run_id=None):
        """log automl artifacts to mlflow
        load back with `automl = mlflow.pyfunc.load_model(model_run_id_or_uri)`, then do prediction with `automl.predict(X)`
        """
        logger.debug(f"logging automl estimator {estimator}")
        # self._pickle_and_log_artifact(
        #     automl.feature_transformer, "feature_transformer", "feature_transformer.pkl", run_id
        # )
        # self._pickle_and_log_artifact(automl.label_transformer, "label_transformer", "label_transformer.pkl", run_id)
        if estimator.endswith("_spark"):
            # spark pipeline is not supported yet
            return
        feature_transformer = automl.feature_transformer
        if isinstance(feature_transformer, Pipeline) and not estimator.endswith("_spark"):
            pipeline = feature_transformer
            pipeline.steps.append(("estimator", model))
        elif isinstance(feature_transformer, SparkPipelineModel) and estimator.endswith("_spark"):
            pipeline = feature_transformer
            pipeline.stages.append(model)
        elif not estimator.endswith("_spark"):
            steps = [("feature_transformer", feature_transformer)]
            steps.append(("estimator", model))
            pipeline = Pipeline(steps)
        else:
            stages = []
            if feature_transformer is not None:
                stages.append(feature_transformer)
            stages.append(model)
            pipeline = SparkPipelineModel(stages=stages)
        if isinstance(pipeline, SparkPipelineModel):
            logger.debug(f"logging spark pipeline {estimator}")
            self._log_pipeline(pipeline, "spark", "model", signature, run_id, estimator)
        else:
            # Add a log named "model" to fit default settings
            logger.debug(f"logging sklearn pipeline {estimator}")
            self._log_pipeline(pipeline, "sklearn", "model", signature, run_id, estimator)
        return f"Successfully pickle_and_log_automl_artifacts {estimator} to run_id {run_id}"

    @time_it
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

        self.automl_user_configurations = safe_json_dumps(automl._automl_user_configurations)

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
                "flaml.automl_user_configurations": self.automl_user_configurations,
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
            _t1 = time.time()
            wait(self.futures_log_model)
            _t2 = time.time() - _t1
            logger.debug(f"wait futures_log_model in record_state took {_t2} seconds")
            with mlflow.start_run(nested=True, run_name=run_name) as child_run:
                future = executor.submit(lambda: self._log_info_to_run(info, child_run.info.run_id, log_params=True))
                self.futures[future] = f"iter_{automl._track_iter}_log_info_to_run"
                future = executor.submit(lambda: self._log_automl_configurations(child_run.info.run_id))
                self.futures[future] = f"iter_{automl._track_iter}_log_automl_configurations"
                if automl._state.model_history:
                    if estimator.endswith("_spark"):
                        future = executor.submit(
                            lambda: self.log_model(
                                search_state.trained_estimator._model,
                                estimator,
                                automl.estimator_signature,
                                child_run.info.run_id,
                            )
                        )
                        self.futures_log_model[future] = f"record_state-log_model_{estimator}"
                    else:
                        future = executor.submit(
                            lambda: self.pickle_and_log_automl_artifacts(
                                automl,
                                search_state.trained_estimator,
                                estimator,
                                automl.pipeline_signature,
                                child_run.info.run_id,
                            )
                        )
                        self.futures_log_model[future] = f"record_state-pickle_and_log_automl_artifacts_{estimator}"
                self.manual_run_ids.append(child_run.info.run_id)
            self.child_counter += 1
        return f"Successfully record_state iteration {automl._track_iter}"

    @time_it
    def log_automl(self, automl):
        self.set_best_iter(automl)
        if self.autolog:
            if self.parent_run_id is not None:
                mlflow.start_run(run_id=self.parent_run_id, experiment_id=self.experiment_id)
                mlflow.log_metrics(
                    {
                        "best_validation_loss": automl._state.best_loss,
                        "best_iteration": automl._best_iteration,
                        "num_child_runs": len(self.infos),
                    }
                )
                if (
                    automl._trained_estimator is not None
                    and not self.has_model
                    and automl._trained_estimator._model is not None
                ):
                    if automl.best_estimator.endswith("_spark"):
                        self.log_model(
                            automl._trained_estimator._model,
                            automl.best_estimator,
                            automl.estimator_signature,
                            self.parent_run_id,
                        )
                    else:
                        self.pickle_and_log_automl_artifacts(
                            automl, automl.model, automl.best_estimator, automl.pipeline_signature, self.parent_run_id
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

                mlflow.log_params({**conf, "best_learner": automl._best_estimator}, run_id=self.parent_run_id)
                if not self.has_summary:
                    logger.info(f"logging best model {automl.best_estimator}")
                    future = executor.submit(lambda: self.copy_mlflow_run(best_mlflow_run_id, self.parent_run_id))
                    self.futures[future] = "log_automl_copy_mlflow_run"
                    future = executor.submit(lambda: self._log_automl_configurations(self.parent_run_id))
                    self.futures[future] = "log_automl_log_automl_configurations"
                    self.has_summary = True
                    _t1 = time.time()
                    wait(self.futures_log_model)
                    _t2 = time.time() - _t1
                    logger.debug(f"wait futures_log_model in log_automl took {_t2} seconds")
                    if (
                        automl._trained_estimator is not None
                        and not self.has_model
                        and automl._trained_estimator._model is not None
                    ):
                        if automl.best_estimator.endswith("_spark"):
                            future = executor.submit(
                                lambda: self.log_model(
                                    automl._trained_estimator._model,
                                    automl.best_estimator,
                                    signature=automl.estimator_signature,
                                    run_id=self.parent_run_id,
                                )
                            )
                            self.futures_log_model[future] = f"log_automl-log_model_{automl.best_estimator}"
                        else:
                            future = executor.submit(
                                lambda: self.pickle_and_log_automl_artifacts(
                                    automl,
                                    automl.model,
                                    automl.best_estimator,
                                    signature=automl.pipeline_signature,
                                    run_id=self.parent_run_id,
                                )
                            )
                            self.futures_log_model[
                                future
                            ] = f"log_automl-pickle_and_log_automl_artifacts_{automl.best_estimator}"
                        self.has_model = True

    def resume_mlflow(self):
        if len(self.resume_params) > 0:
            mlflow.autolog(**self.resume_params)

    def _log_automl_configurations(self, run_id):
        self.mlflow_client.log_text(
            run_id=run_id,
            text=self.automl_user_configurations,
            artifact_file="automl_configurations/automl_user_configurations.json",
        )
        return f"Successfully _log_automl_configurations to run_id {run_id}"

    def _log_info_to_run(self, info, run_id, log_params=False):
        _metrics = [Metric(key, value, int(time.time() * 1000), 0) for key, value in info["metrics"].items()]
        _tags = [
            RunTag(key, str(value)[:5000]) for key, value in info["tags"].items()
        ]  # AML will raise error if value length > 5000
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
        return f"Successfully _log_info_to_run to run_id {run_id}"

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
