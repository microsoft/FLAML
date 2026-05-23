import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, r2_score

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None


def pytest_configure(config):
    os.environ["FLAML_FEATURIZATION"] = os.environ.get("FLAML_FEATURIZATION", "auto")

    # Isolate MLflow tracking per xdist worker (and per process) so that
    # parallel workers don't race on writes to the same ``mlruns/<exp>/<run>/meta.yaml``
    # file. Concurrent writers can leave the file half-written, which then
    # surfaces as ``yaml.parser.ParserError: while parsing a block mapping``
    # in every subsequent test that talks to MLflow on the same worker.
    #
    # ``pytest_configure`` runs in the main pytest process AND in every
    # xdist worker. The main process runs first and exports
    # ``MLFLOW_TRACKING_URI``; workers then inherit that value. Without
    # the override below, all workers would share the main process's
    # tracking dir and race on writes. We therefore force-override the
    # URI in xdist workers when the inherited value is one we previously
    # set ourselves (path contains ``flaml_mlruns_``); a URI set
    # externally by the user is left untouched.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    existing_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    inherited_from_main = bool(worker_id) and "flaml_mlruns_" in existing_uri
    if not existing_uri or inherited_from_main:
        import tempfile
        from pathlib import Path

        tag = worker_id or "main"
        # IMPORTANT: do NOT pre-create this directory. ``FileStore.__init__``
        # only bootstraps the default ``Experiment(id="0")`` (which several
        # tests in ``test/fabric/test_mlflow_coverage.py`` implicitly rely on
        # via ``mlflow.start_run()``) when the root directory does not yet
        # exist. Pre-creating the dir would leave the store without a
        # Default experiment, breaking those tests with
        # ``MlflowException: Could not find experiment with ID 0``.
        tracking_dir = os.path.join(tempfile.gettempdir(), f"flaml_mlruns_{tag}_{os.getpid()}")
        # ``file:`` URIs need POSIX-style forward slashes even on Windows;
        # ``Path.as_uri()`` handles the platform-specific prefix correctly.
        os.environ["MLFLOW_TRACKING_URI"] = Path(tracking_dir).as_uri()

    _patch_mlflow_file_store_for_corrupt_yaml()


def _patch_mlflow_file_store_for_corrupt_yaml():
    """Make mlflow's ``FileStore`` tolerant of corrupt ``meta.yaml`` files.

    FLAML's ``tune.run`` / ``AutoML.fit`` can drive parallel trials
    (``n_concurrent_trials > 1``, ``use_ray=True``, ``use_spark=True``)
    that issue concurrent writes to mlflow's file-backed store. Without
    atomic file writes, a reader can observe a half-written
    ``meta.yaml`` and raise ``yaml.parser.ParserError``. That parse
    error then poisons every subsequent ``search_runs()`` call against
    the same experiment because ``FileStore._list_run_infos`` walks
    every run directory.

    Patch ``_read_yaml`` so that, after a brief retry, parse errors are
    translated into ``MissingConfigException`` which
    ``_list_run_infos`` already catches and skips. This contains the
    blast radius of a single bad write to the affected run instead of
    failing every later FLAML+MLflow test on the worker.
    """
    try:
        import time

        import yaml
        from mlflow.exceptions import MissingConfigException
        from mlflow.store.tracking.file_store import FileStore
    except Exception:
        return

    if getattr(FileStore, "_flaml_corrupt_yaml_patch", False):
        return

    original_read_yaml = FileStore._read_yaml

    @staticmethod
    def _safe_read_yaml(root, file_name, retries=2):
        last_error = None
        for attempt in range(retries + 1):
            try:
                return original_read_yaml(root, file_name, retries=retries)
            except yaml.YAMLError as exc:
                last_error = exc
                time.sleep(0.1 * (attempt + 1))
        raise MissingConfigException(f"Skipping unparsable {file_name} in {root}: {last_error}")

    FileStore._read_yaml = _safe_read_yaml
    FileStore._flaml_corrupt_yaml_patch = True


@pytest.fixture(autouse=True)
def _end_mlflow_runs_between_tests():
    """Ensure no MLflow run leaks between tests.

    A leaked active run causes ``MLflowIntegration.__init__`` (in
    ``flaml/fabric/mlflow.py``) to adopt that run's id as
    ``parent_run_id``, which then triggers
    ``MlflowException: Changing param values is not allowed`` when a
    later test logs ``n_estimators`` (or any other already-logged param)
    to that same parent run. Cleaning up both before and after each test
    keeps cross-test state from leaking in either direction.
    """
    try:
        import mlflow
    except Exception:
        yield
        return

    try:
        while mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass
    yield
    try:
        while mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass


def _is_catboost_model_type(model_type: type) -> bool:
    if CatBoostClassifier is not None and CatBoostRegressor is not None:
        return model_type is CatBoostClassifier or model_type is CatBoostRegressor
    return getattr(model_type, "__module__", "").startswith("catboost")


def evaluate_cv_folds_with_underlying_model(X_train_all, y_train_all, kf, model: Any, task: str) -> List[float]:
    """Mimic the FLAML CV process to calculate the metrics across each fold.

    :param X_train_all: X training data
    :param y_train_all: y training data
    :param kf: The splitter object to use to generate the folds
    :param model: The estimator to fit to the data during the CV process
    :param task: classification or regression
    :return: An array containing the metrics
    """
    rng = np.random.RandomState(2020)
    all_fold_metrics: List[float] = []
    for train_index, val_index in kf.split(X_train_all, y_train_all):
        X_train_split, y_train_split = X_train_all, y_train_all
        train_index = rng.permutation(train_index)
        X_train = X_train_split.iloc[train_index]
        X_val = X_train_split.iloc[val_index]
        y_train, y_val = y_train_split[train_index], y_train_split[val_index]
        model_type = type(model)
        if not _is_catboost_model_type(model_type):
            model.fit(X_train, y_train)
        else:
            if Pool is None:
                pytest.skip("catboost is not installed")
            use_best_model = True
            n = max(int(len(y_train) * 0.9), len(y_train) - 1000) if use_best_model else len(y_train)
            X_tr, y_tr = (X_train)[:n], y_train[:n]
            eval_set = Pool(data=X_train[n:], label=y_train[n:], cat_features=[]) if use_best_model else None
            model.fit(X_tr, y_tr, eval_set=eval_set, use_best_model=True)
        y_pred_classes = model.predict(X_val)
        if task == "classification":
            reproduced_metric = 1 - f1_score(y_val, y_pred_classes)
        else:
            reproduced_metric = 1 - r2_score(y_val, y_pred_classes)
        all_fold_metrics.append(float(reproduced_metric))
    return all_fold_metrics
