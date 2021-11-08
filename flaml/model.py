# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
from contextlib import contextmanager
from functools import partial
import signal
import os
from typing import Callable, List
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from scipy.sparse import issparse
import pandas as pd
import logging
from . import tune
from .data import (
    group_counts,
    CLASSIFICATION,
    TS_FORECAST,
    TS_TIMESTAMP_COL,
    TS_VALUE_COL,
)

try:
    import psutil
except ImportError:
    psutil = None
try:
    import resource
except ImportError:
    resource = None

logger = logging.getLogger("flaml.automl")
FREE_MEM_RATIO = 0.2


def TimeoutHandler(sig, frame):
    raise TimeoutError(sig, frame)


@contextmanager
def limit_resource(memory_limit, time_limit):
    if memory_limit > 0:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft < 0 and (hard < 0 or memory_limit <= hard) or memory_limit < soft:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
    main_thread = False
    if time_limit is not None:
        try:
            signal.signal(signal.SIGALRM, TimeoutHandler)
            signal.alarm(int(time_limit) or 1)
            main_thread = True
        except ValueError:
            pass
    try:
        yield
    finally:
        if main_thread:
            signal.alarm(0)
        if memory_limit > 0:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


class BaseEstimator:
    """The abstract class for all learners.

    Typical examples:
        * XGBoostEstimator: for regression.
        * XGBoostSklearnEstimator: for classification.
        * LGBMEstimator, RandomForestEstimator, LRL1Classifier, LRL2Classifier:
            for both regression and classification.
    """

    def __init__(self, task="binary", **config):
        """Constructor.

        Args:
            task: A string of the task type, one of
                'binary', 'multi', 'regression', 'rank', 'forecast'
            config: A dictionary containing the hyperparameter names
                and 'n_jobs' as keys. n_jobs is the number of parallel threads.
        """
        self.params = self.config2params(config)
        self.estimator_class = self._model = None
        self._task = task
        if "_estimator_type" in config:
            self._estimator_type = self.params.pop("_estimator_type")
        else:
            self._estimator_type = (
                "classifier" if task in CLASSIFICATION else "regressor"
            )

    def get_params(self, deep=False):
        params = self.params.copy()
        params["task"] = self._task
        if hasattr(self, "_estimator_type"):
            params["_estimator_type"] = self._estimator_type
        return params

    @property
    def classes_(self):
        return self._model.classes_

    @property
    def n_features_in_(self):
        return self.model.n_features_in_

    @property
    def model(self):
        """Trained model after fit() is called, or None before fit() is called."""
        return self._model

    @property
    def estimator(self):
        """Trained model after fit() is called, or None before fit() is called."""
        return self._model

    def _preprocess(self, X):
        return X

    def _fit(self, X_train, y_train, **kwargs):

        current_time = time.time()
        if "groups" in kwargs:
            kwargs = kwargs.copy()
            groups = kwargs.pop("groups")
            if self._task == "rank":
                kwargs["group"] = group_counts(groups)
                # groups_val = kwargs.get('groups_val')
                # if groups_val is not None:
                #     kwargs['eval_group'] = [group_counts(groups_val)]
                #     kwargs['eval_set'] = [
                #         (kwargs['X_val'], kwargs['y_val'])]
                #     kwargs['verbose'] = False
                #     del kwargs['groups_val'], kwargs['X_val'], kwargs['y_val']
        X_train = self._preprocess(X_train)
        model = self.estimator_class(**self.params)
        if logger.level == logging.DEBUG:
            logger.debug(f"flaml.model - {model} fit started")
        model.fit(X_train, y_train, **kwargs)
        if logger.level == logging.DEBUG:
            logger.debug(f"flaml.model - {model} fit finished")
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def fit(self, X_train, y_train, budget=None, **kwargs):
        """Train the model from given training data.

        Args:
            X_train: A numpy array or a dataframe of training data in shape n*m.
            y_train: A numpy array or a series of labels in shape n*1.
            budget: A float of the time budget in seconds.

        Returns:
            train_time: A float of the training time in seconds.
        """
        if (
            getattr(self, "limit_resource", None)
            and resource is not None
            and (budget is not None or psutil is not None)
        ):
            start_time = time.time()
            mem = psutil.virtual_memory() if psutil is not None else None
            try:
                with limit_resource(
                    mem.available * (1 - FREE_MEM_RATIO)
                    + psutil.Process(os.getpid()).memory_info().rss
                    if mem is not None
                    else -1,
                    budget,
                ):
                    train_time = self._fit(X_train, y_train, **kwargs)
            except (MemoryError, TimeoutError) as e:
                logger.warning(f"{e.__class__} {e}")
                if self._task in CLASSIFICATION:
                    model = DummyClassifier()
                else:
                    model = DummyRegressor()
                X_train = self._preprocess(X_train)
                model.fit(X_train, y_train)
                self._model = model
                train_time = time.time() - start_time
        else:
            train_time = self._fit(X_train, y_train, **kwargs)
        return train_time

    def predict(self, X_test):
        """Predict label from features.

        Args:
            X_test: A numpy array or a dataframe of featurized instances, shape n*m.

        Returns:
            A numpy array of shape n*1.
            Each element is the label for a instance.
        """
        if self._model is not None:
            X_test = self._preprocess(X_test)
            return self._model.predict(X_test)
        else:
            return np.ones(X_test.shape[0])

    def predict_proba(self, X_test):
        """Predict the probability of each class from features.

        Only works for classification problems

        Args:
            X_test: A numpy array of featurized instances, shape n*m.

        Returns:
            A numpy array of shape n*c. c is the # classes.
            Each element at (i,j) is the probability for instance i to be in
                class j.
        """
        assert (
            self._task in CLASSIFICATION
        ), "predict_prob() only for classification task."
        X_test = self._preprocess(X_test)
        return self._model.predict_proba(X_test)

    def cleanup(self):
        pass

    @classmethod
    def search_space(cls, **params):
        """[required method] search space.

        Returns:
            A dictionary of the search space.
            Each key is the name of a hyperparameter, and value is a dict with
                its domain (required) and low_cost_init_value, init_value,
                cat_hp_cost (if applicable).
                e.g.,
                `{'domain': tune.randint(lower=1, upper=10), 'init_value': 1}.`
        """
        return {}

    @classmethod
    def size(cls, config: dict) -> float:
        """[optional method] memory size of the estimator in bytes.

        Args:
            config: A dict of the hyperparameter config.

        Returns:
            A float of the memory size required by the estimator to train the
            given config.
        """
        return 1.0

    @classmethod
    def cost_relative2lgbm(cls) -> float:
        """[optional method] relative cost compared to lightgbm."""
        return 1.0

    @classmethod
    def init(cls):
        """[optional method] initialize the class."""
        pass

    def config2params(self, config: dict) -> dict:
        """[optional method] config dict to params dict

        Args:
            config: A dict of the hyperparameter config.

        Returns:
            A dict that will be passed to self.estimator_class's constructor.
        """
        return config.copy()


class SKLearnEstimator(BaseEstimator):
    """The base class for tuning scikit-learn estimators."""

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)

    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            cat_columns = X.select_dtypes(include=["category"]).columns
            if not cat_columns.empty:
                X = X.copy()
                X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
        elif isinstance(X, np.ndarray) and X.dtype.kind not in "buif":
            # numpy array is not of numeric dtype
            X = pd.DataFrame(X)
            for col in X.columns:
                if isinstance(X[col][0], str):
                    X[col] = X[col].astype("category").cat.codes
            X = X.to_numpy()
        return X


class LGBMEstimator(BaseEstimator):
    """The class for tuning LGBM, using sklearn API."""

    ITER_HP = "n_estimators"
    HAS_CALLBACK = True

    @classmethod
    def search_space(cls, data_size, **params):
        upper = min(32768, int(data_size))
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "num_leaves": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "min_child_samples": {
                "domain": tune.lograndint(lower=2, upper=2 ** 7 + 1),
                "init_value": 20,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1.0),
                "init_value": 0.1,
            },
            # 'subsample': {
            #     'domain': tune.uniform(lower=0.1, upper=1.0),
            #     'init_value': 1.0,
            # },
            "log_max_bin": {  # log transformed with base 2
                "domain": tune.lograndint(lower=3, upper=11),
                "init_value": 8,
            },
            "colsample_bytree": {
                "domain": tune.uniform(lower=0.01, upper=1.0),
                "init_value": 1.0,
            },
            "reg_alpha": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1024),
                "init_value": 1 / 1024,
            },
            "reg_lambda": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1024),
                "init_value": 1.0,
            },
        }

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        if "log_max_bin" in params:
            params["max_bin"] = (1 << params.pop("log_max_bin")) - 1
        return params

    @classmethod
    def size(cls, config):
        num_leaves = int(round(config.get("num_leaves") or config["max_leaves"]))
        n_estimators = int(round(config["n_estimators"]))
        return (num_leaves * 3 + (num_leaves - 1) * 4 + 1.0) * n_estimators * 8

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        if "verbose" not in self.params:
            self.params["verbose"] = -1
        if "regression" == task:
            from lightgbm import LGBMRegressor

            self.estimator_class = LGBMRegressor
        elif "rank" == task:
            from lightgbm import LGBMRanker

            self.estimator_class = LGBMRanker
        else:
            from lightgbm import LGBMClassifier

            self.estimator_class = LGBMClassifier
        self._time_per_iter = None
        self._train_size = 0
        self._mem_per_iter = 1
        self.HAS_CALLBACK = self.HAS_CALLBACK and self._callbacks(0, 0) is not None

    def _preprocess(self, X):
        if (
            not isinstance(X, pd.DataFrame)
            and issparse(X)
            and np.issubdtype(X.dtype, np.integer)
        ):
            X = X.astype(float)
        elif isinstance(X, np.ndarray) and X.dtype.kind not in "buif":
            # numpy array is not of numeric dtype
            X = pd.DataFrame(X)
            for col in X.columns:
                if isinstance(X[col][0], str):
                    X[col] = X[col].astype("category").cat.codes
            X = X.to_numpy()
        return X

    def fit(self, X_train, y_train, budget=None, **kwargs):
        start_time = time.time()
        deadline = start_time + budget if budget else np.inf
        n_iter = self.params[self.ITER_HP]
        trained = False
        if not self.HAS_CALLBACK:
            mem0 = psutil.virtual_memory().available if psutil is not None else 1
            if (
                (
                    not self._time_per_iter
                    or abs(self._train_size - X_train.shape[0]) > 4
                )
                and budget is not None
                or self._mem_per_iter <= 1
                and psutil is not None
            ) and n_iter > 1:
                self.params[self.ITER_HP] = 1
                self._t1 = self._fit(X_train, y_train, **kwargs)
                if budget is not None and self._t1 >= budget or n_iter == 1:
                    # self.params[self.ITER_HP] = n_iter
                    return self._t1
                mem1 = psutil.virtual_memory().available if psutil is not None else 1
                self._mem1 = mem0 - mem1
                self.params[self.ITER_HP] = min(n_iter, 4)
                self._t2 = self._fit(X_train, y_train, **kwargs)
                mem2 = psutil.virtual_memory().available if psutil is not None else 1
                self._mem2 = max(mem0 - mem2, self._mem1)
                # if self._mem1 <= 0:
                #     self._mem_per_iter = self._mem2 / (self.params[self.ITER_HP] + 1)
                # elif self._mem2 <= 0:
                #     self._mem_per_iter = self._mem1
                # else:
                self._mem_per_iter = min(
                    self._mem1, self._mem2 / self.params[self.ITER_HP]
                )
                if self._mem_per_iter <= 1 and psutil is not None:
                    n_iter = self.params[self.ITER_HP]
                self._time_per_iter = (
                    (self._t2 - self._t1) / (self.params[self.ITER_HP] - 1)
                    if self._t2 > self._t1
                    else self._t1
                    if self._t1
                    else 0.001
                )
                self._train_size = X_train.shape[0]
                if (
                    budget is not None
                    and self._t1 + self._t2 >= budget
                    or n_iter == self.params[self.ITER_HP]
                ):
                    # self.params[self.ITER_HP] = n_iter
                    return time.time() - start_time
                trained = True
            # logger.debug(mem0)
            # logger.debug(self._mem_per_iter)
            if n_iter > 1:
                max_iter = min(
                    n_iter,
                    int(
                        (budget - time.time() + start_time - self._t1)
                        / self._time_per_iter
                        + 1
                    )
                    if budget is not None
                    else n_iter,
                    int((1 - FREE_MEM_RATIO) * mem0 / self._mem_per_iter)
                    if psutil is not None
                    else n_iter,
                )
                if trained and max_iter <= self.params[self.ITER_HP]:
                    return time.time() - start_time
                self.params[self.ITER_HP] = max_iter
        if self.params[self.ITER_HP] > 0:
            if self.HAS_CALLBACK:
                self._fit(
                    X_train,
                    y_train,
                    callbacks=self._callbacks(start_time, deadline),
                    **kwargs,
                )
                best_iteration = (
                    self._model.get_booster().best_iteration
                    if isinstance(self, XGBoostSklearnEstimator)
                    else self._model.best_iteration_
                )
                if best_iteration is not None:
                    self._model.set_params(n_estimators=best_iteration + 1)
            else:
                self._fit(X_train, y_train, **kwargs)
        else:
            self.params[self.ITER_HP] = self._model.n_estimators
        train_time = time.time() - start_time
        return train_time

    def _callbacks(self, start_time, deadline) -> List[Callable]:
        return [partial(self._callback, start_time, deadline)]

    def _callback(self, start_time, deadline, env) -> None:
        from lightgbm.callback import EarlyStopException

        now = time.time()
        if env.iteration == 0:
            self._time_per_iter = now - start_time
        if now + self._time_per_iter > deadline:
            raise EarlyStopException(env.iteration, env.evaluation_result_list)
        if psutil is not None:
            mem = psutil.virtual_memory()
            if mem.available / mem.total < FREE_MEM_RATIO:
                raise EarlyStopException(env.iteration, env.evaluation_result_list)


class XGBoostEstimator(SKLearnEstimator):
    """The class for tuning XGBoost regressor, not using sklearn API."""

    @classmethod
    def search_space(cls, data_size, **params):
        upper = min(32768, int(data_size))
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "max_leaves": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "min_child_weight": {
                "domain": tune.loguniform(lower=0.001, upper=128),
                "init_value": 1,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1.0),
                "init_value": 0.1,
            },
            "subsample": {
                "domain": tune.uniform(lower=0.1, upper=1.0),
                "init_value": 1.0,
            },
            "colsample_bylevel": {
                "domain": tune.uniform(lower=0.01, upper=1.0),
                "init_value": 1.0,
            },
            "colsample_bytree": {
                "domain": tune.uniform(lower=0.01, upper=1.0),
                "init_value": 1.0,
            },
            "reg_alpha": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1024),
                "init_value": 1 / 1024,
            },
            "reg_lambda": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1024),
                "init_value": 1.0,
            },
        }

    @classmethod
    def size(cls, config):
        return LGBMEstimator.size(config)

    @classmethod
    def cost_relative2lgbm(cls):
        return 1.6

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["max_depth"] = params.get("max_depth", 0)
        params["grow_policy"] = params.get("grow_policy", "lossguide")
        params["booster"] = params.get("booster", "gbtree")
        params["use_label_encoder"] = params.get("use_label_encoder", False)
        params["tree_method"] = params.get("tree_method", "hist")
        if "n_jobs" in config:
            params["nthread"] = params.pop("n_jobs")
        return params

    def __init__(
        self,
        task="regression",
        **config,
    ):
        super().__init__(task, **config)
        self.params["verbosity"] = 0

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import xgboost as xgb

        start_time = time.time()
        deadline = start_time + budget if budget else np.inf
        if issparse(X_train):
            self.params["tree_method"] = "auto"
        else:
            X_train = self._preprocess(X_train)
        if "sample_weight" in kwargs:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=kwargs["sample_weight"])
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)

        objective = self.params.get("objective")
        if isinstance(objective, str):
            obj = None
        else:
            obj = objective
            if "objective" in self.params:
                del self.params["objective"]
        _n_estimators = self.params.pop("n_estimators")
        callbacks = XGBoostEstimator._callbacks(start_time, deadline)
        if callbacks:
            self._model = xgb.train(
                self.params,
                dtrain,
                _n_estimators,
                obj=obj,
                callbacks=callbacks,
            )
            self.params["n_estimators"] = self._model.best_iteration + 1
        else:
            self._model = xgb.train(self.params, dtrain, _n_estimators, obj=obj)
            self.params["n_estimators"] = _n_estimators
        self.params["objective"] = objective
        del dtrain
        train_time = time.time() - start_time
        return train_time

    def predict(self, X_test):
        import xgboost as xgb

        if not issparse(X_test):
            X_test = self._preprocess(X_test)
        dtest = xgb.DMatrix(X_test)
        return super().predict(dtest)

    @classmethod
    def _callbacks(cls, start_time, deadline):
        try:
            from xgboost.callback import TrainingCallback
        except ImportError:  # for xgboost<1.3
            return None

        class ResourceLimit(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log) -> bool:
                now = time.time()
                if epoch == 0:
                    self._time_per_iter = now - start_time
                if now + self._time_per_iter > deadline:
                    return True
                if psutil is not None:
                    mem = psutil.virtual_memory()
                    if mem.available / mem.total < FREE_MEM_RATIO:
                        return True
                return False

        return [ResourceLimit()]


class XGBoostSklearnEstimator(SKLearnEstimator, LGBMEstimator):
    """The class for tuning XGBoost (for classification), using sklearn API."""

    @classmethod
    def search_space(cls, data_size, **params):
        return XGBoostEstimator.search_space(data_size)

    @classmethod
    def cost_relative2lgbm(cls):
        return XGBoostEstimator.cost_relative2lgbm()

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["max_depth"] = 0
        params["grow_policy"] = params.get("grow_policy", "lossguide")
        params["booster"] = params.get("booster", "gbtree")
        params["use_label_encoder"] = params.get("use_label_encoder", False)
        params["tree_method"] = params.get("tree_method", "hist")
        return params

    def __init__(
        self,
        task="binary",
        **config,
    ):
        super().__init__(task, **config)
        del self.params["verbose"]
        self.params["verbosity"] = 0
        import xgboost as xgb

        self.estimator_class = xgb.XGBRegressor
        if "rank" == task:
            self.estimator_class = xgb.XGBRanker
        elif task in CLASSIFICATION:
            self.estimator_class = xgb.XGBClassifier

    def fit(self, X_train, y_train, budget=None, **kwargs):
        if issparse(X_train):
            self.params["tree_method"] = "auto"
        return super().fit(X_train, y_train, budget, **kwargs)

    def _callbacks(self, start_time, deadline) -> List[Callable]:
        return XGBoostEstimator._callbacks(start_time, deadline)


class RandomForestEstimator(SKLearnEstimator, LGBMEstimator):
    """The class for tuning Random Forest."""

    HAS_CALLBACK = False

    @classmethod
    def search_space(cls, data_size, task, **params):
        data_size = int(data_size)
        upper = min(2048, data_size)
        space = {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "max_features": {
                "domain": tune.loguniform(lower=0.1, upper=1.0),
                "init_value": 1.0,
            },
            "max_leaves": {
                "domain": tune.lograndint(lower=4, upper=min(32768, data_size)),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
        }
        if task in CLASSIFICATION:
            space["criterion"] = {
                "domain": tune.choice(["gini", "entropy"]),
                # 'init_value': 'gini',
            }
        return space

    @classmethod
    def cost_relative2lgbm(cls):
        return 2.0

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        if "max_leaves" in params:
            params["max_leaf_nodes"] = params.get(
                "max_leaf_nodes", params.pop("max_leaves")
            )
        return params

    def __init__(
        self,
        task="binary",
        **params,
    ):
        super().__init__(task, **params)
        self.params["verbose"] = 0
        self.estimator_class = RandomForestRegressor
        if task in CLASSIFICATION:
            self.estimator_class = RandomForestClassifier


class ExtraTreesEstimator(RandomForestEstimator):
    """The class for tuning Extra Trees."""

    @classmethod
    def cost_relative2lgbm(cls):
        return 1.9

    def __init__(self, task="binary", **params):
        super().__init__(task, **params)
        if "regression" in task:
            self.estimator_class = ExtraTreesRegressor
        else:
            self.estimator_class = ExtraTreesClassifier


class LRL1Classifier(SKLearnEstimator):
    """The class for tuning Logistic Regression with L1 regularization."""

    @classmethod
    def search_space(cls, **params):
        return {
            "C": {
                "domain": tune.loguniform(lower=0.03125, upper=32768.0),
                "init_value": 1.0,
            },
        }

    @classmethod
    def cost_relative2lgbm(cls):
        return 160

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["tol"] = params.get("tol", 0.0001)
        params["solver"] = params.get("solver", "saga")
        params["penalty"] = params.get("penalty", "l1")
        return params

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        assert task in CLASSIFICATION, "LogisticRegression for classification task only"
        self.estimator_class = LogisticRegression


class LRL2Classifier(SKLearnEstimator):
    """The class for tuning Logistic Regression with L2 regularization."""

    limit_resource = True

    @classmethod
    def search_space(cls, **params):
        return LRL1Classifier.search_space(**params)

    @classmethod
    def cost_relative2lgbm(cls):
        return 25

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["tol"] = params.get("tol", 0.0001)
        params["solver"] = params.get("solver", "lbfgs")
        params["penalty"] = params.get("penalty", "l2")
        return params

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        assert task in CLASSIFICATION, "LogisticRegression for classification task only"
        self.estimator_class = LogisticRegression


class CatBoostEstimator(BaseEstimator):
    """The class for tuning CatBoost."""

    ITER_HP = "n_estimators"

    @classmethod
    def search_space(cls, data_size, **params):
        upper = max(min(round(1500000 / data_size), 150), 12)
        return {
            "early_stopping_rounds": {
                "domain": tune.lograndint(lower=10, upper=upper),
                "init_value": 10,
                "low_cost_init_value": 10,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=0.005, upper=0.2),
                "init_value": 0.1,
            },
            "n_estimators": {
                "domain": 8192,
                "init_value": 8192,
            },
        }

    @classmethod
    def size(cls, config):
        n_estimators = config.get("n_estimators", 8192)
        max_leaves = 64
        return (max_leaves * 3 + (max_leaves - 1) * 4 + 1.0) * n_estimators * 8

    @classmethod
    def cost_relative2lgbm(cls):
        return 15

    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            cat_columns = X.select_dtypes(include=["category"]).columns
            if not cat_columns.empty:
                X = X.copy()
                X[cat_columns] = X[cat_columns].apply(
                    lambda x: x.cat.rename_categories(
                        [
                            str(c) if isinstance(c, float) else c
                            for c in x.cat.categories
                        ]
                    )
                )
        elif isinstance(X, np.ndarray) and X.dtype.kind not in "buif":
            # numpy array is not of numeric dtype
            X = pd.DataFrame(X)
            for col in X.columns:
                if isinstance(X[col][0], str):
                    X[col] = X[col].astype("category").cat.codes
            X = X.to_numpy()
        return X

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["n_estimators"] = params.get("n_estimators", 8192)
        if "n_jobs" in params:
            params["thread_count"] = params.pop("n_jobs")
        return params

    def __init__(
        self,
        task="binary",
        **config,
    ):
        super().__init__(task, **config)
        self.params.update(
            {
                "verbose": config.get("verbose", False),
                "random_seed": config.get("random_seed", 10242048),
            }
        )
        from catboost import CatBoostRegressor

        self.estimator_class = CatBoostRegressor
        if task in CLASSIFICATION:
            from catboost import CatBoostClassifier

            self.estimator_class = CatBoostClassifier

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import shutil

        start_time = time.time()
        deadline = start_time + budget if budget else np.inf
        train_dir = f"catboost_{str(start_time)}"
        X_train = self._preprocess(X_train)
        if isinstance(X_train, pd.DataFrame):
            cat_features = list(X_train.select_dtypes(include="category").columns)
        else:
            cat_features = []
        n = max(int(len(y_train) * 0.9), len(y_train) - 1000)
        X_tr, y_tr = X_train[:n], y_train[:n]
        if "sample_weight" in kwargs:
            weight = kwargs["sample_weight"]
            if weight is not None:
                kwargs["sample_weight"] = weight[:n]
        else:
            weight = None
        from catboost import Pool, __version__

        model = self.estimator_class(train_dir=train_dir, **self.params)
        if __version__ >= "0.26":
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=Pool(
                    data=X_train[n:], label=y_train[n:], cat_features=cat_features
                ),
                callbacks=CatBoostEstimator._callbacks(start_time, deadline),
                **kwargs,
            )
        else:
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=Pool(
                    data=X_train[n:], label=y_train[n:], cat_features=cat_features
                ),
                **kwargs,
            )
        shutil.rmtree(train_dir, ignore_errors=True)
        if weight is not None:
            kwargs["sample_weight"] = weight
        self._model = model
        self.params[self.ITER_HP] = self._model.tree_count_
        train_time = time.time() - start_time
        return train_time

    @classmethod
    def _callbacks(cls, start_time, deadline):
        class ResourceLimit:
            def after_iteration(self, info) -> bool:
                now = time.time()
                if info.iteration == 1:
                    self._time_per_iter = now - start_time
                if now + self._time_per_iter > deadline:
                    return False
                if psutil is not None:
                    mem = psutil.virtual_memory()
                    if mem.available / mem.total < FREE_MEM_RATIO:
                        return False
                return True  # can continue

        return [ResourceLimit()]


class KNeighborsEstimator(BaseEstimator):
    @classmethod
    def search_space(cls, data_size, **params):
        upper = min(512, int(data_size / 2))
        return {
            "n_neighbors": {
                "domain": tune.lograndint(lower=1, upper=upper),
                "init_value": 5,
                "low_cost_init_value": 1,
            },
        }

    @classmethod
    def cost_relative2lgbm(cls):
        return 30

    def config2params(cls, config: dict) -> dict:
        params = config.copy()
        params["weights"] = params.get("weights", "distance")
        return params

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        if task in CLASSIFICATION:
            from sklearn.neighbors import KNeighborsClassifier

            self.estimator_class = KNeighborsClassifier
        else:
            from sklearn.neighbors import KNeighborsRegressor

            self.estimator_class = KNeighborsRegressor

    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            cat_columns = X.select_dtypes(["category"]).columns
            if X.shape[1] == len(cat_columns):
                raise ValueError("kneighbor requires at least one numeric feature")
            X = X.drop(cat_columns, axis=1)
        elif isinstance(X, np.ndarray) and X.dtype.kind not in "buif":
            # drop categocial columns if any
            X = pd.DataFrame(X)
            cat_columns = []
            for col in X.columns:
                if isinstance(X[col][0], str):
                    cat_columns.append(col)
            X = X.drop(cat_columns, axis=1)
            X = X.to_numpy()
        return X


class Prophet(SKLearnEstimator):
    """The class for tuning Prophet."""

    @classmethod
    def search_space(cls, **params):
        space = {
            "changepoint_prior_scale": {
                "domain": tune.loguniform(lower=0.001, upper=0.05),
                "init_value": 0.05,
                "low_cost_init_value": 0.001,
            },
            "seasonality_prior_scale": {
                "domain": tune.loguniform(lower=0.01, upper=10),
                "init_value": 10,
            },
            "holidays_prior_scale": {
                "domain": tune.loguniform(lower=0.01, upper=10),
                "init_value": 10,
            },
            "seasonality_mode": {
                "domain": tune.choice(["additive", "multiplicative"]),
                "init_value": "multiplicative",
            },
        }
        return space

    def __init__(self, task=TS_FORECAST, n_jobs=1, **params):
        super().__init__(task, **params)

    def _join(self, X_train, y_train):
        assert TS_TIMESTAMP_COL in X_train, (
            "Dataframe for training ts_forecast model must have column"
            f' "{TS_TIMESTAMP_COL}" with the dates in X_train.'
        )
        y_train = pd.DataFrame(y_train, columns=[TS_VALUE_COL])
        train_df = X_train.join(y_train)
        return train_df

    def fit(self, X_train, y_train, budget=None, **kwargs):
        from prophet import Prophet

        current_time = time.time()
        train_df = self._join(X_train, y_train)
        train_df = self._preprocess(train_df)
        cols = list(train_df)
        cols.remove(TS_TIMESTAMP_COL)
        cols.remove(TS_VALUE_COL)
        model = Prophet(**self.params)
        for regressor in cols:
            model.add_regressor(regressor)
        with suppress_stdout_stderr():
            model.fit(train_df)
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def predict(self, X_test):
        if isinstance(X_test, int):
            raise ValueError(
                "predict() with steps is only supported for arima/sarimax."
                " For Prophet, pass a dataframe with the first column containing"
                " the timestamp values."
            )
        if self._model is not None:
            X_test = self._preprocess(X_test)
            forecast = self._model.predict(X_test)
            return forecast["yhat"]
        else:
            logger.warning(
                "Estimator is not fit yet. Please run fit() before predict()."
            )
            return np.ones(X_test.shape[0])


class ARIMA(Prophet):
    """The class for tuning ARIMA."""

    @classmethod
    def search_space(cls, **params):
        space = {
            "p": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
        }
        return space

    def _join(self, X_train, y_train):
        train_df = super()._join(X_train, y_train)
        train_df.index = pd.to_datetime(train_df[TS_TIMESTAMP_COL])
        train_df = train_df.drop(TS_TIMESTAMP_COL, axis=1)
        return train_df

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import warnings

        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_estimator

        current_time = time.time()
        train_df = self._join(X_train, y_train)
        train_df = self._preprocess(train_df)
        cols = list(train_df)
        cols.remove(TS_VALUE_COL)
        regressors = cols
        if regressors:
            model = ARIMA_estimator(
                train_df[[TS_VALUE_COL]],
                exog=train_df[regressors],
                order=(self.params["p"], self.params["d"], self.params["q"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = ARIMA_estimator(
                train_df,
                order=(self.params["p"], self.params["d"], self.params["q"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        with suppress_stdout_stderr():
            model = model.fit()
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def predict(self, X_test):
        if self._model is not None:
            if isinstance(X_test, int):
                forecast = self._model.forecast(steps=X_test)
            elif isinstance(X_test, pd.DataFrame):
                first_col = X_test.pop(TS_TIMESTAMP_COL)
                X_test.insert(0, TS_TIMESTAMP_COL, first_col)
                start = X_test.iloc[0, 0]
                end = X_test.iloc[-1, 0]
                if len(X_test.columns) > 1:
                    regressors = list(X_test)
                    regressors.remove(TS_TIMESTAMP_COL)
                    X_test = self._preprocess(X_test)
                    forecast = self._model.predict(
                        start=start, end=end, exog=X_test[regressors]
                    )
                else:
                    forecast = self._model.predict(start=start, end=end)
            else:
                raise ValueError(
                    "X_test needs to be either a pd.Dataframe with dates as the first column"
                    " or an int number of periods for predict()."
                )
            return forecast
        else:
            return np.ones(X_test if isinstance(X_test, int) else X_test.shape[0])


class SARIMAX(ARIMA):
    """The class for tuning SARIMA."""

    @classmethod
    def search_space(cls, **params):
        space = {
            "p": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "P": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "D": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "Q": {
                "domain": tune.quniform(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "s": {
                "domain": tune.choice([1, 4, 6, 12]),
                "init_value": 12,
            },
        }
        return space

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import warnings

        warnings.filterwarnings("ignore")
        from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_estimator

        current_time = time.time()
        train_df = self._join(X_train, y_train)
        train_df = self._preprocess(train_df)
        regressors = list(train_df)
        regressors.remove(TS_VALUE_COL)
        if regressors:
            model = SARIMAX_estimator(
                train_df[[TS_VALUE_COL]],
                exog=train_df[regressors],
                order=(self.params["p"], self.params["d"], self.params["q"]),
                seasonality_order=(
                    self.params["P"],
                    self.params["D"],
                    self.params["Q"],
                    self.params["s"],
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = SARIMAX_estimator(
                train_df,
                order=(self.params["p"], self.params["d"], self.params["q"]),
                seasonality_order=(
                    self.params["P"],
                    self.params["D"],
                    self.params["Q"],
                    self.params["s"],
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        with suppress_stdout_stderr():
            model = model.fit()
        train_time = time.time() - current_time
        self._model = model
        return train_time


class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
