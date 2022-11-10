from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd

from flaml.automl.generic_task import GenericTask

# TODO: if your task is not specified in here, define your task as an all-capitalized word
SEQCLASSIFICATION = "seq-classification"
MULTICHOICECLASSIFICATION = "multichoice-classification"
TOKENCLASSIFICATION = "token-classification"

SEQREGRESSION = "seq-regression"

TS_FORECASTREGRESSION = (
    "forecast",
    "ts_forecast",
    "ts_forecast_regression",
)
REGRESSION = ("regression", SEQREGRESSION, *TS_FORECASTREGRESSION)
TS_FORECASTCLASSIFICATION = "ts_forecast_classification"
TS_FORECASTPANEL = "ts_forecast_panel"
TS_FORECAST = (
    *TS_FORECASTREGRESSION,
    TS_FORECASTCLASSIFICATION,
    TS_FORECASTPANEL,
)
CLASSIFICATION = (
    "binary",
    "multiclass",
    "classification",
    SEQCLASSIFICATION,
    MULTICHOICECLASSIFICATION,
    TOKENCLASSIFICATION,
    TS_FORECASTCLASSIFICATION,
)
RANK = ("rank",)
SUMMARIZATION = "summarization"
NLG_TASKS = (SUMMARIZATION,)
NLU_TASKS = (
    SEQREGRESSION,
    SEQCLASSIFICATION,
    MULTICHOICECLASSIFICATION,
    TOKENCLASSIFICATION,
)
NLP_TASKS = (*NLG_TASKS, *NLU_TASKS)


def _is_nlp_task(task):
    if task in NLU_TASKS or task in NLG_TASKS:
        return True
    else:
        return False


def get_classification_objective(num_labels: int) -> str:
    if num_labels == 2:
        objective_name = "binary"
    else:
        objective_name = "multiclass"
    return objective_name


class Task(ABC):
    estimators = {}

    def __init__(
        self,
        task_name: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
    ):
        self.name = task_name
        if X_train is not None:
            self.train_data_size = X_train.shape[0]
        else:
            self.train_data_size = None

    @abstractmethod
    def evaluate_model_CV(
        self,
        config,
        estimator,
        X_train_all,
        y_train_all,
        budget,
        kf,
        eval_metric,
        best_val_loss,
        log_training_metric=False,
        fit_kwargs={},
    ) -> Tuple[float, float, float, float]:
        pass

    @staticmethod
    @abstractmethod
    def _validate_data(
        automl,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        eval_method,
        time_col=None,
        X_val=None,
        y_val=None,
        groups_val=None,
        groups=None,
    ):
        pass

    @abstractmethod
    def _prepare_data(
        self,
        automl,
        eval_method,
        split_ratio,
        n_splits,
        time_col=None,
    ):
        pass

    def is_ts_forecast(self):
        return self.name in TS_FORECAST

    def is_nlp(self):
        return self.name in NLP_TASKS

    def is_nlg(self):
        return self.name in NLG_TASKS

    def is_classification(self):
        return self.name in CLASSIFICATION

    def is_rank(self):
        return self.name in RANK

    def is_binary(self):
        return self.name == "binary"

    def is_seq_regression(self):
        return self.name in SEQREGRESSION

    def is_seq_classification(self):
        return self.name in SEQCLASSIFICATION

    def is_token_classification(self):
        return self.name in TOKENCLASSIFICATION

    def is_summarization(self):
        return self.name in SUMMARIZATION

    def default_estimator_list(self):
        if self.is_rank():
            estimator_list = ["lgbm", "xgboost", "xgb_limitdepth"]
        elif self.is_nlp():
            estimator_list = ["transformer"]
        else:
            estimator_list = [
                "lgbm",
                "rf",
                "xgboost",
                "extra_tree",
                "xgb_limitdepth",
            ]

            try:
                import catboost

                estimator_list += ["catboost"]

            except ImportError:
                pass

        return estimator_list

    def __eq__(self, other):
        """For backward compatibility with all the string comparisons to task"""
        return self.name == other

    @classmethod
    def estimator_class_from_str(cls, estimator_name: str):
        if estimator_name in cls.estimators:
            return cls.estimators[estimator_name]
        else:
            raise ValueError(
                f"{estimator_name} is not a built-in learner for this task type, "
                f"only {list(cls.estimators.keys())} are supported."
                "Please use AutoML.add_learner() to add a customized learner."
            )


def task_factory(
    task_name: str,
    X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_train: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
) -> Task:
    return GenericTask(task_name, X_train, y_train)
