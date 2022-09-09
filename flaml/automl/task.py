from abc import ABC, abstractmethod
from typing import Tuple

# TODO: if your task is not specified in here, define your task as an all-capitalized word
from typing import Union

import pandas as pd
import numpy as np

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
TS_FORECAST = (
    *TS_FORECASTREGRESSION,
    TS_FORECASTCLASSIFICATION,
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
SUMMARIZATION = "summarization"
NLG_TASKS = (SUMMARIZATION,)
NLU_TASKS = (
    SEQREGRESSION,
    SEQCLASSIFICATION,
    MULTICHOICECLASSIFICATION,
    TOKENCLASSIFICATION,
)
NLP_TASKS = (*NLG_TASKS, *NLU_TASKS)


def get_classification_objective(num_labels: int) -> str:
    if num_labels == 2:
        objective_name = "binary"
    else:
        objective_name = "multiclass"
    return objective_name


class Task:
    def __init__(
        self,
        task_name: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
    ):
        self.name = task_name
        if X_train is not None:
            self.train_data_size = len(X_train)
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

    def is_ts_forecast(self):
        return self.name in TS_FORECAST

    def is_nlp(self):
        return self.name in NLP_TASKS

    def is_nlg(self):
        return self.name in NLG_TASKS

    def is_classification(self):
        return self.name in CLASSIFICATION

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
        if self.name == "rank":
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

        # For backward compatibility with all the string comparisons to task

    def __eq__(self, other):
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
