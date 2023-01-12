from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

try:
    import ray
except:
    ray = None

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
        X_train: Optional[Union[np.ndarray, pd.DataFrame]],
        y_train: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]],
    ):
        self.name = task_name
        if hasattr(ray, "ObjectRef") and isinstance(X_train, ray.ObjectRef):
            X_train = ray.get(X_train)

    def __str__(self):
        return self.name

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

    @abstractmethod
    def validate_data(
        self,
        automl,
        state,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        X_val=None,
        y_val=None,
        groups_val=None,
        groups=None,
    ):
        pass

    @abstractmethod
    def prepare_data(
        self,
        state,
        X_train_all,
        y_train_all,
        auto_argument,
        eval_method,
        split_type,
        split_ratio,
        n_splits,
        data_is_df,
        sample_weight_full,
    ):
        pass

    @abstractmethod
    def decide_split_type(
        self, split_type, y_train_all, fit_kwargs, groups=None
    ) -> str:
        pass

    # TODO Remove private marker
    @abstractmethod
    def preprocess(self, X, transformer=None):
        pass

    @abstractmethod
    def default_estimator_list(self, estimator_list: List[str]) -> List[str]:
        pass

    @abstractmethod
    def default_metric(self, metric: str) -> str:
        pass

    def is_ts_forecast(self):
        return self.name in TS_FORECAST

    def is_ts_forecastpanel(self):
        return self.name == TS_FORECASTPANEL

    def is_ts_forecastregression(self):
        return self.name in TS_FORECASTREGRESSION

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
        return self.name == SEQREGRESSION

    def is_seq_classification(self):
        return self.name == SEQCLASSIFICATION

    def is_token_classification(self):
        return self.name == TOKENCLASSIFICATION

    def is_summarization(self):
        return self.name == SUMMARIZATION

    def is_multiclass(self):
        return "multiclass" in self.name

    def is_regression(self):
        return self.name in REGRESSION

    def __eq__(self, other):
        """For backward compatibility with all the string comparisons to task"""
        return self.name == other

    def estimator_class_from_str(self, estimator_name: str):
        if estimator_name in self.estimators:
            return self.estimators[estimator_name]
        else:
            raise ValueError(
                f"{estimator_name} is not a built-in learner for this task type, "
                f"only {list(self.estimators.keys())} are supported."
                "Please use AutoML.add_learner() to add a customized learner."
            )
