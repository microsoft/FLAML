from .automl import AutoML as AutoMLGeneric
from .time_series import AutoMLTS
from ..data import TS_FORECAST, CLASSIFICATION, NLP_TASKS
from ..model import (
    XGBoostSklearnEstimator,
    XGBoostLimitDepthEstimator,
    RandomForestEstimator,
    LGBMEstimator,
    LRL1Classifier,
    LRL2Classifier,
    CatBoostEstimator,
    ExtraTreesEstimator,
    KNeighborsEstimator,
    TransformersEstimator,
    TransformersEstimatorModelSelection,
)
from ..ts_model import (
    Prophet,
    Orbit,
    ARIMA,
    SARIMAX,
    LGBM_TS,
    XGBoost_TS,
    RF_TS,
    ExtraTrees_TS,
    XGBoostLimitDepth_TS,
)


class Task:
    AUTOML_CLASS = AutoMLGeneric

    estimators = {
        "xgboost": XGBoostSklearnEstimator,
        "xgb_limitdepth": XGBoostLimitDepthEstimator,
        "rf": RandomForestEstimator,
        "lgbm": LGBMEstimator,
        "lrl1": LRL1Classifier,
        "lrl2": LRL2Classifier,
        "catboost": CatBoostEstimator,
        "extra_tree": ExtraTreesEstimator,
        "kneighbor": KNeighborsEstimator,
        "transformer": TransformersEstimator,
        "transformer_ms": TransformersEstimatorModelSelection,
    }

    def __init__(self, task_name):
        self.name = task_name

    def is_ts_forecast(self):
        return self.name in TS_FORECAST

    def is_nlp(self):
        return self.name in NLP_TASKS

    def is_classification(self):
        return self.name in CLASSIFICATION

    # For backward compatibility with all the string-valued "task" variables
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
        return estimator_class


class TaskTS(Task):
    AUTOML_CLASS = AutoMLTS

    estimators = {
        "xgboost": XGBoost_TS,
        "xgb_limitdepth": XGBoostLimitDepth_TS,
        "rf": RF_TS,
        "lgbm": LGBM_TS,
        "extra_tree": ExtraTrees_TS,
        "prophet": Prophet,
        "orbit": Orbit,
        "arima": ARIMA,
        "sarimax": SARIMAX,
    }


def task_factory(task_name: str) -> Task:
    if task_name in TS_FORECAST:
        return TaskTS(task_name)
    return Task(task_name)
