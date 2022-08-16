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


class TaskParent:
    def __init__(self, task_name):
        self.name = task_name

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
