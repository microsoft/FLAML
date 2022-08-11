from .automl import AutoML as AutoMLGeneric
from .time_series import AutoMLTS
from ..data import TS_FORECASTREGRESSION


class Task:
    AUTOML_CLASS = AutoMLGeneric

    def __init__(self, task_name):
        self.task_name = task_name


class TaskTS(Task):
    AUTOML_CLASS = AutoMLTS


def task_factory(task_name: str) -> Task:
    if task_name in TS_FORECASTREGRESSION:
        return TaskTS(task_name)
    return Task(task_name)
