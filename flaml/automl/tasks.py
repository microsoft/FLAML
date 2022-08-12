from .automl import AutoML as AutoMLGeneric
from .time_series import AutoMLTS
from ..data import TS_FORECAST


class Task:
    AUTOML_CLASS = AutoMLGeneric

    def __init__(self, task_name):
        self.name = task_name


class TaskTS(Task):
    AUTOML_CLASS = AutoMLTS


def task_factory(task_name: str) -> Task:
    if task_name in TS_FORECAST:
        return TaskTS(task_name)
    return Task(task_name)
