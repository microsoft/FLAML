from .task.generic import Task
from .task.time_series import TaskTS
from ..data import TS_FORECAST


def task_factory(task_name: str) -> Task:
    if task_name in TS_FORECAST:
        return TaskTS(task_name)
    return Task(task_name)
