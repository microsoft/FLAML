from flaml.automl.tasks.generic import GenericTask
from flaml.time_series.time_series_task import TaskTS
from flaml.automl.tasks.task import Task, TS_FORECAST


def task_factory(task_name: str) -> Task:
    if task_name in TS_FORECAST:
        return TaskTS(task_name)
    return GenericTask(task_name)
