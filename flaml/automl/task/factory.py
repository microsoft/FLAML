from flaml.automl.task.generic import Task
from flaml.automl.task.time_series import TaskTS
from flaml.automl.tasks import TaskParent, TS_FORECAST


def task_factory(task_name: str) -> TaskParent:
    if task_name in TS_FORECAST:
        return TaskTS(task_name)
    return Task(task_name)
