from typing import Optional, Union
import numpy as np
from flaml.automl.task.generic_task import GenericTask
from flaml.automl.task.task import Task

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None


def task_factory(
    task_name: str,
    X_train: Optional[Union[np.ndarray, DataFrame]] = None,
    y_train: Optional[Union[np.ndarray, DataFrame, Series]] = None,
) -> Task:
    return GenericTask(task_name, X_train, y_train)
