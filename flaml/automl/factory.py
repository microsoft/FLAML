from typing import Union, Optional

import numpy as np
import pandas as pd

from flaml.automl.generic_task import GenericTask
from flaml.time_series.ts_task import TSTask
from flaml.automl.task import Task, TS_FORECAST


def task_factory(
    task_name: str,
    X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_train: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
) -> Task:
    if task_name in TS_FORECAST:
        return TSTask(task_name, X_train, y_train)
    return GenericTask(task_name, X_train, y_train)
