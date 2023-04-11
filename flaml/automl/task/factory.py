from typing import Optional, Union

import numpy as np
import pandas as pd


from flaml.automl.task.task import Task, TS_FORECAST


def task_factory(
    task_name: str,
    X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_train: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
) -> Task:
    from flaml.automl.task.generic_task import GenericTask
    from flaml.automl.task.time_series_task import TimeSeriesTask

    if task_name in TS_FORECAST:
        return TimeSeriesTask(task_name, X_train, y_train)
    else:
        return GenericTask(task_name, X_train, y_train)