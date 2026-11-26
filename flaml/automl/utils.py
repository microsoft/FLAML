from typing import Any, Tuple, Union

import numpy as np


def len_labels(y: Any, return_labels=False) -> Union[int, Tuple[int, Any]]:
    """Count labels locally, importing Spark only for an already-distributed input."""
    if any(base.__module__.startswith("pyspark.pandas.") for base in type(y).__mro__):
        from flaml.automl.spark.utils import len_labels as spark_len_labels

        return spark_len_labels(y, return_labels=return_labels)
    labels = np.unique(y)
    if return_labels:
        return len(labels), labels
    return len(labels)


def unique_value_first_index(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unique values and indices of a pandas series or numpy array.
    The non-spark version of flaml.automl.spark.utils.unique_value_first_index"""
    label_set, first_index = np.unique(y, return_index=True)
    return label_set, first_index
