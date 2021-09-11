"""!
 * Copyright (c) 2020-2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
"""

import numpy as np
from scipy.sparse import vstack, issparse
import pandas as pd
from .training_log import training_log_reader

from datetime import datetime


def load_openml_dataset(
    dataset_id, data_dir=None, random_state=0, dataset_format="dataframe"
):
    """Load dataset from open ML.

    If the file is not cached locally, download it from open ML.

    Args:
        dataset_id: An integer of the dataset id in openml
        data_dir: A string of the path to store and load the data
        random_state: An integer of the random seed for splitting data
        dataset_format: A string specifying the format of returned dataset. Default is 'dataframe'.
            Can choose from ['dataframe', 'array'].
            If 'dataframe', the returned dataset will be a Pandas DataFrame.
            If 'array', the returned dataset will be a NumPy array or a SciPy sparse matrix.
    Returns:
        X_train: Training data
        X_test:  Test data
        y_train: A series or array of labels for training data
        y_test:  A series or array of labels for test data
    """
    import os
    import openml
    import pickle
    from sklearn.model_selection import train_test_split

    filename = "openml_ds" + str(dataset_id) + ".pkl"
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        print("load dataset from", filepath)
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("download dataset from openml")
        dataset = openml.datasets.get_dataset(dataset_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print("Dataset name:", dataset.name)
    X, y, *__ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format=dataset_format
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    print(
        "X_train.shape: {}, y_train.shape: {};\nX_test.shape: {}, y_test.shape: {}".format(
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
        )
    )
    return X_train, X_test, y_train, y_test


def load_openml_task(task_id, data_dir):
    """Load task from open ML.

    Use the first fold of the task.
    If the file is not cached locally, download it from open ML.

    Args:
        task_id: An integer of the task id in openml
        data_dir: A string of the path to store and load the data

    Returns:
        X_train: A dataframe of training data
        X_test:  A dataframe of test data
        y_train: A series of labels for training data
        y_test:  A series of labels for test data
    """
    import os
    import openml
    import pickle

    task = openml.tasks.get_task(task_id)
    filename = "openml_task" + str(task_id) + ".pkl"
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        print("load dataset from", filepath)
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("download dataset from openml")
        dataset = task.get_dataset()
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    X, y, _, _ = dataset.get_data(task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0,
        fold=0,
        sample=0,
    )
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    print(
        "X_train.shape: {}, y_train.shape: {},\nX_test.shape: {}, y_test.shape: {}".format(
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
        )
    )
    return X_train, X_test, y_train, y_test


def get_output_from_log(filename, time_budget):
    """Get output from log file

    Args:
        filename: A string of the log file name
        time_budget: A float of the time budget in seconds

    Returns:
        search_time_list: A list of the finished time of each logged iter
        best_error_list:
            A list of the best validation error after each logged iter
        error_list: A list of the validation error of each logged iter
        config_list:
            A list of the estimator, sample size and config of each logged iter
        logged_metric_list: A list of the logged metric of each logged iter
    """

    best_config = None
    best_learner = None
    best_val_loss = float("+inf")

    search_time_list = []
    config_list = []
    best_error_list = []
    error_list = []
    logged_metric_list = []
    best_config_list = []
    with training_log_reader(filename) as reader:
        for record in reader.records():
            time_used = record.wall_clock_time
            val_loss = record.validation_loss
            config = record.config
            learner = record.learner.split("_")[0]
            sample_size = record.sample_size
            metric = record.logged_metric

            if time_used < time_budget and np.isfinite(val_loss):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = config
                    best_learner = learner
                    best_config_list.append(best_config)
                search_time_list.append(time_used)
                best_error_list.append(best_val_loss)
                logged_metric_list.append(metric)
                error_list.append(val_loss)
                config_list.append(
                    {
                        "Current Learner": learner,
                        "Current Sample": sample_size,
                        "Current Hyper-parameters": record.config,
                        "Best Learner": best_learner,
                        "Best Hyper-parameters": best_config,
                    }
                )

    return (
        search_time_list,
        best_error_list,
        error_list,
        config_list,
        logged_metric_list,
    )


def concat(X1, X2):
    """concatenate two matrices vertically"""
    if isinstance(X1, pd.DataFrame) or isinstance(X1, pd.Series):
        df = pd.concat([X1, X2], sort=False)
        df.reset_index(drop=True, inplace=True)
        if isinstance(X1, pd.DataFrame):
            cat_columns = X1.select_dtypes(include="category").columns
            if len(cat_columns):
                df[cat_columns] = df[cat_columns].astype("category")
        return df
    if issparse(X1):
        return vstack((X1, X2))
    else:
        return np.concatenate([X1, X2])


class DataTransformer:
    """transform X, y"""

    def fit_transform(self, X, y, task):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            n = X.shape[0]
            cat_columns, num_columns, datetime_columns = [], [], []
            drop = False
            for column in X.columns:
                # sklearn\utils\validation.py needs int/float values
                if X[column].dtype.name in ("object", "category"):
                    if (
                        X[column].nunique() == 1
                        or X[column].nunique(dropna=True)
                        == n - X[column].isnull().sum()
                    ):
                        X.drop(columns=column, inplace=True)
                        drop = True
                    elif X[column].dtype.name == "category":
                        current_categories = X[column].cat.categories
                        if "__NAN__" not in current_categories:
                            X[column] = (
                                X[column]
                                .cat.add_categories("__NAN__")
                                .fillna("__NAN__")
                            )
                        cat_columns.append(column)
                    else:
                        X[column] = X[column].fillna("__NAN__")
                        cat_columns.append(column)
                else:
                    # print(X[column].dtype.name)
                    if X[column].nunique(dropna=True) < 2:
                        X.drop(columns=column, inplace=True)
                        drop = True
                    else:
                        if X[column].dtype.name == "datetime64[ns]":
                            tmp_dt = X[column].dt
                            new_columns_dict = {
                                f"year_{column}": tmp_dt.year,
                                f"month_{column}": tmp_dt.month,
                                f"day_{column}": tmp_dt.day,
                                f"hour_{column}": tmp_dt.hour,
                                f"minute_{column}": tmp_dt.minute,
                                f"second_{column}": tmp_dt.second,
                                f"dayofweek_{column}": tmp_dt.dayofweek,
                                f"dayofyear_{column}": tmp_dt.dayofyear,
                                f"quarter_{column}": tmp_dt.quarter,
                            }
                            for new_col_name in new_columns_dict.keys():
                                if (
                                    new_col_name not in X.columns
                                    and new_columns_dict.get(new_col_name).nunique(
                                        dropna=False
                                    )
                                    >= 2
                                ):
                                    X[new_col_name] = new_columns_dict.get(new_col_name)
                                    num_columns.append(new_col_name)
                            X[column] = X[column].map(datetime.toordinal)
                            datetime_columns.append(column)
                            del tmp_dt
                        else:
                            X[column] = X[column].fillna(np.nan)
                            num_columns.append(column)
            X = X[cat_columns + num_columns]
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns]
                if np.issubdtype(X_num.columns.dtype, np.integer) and (
                    drop
                    or min(X_num.columns) != 0
                    or max(X_num.columns) != X_num.shape[1] - 1
                ):
                    X_num.columns = range(X_num.shape[1])
                    drop = True
                else:
                    drop = False
                from sklearn.impute import SimpleImputer
                from sklearn.compose import ColumnTransformer

                self.transformer = ColumnTransformer(
                    [
                        (
                            "continuous",
                            SimpleImputer(missing_values=np.nan, strategy="median"),
                            X_num.columns,
                        )
                    ]
                )
                X[num_columns] = self.transformer.fit_transform(X_num)
            self._cat_columns, self._num_columns, self._datetime_columns = (
                cat_columns,
                num_columns,
                datetime_columns,
            )
            self._drop = drop

        if task in (
            "binary",
            "multi",
            "classification",
        ) or not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder

            self.label_transformer = LabelEncoder()
            y = self.label_transformer.fit_transform(y)
        else:
            self.label_transformer = None
        return X, y

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            cat_columns, num_columns, datetime_columns = (
                self._cat_columns,
                self._num_columns,
                self._datetime_columns,
            )
            if datetime_columns:
                for column in datetime_columns:
                    tmp_dt = X[column].dt
                    new_columns_dict = {
                        f"year_{column}": tmp_dt.year,
                        f"month_{column}": tmp_dt.month,
                        f"day_{column}": tmp_dt.day,
                        f"hour_{column}": tmp_dt.hour,
                        f"minute_{column}": tmp_dt.minute,
                        f"second_{column}": tmp_dt.second,
                        f"dayofweek_{column}": tmp_dt.dayofweek,
                        f"dayofyear_{column}": tmp_dt.dayofyear,
                        f"quarter_{column}": tmp_dt.quarter,
                    }
                    for new_col_name in new_columns_dict.keys():
                        if (
                            new_col_name not in X.columns
                            and new_columns_dict.get(new_col_name).nunique(dropna=False)
                            >= 2
                        ):
                            X[new_col_name] = new_columns_dict.get(new_col_name)
                    X[column] = X[column].map(datetime.toordinal)
                    del tmp_dt
            X = X[cat_columns + num_columns].copy()
            for column in cat_columns:
                if X[column].dtype.name == "object":
                    X[column] = X[column].fillna("__NAN__")
                elif X[column].dtype.name == "category":
                    current_categories = X[column].cat.categories
                    if "__NAN__" not in current_categories:
                        X[column] = (
                            X[column].cat.add_categories("__NAN__").fillna("__NAN__")
                        )
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns].fillna(np.nan)
                if self._drop:
                    X_num.columns = range(X_num.shape[1])
                X[num_columns] = self.transformer.transform(X_num)
        return X


def group_counts(groups):
    _, i, c = np.unique(groups, return_counts=True, return_index=True)
    return c[np.argsort(i)]
