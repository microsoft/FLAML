# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
import os
from datetime import datetime
from typing import TYPE_CHECKING, Union

import numpy as np

from flaml.automl.spark import DataFrame, Series, pd, ps, psDataFrame, psSeries
from flaml.automl.training_log import training_log_reader

try:
    from scipy.sparse import issparse, vstack
except ImportError:
    pass

if TYPE_CHECKING:
    from flaml.automl.task import Task

TS_TIMESTAMP_COL = "ds"
TS_VALUE_COL = "y"


def load_openml_dataset(dataset_id, data_dir=None, random_state=0, dataset_format="dataframe"):
    """Load dataset from open ML.

    If the file is not cached locally, download it from open ML.

    Args:
        dataset_id: An integer of the dataset id in openml.
        data_dir: A string of the path to store and load the data.
        random_state: An integer of the random seed for splitting data.
        dataset_format: A string specifying the format of returned dataset. Default is 'dataframe'.
            Can choose from ['dataframe', 'array'].
            If 'dataframe', the returned dataset will be a Pandas DataFrame.
            If 'array', the returned dataset will be a NumPy array or a SciPy sparse matrix.

    Returns:
        X_train: Training data.
        X_test:  Test data.
        y_train: A series or array of labels for training data.
        y_test:  A series or array of labels for test data.
    """
    import pickle

    import openml
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
    try:
        X, y, *__ = dataset.get_data(target=dataset.default_target_attribute, dataset_format=dataset_format)
    except ValueError:
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(data_id=dataset_id, return_X_y=True)
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
        task_id: An integer of the task id in openml.
        data_dir: A string of the path to store and load the data.

    Returns:
        X_train: A dataframe of training data.
        X_test:  A dataframe of test data.
        y_train: A series of labels for training data.
        y_test:  A series of labels for test data.
    """
    import pickle

    import openml

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
    """Get output from log file.

    Args:
        filename: A string of the log file name.
        time_budget: A float of the time budget in seconds.

    Returns:
        search_time_list: A list of the finished time of each logged iter.
        best_error_list: A list of the best validation error after each logged iter.
        error_list: A list of the validation error of each logged iter.
        config_list: A list of the estimator, sample size and config of each logged iter.
        logged_metric_list: A list of the logged metric of each logged iter.
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
    """concatenate two matrices vertically."""
    if type(X1) != type(X2):
        if isinstance(X2, (psDataFrame, psSeries)):
            X1 = ps.from_pandas(pd.DataFrame(X1))
        elif isinstance(X1, (psDataFrame, psSeries)):
            X2 = ps.from_pandas(pd.DataFrame(X2))
        else:
            X1 = pd.DataFrame(X1)
            X2 = pd.DataFrame(X2)

    if isinstance(X1, (DataFrame, Series)):
        df = pd.concat([X1, X2], sort=False)
        df.reset_index(drop=True, inplace=True)
        if isinstance(X1, DataFrame):
            cat_columns = X1.select_dtypes(include="category").columns
            if len(cat_columns):
                df[cat_columns] = df[cat_columns].astype("category")
        return df
    if isinstance(X1, (psDataFrame, psSeries)):
        df = ps.concat([X1, X2], ignore_index=True)
        if isinstance(X1, psDataFrame):
            cat_columns = X1.select_dtypes(include="category").columns.values.tolist()
            if len(cat_columns):
                df[cat_columns] = df[cat_columns].astype("category")
        return df
    if issparse(X1):
        return vstack((X1, X2))
    else:
        return np.concatenate([X1, X2])


def add_time_idx_col(X):
    unique_dates = X[TS_TIMESTAMP_COL].drop_duplicates().sort_values(ascending=True)
    # assume no missing timestamps
    freq = pd.infer_freq(unique_dates)
    if freq == "MS":
        X["time_idx"] = X[TS_TIMESTAMP_COL].dt.year * 12 + X[TS_TIMESTAMP_COL].dt.month
    elif freq == "Y":
        X["time_idx"] = X[TS_TIMESTAMP_COL].dt.year
    else:
        # using time frequency to generate all time stamps and then indexing for time_idx
        # full_range = pd.date_range(X[TS_TIMESTAMP_COL].min(), X[TS_TIMESTAMP_COL].max(), freq=freq).to_list()
        # X["time_idx"] = [full_range.index(time) for time in X[TS_TIMESTAMP_COL]]
        # taking minimum difference in timestamp
        timestamps = unique_dates.view("int64")
        freq = int(timestamps.diff().mode())
        X["time_idx"] = timestamps - timestamps.min() / freq
        X["time_idx"] = X["time_idx"].astype("int")
    return X


class DataTransformer:
    """Transform input training data."""

    def fit_transform(self, X: Union[DataFrame, np.ndarray], y, task: Union[str, "Task"]):
        """Fit transformer and process the input training data according to the task type.

        Args:
            X: A numpy array or a pandas dataframe of training data.
            y: A numpy array or a pandas series of labels.
            task: An instance of type Task, or a str such as 'classification', 'regression'.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
            y: Processed numpy array or pandas series of labels.
        """
        if isinstance(task, str):
            from flaml.automl.task.factory import task_factory

            task = task_factory(task, X, y)

        if task.is_nlp():
            # if the mode is NLP, check the type of input, each column must be either string or
            # ids (input ids, token type id, attention mask, etc.)
            str_columns = []
            for column in X.columns:
                if isinstance(X[column].iloc[0], str):
                    str_columns.append(column)
            if len(str_columns) > 0:
                X[str_columns] = X[str_columns].astype("string")
            self._str_columns = str_columns
        elif isinstance(X, DataFrame):
            X = X.copy()
            n = X.shape[0]
            cat_columns, num_columns, datetime_columns = [], [], []
            drop = False
            if task.is_ts_forecast():
                X = X.rename(columns={X.columns[0]: TS_TIMESTAMP_COL})
                if task.is_ts_forecastpanel():
                    if "time_idx" not in X:
                        X = add_time_idx_col(X)
                ds_col = X.pop(TS_TIMESTAMP_COL)
                if isinstance(y, Series):
                    y = y.rename(TS_VALUE_COL)
            for column in X.columns:
                # sklearn\utils\validation.py needs int/float values
                if X[column].dtype.name in ("object", "category", "string"):
                    if X[column].nunique() == 1 or X[column].nunique(dropna=True) == n - X[column].isnull().sum():
                        X.drop(columns=column, inplace=True)
                        drop = True
                    elif X[column].dtype.name == "category":
                        current_categories = X[column].cat.categories
                        if "__NAN__" not in current_categories:
                            X[column] = X[column].cat.add_categories("__NAN__").fillna("__NAN__")
                        cat_columns.append(column)
                    else:
                        X[column] = X[column].fillna("__NAN__")
                        cat_columns.append(column)
                elif X[column].nunique(dropna=True) < 2:
                    X.drop(columns=column, inplace=True)
                    drop = True
                else:  # datetime or numeric
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
                        for key, value in new_columns_dict.items():
                            if key not in X.columns and value.nunique(dropna=False) >= 2:
                                X[key] = value
                                num_columns.append(key)
                        X[column] = X[column].map(datetime.toordinal)
                        datetime_columns.append(column)
                        del tmp_dt
                    X[column] = X[column].fillna(np.nan)
                    num_columns.append(column)
            X = X[cat_columns + num_columns]
            if task.is_ts_forecast():
                X.insert(0, TS_TIMESTAMP_COL, ds_col)
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns]
                if np.issubdtype(X_num.columns.dtype, np.integer) and (
                    drop or min(X_num.columns) != 0 or max(X_num.columns) != X_num.shape[1] - 1
                ):
                    X_num.columns = range(X_num.shape[1])
                    drop = True
                else:
                    drop = False
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer

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
        if task.is_classification() or not pd.api.types.is_numeric_dtype(y) and not task.is_nlg():
            if not task.is_token_classification():
                from sklearn.preprocessing import LabelEncoder

                self.label_transformer = LabelEncoder()
            else:
                from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

                self.label_transformer = LabelEncoderforTokenClassification()
            y = self.label_transformer.fit_transform(y)
        else:
            self.label_transformer = None
        self._task = task
        return X, y

    def transform(self, X: Union[DataFrame, np.array]):
        """Process data using fit transformer.

        Args:
            X: A numpy array or a pandas dataframe of training data.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
        """
        X = X.copy()

        if self._task.is_nlp():
            # if the mode is NLP, check the type of input, each column must be either string or
            # ids (input ids, token type id, attention mask, etc.)
            if len(self._str_columns) > 0:
                X[self._str_columns] = X[self._str_columns].astype("string")
        elif isinstance(X, DataFrame):
            cat_columns, num_columns, datetime_columns = (
                self._cat_columns,
                self._num_columns,
                self._datetime_columns,
            )
            if self._task.is_ts_forecast():
                X = X.rename(columns={X.columns[0]: TS_TIMESTAMP_COL})
                ds_col = X.pop(TS_TIMESTAMP_COL)
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
                for new_col_name, new_col_value in new_columns_dict.items():
                    if new_col_name not in X.columns and new_col_name in num_columns:
                        X[new_col_name] = new_col_value
                X[column] = X[column].map(datetime.toordinal)
                del tmp_dt
            X = X[cat_columns + num_columns].copy()
            if self._task.is_ts_forecast():
                X.insert(0, TS_TIMESTAMP_COL, ds_col)
            for column in cat_columns:
                if X[column].dtype.name == "object":
                    X[column] = X[column].fillna("__NAN__")
                elif X[column].dtype.name == "category":
                    current_categories = X[column].cat.categories
                    if "__NAN__" not in current_categories:
                        X[column] = X[column].cat.add_categories("__NAN__").fillna("__NAN__")
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
