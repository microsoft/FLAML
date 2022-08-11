import logging

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle

from .automl import AutoML
from ..config import RANDOM_SEED

from ..data import TS_FORECAST, DataTransformer


logger = logging.getLogger(__name__)


class AutoMLTS(AutoML):
    """AutoML for Time Series"""

    def _validate_ts_data(
        self,
        dataframe,
        y_train_all=None,
    ):
        assert (
            dataframe[dataframe.columns[0]].dtype.name == "datetime64[ns]"
        ), f"For '{TS_FORECAST}' task, the first column must contain timestamp values."
        if y_train_all is not None:
            y_df = (
                pd.DataFrame(y_train_all)
                if isinstance(y_train_all, pd.Series)
                else pd.DataFrame(y_train_all, columns=["labels"])
            )
            dataframe = dataframe.join(y_df)
        duplicates = dataframe.duplicated()
        if any(duplicates):
            logger.warning(
                "Duplicate timestamp values found in timestamp column. "
                f"\n{dataframe.loc[duplicates, dataframe][dataframe.columns[0]]}"
            )
            dataframe = dataframe.drop_duplicates()
            logger.warning("Removed duplicate rows based on all columns")
            assert (
                dataframe[[dataframe.columns[0]]].duplicated() is None
            ), "Duplicate timestamp values with different values for other columns."
        ts_series = pd.to_datetime(dataframe[dataframe.columns[0]])
        inferred_freq = pd.infer_freq(ts_series)
        if inferred_freq is None:
            logger.warning(
                "Missing timestamps detected. To avoid error with estimators, set estimator list to ['prophet']. "
            )
        if y_train_all is not None:
            return dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
        return dataframe

    def _validate_data(
        self,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        X_val=None,
        y_val=None,
        groups_val=None,
        groups=None,
    ):
        if X_train_all is not None and y_train_all is not None:
            assert (
                isinstance(X_train_all, np.ndarray)
                or issparse(X_train_all)
                or isinstance(X_train_all, pd.DataFrame)
            ), (
                "X_train_all must be a numpy array, a pandas dataframe, "
                "or Scipy sparse matrix."
            )
            assert isinstance(y_train_all, np.ndarray) or isinstance(
                y_train_all, pd.Series
            ), "y_train_all must be a numpy array or a pandas series."
            assert (
                X_train_all.size != 0 and y_train_all.size != 0
            ), "Input data must not be empty."
            if isinstance(X_train_all, np.ndarray) and len(X_train_all.shape) == 1:
                X_train_all = np.reshape(X_train_all, (X_train_all.size, 1))
            if isinstance(y_train_all, np.ndarray):
                y_train_all = y_train_all.flatten()
            assert (
                X_train_all.shape[0] == y_train_all.shape[0]
            ), "# rows in X_train must match length of y_train."
            self._df = isinstance(X_train_all, pd.DataFrame)
            self._nrow, self._ndim = X_train_all.shape
            X_train_all = pd.DataFrame(X_train_all)
            X_train_all, y_train_all = self._validate_ts_data(X_train_all, y_train_all)
            X, y = X_train_all, y_train_all

        elif dataframe is not None and label is not None:
            assert isinstance(
                dataframe, pd.DataFrame
            ), "dataframe must be a pandas DataFrame"
            assert label in dataframe.columns, "label must a column name in dataframe"
            self._df = True
            dataframe = self._validate_ts_data(dataframe)
            X = dataframe.drop(columns=label)
            self._nrow, self._ndim = X.shape
            y = dataframe[label]

        if issparse(X_train_all):
            self._transformer = self._label_transformer = False
            self._X_train_all, self._y_train_all = X, y
        else:
            self._transformer = DataTransformer()

            self._X_train_all, self._y_train_all = self._transformer.fit_transform(
                X, y, self._state.task
            )
            self._label_transformer = self._transformer.label_transformer
            self._feature_names_in_ = (
                self._X_train_all.columns.to_list()
                if hasattr(self._X_train_all, "columns")
                else None
            )

        self._sample_weight_full = self._state.fit_kwargs.get(
            "sample_weight"
        )  # NOTE: _validate_data is before kwargs is updated to fit_kwargs_by_estimator
        if X_val is not None and y_val is not None:
            assert (
                isinstance(X_val, np.ndarray)
                or issparse(X_val)
                or isinstance(X_val, pd.DataFrame)
            ), (
                "X_val must be None, a numpy array, a pandas dataframe, "
                "or Scipy sparse matrix."
            )
            assert isinstance(y_val, np.ndarray) or isinstance(
                y_val, pd.Series
            ), "y_val must be None, a numpy array or a pandas series."
            assert X_val.size != 0 and y_val.size != 0, (
                "Validation data are expected to be nonempty. "
                "Use None for X_val and y_val if no validation data."
            )
            if isinstance(y_val, np.ndarray):
                y_val = y_val.flatten()
            assert (
                X_val.shape[0] == y_val.shape[0]
            ), "# rows in X_val must match length of y_val."
            if self._transformer:
                self._state.X_val = self._transformer.transform(X_val)
            else:
                self._state.X_val = X_val
            # If it's NLG_TASKS, y_val is a pandas series containing the output sequence tokens,
            # so we cannot use label_transformer.transform to process it
            if self._label_transformer:
                self._state.y_val = self._label_transformer.transform(y_val)
            else:
                self._state.y_val = y_val
        else:
            self._state.X_val = self._state.y_val = None

        if groups is not None and len(groups) != self._nrow:
            # groups is given as group counts
            self._state.groups = np.concatenate([[i] * c for i, c in enumerate(groups)])
            assert (
                len(self._state.groups) == self._nrow
            ), "the sum of group counts must match the number of examples"
            self._state.groups_val = (
                np.concatenate([[i] * c for i, c in enumerate(groups_val)])
                if groups_val is not None
                else None
            )
        else:
            self._state.groups_val = groups_val
            self._state.groups = groups

    def _prepare_data(self, eval_method, split_ratio, n_splits):
        X_val, y_val = self._state.X_val, self._state.y_val
        if issparse(X_val):
            X_val = X_val.tocsr()
        X_train_all, y_train_all = self._X_train_all, self._y_train_all
        if issparse(X_train_all):
            X_train_all = X_train_all.tocsr()

        SHUFFLE_SPLIT_TYPES = ["uniform", "stratified"]
        if self._split_type in SHUFFLE_SPLIT_TYPES:
            if self._sample_weight_full is not None:
                X_train_all, y_train_all, self._state.sample_weight_all = shuffle(
                    X_train_all,
                    y_train_all,
                    self._sample_weight_full,
                    random_state=RANDOM_SEED,
                )
                self._state.fit_kwargs[
                    "sample_weight"
                ] = (
                    self._state.sample_weight_all
                )  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
            else:
                X_train_all, y_train_all = shuffle(
                    X_train_all, y_train_all, random_state=RANDOM_SEED
                )
            if self._df:
                X_train_all.reset_index(drop=True, inplace=True)
                if isinstance(y_train_all, pd.Series):
                    y_train_all.reset_index(drop=True, inplace=True)

        X_train, y_train = X_train_all, y_train_all
        self._state.groups_all = self._state.groups
        if X_val is None and eval_method == "holdout":
            # if eval_method = holdout, make holdout data
            num_samples = X_train_all.shape[0]
            period = self._state.fit_kwargs[
                "period"
            ]  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
            assert period < num_samples, f"period={period}>#examples={num_samples}"
            split_idx = num_samples - period
            X_train = X_train_all[:split_idx]
            y_train = y_train_all[:split_idx]
            X_val = X_train_all[split_idx:]
            y_val = y_train_all[split_idx:]
        self._state.data_size = X_train.shape
        self.data_size_full = len(y_train_all)
        self._state.X_train, self._state.y_train = X_train, y_train
        self._state.X_val, self._state.y_val = X_val, y_val
        self._state.X_train_all = X_train_all
        self._state.y_train_all = y_train_all
        period = self._state.fit_kwargs[
            "period"
        ]  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
        if period * (n_splits + 1) > y_train_all.size:
            n_splits = int(y_train_all.size / period - 1)
            assert n_splits >= 2, (
                f"cross validation for forecasting period={period}"
                f" requires input data with at least {3 * period} examples."
            )
            logger.info(f"Using nsplits={n_splits} due to data size limit.")
        self._state.kf = TimeSeriesSplit(n_splits=n_splits, test_size=period)

    def _decide_split_type(self, split_type):
        assert split_type in ["auto", "time"]
        self._split_type = "time"
        assert isinstance(
            self._state.fit_kwargs.get("period"),
            int,  # NOTE: _decide_split_type is before kwargs is updated to fit_kwargs_by_estimator
        ), f"missing a required integer 'period' for '{TS_FORECAST}' task."
