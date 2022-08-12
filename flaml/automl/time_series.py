import logging
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle

from .automl import AutoML
from .ts_data import TimeSeriesDataset
from ..config import RANDOM_SEED

from ..data import TS_FORECAST, DataTransformerTS


logger = logging.getLogger(__name__)


class AutoMLTS(AutoML):
    """AutoML for Time Series"""

    def _validate_ts_data(
        self,
        dataframe,
        time_col,
        y_train_all=None,
    ):
        assert (
            dataframe.dtypes[time_col].name == "datetime64[ns]"
        ), f"For '{TS_FORECAST}' task, time_col must contain timestamp values."
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
        if y_train_all is not None:
            return dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
        return dataframe

    def _validate_data(
        self,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        time_col=None,
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
            assert (
                isinstance(y_train_all, np.ndarray)
                or isinstance(y_train_all, pd.Series)
                or isinstance(y_train_all, pd.DataFrame)
            ), "y_train_all must be a numpy array or a pandas series or DataFrame."
            assert (
                X_train_all.size != 0 and y_train_all.size != 0
            ), "Input data must not be empty."
            if isinstance(X_train_all, np.ndarray) and len(X_train_all.shape) == 1:
                X_train_all = np.reshape(X_train_all, (X_train_all.size, 1))
            assert (
                X_train_all.shape[0] == y_train_all.shape[0]
            ), "# rows in X_train must match length of y_train."

            if isinstance(X_train_all, np.ndarray):
                X_train_all = pd.DataFrame(
                    X_train_all,
                    columns=["ds"] + [f"x{i}" for i in range(X_train_all.shape[1] - 1)],
                )
                time_col = "ds"

            if isinstance(y_train_all, np.ndarray):
                # TODO: will need to revisit this when doing multivariate y
                y_train_all = pd.Series(y_train_all.reshape(-1), name="y")
                label = "y"

            self._df = isinstance(X_train_all, pd.DataFrame)
            self._nrow, self._ndim = X_train_all.shape
            X_train_all, y_train_all = self._validate_ts_data(
                X_train_all, time_col, y_train_all
            )
            X, y = X_train_all, y_train_all

        elif dataframe is not None:
            assert label is not None, "A label or list of labels must be provided."
            assert isinstance(
                dataframe, pd.DataFrame
            ), "dataframe must be a pandas DataFrame"
            assert label in dataframe.columns, "label must a column name in dataframe"
            self._df = True
            dataframe = self._validate_ts_data(dataframe, time_col)
            X = dataframe.drop(columns=label)
            self._nrow, self._ndim = X.shape
            y = dataframe[label]

        if issparse(X_train_all):
            self._transformer = self._label_transformer = False
            self._X_train_all, self._y_train_all = X, y
        else:
            self._transformer = DataTransformerTS(time_col, label)

            self._X_train_all, self._y_train_all = self._transformer.fit_transform(X, y)
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
            assert (
                isinstance(y_val, np.ndarray)
                or isinstance(y_val, pd.Series)
                or isinstance(y_val, pd.DataFrame)
            ), "y_val must be None, a numpy array or a pandas series."
            assert X_val.size != 0 and y_val.size != 0, (
                "Validation data are expected to be nonempty. "
                "Use None for X_val and y_val if no validation data."
            )
            assert (
                X_val.shape[0] == y_val.shape[0]
            ), "# rows in X_val must match length of y_val."
            if self._transformer:
                self._state.X_val = self._transformer.transform(X_val)
            else:
                self._state.X_val = X_val
            self._state.y_val = self._label_transformer.transform(y_val)
        else:
            self._state.X_val = self._state.y_val = None

    def _prepare_data(
        self,
        eval_method,
        split_ratio,
        n_splits,
        time_col=None,
    ):
        if time_col is None:
            time_col = "ds"
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
        self._state.groups = None
        self._state.groups_all = None
        self._state.groups_val = None
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
            dataframe_val = X_val.merge(y_val, left_index=True, right_index=True)
        else:
            dataframe_val = None

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

        dataframe = X_train.merge(y_train, left_index=True, right_index=True)

        frequency = pd.infer_freq(dataframe[time_col])
        target_names = list(pd.DataFrame(y_train).columns)
        assert (
            frequency is not None
        ), "Only time series of regular frequency are currently supported."
        time_varying_known_reals = list(
            X_train.select_dtypes(include=["floating"]).columns
        )
        time_varying_known_categoricals = list(
            set(X_train.columns)
            - set(time_varying_known_reals)
            - set(pd.DataFrame(y_train).columns)
            - {time_col}
        )
        data = TimeSeriesDataset(
            train_data=dataframe,
            time_idx="index",
            time_col=time_col,
            target_names=target_names,
            frequency=frequency,
            test_data=dataframe_val,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
        )

        self._state.X_val = data
        self._state.X_train = data
        self._state.y_train = None
        self._state.y_val = None
        if data.test_data is not None:
            self._state.X_train_all = data.move_validation_boundary(len(data.test_data))
        else:
            self._state.X_train_all = data
        self._state.y_train_all = None

    def _decide_split_type(self, split_type):
        assert split_type in ["auto", "time"]
        self._split_type = "time"
        assert isinstance(
            self._state.fit_kwargs.get("period"),
            int,  # NOTE: _decide_split_type is before kwargs is updated to fit_kwargs_by_estimator
        ), f"missing a required integer 'period' for '{TS_FORECAST}' task."
