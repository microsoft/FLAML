import logging
import time

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import TimeSeriesSplit

# from .automl import AutoML
from flaml.automl.task import CLASSIFICATION, Task, TS_FORECASTREGRESSION
from flaml.time_series.ts_data import TimeSeriesDataset, DataTransformerTS

from flaml.automl.task import TS_FORECAST
from flaml.ml import get_val_loss
from flaml.time_series import (
    XGBoost_TS,
    XGBoostLimitDepth_TS,
    RF_TS,
    LGBM_TS,
    ExtraTrees_TS,
    Prophet,
    Orbit,
    ARIMA,
    SARIMAX,
)
from flaml.time_series.multiscale import MultiscaleModel

logger = logging.getLogger(__name__)


class TaskTS(Task):
    estimators = {
        "xgboost": XGBoost_TS,
        "xgb_limitdepth": XGBoostLimitDepth_TS,
        "rf": RF_TS,
        "lgbm": LGBM_TS,
        "extra_tree": ExtraTrees_TS,
        "prophet": Prophet,
        "orbit": Orbit,
        "arima": ARIMA,
        "sarimax": SARIMAX,
        "multiscale": MultiscaleModel,
    }

    @staticmethod
    def _validate_data(
        automl,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        eval_method,
        time_col=None,
        X_val=None,
        y_val=None,
        groups_val=None,
        groups=None,
    ):
        if label is None:
            label = "y"  # Prophet convention

        if isinstance(label, str):
            target_names = [label]
        else:
            target_names = label

        automl.task.time_col = time_col
        automl.task.target_names = target_names

        # we will cast any X to a dataframe,
        automl._df = True

        if X_train_all is not None and y_train_all is not None:
            time_col = time_col or "ds"
            validate_data_basic(X_train_all, y_train_all)
            dataframe = normalize_ts_data(
                X_train_all, target_names, time_col, y_train_all
            )

        elif dataframe is not None:
            assert label is not None, "A label or list of labels must be provided."
            assert isinstance(
                dataframe, pd.DataFrame
            ), "dataframe must be a pandas DataFrame"
            assert label in dataframe.columns, "label must a column name in dataframe"
        else:
            raise ValueError(
                "Must supply either X_train_all and y_train_all, or dataframe and label"
            )

        assert (
            dataframe.dtypes[time_col].name == "datetime64[ns]"
        ), f"For '{TS_FORECAST}' task, time_col must contain timestamp values."

        dataframe = remove_ts_duplicates(dataframe, time_col)
        X = dataframe.drop(columns=target_names)
        automl._nrow, automl._ndim = X.shape

        # transform them all together to guarantee consistency
        if X_val is not None and y_val is not None:
            validate_data_basic(X_val, y_val)
            val_df = normalize_ts_data(X_val, y_val, target_names, time_col)
            all_df = pd.concat([dataframe, val_df], axis=0)
            val_len = len(val_df)
        else:
            all_df = dataframe
            val_len = 0

        automl._transformer = DataTransformerTS(time_col, label)
        Xt, yt = automl._transformer.fit_transform(
            all_df.drop(columns=target_names), all_df[target_names]
        )
        df_t = pd.concat([Xt, yt], axis=1)

        if val_len == 0:
            df_train, df_val = df_t, None
        else:
            df_train, df_val = df_t[:-val_len], df_t[-val_len:]

        automl._X_train_all, automl._y_train_all = Xt, yt

        automl._label_transformer = automl._transformer.label_transformer

        automl._feature_names_in_ = (
            automl._X_train_all.columns.to_list()
            if hasattr(automl._X_train_all, "columns")
            else None
        )
        # NOTE: _validate_data is before kwargs is updated to fit_kwargs_by_estimator

        data = TimeSeriesDataset(
            train_data=df_train,
            time_col=time_col,
            target_names=target_names,
            test_data=df_val,
        )

        automl._state.X_val = data
        automl._state.X_train = data
        automl._state.y_train = None
        automl._state.y_val = None
        if data.test_data is not None and len(data.test_data) > 0:
            automl._state.X_train_all = data.move_validation_boundary(
                len(data.test_data)
            )
        else:
            automl._state.X_train_all = data
        automl._state.y_train_all = None

        automl._state.data_size = data.train_data.shape
        automl.data_size_full = len(data.all_data)

    @staticmethod
    def _prepare_data(
        automl,
        eval_method,
        split_ratio,
        n_splits,
        time_col=None,
    ):

        automl._state.kf = None
        automl._sample_weight_full = None

        SHUFFLE_SPLIT_TYPES = ["uniform", "stratified"]
        if automl._split_type in SHUFFLE_SPLIT_TYPES:
            raise ValueError(
                f"Split type {automl._split_type} is not valid for time series"
            )

        automl._state.groups = None
        automl._state.groups_all = None
        automl._state.groups_val = None

        ts_data = automl._state.X_val
        no_test_data = (
            ts_data is None or ts_data.test_data is None or len(ts_data.test_data) == 0
        )
        if no_test_data and eval_method == "holdout":
            # if eval_method = holdout, make holdout data
            num_samples = ts_data.train_data.shape[0]
            period = automl._state.fit_kwargs[
                "period"
            ]  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
            assert period < num_samples, f"period={period}>#examples={num_samples}"
            automl._state.X_val = ts_data.move_validation_boundary(-period)
            automl._state.X_train = automl._state.X_val

        if eval_method != "holdout":
            period = automl._state.fit_kwargs[
                "period"
            ]  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator

            ts_data = automl._state.X_train
            if period * (n_splits + 1) > ts_data.y_train.size:
                n_splits = int(automl._state.y_train.size / period - 1)
                assert n_splits >= 2, (
                    f"cross validation for forecasting period={period}"
                    f" requires input data with at least {3 * period} examples."
                )
                logger.info(f"Using nsplits={n_splits} due to data size limit.")
            automl._state.kf = TimeSeriesSplit(n_splits=n_splits, test_size=period)

    @staticmethod
    def _decide_split_type(automl, split_type):
        assert split_type in ["auto", "time"]
        automl._split_type = "time"
        assert isinstance(
            automl._state.fit_kwargs.get("period"),
            int,  # NOTE: _decide_split_type is before kwargs is updated to fit_kwargs_by_estimator
        ), f"missing a required integer 'period' for '{TS_FORECAST}' task."

    @staticmethod
    def _prepare_sample_train_data(automlstate, sample_size):
        # we take the tail, rather than the head, for compatibility with time series

        shift = sample_size - len(automlstate.X_train.train_data)
        sampled_X_train = automlstate.X_train.move_validation_boundary(shift)

        return sampled_X_train, None, None, None

    @staticmethod
    def _preprocess(automl, X):
        X = normalize_ts_data(X, automl.task.target_names, automl.task.time_col)
        return automl._preprocess(X)

    def default_estimator_list(self):
        estimator_list = super().default_estimator_list()

        # catboost is removed because it has a `name` parameter, making it incompatible with hcrystalball
        if "catboost" in estimator_list:
            estimator_list.remove("catboost")
        if self.name in TS_FORECASTREGRESSION:
            estimator_list += ["arima", "sarimax"]
            # Multiscale struggled with multivariate categorical forecast test
            # if self.train_data_size <= 365 * 10:  # at most 10 years
            #     # matrix inversion of the multiscale transform gets too expensive otherwise
            #     estimator_list = ["multiscale"]
            try:
                import prophet

                estimator_list += ["prophet"]
            except ImportError:
                pass

            # try:
            #     import orbit
            #
            #     estimator_list += ["orbit"]
            # except ImportError:
            #     pass
        return estimator_list

    def evaluate_model_CV(
        self,
        config,
        estimator,
        X_train_all,
        y_train_all,
        budget,
        kf,
        eval_metric,
        best_val_loss,
        log_training_metric=False,
        fit_kwargs={},
    ):
        start_time = time.time()
        total_val_loss = 0
        total_metric = None
        metric = None
        train_time = pred_time = 0
        valid_fold_num = total_fold_num = 0
        n = kf.get_n_splits()
        if self.name in CLASSIFICATION:
            labels = np.unique(y_train_all)
        else:
            labels = fit_kwargs.get(
                "label_list"
            )  # pass the label list on to compute the evaluation metric
        ts_data = X_train_all
        val_loss_list = []
        budget_per_train = budget / n
        ts_data = X_train_all
        for data in ts_data.cv_train_val_sets(kf.n_splits, kf.test_size):
            estimator.cleanup()
            val_loss_i, metric_i, train_time_i, pred_time_i = get_val_loss(
                config,
                estimator,
                X_train=data,
                y_train=None,
                X_val=data,
                y_val=None,
                eval_metric=eval_metric,
                labels=labels,
                budget=budget_per_train,
                log_training_metric=log_training_metric,
                fit_kwargs=fit_kwargs,
                task=self,
                weight_val=None,
                groups_val=None,
            )
            valid_fold_num += 1
            total_fold_num += 1
            total_val_loss += val_loss_i
            if log_training_metric or not isinstance(eval_metric, str):
                if isinstance(total_metric, dict):
                    total_metric = {k: total_metric[k] + v for k, v in metric_i.items()}
                elif total_metric is not None:
                    total_metric += metric_i
                else:
                    total_metric = metric_i
            train_time += train_time_i
            pred_time += pred_time_i
            if valid_fold_num == n:
                val_loss_list.append(total_val_loss / valid_fold_num)
                total_val_loss = valid_fold_num = 0
            elif time.time() - start_time >= budget:
                val_loss_list.append(total_val_loss / valid_fold_num)
                break
        val_loss = np.max(val_loss_list)
        n = total_fold_num
        if log_training_metric or not isinstance(eval_metric, str):
            if isinstance(total_metric, dict):
                metric = {k: v / n for k, v in total_metric.items()}
            else:
                metric = total_metric / n
        pred_time /= n
        return val_loss, metric, train_time, pred_time


def validate_data_basic(X_train_all, y_train_all):

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
    ), "Input data must not be empty, use None if no data"

    assert (
        X_train_all.shape[0] == y_train_all.shape[0]
    ), "# rows in X_train must match length of y_train."


def normalize_ts_data(X_train_all, target_names, time_col, y_train_all=None):
    if issparse(X_train_all):
        X_train_all = X_train_all.tocsr()

    if isinstance(X_train_all, np.ndarray) and len(X_train_all.shape) == 1:
        X_train_all = np.reshape(X_train_all, (X_train_all.size, 1))

    if isinstance(X_train_all, np.ndarray):
        X_train_all = pd.DataFrame(
            X_train_all,
            columns=[time_col] + [f"x{i}" for i in range(X_train_all.shape[1] - 1)],
        )

    if y_train_all is None:
        return X_train_all
    else:
        if isinstance(y_train_all, np.ndarray):
            # TODO: will need to revisit this when doing multivariate y
            y_train_all = pd.DataFrame(
                y_train_all.reshape(len(X_train_all), -1),
                columns=target_names,
                index=X_train_all.index,
            )
        elif isinstance(y_train_all, pd.Series):
            y_train_all = pd.DataFrame(y_train_all)
            y_train_all.index = X_train_all.index

        dataframe = pd.concat([X_train_all, y_train_all], axis=1)

        return dataframe


def remove_ts_duplicates(
    X,
    time_col,
):
    """
    Assumes the targets are included
    @param X:
    @param time_col:
    @param y:
    @return:
    """

    duplicates = X.duplicated()

    if any(duplicates):
        logger.warning(
            "Duplicate timestamp values found in timestamp column. "
            f"\n{X.loc[duplicates, X][time_col]}"
        )
        X = X.drop_duplicates()
        logger.warning("Removed duplicate rows based on all columns")
        assert (
            X[[X.columns[0]]].duplicated() is None
        ), "Duplicate timestamp values with different values for other columns."

    return X
