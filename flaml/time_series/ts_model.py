import time
import logging
import os
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame, Series, to_datetime

from flaml import tune

# This may be needed to get PyStan to run, needed for Orbit
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd


from flaml.model import (
    suppress_stdout_stderr,
    SKLearnEstimator,
    logger,
    LGBMEstimator,
    XGBoostSklearnEstimator,
    RandomForestEstimator,
    ExtraTreesEstimator,
    XGBoostLimitDepthEstimator,
)
from flaml.data import TS_TIMESTAMP_COL, TS_VALUE_COL
from flaml.time_series.ts_data import TimeSeriesDataset, enrich, create_forward_frame
from flaml.automl.task import Task


class TimeSeriesEstimator(SKLearnEstimator):
    def __init__(self, task="ts_forecast", n_jobs=1, **params):
        super().__init__(task, **params)
        self.time_col: Optional[str] = None
        self.target_names: Optional[Union[str, List[str]]] = None
        self.frequency: Optional[str] = None
        self.end_date: Optional[datetime] = None

    @classmethod
    def search_space(cls, data: TimeSeriesDataset, task: Task, pred_horizon: int):
        space = cls._search_space(data=data, task=task, pred_horizon=pred_horizon)
        space.update(
            {
                "monthly_fourier_degree": {
                    "domain": tune.randint(lower=0, upper=8),
                    "init_value": 2,
                    "low_cost_init_value": 1,
                }
            }
        )
        return space

    @classmethod
    def top_level_params(cls):
        return ["monthly_fourier_degree"]

    def _join(self, X_train, y_train):
        assert TS_TIMESTAMP_COL in X_train, (
            "Dataframe for training ts_forecast model must have column"
            f' "{TS_TIMESTAMP_COL}" with the dates in X_train.'
        )
        y_train = DataFrame(y_train, columns=[TS_VALUE_COL])
        train_df = X_train.join(y_train)
        return train_df

    def fit(self, X_train: TimeSeriesDataset, y_train=None, budget=None, **kwargs):
        # TODO purge y_train
        self.time_col = X_train.time_col
        self.target_names = X_train.target_names
        self.X_train = X_train
        self.frequency = self.X_train.frequency
        self.end_date = self.X_train.end_date

    def score(self, X_val: DataFrame, y_val: Series, **kwargs):
        from sklearn.metrics import r2_score
        from ..ml import metric_loss_score

        y_pred = self.predict(X_val, **kwargs)
        if isinstance(X_val, TimeSeriesDataset):
            y_val = X_val.test_data[X_val.target_names[0]]
        self._metric = kwargs.get("metric", None)
        if self._metric:
            return metric_loss_score(self._metric, y_pred, y_val)
        else:
            return r2_score(y_pred, y_val)


class Orbit(TimeSeriesEstimator):
    def fit(self, X_train: TimeSeriesDataset, y_train=None, budget=None, **kwargs):
        from orbit.models import DLT

        # y_train is ignored, just need it for signature compatibility with other classes
        super().fit(X_train, y_train, budget=budget, **kwargs)
        current_time = time.time()
        self.logger = logging.getLogger("orbit").setLevel(logging.WARNING)

        model_class = self.params.get("model_class", DLT)
        self._model = model_class(
            response_col=X_train.target_names[0],
            date_col=X_train.time_col,
            regressor_col=X_train.regressors,
            # TODO: infer seasonality from frequency
            **self.params,
        )

        with suppress_stdout_stderr():
            self._model.fit(df=X_train.train_data.copy())

        train_time = time.time() - current_time
        return train_time

    def predict(self, X: Union[TimeSeriesDataset, pd.DataFrame], **kwargs):
        if isinstance(X, int):
            X = create_forward_frame(
                self.frequency,
                X,
                self.end_date,
                self.time_col,
            )

        elif isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data[[self.time_col] + X.regressors]

        if self._model is not None:

            forecast = self._model.predict(X, **kwargs)
            out = (
                pd.DataFrame(
                    forecast[
                        [
                            self.time_col,
                            "prediction",
                            "prediction_5",
                            "prediction_95",
                        ]
                    ]
                )
                .reset_index(drop=True)
                .rename(
                    columns={
                        "prediction": self.target_names[0],
                    }
                )
            )

            return out
        else:
            self.logger.warning(
                "Estimator is not fit yet. Please run fit() before predict()."
            )
            return None

    @classmethod
    def _search_space(cls, **params):
        # TODO: fill in a proper search space
        space = {}
        return space


class Prophet(TimeSeriesEstimator):
    """The class for tuning Prophet."""

    @classmethod
    def _search_space(cls, **params):
        space = {
            "changepoint_prior_scale": {
                "domain": tune.loguniform(lower=0.001, upper=0.05),
                "init_value": 0.05,
                "low_cost_init_value": 0.001,
            },
            "seasonality_prior_scale": {
                "domain": tune.loguniform(lower=0.01, upper=10),
                "init_value": 10,
            },
            "holidays_prior_scale": {
                "domain": tune.loguniform(lower=0.01, upper=10),
                "init_value": 10,
            },
            "seasonality_mode": {
                "domain": tune.choice(["additive", "multiplicative"]),
                "init_value": "multiplicative",
            },
        }
        return space

    def fit(self, X_train, y_train=None, budget=None, **kwargs):
        from prophet import Prophet

        X_train = enrich(X_train, self.params["monthly_fourier_degree"], self.time_col)
        super().fit(X_train, y_train, budget=budget, **kwargs)

        current_time = time.time()

        if isinstance(X_train, TimeSeriesDataset):
            data = X_train
            target_col = data.target_names[0]
            time_col = data.time_col
            regressors = data.regressors
            # this class only supports univariate regression
            train_df = data.train_data[regressors + [target_col, time_col]]
            train_df = train_df.rename(columns={target_col: "y", time_col: "ds"})
        else:
            train_df = self._join(X_train, y_train)

            regressors = list(train_df.columns)
            regressors.remove(TS_TIMESTAMP_COL)
            regressors.remove(TS_VALUE_COL)

        train_df = self._preprocess(train_df)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        nice_params = {
            k: v for k, v in self.params.items() if k in self._search_space()
        }
        model = Prophet(**nice_params)
        for regressor in regressors:
            model.add_regressor(regressor)
        with suppress_stdout_stderr():
            model.fit(train_df)
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def predict(self, X, **kwargs):
        X = enrich(
            X,
            self.params["monthly_fourier_degree"],
            self.time_col,
            frequency=self.frequency,
            test_end_date=self.end_date,
        )
        if isinstance(X, int):
            raise ValueError(
                "predict() with steps is only supported for arima/sarimax."
                " For Prophet, pass a dataframe with the first column containing"
                " the timestamp values."
            )

        if isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data[data.regressors + [data.time_col]].rename(
                columns={data.time_col: "ds"}
            )

        if self._model is not None:
            X = self._preprocess(X)
            forecast = self._model.predict(X, **kwargs)
            out = forecast["yhat"]
            out.name = self.target_names[0]
            return out

        else:
            logger.warning(
                "Estimator is not fit yet. Please run fit() before predict()."
            )
            return np.ones(X.shape[0])


class ARIMA(TimeSeriesEstimator):
    """The class for tuning ARIMA."""

    @classmethod
    def _search_space(cls, data: TimeSeriesDataset, **params):
        scale = data.next_scale()
        space = {
            "p": {
                "domain": tune.qrandint(lower=0, upper=2 * scale, q=1),
                "init_value": scale,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.qrandint(lower=0, upper=4, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.qrandint(lower=0, upper=2 * scale, q=1),
                "init_value": scale,
                "low_cost_init_value": 0,
            },
        }
        return space

    def _join(self, X_train, y_train):
        train_df = super()._join(X_train, y_train)
        train_df.index = to_datetime(train_df[TS_TIMESTAMP_COL])
        train_df = train_df.drop(TS_TIMESTAMP_COL, axis=1)
        return train_df

    def fit(self, X_train, y_train=None, budget=None, **kwargs):
        import warnings

        super().fit(X_train, y_train, budget=budget, **kwargs)
        X_train = enrich(X_train, self.params["monthly_fourier_degree"], self.time_col)

        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_estimator

        current_time = time.time()

        if isinstance(X_train, TimeSeriesDataset):
            data = X_train
            # this class only supports univariate regression
            target_col = (
                data.target_names[0]
                if isinstance(data.target_names, list)
                else data.target_names
            )
            self.regressors = data.regressors
            train_df = data.train_data[self.regressors + [target_col]]
            train_df.index = to_datetime(data.train_data[data.time_col])
            self.time_col = data.time_col
            self.target_names = target_col
        else:
            target_col = TS_VALUE_COL
            train_df = self._join(X_train, y_train)
            self.regressors = list(train_df)
            self.regressors.remove(TS_VALUE_COL)

        train_df = self._preprocess(train_df)

        if len(self.regressors):
            model = ARIMA_estimator(
                train_df[[target_col]],
                exog=train_df[self.regressors],
                order=(self.params["p"], self.params["d"], self.params["q"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = ARIMA_estimator(
                train_df,
                order=(self.params["p"], self.params["d"], self.params["q"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        with suppress_stdout_stderr():
            model = model.fit()
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def predict(self, X, **kwargs) -> pd.Series:
        X = enrich(
            X,
            self.params["monthly_fourier_degree"],
            self.time_col,
            frequency=self.frequency,
            test_end_date=self.end_date,
        )
        if self._model is None:
            return np.ones(X if isinstance(X, int) else X.shape[0])

        if isinstance(X, int):
            return self._model.forecast(steps=X)

        if isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data[data.regressors + [data.time_col]]

        if isinstance(X, DataFrame):
            start = X[self.time_col].iloc[0]
            end = X[self.time_col].iloc[-1]
            if len(self.regressors):
                exog = self._preprocess(X[self.regressors])
                forecast = self._model.predict(
                    start=start, end=end, exog=exog.values, **kwargs
                )
            else:
                forecast = self._model.predict(start=start, end=end, **kwargs)
        else:
            raise ValueError(
                "X needs to be either a pandas Dataframe with dates as the first column"
                " or an int number of periods for predict()."
            )
        forecast.name = self.target_names[0]
        return forecast


class SARIMAX(ARIMA):
    """The class for tuning SARIMA."""

    @classmethod
    def _search_space(cls, data: TimeSeriesDataset, **params):
        scale = data.next_scale()
        max_steps = int(len(data.train_data) / scale - 0.5)  # rounding down

        if max_steps < 4:  # soft fallback if TS too short
            scale = 1
            max_steps = int(len(data.train_data) - 0.5)
        # TODO: instead, downscale the dataset and take next_scale from that for P and Q
        space = {
            "p": {
                "domain": tune.qrandint(lower=0, upper=scale - 1, q=1),
                "init_value": scale - 1,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.qrandint(lower=0, upper=2, q=1),
                "init_value": 0,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.qrandint(lower=0, upper=scale - 1, q=1),
                "init_value": scale - 1,
                "low_cost_init_value": 0,
            },
            "P": {
                "domain": tune.qrandint(lower=0, upper=min(10, max_steps), q=1),
                "init_value": 3,
                "low_cost_init_value": 0,
            },
            "D": {
                "domain": tune.qrandint(lower=0, upper=2, q=1),
                "init_value": 0,
                "low_cost_init_value": 0,
            },
            "Q": {
                "domain": tune.qrandint(lower=0, upper=min(10, max_steps), q=1),
                "init_value": 3,
                "low_cost_init_value": 0,
            },
            "s": {
                "domain": tune.choice([scale, 2 * scale, 3 * scale, 4 * scale]),
                "init_value": scale,
            },
        }
        return space

    def fit(self, X_train, y_train=None, budget=None, **kwargs):
        import warnings

        super().fit(X_train, y_train, budget=budget, **kwargs)
        X_train = enrich(X_train, self.params["monthly_fourier_degree"], self.time_col)

        warnings.filterwarnings("ignore")
        from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_estimator

        current_time = time.time()

        if isinstance(X_train, TimeSeriesDataset):
            data = X_train
            target_col = data.target_names[0]
            self.regressors = data.regressors
            # this class only supports univariate regression
            train_df = data.train_data[self.regressors + [target_col]]
            train_df.index = to_datetime(data.train_data[data.time_col])
        else:
            target_col = TS_VALUE_COL
            train_df = self._join(X_train, y_train)
            self.regressors = list(train_df)
            self.regressors.remove(TS_VALUE_COL)

        train_df = self._preprocess(train_df)
        # regressors = list(train_df)
        # regressors.remove(target_col)
        if self.regressors:
            model = SARIMAX_estimator(
                train_df[[target_col]],
                exog=train_df[self.regressors],
                order=(self.params["p"], self.params["d"], self.params["q"]),
                seasonal_order=(
                    self.params["P"],
                    self.params["D"],
                    self.params["Q"],
                    self.params["s"],
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = SARIMAX_estimator(
                train_df,
                order=(self.params["p"], self.params["d"], self.params["q"]),
                seasonal_order=(
                    self.params["P"],
                    self.params["D"],
                    self.params["Q"],
                    self.params["s"],
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        with suppress_stdout_stderr():
            model = model.fit()
        train_time = time.time() - current_time
        self._model = model
        return train_time


class TS_SKLearn(TimeSeriesEstimator):
    """The class for tuning SKLearn Regressors for time-series forecasting, using hcrystalball"""

    base_class = SKLearnEstimator

    @classmethod
    def _search_space(
        cls, data: TimeSeriesDataset, task: Task, pred_horizon: int, **params
    ):
        data_size = data.train_data.shape
        space = cls.base_class.search_space(data_size=data_size, task=task, **params)
        scale = data.next_scale()
        max_lags = max(scale, int(np.sqrt(data_size[0])))

        space.update(
            {
                "optimize_for_horizon": {
                    "domain": tune.choice([True, False]),
                    "init_value": False,
                    "low_cost_init_value": False,
                },
                "lags": {
                    "domain": tune.randint(lower=1, upper=max_lags),
                    "init_value": scale,
                },
            }
        )
        return space

    def __init__(self, task="ts_forecast", **params):
        # TODO: pass task objects throughout
        super().__init__(task, **params)
        self.hcrystaball_model = None
        self.ts_task = task

    def transform_X(self, X: pd.DataFrame):
        cols = list(X)

        if len(cols) == 1:
            X = DataFrame(index=X[self.time_col])
        elif len(cols) > 1:
            exog_cols = [c for c in cols if c != self.time_col]
            X = X.set_index(self.time_col)[exog_cols]
        return X

    def _fit(self, X_train, y_train, budget=None, time_col=None, **kwargs):
        from hcrystalball.wrappers import get_sklearn_wrapper

        X_train = self.transform_X(X_train)
        self.regressors = list(X_train.columns)
        X_train = self._preprocess(X_train)

        params = self.params.copy()
        lags = params.pop("lags")
        optimize_for_horizon = params.pop("optimize_for_horizon")
        est_params = {
            k: v for k, v in params.items() if k not in self.top_level_params()
        }
        estimator = self.base_class(task=self.ts_task, **est_params)
        self.hcrystaball_model = get_sklearn_wrapper(estimator.estimator_class)
        self.hcrystaball_model.lags = int(lags)
        self.hcrystaball_model.fit(X_train, y_train)
        if optimize_for_horizon:
            # Direct Multi-step Forecast Strategy - fit a seperate model for each horizon
            model_list = []
            for i in range(1, kwargs["period"] + 1):
                (
                    X_fit,
                    y_fit,
                ) = self.hcrystaball_model._transform_data_to_tsmodel_input_format(
                    X_train, y_train, i
                )
                self.hcrystaball_model.model.set_params(**estimator.params)
                model = self.hcrystaball_model.model.fit(X_fit, y_fit)
                model_list.append(model)
            self._model = model_list
        else:
            (
                X_fit,
                y_fit,
            ) = self.hcrystaball_model._transform_data_to_tsmodel_input_format(
                X_train, y_train, kwargs["period"]
            )
            self.hcrystaball_model.model.set_params(**estimator.params)
            model = self.hcrystaball_model.model.fit(X_fit, y_fit)
            self._model = model

    def fit(self, X_train, y_train=None, budget=None, **kwargs):
        # self.time_col = kwargs.pop("time_col", "ds")  # Doesn't work for some reason
        super().fit(X_train, y_train, budget=budget, **kwargs)
        X_train = enrich(X_train, self.params["monthly_fourier_degree"], self.time_col)

        current_time = time.time()
        if isinstance(X_train, TimeSeriesDataset):
            data = X_train
            X_train = data.train_data[data.regressors + [data.time_col]]
            # this class only supports univariate regression
            y_train = data.y_train
            self.time_col = data.time_col
            self.target_names = data.target_names
        elif isinstance(X_train, pd.DataFrame):
            self.time_col = X_train.columns.tolist()[0]

        self._fit(X_train, y_train, budget=budget, time_col=self.time_col, **kwargs)
        train_time = time.time() - current_time
        return train_time

    def predict(self, X, **kwargs):
        X = enrich(
            X,
            self.params["monthly_fourier_degree"],
            self.time_col,
            frequency=self.frequency,
            test_end_date=self.end_date,
        )
        if isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data

        X = X[self.regressors + [self.time_col]]

        if self._model is not None:
            X = self.transform_X(X)
            X = self._preprocess(X)
            if isinstance(self._model, list):
                assert len(self._model) == len(
                    X
                ), "Model is optimized for horizon, length of X must be equal to `period`."
                preds = []
                for i in range(1, len(self._model) + 1):
                    (
                        X_pred,
                        _,
                    ) = self.hcrystaball_model._transform_data_to_tsmodel_input_format(
                        X.iloc[:i, :]
                    )
                    preds.append(self._model[i - 1].predict(X_pred, **kwargs)[-1])
                forecast = DataFrame(
                    data=np.asarray(preds).reshape(-1, 1),
                    columns=self.target_names,  # [self.hcrystaball_model.name],
                    index=X.index,
                )
            else:
                (
                    X_pred,
                    _,
                ) = self.hcrystaball_model._transform_data_to_tsmodel_input_format(X)
                forecast = self._model.predict(X_pred, **kwargs)
            if isinstance(forecast, pd.Series):
                forecast.name = self.target_names[0]

            return forecast
        else:
            logger.warning(
                "Estimator is not fit yet. Please run fit() before predict()."
            )
            return np.ones(X.shape[0])


class LGBM_TS(TS_SKLearn):
    """The class for tuning LGBM Regressor for time-series forecasting"""

    base_class = LGBMEstimator


class XGBoost_TS(TS_SKLearn):
    """The class for tuning XGBoost Regressor for time-series forecasting"""

    base_class = XGBoostSklearnEstimator


class RF_TS(TS_SKLearn):
    """The class for tuning Random Forest Regressor for time-series forecasting"""

    base_class = RandomForestEstimator


class ExtraTrees_TS(TS_SKLearn):
    """The class for tuning Extra Trees Regressor for time-series forecasting"""

    base_class = ExtraTreesEstimator


class XGBoostLimitDepth_TS(TS_SKLearn):
    """The class for tuning XGBoost Regressor with unlimited depth for time-series forecasting"""

    base_class = XGBoostLimitDepthEstimator


# catboost regressor is invalid because it has a `name` parameter, making it incompatible with hcrystalball
# class CatBoost_TS_Regressor(TS_Regressor):
#     base_class = CatBoostEstimator
