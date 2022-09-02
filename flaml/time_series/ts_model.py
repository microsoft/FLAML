import time
import logging
import os
from typing import List, Optional, Union


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
from flaml.time_series.ts_data import TimeSeriesDataset


class TimeSeriesEstimator(SKLearnEstimator):
    def __init__(self, task="ts_forecast", n_jobs=1, **params):
        super().__init__(task, **params)
        self.time_col: Optional[str] = None
        self.target_names: Optional[Union[str, List[str]]] = None

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
            # TODO: do prediction like that too, with TimeSeriesDataset
            raise ValueError(
                "predict() with steps is only supported for arima/sarimax."
                " For Prophet, pass a dataframe with the first column containing"
                " the timestamp values."
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
    def search_space(cls, **params):
        space = {}
        return space


class Prophet(TimeSeriesEstimator):
    """The class for tuning Prophet."""

    @classmethod
    def search_space(cls, **params):
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
        model = Prophet(**self.params)
        for regressor in regressors:
            model.add_regressor(regressor)
        with suppress_stdout_stderr():
            model.fit(train_df)
        train_time = time.time() - current_time
        self._model = model
        return train_time

    def predict(self, X, **kwargs):
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
            return forecast["yhat"]
        else:
            logger.warning(
                "Estimator is not fit yet. Please run fit() before predict()."
            )
            return np.ones(X.shape[0])


class ARIMA(TimeSeriesEstimator):
    """The class for tuning ARIMA."""

    @classmethod
    def search_space(cls, **params):
        space = {
            "p": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 1,
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

    def predict(self, X, **kwargs):
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
                X = self._preprocess(X[self.regressors])
                forecast = self._model.predict(start=start, end=end, exog=X, **kwargs)
            else:
                forecast = self._model.predict(start=start, end=end, **kwargs)
        else:
            raise ValueError(
                "X needs to be either a pandas Dataframe with dates as the first column"
                " or an int number of periods for predict()."
            )
        return forecast


class SARIMAX(ARIMA):
    """The class for tuning SARIMA."""

    @classmethod
    def search_space(cls, **params):
        space = {
            "p": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "d": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 2,
                "low_cost_init_value": 0,
            },
            "q": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "P": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "D": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "Q": {
                "domain": tune.qrandint(lower=0, upper=10, q=1),
                "init_value": 1,
                "low_cost_init_value": 0,
            },
            "s": {
                "domain": tune.choice([1, 4, 6, 12]),
                "init_value": 12,
            },
        }
        return space

    def fit(self, X_train, y_train=None, budget=None, **kwargs):
        import warnings

        super().fit(X_train, y_train, budget=budget, **kwargs)

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
    def search_space(cls, data_size, pred_horizon, **params):
        space = cls.base_class.search_space(data_size, **params)
        space.update(
            {
                "optimize_for_horizon": {
                    "domain": tune.choice([True, False]),
                    "init_value": False,
                    "low_cost_init_value": False,
                },
                "lags": {
                    "domain": tune.randint(
                        lower=1, upper=max(2, int(np.sqrt(data_size[0])))
                    ),
                    "init_value": 3,
                },
            }
        )
        return space

    def __init__(self, task="ts_forecast", **params):
        # TODO: pass task objects throughout
        super().__init__(task, **params)
        self.hcrystaball_model = None
        self.ts_task = task
        # (
        #     # TODO: wire this into task class
        #     "regression"
        #     if task.name in TS_FORECASTREGRESSION
        #     else "classification"
        # )

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
        X_train = self._preprocess(X_train)
        params = self.params.copy()
        lags = params.pop("lags")
        optimize_for_horizon = params.pop("optimize_for_horizon")
        estimator = self.base_class(task=self.ts_task, **params)
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
        current_time = time.time()
        if isinstance(X_train, TimeSeriesDataset):
            data = X_train
            X_train = data.train_data[data.regressors + [data.time_col]]
            # this class only supports univariate regression
            y_train = data.train_data[data.target_names[0]]
            self.time_col = data.time_col
            self.target_names = data.target_names
        elif isinstance(X_train, pd.DataFrame):
            self.time_col = X_train.columns.tolist()[0]

        self._fit(X_train, y_train, budget=budget, time_col=self.time_col, **kwargs)
        train_time = time.time() - current_time
        return train_time

    def predict(self, X, **kwargs):
        if isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data[data.regressors + [data.time_col]]

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
                    columns=[self.hcrystaball_model.name],
                    index=X.index,
                )
            else:
                (
                    X_pred,
                    _,
                ) = self.hcrystaball_model._transform_data_to_tsmodel_input_format(X)
                forecast = self._model.predict(X_pred, **kwargs)
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
