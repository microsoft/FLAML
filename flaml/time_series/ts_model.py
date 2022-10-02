import time
import logging
import os
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame, Series, to_datetime

from flaml import tune

import numpy as np


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
from flaml.data import TS_TIMESTAMP_COL, TS_VALUE_COL, add_time_idx_col
from flaml.time_series.ts_data import TimeSeriesDataset, enrich
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
                    "init_value": 4,
                    "low_cost_init_value": 2,
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

        X_train = enrich(
            X_train, self.params.get("monthly_fourier_degree", None), self.time_col
        )
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
            self.params.get("monthly_fourier_degree", None),
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
                "domain": tune.qrandint(lower=0, upper=6, q=1),
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
        X_train = enrich(
            X_train,
            self.params.get("monthly_fourier_degree", None),
            self.time_col,
            remove_constants=True,
        )

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
            self.params.get("monthly_fourier_degree", None),
            self.time_col,
            frequency=self.frequency,
            test_end_date=self.end_date,
        )
        if self._model is None or self._model is False:
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
                "domain": tune.qrandint(lower=0, upper=6, q=1),
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
                "domain": tune.qrandint(lower=0, upper=6, q=1),
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
        X_train = enrich(
            X_train,
            self.params.get("monthly_fourier_degree", None),
            self.time_col,
            remove_constants=True,
        )

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
        super().fit(X_train, y_train, budget=budget, **kwargs)
        X_train = enrich(
            X_train,
            self.params.get(
                "monthly_fourier_degree",
                None,
            ),
            self.time_col,
        )

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
            self.params.get("monthly_fourier_degree", None),
            self.time_col,
            frequency=self.frequency,
            test_end_date=self.end_date,
        )
        if isinstance(X, TimeSeriesDataset):
            data = X
            X = data.test_data

        if self._model is not None:
            X = X[self.regressors + [self.time_col]]
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
                    columns=self.target_names,
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


class TemporalFusionTransformerEstimator(TimeSeriesEstimator):
    """The class for tuning Temporal Fusion Transformer"""

    @classmethod
    def search_space(cls, data, task, pred_horizon, **params):
        space = {
            "gradient_clip_val": {
                "domain": tune.loguniform(lower=0.01, upper=100.0),
                "init_value": 0.01,
            },
            "hidden_size": {
                "domain": tune.lograndint(lower=8, upper=512),
                "init_value": 16,
            },
            "hidden_continuous_size": {
                "domain": tune.randint(lower=1, upper=65),
                "init_value": 8,
            },
            "attention_head_size": {
                "domain": tune.randint(lower=1, upper=5),
                "init_value": 4,
            },
            "dropout": {
                "domain": tune.uniform(lower=0.1, upper=0.3),
                "init_value": 0.1,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=0.00001, upper=1.0),
                "init_value": 0.001,
            },
        }
        return space

    def transform_ds(self, X_train, y_train, **kwargs):
        self.data = X_train.train_data

        max_prediction_length = kwargs["period"]
        self.max_encoder_length = kwargs["max_encoder_length"]
        training_cutoff = self.data["time_idx"].max() - max_prediction_length

        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        self.group_ids = kwargs["group_ids"].copy()
        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=X_train.target_names[0],
            group_ids=self.group_ids,
            min_encoder_length=kwargs.get(
                "min_encoder_length", self.max_encoder_length // 2
            ),  # keep encoder length long (as it is in the validation set)
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=kwargs.get("static_categoricals", []),
            static_reals=kwargs.get("static_reals", []),
            time_varying_known_categoricals=kwargs.get(
                "time_varying_known_categoricals", []
            ),
            time_varying_known_reals=kwargs.get("time_varying_known_reals", []),
            time_varying_unknown_categoricals=kwargs.get(
                "time_varying_unknown_categoricals", []
            ),
            time_varying_unknown_reals=kwargs.get("time_varying_unknown_reals", []),
            variable_groups=kwargs.get(
                "variable_groups", {}
            ),  # group of categorical variables can be treated as one variable
            lags=kwargs.get("lags", {}),
            target_normalizer=GroupNormalizer(
                groups=kwargs["group_ids"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation = TimeSeriesDataSet.from_dataset(
            training, self.data, predict=True, stop_randomization=True
        )

        # create dataloaders for model
        batch_size = kwargs.get("batch_size", 64)
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=0
        )

        return training, train_dataloader, val_dataloader

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import warnings
        import pytorch_lightning as pl
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger

        warnings.filterwarnings("ignore")
        current_time = time.time()
        super().fit(X_train, **kwargs)
        training, train_dataloader, val_dataloader = self.transform_ds(
            X_train, y_train, **kwargs
        )
        params = self.params.copy()
        gradient_clip_val = params.pop("gradient_clip_val", None)
        params.pop("n_jobs", None)
        max_epochs = kwargs.get("max_epochs", 20)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger(
            kwargs.get("log_dir", "lightning_logs")
        )  # logging results to a tensorboard
        default_trainer_kwargs = dict(
            gpus=kwargs.get("gpu_per_trial", [0])
            if torch.cuda.is_available()
            else None,
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )
        tft = TemporalFusionTransformer.from_dataset(
            training,
            **params,
            lstm_layers=2,  # 2 is mostly optimal according to documentation
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        )
        # fit network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        train_time = time.time() - current_time
        self._model = best_tft
        return train_time

    def predict(self, X):
        ids = self.group_ids.copy()
        ids.append(self.time_col)
        encoder_data = self.data[
            lambda x: x.time_idx > x.time_idx.max() - self.max_encoder_length
        ]
        # following pytorchforecasting example, make all target values equal to the last data
        last_data_cols = self.group_ids.copy()
        last_data_cols.append(self.target_names[0])
        last_data = self.data[lambda x: x.time_idx == x.time_idx.max()][last_data_cols]
        decoder_data = X.X_val if isinstance(X, TimeSeriesDataset) else X
        if "time_idx" not in decoder_data:
            decoder_data = add_time_idx_col(decoder_data)
        decoder_data["time_idx"] += (
            encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
        )
        decoder_data = decoder_data.merge(last_data, how="inner", on=self.group_ids)
        decoder_data = decoder_data.sort_values(ids)
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        new_prediction_data["time_idx"] = new_prediction_data["time_idx"].astype("int")
        new_raw_predictions = self._model.predict(new_prediction_data)
        index = [decoder_data[idx].to_numpy() for idx in ids]
        predictions = pd.Series(new_raw_predictions.numpy().ravel(), index=index)
        return predictions


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
