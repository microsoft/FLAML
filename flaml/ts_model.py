import time
import logging

import numpy as np
import pandas as pd

from orbit.models import DLT, LGT, ETS


from .model import TimeSeriesEstimator, suppress_stdout_stderr
from .data import TS_TIMESTAMP_COL, TS_VALUE_COL


class TimeSeriesDataset:
    @property
    def regressors(self):
        return self.time_varying_known_categoricals + self.time_varying_known_reals


class Orbit(TimeSeriesEstimator):
    def fit(self, X_train: TimeSeriesDataset, y_train, budget=None, **kwargs):
        # y_train is ignored, just need it for signature compatibility with other classes
        current_time = time.time()
        self.logger = logging.getLogger("orbit").setLevel(logging.WARNING)

        model_class = self.params.get("model_class", DLT)
        self._model = model_class(
            response_col=X_train.target_names[0],
            date_col=X_train.time_col,
            regressor_col=X_train.regressors,
            # TODO: infer seasonality from frequency
            **self.params
        )

        with suppress_stdout_stderr():
            self._model.fit(df=X_train.train_data.copy())

        train_time = time.time() - current_time
        return train_time

    def predict(self, X: TimeSeriesDataset, **kwargs):
        if isinstance(X, int):
            raise ValueError(
                "predict() with steps is only supported for arima/sarimax."
                " For Prophet, pass a dataframe with the first column containing"
                " the timestamp values."
            )

        if self._model is not None:

            forecast = self._model.predict(X, **kwargs)
            out = (
                pd.DataFrame(
                    forecast[
                        [
                            X.metadata["time_col"],
                            "prediction",
                            "prediction_5",
                            "prediction_95",
                        ]
                    ]
                )
                .reset_index(drop=True)
                .rename(
                    columns={
                        "TIME_BUCKET": X.time_col,
                        "prediction": self.target_name,
                        "prediction_5": "lower",
                        "prediction_95": "upper",
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
