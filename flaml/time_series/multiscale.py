import math
from typing import Union, Callable


import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from flaml import tune

from flaml.time_series.smooth import expsmooth, moving_window_smooth
from flaml.time_series import TimeSeriesEstimator, TimeSeriesDataset, ARIMA, SARIMAX
from flaml.automl.task import Task


def scale_transform(points: int, step: int, smooth_fun: Callable, offset: int = -1):
    I = np.eye(points)
    T = smooth_fun(I)
    S = np.zeros((math.ceil(points / step), points))

    offset = offset % points

    i = 0
    sample_indices = []
    for j in range(points):
        if j % step == offset % step:
            S[i, j] = 1.0
            sample_indices.append(j)
            i += 1
    lo = S.dot(T)
    hi = I - T
    mat = np.concatenate([hi, lo], axis=0)
    myinv = np.linalg.pinv(mat)
    return T, mat, myinv, np.array(sample_indices)


def split_cols(X: pd.DataFrame):
    float_cols, other_cols = [], []
    for c in X.columns:
        (float_cols if is_float_dtype(X[c]) else other_cols).append(c)
    assert len(float_cols), "Need float columns to transform"
    return float_cols, other_cols


class ScaleTransform:
    def __init__(
        self, step: int, smooth_type: str = "moving_window", scale_tweak: float = 0.5
    ):
        self.step = step
        if smooth_type == "moving_window":
            self.smooth_fun = lambda x: moving_window_smooth(x, step, True)
        elif smooth_type == "exponential":
            self.smooth_fun = lambda x: expsmooth(x, 1 / (step * scale_tweak))
        else:
            raise ValueError(
                "Only smooth types 'moving_window' and 'exponential' are supported"
            )
        self.mat = None
        self.sample_indices = None

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        self.fit(X)
        return self.transform(X)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, TimeSeriesDataset]):
        if isinstance(X, TimeSeriesDataset):
            num_rows = len(X.train_data) + len(X.test_data)
            # want to sample the last timestamp in the _train_ set
            offset = -len(X.test_data) - 1
        else:
            num_rows = len(X)
            offset = -1

        _, self.mat, _, self.sample_indices = scale_transform(
            num_rows, self.step, self.smooth_fun, offset
        )

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, np.ndarray):
            out = self.mat.dot(X)
            return out[len(X) :], out[: len(X)]
        elif isinstance(X, pd.DataFrame):
            float_cols, other_cols = split_cols(X)
            lo_, hi_ = self.transform(X[float_cols].values)
            hi = pd.concat(
                [
                    X[other_cols],
                    pd.DataFrame(data=hi_, columns=float_cols, index=X.index),
                ],
                axis=1,
            )
            Xlo = X.iloc[self.sample_indices][other_cols]
            lo = pd.concat(
                [
                    Xlo,
                    pd.DataFrame(data=lo_, columns=float_cols, index=Xlo.index),
                ],
                axis=1,
            )
            return lo, hi
        elif isinstance(X, TimeSeriesDataset):
            lo_, hi_ = self.transform(pd.concat([X.train_data, X.test_data], axis=0))
            hi = TimeSeriesDataset(
                train_data=hi_[: len(X.train_data)],
                test_data=hi_[len(X.train_data) :],
                time_col=X.time_col,
                target_names=X.target_names,
                time_idx=X.time_idx,
            )
            lo_train = lo_[lo_[X.time_col] <= hi.train_data[hi.time_col].max()]
            lo_test = lo_[lo_[X.time_col] > hi.train_data[hi.time_col].max()]

            lo = TimeSeriesDataset(
                train_data=lo_train,
                test_data=lo_test,
                time_col=X.time_col,
                target_names=X.target_names,
                time_idx=X.time_idx,
            )
            return lo, hi
        else:
            raise ValueError(
                "Only np.ndarray, pd.DataFrame, and TimeSeriesDataset are supported"
            )

    def inverse_transform(
        self, lo: Union[np.ndarray, pd.DataFrame], hi: Union[np.ndarray, pd.DataFrame]
    ):
        if isinstance(lo, np.ndarray) and isinstance(hi, np.ndarray):
            X = np.concatenate([hi, lo], axis=0)
            return np.linalg.pinv(self.mat).dot(X)
        elif isinstance(lo, pd.DataFrame) and isinstance(hi, pd.DataFrame):
            float_cols, other_cols = split_cols(hi)
            vals = self.inverse_transform(lo[float_cols].values, hi[float_cols].values)
            out = pd.concat(
                [
                    hi[other_cols],
                    pd.DataFrame(data=vals, columns=float_cols, index=hi.index),
                ],
                axis=1,
            )
            return out
        elif isinstance(lo, TimeSeriesDataset) and isinstance(hi, TimeSeriesDataset):
            merged = self.inverse_transform(lo.all_data, hi.all_data)
            out = TimeSeriesDataset(
                train_data=merged[: len(hi.train_data)],
                test_data=merged[len(hi.train_data) :],
                time_col=hi.time_col,
                target_names=hi.target_names,
                time_idx=hi.time_idx,
            )
            return out
        else:
            raise ValueError(
                "hi and lo must either both be np.ndarray,pd.DataFrame, or TimeSeriesDataset"
            )


class MultiscaleModel(TimeSeriesEstimator):
    def __init__(
        self,
        model_lo: Union[dict, TimeSeriesEstimator],
        model_hi: Union[dict, TimeSeriesEstimator],
        scale: int = None,
        task: Union[Task, str] = "ts_forecast",
    ):
        super().__init__(task=task)

        self.model_lo = (
            model_lo
            if isinstance(model_lo, TimeSeriesEstimator)
            else self._init_submodel(model_lo)
        )
        self.model_hi = (
            model_hi
            if isinstance(model_hi, TimeSeriesEstimator)
            else self._init_submodel(model_hi)
        )
        self.scale = scale
        self.scale_transform: ScaleTransform = None

    def _init_submodel(self, config: dict):
        est_name = config.pop("estimator")
        est_class = self._task.estimator_class_from_str(est_name)
        return est_class(task=self._task, **config)

    def _search_space(
        self, data: TimeSeriesDataset, task: Task, pred_horizon: int, **params
    ):
        estimators = {
            "model_lo": ["arima", "sarimax"],
            "model_hi": ["arima", "sarimax"],
        }
        out = {}
        for mdl, ests in estimators.items():
            est_cfgs = []
            for est in ests:
                est_class = task.estimator_class_from_str(est)
                est_cfgs.append(
                    {
                        "estimator": est,
                        **(est_class.search_space(data, task, pred_horizon)),
                    }
                )
            out[mdl] = tune.choice(est_cfgs)
        return out

    def fit(self, X_train: TimeSeriesDataset, y_train=None, **kwargs):
        super().fit(X_train)
        if self.scale is None:
            self.scale = X_train.next_scale()
        self.scale_transform = ScaleTransform(self.scale)
        self.X_train = X_train
        X_lo, X_hi = self.scale_transform.fit_transform(X_train)
        self.model_lo.fit(X_lo)
        self.model_hi.fit(X_hi)

    def predict(self, X: TimeSeriesDataset):
        # X has all the known past in train_data, genuine future in test_data
        X_lo, X_hi = self.scale_transform.fit_transform(X)
        y_pred_lo = self.model_lo.predict(X_lo.test_data)
        y_lo = X_lo.merge_prediction_with_target(y_pred_lo)

        y_pred_hi = self.model_hi.predict(X_hi.test_data)
        y_hi = X_hi.merge_prediction_with_target(y_pred_hi)

        out = self.scale_transform.inverse_transform(y_lo, y_hi)
        return out[len(X_hi.train_data) :]


if __name__ == "__main__":
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=300))
    df = pd.DataFrame(y)

    t = np.array(range(len(y)))
    data = (
        np.sin(2.0 * math.pi * t / 7.0)
        + t**0.5
        + 0.1 * np.random.normal(size=len(df))
    )
    df["data"] = pd.Series(data=data, index=df.index)

    test_rows = 100

    ts_data = TimeSeriesDataset(
        train_data=df[:-test_rows],
        time_col="date",
        target_names="data",
        test_data=df[-test_rows:],
    )

    model_lo = ARIMA(p=3, d=1, q=1)
    model_hi = SARIMAX(p=1, d=0, q=1, P=3, D=0, Q=3, s=7)
    model = MultiscaleModel(model_lo, model_hi)
    model.fit(ts_data)
    out = model.predict(ts_data)
    import matplotlib.pyplot as plt

    plt.plot(ts_data.all_data.date, ts_data.all_data.data)
    plt.plot(out.date, out.data)
    plt.show()
    print("yahoo!")
