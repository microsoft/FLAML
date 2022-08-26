import copy
import math
from typing import Union, Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from flaml.multiscale.smooth import expsmooth, moving_window_smooth
from flaml.ts_model import TimeSeriesEstimator, TimeSeriesDataset


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

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], offset: int = -1):
        self.fit(X, offset)
        return self.transform(X)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], offset: int = -1):
        _, self.mat, _, self.sample_indices = scale_transform(
            len(X), self.step, self.smooth_fun, offset
        )
        self.sample_indices = self.sample_indices

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
            raise NotImplementedError
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
        else:
            raise ValueError(
                "hi and lo must either both be np.ndarray,pd.DataFrame, or TimeSeriesDataset"
            )


class MultiscaleModel(TimeSeriesEstimator):
    def __init__(
        self,
        model_lo: TimeSeriesEstimator,
        model_hi: TimeSeriesEstimator,
        scale=None,
        task="ts_forecast",
    ):
        super().__init__(task=task)
        self.model_lo = model_lo
        self.model_hi = model_hi
        self.scale = scale
        self.scale_transform: ScaleTransform = None

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

        y_pred_hi = self.model_lo.predict(X_hi.test_data)
        y_hi = X_hi.merge_prediction_with_target(y_pred_hi)

        out = self.scale_transform.inverse_transform(y_lo, y_hi)
        return out


if __name__ == "__main__":
    st = ScaleTransform(step=7)
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=300))
    df = pd.DataFrame(y)
    df["data"] = pd.Series(data=np.random.normal(size=len(df)), index=df.index)
    lo, hi = st.fit_transform(df)
    out = st.inverse_transform(lo, hi)
    error = df["data"] - out["data"]
    error.abs().max()
    print("yahoo!")
