import numpy as np
import pandas as pd
from flaml.time_series.multiscale import ScaleTransform, MultiscaleModel
from flaml.time_series.ts_data import TimeSeriesDataset
from flaml.time_series.ts_model import ARIMA, SARIMAX


def test_multiscale_transform():
    st = ScaleTransform(step=7)
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=30))
    df = pd.DataFrame(y)
    df["data"] = pd.Series(data=np.random.normal(size=len(df)), index=df.index)
    lo, hi = st.fit_transform(df)
    out = st.inverse_transform(lo, hi)
    error = df["data"] - out["data"]
    assert error.abs().max() <= 1e-10


def test_multiscale_transform_dataset():
    st = ScaleTransform(step=7)
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=300))
    df = pd.DataFrame(y)
    df["data"] = pd.Series(data=np.random.normal(size=len(df)), index=df.index)

    ts_data = TimeSeriesDataset(
        train_data=df[:-50], time_col="date", target_names="data", test_data=df[-50:]
    )

    lo, hi = st.fit_transform(ts_data)
    re_data = st.inverse_transform(lo, hi)

    test_df = ts_data.all_data.merge(re_data.all_data, on=ts_data.time_col)

    assert len(test_df) == len(ts_data.all_data)
    assert (test_df["data_x"] - test_df["data_y"]).abs().max() < 1e-10


def test_multiscale_arima():
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=300))
    df = pd.DataFrame(y)
    df["data"] = pd.Series(data=np.random.normal(size=len(df)), index=df.index)

    ts_data = TimeSeriesDataset(
        train_data=df[:-50], time_col="date", target_names="data", test_data=df[-50:]
    )
    model_lo = ARIMA(p=2, d=2, q=1)
    model_hi = SARIMAX(p=1, d=0, q=1, P=3, D=0, Q=3, s=7)
    model = MultiscaleModel(model_lo, model_hi)
    model.fit(ts_data)
    model.predict(ts_data)


if __name__ == "__main__":
    test_multiscale_transform()
