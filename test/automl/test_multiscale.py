import numpy as np
import pandas as pd
from flaml.multiscale import ScaleTransform


def test_multiscale():
    st = ScaleTransform(step=7)
    y = pd.Series(name="date", data=pd.date_range(start="1/1/2018", periods=30))
    df = pd.DataFrame(y)
    df["data"] = pd.Series(data=np.random.normal(size=len(df)), index=df.index)
    lo, hi = st.fit_transform(df)
    out = st.inverse_transform(lo, hi)
    error = df["data"] - out["data"]
    assert error.abs().max() <= 1e-10


if __name__ == "__main__":
    test_multiscale()
