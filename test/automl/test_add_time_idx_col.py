import pandas as pd

from flaml.automl.data import add_time_idx_col


def test_daily_time_idx_counts_steps():
    X = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=5, freq="D")})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 3, 4]


def test_daily_time_idx_with_shared_timestamps():
    # panel data: every group repeats the same timestamps
    dates = list(pd.date_range("2024-01-01", periods=3, freq="D"))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 3 + ["b"] * 3})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 0, 1, 2]


def test_month_end_time_idx_with_shared_timestamps():
    dates = list(pd.date_range("2021-01-31", periods=6, freq=pd.offsets.MonthEnd()))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 6 + ["b"] * 6})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 3, 4, 5] * 2


def test_tz_aware_daily_time_idx_across_dst():
    dates = list(pd.date_range("2021-03-12", periods=6, freq="D", tz="America/New_York"))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 6 + ["b"] * 6})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 3, 4, 5] * 2


def test_irregular_time_idx_keeps_gaps():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05"])
    X = pd.DataFrame({"ds": dates})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 4]
