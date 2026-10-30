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


def test_tz_aware_business_day_time_idx_across_dst():
    dates = list(pd.date_range("2021-03-10", periods=6, freq="B", tz="America/New_York"))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 6 + ["b"] * 6})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 3, 4, 5] * 2


def test_irregular_time_idx_keeps_gaps():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05"])
    X = pd.DataFrame({"ds": dates})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 4]


def test_irregular_time_idx_uses_a_step_that_divides_every_gap():
    # the most common gap (2 days) is larger than the smallest one (1 day)
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-06"])
    X = pd.DataFrame({"ds": dates})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 3, 5]


def test_tz_aware_hourly_time_idx_keeps_a_missing_dst_fold_hour():
    # drop the second 01:00 (01:00-05:00) of the fall-back day
    dates = list(pd.date_range("2021-11-07 00:00", periods=7, freq="h", tz="America/New_York").delete(2))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 6 + ["b"] * 6})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 3, 4, 5, 6] * 2


def test_tz_aware_irregular_daily_time_idx_across_dst():
    # 2021-03-14 is missing and is also the 23 hour day
    dates = list(
        pd.to_datetime(["2021-03-12", "2021-03-13", "2021-03-15", "2021-03-16"]).tz_localize("America/New_York")
    )
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 4 + ["b"] * 4})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 3, 4] * 2
