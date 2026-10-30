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


def test_sparse_month_end_time_idx_counts_months():
    # April 30 is missing; months differ in length, so the step must stay one month
    dates = list(pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31", "2021-05-31"]))
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 4 + ["b"] * 4})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 2, 4] * 2


def test_sparse_quarter_end_time_idx_counts_quarters():
    dates = pd.to_datetime(["2021-03-31", "2021-06-30", "2021-12-31", "2022-03-31"])
    X = pd.DataFrame({"ds": dates})
    X = add_time_idx_col(X)
    assert X["time_idx"].tolist() == [0, 1, 3, 4]


def test_tz_aware_same_day_of_month_time_idx_keeps_both_dst_fold_hours():
    # 01:30 on the 7th each month, and the 7th of November 2021 has two 01:30s in New York
    dates = list(
        pd.to_datetime(["2021-10-07 05:30", "2021-11-07 05:30", "2021-11-07 06:30"])
        .tz_localize("UTC")
        .tz_convert("America/New_York")
    )
    X = pd.DataFrame({"ds": dates * 2, "group": ["a"] * 3 + ["b"] * 3}).sample(frac=1, random_state=0)
    X = add_time_idx_col(X)
    assert X.sort_values(["group", "ds"])["time_idx"].tolist() == [0, 744, 745] * 2


def test_tz_aware_dst_fold_hours_then_same_day_of_month_get_distinct_time_idx():
    dates = list(
        pd.to_datetime(["2021-11-07 05:30", "2021-11-07 06:30", "2021-12-07 06:30", "2022-01-07 06:30"])
        .tz_localize("UTC")
        .tz_convert("America/New_York")
    )
    X = add_time_idx_col(pd.DataFrame({"ds": dates}))
    assert X["time_idx"].tolist() == [0, 1, 721, 1465]


def test_regular_grid_check_builds_one_candidate_per_date(monkeypatch):
    # 1 ms apart on the wall clock, but across the New York fall-back hour
    dates = list(
        pd.to_datetime(["2021-11-07 05:00:00.000", "2021-11-07 06:00:00.001", "2021-11-07 06:00:00.002"])
        .tz_localize("UTC")
        .tz_convert("America/New_York")
    )
    real_date_range = pd.date_range
    sizes = []

    def recording_date_range(*args, **kwargs):
        grid = real_date_range(*args, **kwargs)
        sizes.append(len(grid))
        return grid

    monkeypatch.setattr(pd, "date_range", recording_date_range)
    X = add_time_idx_col(pd.DataFrame({"ds": dates}))
    assert sizes and max(sizes) <= len(dates)
    assert X["time_idx"].tolist() == [0, 3600001, 3600002]
