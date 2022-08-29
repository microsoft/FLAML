import math
import datetime
from functools import lru_cache

import holidays
import pandas as pd


def naive_date_features(timestamps: pd.Series, month_fourier_degree: int = 2):
    data = pd.DataFrame({"time": timestamps})
    data["month"] = timestamps.apply(lambda x: x.month).astype(str).astype("category")
    data["day"] = timestamps.apply(lambda x: x.day).astype(str).astype("category")
    data["dayofweek"] = (
        timestamps.apply(lambda x: x.dayofweek).astype(str).astype("category")
    )
    data["hourofday"] = (
        timestamps.apply(lambda x: x.hour).astype(str).astype("category")
    )

    month_pos = timestamps.apply(
        lambda x: position_in_month(datetime.date(x.year, x.month, x.day))
    )
    for d in range(month_fourier_degree):
        data[f"cos{d+1}"] = (2 * (d + 1) * math.pi * month_pos).apply(math.cos)
        data[f"sin{d + 1}"] = (2 * (d + 1) * math.pi * month_pos).apply(math.sin)

    drop_cols = ["time"]
    for col in data.select_dtypes(include=["floating"]).columns:
        if data[col].std() < 1e-10:
            drop_cols.append(col)

    data = data.drop(columns=drop_cols)

    return data


def add_naive_date_features(timestamps: pd.Series, month_fourier_degree: int = 2):

    df_with_date_features = naive_date_features(
        pd.to_datetime(timestamps), month_fourier_degree
    )

    holidays = create_holidays(timestamps)[
        [
            "is_UK_holiday",
            # "uk_holiday_type",
            "is_US_holiday",
            # "us_holiday_type",
        ]
    ]
    data = pd.concat([df_with_date_features, holidays], axis=1)

    return data


@lru_cache(maxsize=4096)
def position_in_month(d: datetime.date):
    prev = datetime.date(d.year, d.month, 1) - datetime.timedelta(days=1)
    nxt = datetime.date(
        d.year + 1 if d.month == 12 else d.year, 1 if d.month == 12 else d.month + 1, 1
    ) - datetime.timedelta(days=1)
    delta = (d - prev).days / (nxt - prev).days
    return delta


def country_holidays(time_column_series: pd.Series, country: str):
    country_lookup = {"UK": holidays.UK(), "US": holidays.US()}

    data = {}
    hols = country_lookup[country]

    data[f"is_{country}_holiday"] = (
        time_column_series.apply(lambda x: 1 if x.date() in hols else 0)
        .astype(str)
        .astype("category")
    )

    # data[f"{country}_holiday_type"] = (
    #     time_column_series.apply(lambda x: hols[x.date()] if x.date() in hols else 0)
    #     .astype(str)
    #     .astype("category")
    # )
    return data


def create_holidays(time_column_series: pd.Series):

    uk_holidays = country_holidays(time_column_series, "UK")
    us_holidays = country_holidays(time_column_series, "US")

    data = pd.DataFrame({**uk_holidays, **us_holidays})

    return data


if __name__ == "__main__":
    y = pd.Series(
        name="date", data=pd.date_range(start="1/1/2018", periods=300, freq="H")
    )
    f = add_naive_date_features(y, 3)
    print("yahoo!")
