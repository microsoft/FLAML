import copy
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict

import pandas as pd
import numpy as np


@dataclass
class BasicDataset:
    data: pd.DataFrame
    metadata: dict


@dataclass
class TimeSeriesDataset:
    train_data: pd.DataFrame
    time_idx: str
    time_col: str
    target_names: List[str]
    frequency: str
    time_varying_known_categoricals: List[str] = field(default_factory=lambda: [])
    time_varying_known_reals: List[str] = field(default_factory=lambda: [])
    time_varying_unknown_categoricals: List[str] = field(default_factory=lambda: [])
    time_varying_unknown_reals: List[str] = field(default_factory=lambda: [])
    test_data: Optional[pd.DataFrame] = None

    @property
    def regressors(self):
        return self.time_varying_known_categoricals + self.time_varying_known_reals

    def days_to_periods_mult(self):
        freq = self.frequency
        if freq == "H":
            return 24
        elif freq == "D":
            return 1
        elif freq == "30min":
            return 48
        else:
            raise ValueError(f"Frequency '{freq}' is not supported")

    def known_features_to_floats(
        self, train: bool, drop_first: bool = True
    ) -> np.ndarray:
        # this is a bit tricky as shapes for train and test data must match, so need to encode together
        combined = pd.concat(
            [
                self.train_data,
                self.test_data,
            ],
            ignore_index=True,
        )

        cat_one_hots = pd.get_dummies(
            combined[self.time_varying_known_categoricals],
            columns=self.time_varying_known_categoricals,
            drop_first=drop_first,
        ).values.astype(float)

        reals = combined[self.time_varying_known_reals].values.astype(float)
        both = np.concatenate([reals, cat_one_hots], axis=1)

        if train:
            return both[: len(self.train_data)]
        else:
            return both[len(self.train_data) :]

    def unique_dimension_values(self) -> np.ndarray:
        # this is the same set for train and test data, by construction
        return self.combine_dims(self.train_data).unique()

    def combine_dims(self, df):
        return df.apply(lambda row: tuple([row[d] for d in self.dimensions]), axis=1)

    def to_univariate(self) -> Dict[str, "TimeSeriesDataset"]:
        """
        Convert a multivariate TrainingData  to a dict of univariate ones
        @param df:
        @return:
        """

        train_dims = self.combine_dims(self.train_data)
        test_dims = self.combine_dims(self.test_data)

        out = {}
        for d in train_dims.unique():
            out[d] = copy.copy(self)
            out[d].train_data = self.train_data[train_dims == d]
            out[d].test_data = self.test_data[test_dims == d]
        return out

    def split_validation(self, days: int) -> "TimeSeriesDataset":
        out = copy.copy(self)
        last_periods = days * self.days_to_periods_mult()
        split_idx = self.train_data[self.time_idx].max() - last_periods + 1
        max_idx = self.test_data[self.time_idx].max() - last_periods + 1
        all_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        new_train = all_data[self.time_idx] < split_idx
        keep = all_data[self.time_idx] < max_idx
        out.train_data = all_data[new_train]
        out.test_data = all_data[(~new_train) & keep]
        return out

    def filter(self, filter_fun: Callable) -> "TimeSeriesDataset":
        if filter_fun is None:
            return self
        out = copy.copy(self)
        out.train_data = self.train_data[filter_fun]
        out.test_data = self.test_data[filter_fun]
        return out


def add_time_idx_new(data: BasicDataset, time_col: str) -> pd.DataFrame:
    pivoted = data.data.pivot(
        index=time_col,
        columns=data.metadata["dimensions"],
        values=data.metadata["metrics"],
    ).fillna(0.0)

    dt_index = pd.date_range(
        start=pivoted.index.min(),
        end=pivoted.index.max(),
        freq=data.metadata["frequency"],
    )
    indexes = pd.DataFrame(dt_index).reset_index()
    indexes.columns = ["time_idx", time_col]

    # this join is just to make sure that we get all the row timestamps in
    out = pd.merge(
        indexes, pivoted, left_on=time_col, right_index=True, how="left"
    ).fillna(0.0)

    # now flatten back

    melted = pd.melt(out, id_vars=[time_col, "time_idx"])

    for i, d in enumerate(["metric"] + data.metadata["dimensions"]):
        melted[d] = melted["variable"].apply(lambda x: x[i])

    # and finally, move metrics to separate columns
    re_pivoted = melted.pivot(
        index=["time_idx", time_col] + data.metadata["dimensions"],
        columns="metric",
        values="value",
    ).reset_index()

    return re_pivoted


def enrich_data(
    data,
    periods_to_forecast: int,
    time_col: str = "TIME_BUCKET",
    known_feature_function: Optional[Callable] = None,
    unknown_feature_function: Optional[Callable] = None,
) -> TimeSeriesDataset:

    dimension_values = data.data.apply(
        lambda row: tuple([row[d] for d in data.metadata["dimensions"]]), axis=1
    ).unique()

    # 2. fill in missing time periods and values, add integer time index
    df_with_idx = add_time_idx_new(data, time_col)

    # 3. Generate timestamps and time index for forecasting
    freq = data.metadata["frequency"]
    if freq == "H":
        forecasting_end = df_with_idx[time_col].max() + datetime.timedelta(
            hours=periods_to_forecast * 24
        )
    elif freq == "D":
        forecasting_end = df_with_idx[time_col].max() + datetime.timedelta(
            days=periods_to_forecast
        )
    elif freq == "30min":
        forecasting_end = df_with_idx[time_col].max() + datetime.timedelta(
            minutes=30 * 24 * periods_to_forecast
        )
    else:
        raise ValueError(f"Unknown frequency {freq}")

    dt_index = pd.date_range(
        start=df_with_idx[time_col].min(),
        end=forecasting_end,
        freq=data.metadata["frequency"],
    )
    indexes = pd.DataFrame(dt_index).reset_index()
    indexes.columns = ["time_idx", time_col]
    # now multiplex that for every dimension combination
    dfs = []
    for dims in dimension_values:
        this_df = indexes.copy()
        for d_name, d_value in zip(data.metadata["dimensions"], dims):
            this_df[d_name] = d_value
        dfs.append(this_df)

    dfs_all = pd.concat(dfs)

    # 4.5 split that df into train and test dataframes
    if known_feature_function is None:
        known_df = dfs_all
        known_categoricals = []
        known_reals = ["time_idx"]
    else:

        (
            known_df,
            known_categoricals,
            known_reals,
        ) = known_feature_function(dfs_all, time_col=time_col)
        known_reals = list(set(known_reals + ["time_idx"]))

    pre_train_df_ = known_df[known_df[time_col] <= df_with_idx[time_col].max()]
    # merge the metric values back in
    pre_train_df = pd.merge(
        df_with_idx,
        pre_train_df_,
        on=[time_col, "time_idx"] + data.metadata["dimensions"],
    )

    test_df = known_df[known_df[time_col] > df_with_idx[time_col].max()]

    # 5. Enrich train dataset with unknown features (eg currency volumes for call volumes)
    if unknown_feature_function is None:
        train_df = pre_train_df
        unknown_categoricals = []
        unknown_reals = []
    else:
        train_df, unknown_categoricals, unknown_reals = unknown_feature_function(
            pre_train_df
        )

    out = TimeSeriesDataset(
        train_data=train_df,
        test_data=test_df,
        time_idx="time_idx",
        time_col=time_col,
        metadata=data.metadata,
        time_varying_known_categoricals=known_categoricals,
        time_varying_unknown_categoricals=unknown_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
    )

    return out


def enrich_unrelated_values(
    main_df: pd.DataFrame, extra_data: BasicDataset, columns_name: str, values_name: str
):
    time_col = extra_data.metadata["time_col"]
    df = (
        extra_data.data.pivot(
            index=time_col,
            columns=columns_name,
            values=values_name,
        )
        .fillna(0.0)
        .reset_index()
    )

    unknown_reals = list(df.columns)[1:]
    df1 = pd.merge(main_df, df, how="left", on=[time_col])
    for c in unknown_reals:
        df1[c] = df1[c].fillna(0.0)

    return df1, [], unknown_reals
