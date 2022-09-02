import copy
import datetime
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable, Dict, Union

import pandas as pd
import numpy as np
from pandas import DataFrame, Series


from .feature import naive_date_features

# @dataclass
# class BasicDataset:
#     data: pd.DataFrame
#     metadata: dict


@dataclass
class TimeSeriesDataset:
    train_data: pd.DataFrame
    time_idx: str
    time_col: str
    target_names: List[str]
    frequency: str  # TODO: should be derived from dataset
    time_varying_known_categoricals: List[str] = field(default_factory=lambda: [])
    time_varying_known_reals: List[str] = field(default_factory=lambda: [])
    time_varying_unknown_categoricals: List[str] = field(default_factory=lambda: [])
    time_varying_unknown_reals: List[str] = field(default_factory=lambda: [])
    test_data: Optional[pd.DataFrame] = None

    def __init__(
        self,
        train_data: pd.DataFrame,
        time_col: str,
        target_names: Union[str, list[str]],
        time_idx: str = "index",
        test_data: Optional[pd.DataFrame] = None,
    ):
        self.train_data = train_data
        self.time_col = time_col
        self.time_idx = time_idx
        self.target_names = (
            [target_names] if isinstance(target_names, str) else target_names
        )
        self.frequency = pd.infer_freq(train_data[time_col])
        assert (
            self.frequency is not None
        ), "Only time series of regular frequency are currently supported."

        float_cols = list(train_data.select_dtypes(include=["floating"]).columns)
        self.time_varying_known_reals = list(set(float_cols) - set(self.target_names))

        self.time_varying_known_categoricals = list(
            set(train_data.columns)
            - set(self.time_varying_known_reals)
            - set(self.target_names)
            - {time_col}
        )
        if test_data is not None:
            self.test_data = test_data
        else:
            self.test_data = pd.DataFrame(columns=self.train_data.columns)

    @property
    def all_data(self):
        return pd.concat([self.train_data, self.test_data], axis=0)

    @property
    def regressors(self):
        return self.time_varying_known_categoricals + self.time_varying_known_reals

    def next_scale(self) -> int:
        # TODO get from self.frequency()
        return 7

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

    def move_validation_boundary(self, steps: int) -> "TimeSeriesDataset":
        out = copy.copy(self)
        if steps > 0:
            out.train_data = pd.concat([self.train_data, self.test_data[:steps]])
            out.test_data = self.test_data[steps:]
        elif steps < 0:
            out.train_data = self.train_data[:steps]
            out.test_data = pd.concat([self.train_data[steps:], self.test_data])

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

    def prettify_prediction(self, y_pred: Union[pd.DataFrame, pd.Series, np.ndarray]):
        if self.test_data is not None and len(self.test_data):
            assert len(y_pred) == len(self.test_data)

            if isinstance(y_pred, np.ndarray):
                y_pred = pd.DataFrame(
                    data=y_pred, columns=self.target_names, index=self.test_data.index
                )
            elif isinstance(y_pred, pd.Series):
                assert len(self.target_names) == 1, "Not enough columns in y_pred"
                y_pred = pd.DataFrame(y_pred).rename(
                    columns={y_pred.name: self.target_names[0]}
                )
                y_pred.index = self.test_data.index
            elif isinstance(y_pred, pd.DataFrame):
                y_pred.index = self.test_data.index

            if self.time_col not in y_pred.columns:
                y_pred[self.time_col] = self.test_data[self.time_col]

        else:
            if isinstance(y_pred, np.ndarray):
                raise ValueError("Can't enrich np.ndarray as self.test_data is None")
            elif isinstance(y_pred, pd.Series):
                assert len(self.target_names) == 1, "Not enough columns in y_pred"
                y_pred = pd.DataFrame({self.target_names[0]: y_pred})
            # TODO auto-create the timestamps for the time column instead of throwing
            raise NotImplementedError(
                "Need a non-None test_data for this to work, for now"
            )

        assert isinstance(y_pred, pd.DataFrame)
        assert self.time_col in y_pred.columns
        assert all([t in y_pred.columns for t in self.target_names])
        return y_pred

    def merge_prediction_with_target(
        self, y_pred: Union[pd.DataFrame, pd.Series, np.ndarray]
    ):
        y_pred = self.prettify_prediction(y_pred)
        return pd.concat(
            [self.train_data[[self.time_col] + self.target_names], y_pred], axis=0
        )


# def add_time_idx_new(data: BasicDataset, time_col: str) -> pd.DataFrame:
#     pivoted = data.data.pivot(
#         index=time_col,
#         columns=data.metadata["dimensions"],
#         values=data.metadata["metrics"],
#     ).fillna(0.0)
#
#     dt_index = pd.date_range(
#         start=pivoted.index.min(),
#         end=pivoted.index.max(),
#         freq=data.metadata["frequency"],
#     )
#     indexes = pd.DataFrame(dt_index).reset_index()
#     indexes.columns = ["time_idx", time_col]
#
#     # this join is just to make sure that we get all the row timestamps in
#     out = pd.merge(
#         indexes, pivoted, left_on=time_col, right_index=True, how="left"
#     ).fillna(0.0)
#
#     # now flatten back
#
#     melted = pd.melt(out, id_vars=[time_col, "time_idx"])
#
#     for i, d in enumerate(["metric"] + data.metadata["dimensions"]):
#         melted[d] = melted["variable"].apply(lambda x: x[i])
#
#     # and finally, move metrics to separate columns
#     re_pivoted = melted.pivot(
#         index=["time_idx", time_col] + data.metadata["dimensions"],
#         columns="metric",
#         values="value",
#     ).reset_index()
#
#     return re_pivoted


def enrich(
    X: Union[TimeSeriesDataset, pd.DataFrame], fourier_degree: int, time_col: str
):
    if isinstance(X, TimeSeriesDataset):
        return enrich_dataset(X, fourier_degree)
    else:
        return enrich_dataframe(X, time_col, fourier_degree)


def enrich_dataframe(
    df: pd.DataFrame, time_col: str, fourier_degree: int
) -> pd.DataFrame:
    extras = naive_date_features(df[time_col], fourier_degree)
    extras.columns = [f"{time_col}_{c}" for c in extras.columns]
    extras.index = df.index

    return pd.concat([df, extras], axis=1)


def enrich_dataset(X: TimeSeriesDataset, fourier_degree: int = 0) -> TimeSeriesDataset:
    new_train = enrich_dataframe(X.train_data, X.time_col, fourier_degree)
    new_test = (
        None
        if X.test_data is None
        else enrich_dataframe(X.test_data, X.time_col, fourier_degree)
    )
    return TimeSeriesDataset(
        train_data=new_train,
        time_col=X.time_col,
        target_names=X.target_names,
        time_idx=X.time_idx,
        test_data=new_test,
    )


def enrich_data_old(
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


#
# def enrich_unrelated_values(
#     main_df: pd.DataFrame, extra_data: BasicDataset, columns_name: str, values_name: str
# ):
#     time_col = extra_data.metadata["time_col"]
#     df = (
#         extra_data.data.pivot(
#             index=time_col,
#             columns=columns_name,
#             values=values_name,
#         )
#         .fillna(0.0)
#         .reset_index()
#     )
#
#     unknown_reals = list(df.columns)[1:]
#     df1 = pd.merge(main_df, df, how="left", on=[time_col])
#     for c in unknown_reals:
#         df1[c] = df1[c].fillna(0.0)
#
#     return df1, [], unknown_reals


class DataTransformerTS:
    """Transform input time series training data."""

    def __init__(self, time_col: str, label: Union[str, List[str]]):
        self.time_col = time_col
        self.label = label

    def fit_transform(self, X: Union[DataFrame, np.array], y):
        """Fit transformer and process the input training data according to the task type.

        Args:
            X: A numpy array or a pandas dataframe of training data.
            y: A numpy array or a pandas series of labels.
            task: A string of the task type, e.g.,
                'classification', 'regression', 'ts_forecast', 'rank'.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
            y: Processed numpy array or pandas series of labels.
        """
        if isinstance(X, DataFrame):
            X = X.copy()
            n = X.shape[0]
            cat_columns, num_columns, datetime_columns = [], [], []
            drop = False
            ds_col = X.pop(self.time_col)
            if isinstance(y, Series):
                y = y.rename(self.label)
            for column in X.columns:
                # sklearn/utils/validation.py needs int/float values
                if X[column].dtype.name in ("object", "category"):
                    if (
                        X[column].nunique() == 1
                        or X[column].nunique(dropna=True)
                        == n - X[column].isnull().sum()
                    ):
                        X.drop(columns=column, inplace=True)
                        drop = True
                    elif X[column].dtype.name == "category":
                        current_categories = X[column].cat.categories
                        if "__NAN__" not in current_categories:
                            X[column] = (
                                X[column]
                                .cat.add_categories("__NAN__")
                                .fillna("__NAN__")
                            )
                        cat_columns.append(column)
                    else:
                        X[column] = X[column].fillna("__NAN__")
                        cat_columns.append(column)
                elif X[column].nunique(dropna=True) < 2:
                    X.drop(columns=column, inplace=True)
                    drop = True
                else:  # datetime or numeric
                    if X[column].dtype.name == "datetime64[ns]":
                        tmp_dt = X[column].dt
                        new_columns_dict = {
                            f"year_{column}": tmp_dt.year,
                            f"month_{column}": tmp_dt.month,
                            f"day_{column}": tmp_dt.day,
                            f"hour_{column}": tmp_dt.hour,
                            f"minute_{column}": tmp_dt.minute,
                            f"second_{column}": tmp_dt.second,
                            f"dayofweek_{column}": tmp_dt.dayofweek,
                            f"dayofyear_{column}": tmp_dt.dayofyear,
                            f"quarter_{column}": tmp_dt.quarter,
                        }
                        for key, value in new_columns_dict.items():
                            if (
                                key not in X.columns
                                and value.nunique(dropna=False) >= 2
                            ):
                                X[key] = value
                                num_columns.append(key)
                        X[column] = X[column].map(datetime.toordinal)
                        datetime_columns.append(column)
                        del tmp_dt
                    X[column] = X[column].fillna(np.nan)
                    num_columns.append(column)
            X = X[cat_columns + num_columns]
            X.insert(0, self.time_col, ds_col)
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns]
                if np.issubdtype(X_num.columns.dtype, np.integer) and (
                    drop
                    or min(X_num.columns) != 0
                    or max(X_num.columns) != X_num.shape[1] - 1
                ):
                    X_num.columns = range(X_num.shape[1])
                    drop = True
                else:
                    drop = False
                from sklearn.impute import SimpleImputer
                from sklearn.compose import ColumnTransformer

                self.transformer = ColumnTransformer(
                    [
                        (
                            "continuous",
                            SimpleImputer(missing_values=np.nan, strategy="median"),
                            X_num.columns,
                        )
                    ]
                )
                X[num_columns] = self.transformer.fit_transform(X_num)
            self._cat_columns, self._num_columns, self._datetime_columns = (
                cat_columns,
                num_columns,
                datetime_columns,
            )
            self._drop = drop

        # TODO: revisit for multivariate series, and recast for a single df input anyway
        ycol = y[y.columns[0]]
        if not pd.api.types.is_numeric_dtype(ycol):
            from sklearn.preprocessing import LabelEncoder

            self.label_transformer = LabelEncoder()
            y_tr = self.label_transformer.fit_transform(ycol)
            y.iloc[:] = y_tr.reshape(y.shape)
        else:
            self.label_transformer = None
        return X, y

    def transform(self, X: Union[DataFrame, np.array]):
        """Process data using fit transformer.

        Args:
            X: A numpy array or a pandas dataframe of training data.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
        """
        X = X.copy()

        if isinstance(X, DataFrame):
            cat_columns, num_columns, datetime_columns = (
                self._cat_columns,
                self._num_columns,
                self._datetime_columns,
            )
            ds_col = X.pop(self.time_col)
            for column in datetime_columns:
                tmp_dt = X[column].dt
                new_columns_dict = {
                    f"year_{column}": tmp_dt.year,
                    f"month_{column}": tmp_dt.month,
                    f"day_{column}": tmp_dt.day,
                    f"hour_{column}": tmp_dt.hour,
                    f"minute_{column}": tmp_dt.minute,
                    f"second_{column}": tmp_dt.second,
                    f"dayofweek_{column}": tmp_dt.dayofweek,
                    f"dayofyear_{column}": tmp_dt.dayofyear,
                    f"quarter_{column}": tmp_dt.quarter,
                }
                for new_col_name, new_col_value in new_columns_dict.items():
                    if new_col_name not in X.columns and new_col_name in num_columns:
                        X[new_col_name] = new_col_value
                X[column] = X[column].map(datetime.toordinal)
                del tmp_dt
            X = X[cat_columns + num_columns].copy()
            X.insert(0, self.time_col, ds_col)
            for column in cat_columns:
                if X[column].dtype.name == "object":
                    X[column] = X[column].fillna("__NAN__")
                elif X[column].dtype.name == "category":
                    current_categories = X[column].cat.categories
                    if "__NAN__" not in current_categories:
                        X[column] = (
                            X[column].cat.add_categories("__NAN__").fillna("__NAN__")
                        )
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns].fillna(np.nan)
                if self._drop:
                    X_num.columns = range(X_num.shape[1])
                X[num_columns] = self.transformer.transform(X_num)
        return X
