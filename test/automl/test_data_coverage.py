"""Tests to improve coverage for flaml/automl/data.py."""

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse

from flaml.automl.data import (
    DataTransformer,
    add_time_idx_col,
    auto_convert_dtypes_pandas,
    concat,
    get_output_from_log,
    get_random_dataframe,
    load_openml_dataset,
)


# Module-level classes for pickling support
class _FakeDatasetCached:
    name = "test_ds"
    default_target_attribute = "target"

    def get_data(self, target=None, dataset_format=None):
        X = pd.DataFrame({"a": range(100), "b": range(100)})
        y = pd.Series(np.random.randint(0, 2, 100), name="target")
        return X, y, None, None


class _FakeDatasetDownload:
    name = "test_ds"
    default_target_attribute = "target"

    def get_data(self, target=None, dataset_format=None):
        X = pd.DataFrame({"a": range(40), "b": range(40)})
        y = pd.Series(np.random.randint(0, 2, 40))
        return X, y, None, None


class _FakeDatasetFallback:
    name = "fallback_ds"
    default_target_attribute = "target"

    def get_data(self, target=None, dataset_format=None):
        raise ValueError("bad")


# ---------- concat utility ----------


class TestConcat:
    def test_concat_sparse_matrices(self):
        """Cover line 234-235: issparse branch in concat."""
        a = csr_matrix(np.array([[1, 2], [3, 4]]))
        b = csr_matrix(np.array([[5, 6]]))
        result = concat(a, b)
        assert issparse(result)
        assert result.shape == (3, 2)

    def test_concat_numpy_arrays(self):
        """Cover line 237: np.concatenate branch."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6]])
        result = concat(a, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_concat_mismatched_types_to_dataframe(self):
        """Cover lines 216-217: type mismatch falls back to pd.DataFrame."""
        a = np.array([[1, 2], [3, 4]])
        b = pd.DataFrame({"x": [5], "y": [6]})
        result = concat(a, b)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_concat_series(self):
        """Cover lines 219-226: concat two Series."""
        a = pd.Series([1, 2, 3], name="val")
        b = pd.Series([4, 5], name="val")
        result = concat(a, b)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_concat_dataframes_with_category(self):
        """Cover lines 222-226: concat DataFrames preserving category dtype."""
        df1 = pd.DataFrame({"cat": pd.Categorical(["a", "b"]), "num": [1, 2]})
        df2 = pd.DataFrame({"cat": pd.Categorical(["c"]), "num": [3]})
        result = concat(df1, df2)
        assert result["cat"].dtype.name == "category"
        assert len(result) == 3


# ---------- load_openml_dataset ----------


class TestLoadOpenmlDataset:
    def test_load_openml_dataset_cached(self):
        """Cover lines 62-65: load from cached pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "openml_ds999.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(_FakeDatasetCached(), f)

            mock_openml = MagicMock()
            with patch.dict("sys.modules", {"openml": mock_openml}):
                X_train, X_test, y_train, y_test = load_openml_dataset(999, data_dir=tmpdir)
            assert len(X_train) + len(X_test) == 100

    def test_load_openml_dataset_download(self):
        """Cover lines 67-72: download and cache; also line 70 (makedirs)."""
        mock_openml = MagicMock()
        mock_openml.datasets.get_dataset.return_value = _FakeDatasetDownload()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "subdir")
            with patch.dict("sys.modules", {"openml": mock_openml}):
                X_train, X_test, y_train, y_test = load_openml_dataset(998, data_dir=data_dir)
                assert os.path.isfile(os.path.join(data_dir, "openml_ds998.pkl"))
                assert len(X_train) + len(X_test) == 40

    def test_load_openml_dataset_fallback(self):
        """Cover lines 76-79: fallback to fetch_openml on ValueError."""
        X = pd.DataFrame({"a": range(40), "b": range(40)})
        y = pd.Series(np.random.randint(0, 2, 40), name="target")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "openml_ds997.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(_FakeDatasetFallback(), f)

            mock_openml = MagicMock()
            with patch.dict("sys.modules", {"openml": mock_openml}):
                with patch("sklearn.datasets.fetch_openml", return_value=(X, y)) as mock_fetch:
                    X_train, X_test, y_train, y_test = load_openml_dataset(997, data_dir=tmpdir)
                    mock_fetch.assert_called_once_with(data_id=997, return_X_y=True)


# ---------- DataTransformer ----------


class TestDataTransformer:
    def test_fit_transform_with_datetime_column(self):
        """Cover datetime expansion (lines 321-340, 344-345 for ts_forecast)."""
        dt = DataTransformer()
        X = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2020-01-01", "2020-02-15", "2020-06-20", "2021-01-01"]),
                "val": [1, 2, 3, 4],
            }
        )
        y = pd.Series([0, 1, 0, 1])
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert "year_dt" in X_out.columns or "dt" in X_out.columns
        assert len(y_out) == 4

    def test_fit_transform_drop_single_value_columns(self):
        """Cover lines 306-308, 317-319: drop cols with nunique==1 or <2."""
        dt = DataTransformer()
        X = pd.DataFrame(
            {
                "const_cat": ["same"] * 20,
                "const_num": [5.0] * 20,
                "useful": np.random.randn(20),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 20))
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert "const_cat" not in X_out.columns
        assert "const_num" not in X_out.columns

    def test_fit_transform_category_with_nan(self):
        """Cover lines 309-316: category and object columns with NaN."""
        dt = DataTransformer()
        X = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["a", "b", None, "a", "b", "a", "b", "a"]),
                "obj_col": ["x", "y", None, "x", "y", "x", "y", "x"],
                "num": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert X_out["cat_col"].dtype.name == "category"

    def test_transform_preserves_structure(self):
        """Cover lines 400-451: DataTransformer.transform."""
        dt = DataTransformer()
        X = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
                "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        dt.fit_transform(X, y, "classification")

        X_new = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b"]),
                "num": [10.0, np.nan],
            }
        )
        X_transformed = dt.transform(X_new)
        assert len(X_transformed) == 2

    def test_fit_transform_regression_no_label_encoder(self):
        """Cover line 387: label_transformer is None for regression."""
        dt = DataTransformer()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([1.5, 2.5, 3.5, 4.5])
        X_out, y_out = dt.fit_transform(X, y, "regression")
        assert dt.label_transformer is None

    def test_fit_transform_column_with_all_unique_strings(self):
        """Cover line 306: object column with nunique == n (all unique) gets dropped."""
        dt = DataTransformer()
        n = 10
        X = pd.DataFrame(
            {
                "unique_str": [f"val_{i}" for i in range(n)],
                "num": np.arange(n, dtype=float),
            }
        )
        y = pd.Series(np.random.randint(0, 2, n))
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert "unique_str" not in X_out.columns

    def test_fit_transform_numpy_input(self):
        """Cover branch where X is ndarray (not DataFrame)."""
        dt = DataTransformer()
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert isinstance(X_out, np.ndarray)

    def test_fit_transform_integer_column_renumbering(self):
        """Cover lines 350-354: integer columns that need renumbering."""
        dt = DataTransformer()
        X = pd.DataFrame(
            {
                10: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                20: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        X_out, y_out = dt.fit_transform(X, y, "classification")
        assert dt._drop is True


# ---------- add_time_idx_col ----------


class TestAddTimeIdxCol:
    def test_monthly_frequency(self):
        """Cover lines 244-245: freq == 'MS'."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        X = pd.DataFrame({"ds": dates, "val": range(12)})
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns

    def test_yearly_frequency(self):
        """Cover lines 246-247: freq == 'Y'."""
        dates = pd.date_range("2015-01-01", periods=5, freq="YS")
        X = pd.DataFrame({"ds": dates, "val": range(5)})
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert result["time_idx"].tolist() == [2015, 2016, 2017, 2018, 2019]

    def test_daily_frequency(self):
        """Cover lines 253-257: other freq (daily)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        X = pd.DataFrame({"ds": dates, "val": range(30)})
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert result["time_idx"].tolist() == list(range(30))

    def test_business_day_with_holidays(self):
        """Irregular data: business days with US holidays removed (stock data)."""
        us_holidays_h1_2024 = pd.to_datetime(
            [
                "2024-01-15",  # MLK Day
                "2024-02-19",  # Presidents' Day
                "2024-03-29",  # Good Friday
                "2024-05-27",  # Memorial Day
            ]
        )
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        dates = dates[~dates.isin(us_holidays_h1_2024)]
        X = pd.DataFrame({"ds": dates, "price": np.arange(len(dates), dtype=float)})

        result = add_time_idx_col(X)

        assert "time_idx" in result.columns
        assert result["time_idx"].is_monotonic_increasing
        assert result["time_idx"].iloc[0] == 0
        # Step is 1 calendar day, so final time_idx == calendar span.
        expected_span = (dates.max() - dates.min()).days
        assert result["time_idx"].iloc[-1] == expected_span
        # Holidays produce gaps.
        assert len(result) < expected_span + 1

    def test_two_unique_dates(self):
        """Fewer than 3 dates: fall back instead of raising."""
        X = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "val": [10, 20],
            }
        )
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert result["time_idx"].tolist() == [0, 1]

    def test_single_unique_date(self):
        """A single unique timestamp maps every row to 0."""
        X = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
                "val": [1, 2, 3],
            }
        )
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert (result["time_idx"] == 0).all()

    def test_panel_data_with_duplicate_dates(self):
        """Panel data: each row gets time_idx based on its date."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df_a = pd.DataFrame({"ds": dates, "sku": "A", "y": np.arange(5)})
        df_b = pd.DataFrame({"ds": dates, "sku": "B", "y": np.arange(5) + 100})
        X = pd.concat([df_a, df_b], ignore_index=True)
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        per_date = result.groupby("ds")["time_idx"].nunique()
        assert (per_date == 1).all()
        for _, grp in result.groupby("sku"):
            assert sorted(grp["time_idx"].tolist()) == [0, 1, 2, 3, 4]

    def test_unsorted_input(self):
        """Unsorted rows still produce correct per-row indices."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({"ds": dates, "val": np.arange(10)})
        X_shuffled = X.sample(frac=1, random_state=0).reset_index(drop=True)
        result = add_time_idx_col(X_shuffled)
        for ts, idx in zip(result["ds"], result["time_idx"]):
            assert idx == (ts - dates.min()).days

    def test_nat_timestamps_are_dropped_from_inference(self):
        """Rows whose timestamp is NaT must not poison the int64-cast path
        with the sentinel min-int64. Valid rows still receive sensible
        non-negative time_idx values, and NaT rows get pd.NA rather than
        the integer-overflow garbage of the unmasked cast."""
        X = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01", "2024-01-02", None, "2024-01-04", "2024-01-05"]),
                "val": np.arange(5),
            }
        )
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        valid_mask = result["ds"].notna()
        # All valid rows have non-negative, monotonically increasing indices
        # rooted at the earliest non-NaT timestamp.
        assert (result.loc[valid_mask, "time_idx"] >= 0).all()
        assert result.loc[valid_mask, "time_idx"].is_monotonic_increasing
        assert result.loc[valid_mask, "time_idx"].iloc[0] == 0
        # NaT rows: time_idx is pd.NA, not a garbage integer.
        assert result.loc[~valid_mask, "time_idx"].isna().all()

    def test_tz_aware_timestamps(self):
        """Timezone-aware datetime columns must produce the same 0..n-1
        sequential indices as their tz-naive counterparts."""
        X = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "val": np.arange(10),
            }
        )
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert result["time_idx"].tolist() == list(range(10))

    def test_tz_aware_when_infer_freq_raises_typeerror(self, monkeypatch):
        """Older pandas versions raise ``TypeError`` (not ``ValueError``) when
        ``pd.infer_freq`` is called on a tz-aware ``Series``. The fallback
        mode-of-deltas branch must still produce sequential 0..n-1 indices."""

        def fake_infer_freq(idx):
            raise TypeError("cannot infer freq from a non-convertible dtype on a Series of datetime64[ns, UTC]")

        monkeypatch.setattr("flaml.automl.data.pd.infer_freq", fake_infer_freq)
        X = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "val": np.arange(10),
            }
        )
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns
        assert result["time_idx"].tolist() == list(range(10))


# ---------- auto_convert_dtypes_pandas ----------


class TestAutoConvertDtypesPandas:
    def test_numeric_conversion(self):
        """Cover lines 754-764: numeric int and double conversion."""
        df = pd.DataFrame({"ints": ["1", "2", "3", "4"], "floats": ["1.1", "2.2", "3.3", "4.4"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["ints"] == "int"
        assert schema["floats"] == "double"

    def test_datetime_conversion(self):
        """Cover lines 768-772: datetime conversion."""
        df = pd.DataFrame({"dates": ["2020-01-01", "2020-02-01", "2020-03-01"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["dates"] == "timestamp"

    def test_category_conversion(self):
        """Cover lines 786-790: category when low cardinality."""
        df = pd.DataFrame({"cat": ["a", "a", "b", "b", "a", "b", "a", "b", "a", "b"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["cat"] == "category"

    def test_keep_non_object_dtype(self):
        """Cover lines 746-748: skip conversion for non-object dtypes."""
        df = pd.DataFrame({"num": [1, 2, 3, 4]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert "num" in schema

    def test_na_replacement(self):
        """Cover line 719: empty pattern branch."""
        df = pd.DataFrame({"col": ["a", "NA", "b", "null", "c"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert "col" in schema

    def test_sample_ratio(self):
        """``sample_ratio`` only changes how many rows feed the inference
        heuristics; the converted DataFrame must still contain every input
        row. Previously the sampled subset was written back to the full-size
        result, silently coercing non-sampled rows to NaN.

        Uses ``n=300`` and ``sample_ratio=0.5`` so that ``n * sample_ratio
        = 150 > 100`` and the floor formula does NOT clamp the ratio up to
        1.0 — sampling actually takes effect (150-row inference subset).
        """
        n = 300
        df = pd.DataFrame({"col": [str(i) for i in range(n)]})
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.5)
        assert schema["col"] == "int"
        assert len(result) == n
        assert result["col"].isna().sum() == 0
        assert result["col"].tolist() == list(range(n))

    def test_timedelta_conversion(self):
        """Cover lines 776-782: timedelta conversion."""
        df = pd.DataFrame({"td": ["1 days", "2 days", "3 days", "4 days"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["td"] in ("timedelta", "int")

    def test_string_fallback(self):
        """Cover line 793-794: string fallback for high-cardinality strings."""
        df = pd.DataFrame({"uid": [f"unique_{i}" for i in range(20)]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["uid"] == "string"

    def test_boolean_dtype_converted_to_int(self):
        """Boolean columns are converted to ``Int64`` (nullable int).

        Leaving the column as ``boolean`` (or coercing it to a bool-typed
        Categorical) trips sklearn's pandas early-conversion path in
        ``check_array`` and breaks ``SimpleImputer`` whenever the same
        ``ColumnTransformer`` also contains string-typed categoricals.
        FLAML's ``DataTransformer`` treats booleans as numeric features
        anyway, so emitting ``Int64`` (True→1, False→0, NA→pd.NA) is
        consistent end-to-end and avoids the sklearn issue.
        """
        df = pd.DataFrame(
            {
                "flag": pd.array(
                    [True, False, True, False, True, False, True, False, True, pd.NA],
                    dtype="boolean",
                )
            }
        )
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["flag"] == "int"
        assert isinstance(result["flag"].dtype, pd.Int64Dtype)
        # Value mapping preserved, including NA round-trip.
        expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, pd.NA]
        got = result["flag"].tolist()
        for g, e in zip(got, expected):
            if pd.isna(e):
                assert pd.isna(g)
            else:
                assert g == e

    def test_boolean_mixed_with_string_categories_sklearn_imputer(self):
        """Regression for the user-reported sklearn interop failure:

        With ``employment_stable`` cast to ``boolean`` and the rest of the
        frame containing string-typed categoricals (e.g., ``home_ownership``,
        ``loan_purpose``), passing the result through a
        ``ColumnTransformer`` of ``SimpleImputer`` instances used to fail
        with ``ValueError: Cannot cast string dtype to float64`` because the
        boolean column carried ``is_bool_dtype``-True semantics. With the
        bool→``Int64`` conversion, the column is plain numeric and the
        fit succeeds without any caller-side workaround.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        n = 40
        df = pd.DataFrame(
            {
                "home_ownership": (["MORTGAGE", "RENT", "OWN"] * (n // 3 + 1))[:n],
                "loan_purpose": (["auto", "medical", "home"] * (n // 3 + 1))[:n],
                "employment_stable": pd.array([True, False] * (n // 2), dtype="boolean"),
                "age_group": (["21-30", "31-40", "41-50", "51-60", "60+"] * (n // 5))[:n],
                "annual_income": [50000 + i * 1000 for i in range(n)],
            }
        )
        df = df.convert_dtypes()
        result, schema = auto_convert_dtypes_pandas(df)

        # Boolean column is now numeric (Int64), not BooleanDtype / bool-typed
        # category.
        assert schema["employment_stable"] == "int"
        assert isinstance(result["employment_stable"].dtype, pd.Int64Dtype)

        # Mimic the user's pipeline: build a ColumnTransformer with separate
        # mean / most_frequent imputers over a mix of numeric and categorical
        # columns and ensure ``fit`` succeeds end-to-end.
        cat_features = [c for c in result.columns if isinstance(result[c].dtype, pd.CategoricalDtype)]
        num_features = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c].dtype)]
        # employment_stable should be in the numeric group.
        assert "employment_stable" in num_features
        transformers = []
        if num_features:
            transformers.append(("mean_imputer", SimpleImputer(strategy="mean"), num_features))
        if cat_features:
            transformers.append(("mode_imputer", SimpleImputer(strategy="most_frequent"), cat_features))
        ColumnTransformer(transformers=transformers).fit(result)

    def test_int_with_nan_becomes_double(self):
        """Cover line 761-763: int conversion failure falls back to double."""
        df = pd.DataFrame({"col": ["1", "2", "3.0", None, "5"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["col"] in ("int", "double")

    def test_skip_columns_preserves_listed_columns(self):
        """``skip_columns`` must leave listed columns completely untouched:
        same dtype, same values, no NA-token normalization, while other
        columns are still inferred normally."""
        df = pd.DataFrame(
            {
                # Would normally be inferred as int.
                "id": [str(i) for i in range(20)],
                # Would normally be inferred as category (low-cardinality
                # strings with NA-like tokens that would otherwise be
                # normalized to NaN).
                "raw_label": (["A", "B", "NULL", "C", "A"] * 5)[:20],
                # Not skipped — should still be inferred as category.
                "label": (["X", "Y", "Z", "X", "Y"] * 5)[:20],
            }
        )

        result, schema = auto_convert_dtypes_pandas(df, skip_columns=["id", "raw_label"])

        # Skipped columns: identical to input.
        assert schema["id"] == str(df["id"].dtype)
        assert result["id"].dtype == df["id"].dtype
        assert result["id"].tolist() == df["id"].tolist()
        assert schema["raw_label"] == str(df["raw_label"].dtype)
        assert result["raw_label"].dtype == df["raw_label"].dtype
        # NA-like token "NULL" preserved as a string, not coerced to NaN.
        assert result["raw_label"].tolist() == df["raw_label"].tolist()

        # Non-skipped column: still inferred.
        assert schema["label"] == "category"
        assert isinstance(result["label"].dtype, pd.CategoricalDtype)

    def test_skip_columns_unknown_names_are_ignored(self):
        """Column names in ``skip_columns`` that don't exist in the frame
        must be silently ignored (no KeyError)."""
        df = pd.DataFrame({"a": ["1", "2", "3", "4"] * 3})
        result, schema = auto_convert_dtypes_pandas(df, skip_columns=["does_not_exist"])
        assert schema["a"] in ("int", "category")
        assert list(result.columns) == ["a"]

    def test_skip_columns_none_matches_default_behavior(self):
        """Passing ``skip_columns=None`` (the default) must not change behavior."""
        df = pd.DataFrame(
            {
                "n": [str(i) for i in range(20)],
                "lab": (["A", "B", "A", "B"] * 5)[:20],
            }
        )
        baseline_result, baseline_schema = auto_convert_dtypes_pandas(df)
        result, schema = auto_convert_dtypes_pandas(df, skip_columns=None)
        assert schema == baseline_schema
        for col in df.columns:
            assert result[col].dtype == baseline_result[col].dtype

    def test_category_like_strings_not_timedelta(self):
        """Regression: strings like "31-40" / "60+" must NOT be parsed as timedelta.

        ``pd.to_timedelta`` is overly permissive (it parses "31-40" as
        ``31ns - 40ns`` and "60+" as ``+60ns``), which previously caused
        categorical labels (e.g., age groups, ratings) to be silently coerced
        to ``timedelta64[ns]`` instead of ``category``.
        """
        df = pd.DataFrame(
            {
                "age_group": ["21-30", "31-40", "41-50", "51-60", "60+"] * 4,
                "rating_range": ["1-2", "3-4", "5-6", "7-8", "9-10"] * 4,
                "size_bucket": ["S+", "M+", "L+", "XL+", "S+"] * 4,
            }
        )
        result, schema = auto_convert_dtypes_pandas(df)
        for col in df.columns:
            assert schema[col] == "category", (
                f"Expected '{col}' to be 'category', got '{schema[col]}' " f"(dtype={result[col].dtype})"
            )
            assert isinstance(result[col].dtype, pd.CategoricalDtype)

    def test_mixed_dataframe_all_dtypes(self):
        """End-to-end: a DataFrame with every supported kind of raw data is
        converted to the expected dtypes, including category-like strings such
        as "31-40" that would otherwise be mis-parsed as timedeltas.
        """
        n = 20
        df = pd.DataFrame(
            {
                # Plain integer strings -> int
                "int_str": [str(i) for i in range(n)],
                # Decimal strings -> double
                "float_str": [f"{i + 0.5}" for i in range(n)],
                # Native int column (non-object) is preserved as-is
                "native_int": list(range(n)),
                # ISO date strings -> timestamp
                "date_str": [f"2020-01-{(i % 28) + 1:02d}" for i in range(n)],
                # True timedelta strings (with unit) -> timedelta
                "td_str": ["1 days", "2 days", "3 hours", "30 min"] * (n // 4),
                # Low-cardinality labels -> category
                "label": (["Good", "Bad", "Excellent", "Poor"] * (n // 4 + 1))[:n],
                # The regression case: looks numeric-ish but is categorical
                "age_group": (["21-30", "31-40", "41-50", "51-60", "60+"] * (n // 5 + 1))[:n],
                # Unique high-cardinality strings -> string
                "uid": [f"uid_{i:04d}" for i in range(n)],
                # NA-like tokens mixed with a low-cardinality label -> category
                "with_na": (["x", "NA", "y", "null", "x", "y", ""] * (n // 7 + 1))[:n],
            }
        )

        result, schema = auto_convert_dtypes_pandas(df)

        assert schema["int_str"] == "int"
        assert schema["float_str"] == "double"
        # native_int is preserved as-is (non-object dtype branch)
        assert "int" in schema["native_int"].lower()
        assert schema["date_str"] == "timestamp"
        assert schema["td_str"] == "timedelta"
        assert schema["label"] == "category"
        # The key assertion for this fix: age_group is NOT timedelta.
        assert schema["age_group"] == "category", (
            f"age_group should be 'category', got '{schema['age_group']}' " f"(dtype={result['age_group'].dtype})"
        )
        assert not pd.api.types.is_timedelta64_dtype(result["age_group"])
        assert schema["uid"] == "string"
        assert schema["with_na"] == "category"

        # Verify the actual converted dtypes line up with the schema.
        assert pd.api.types.is_integer_dtype(result["int_str"])
        assert pd.api.types.is_float_dtype(result["float_str"])
        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])
        assert pd.api.types.is_timedelta64_dtype(result["td_str"])
        assert isinstance(result["label"].dtype, pd.CategoricalDtype)
        assert isinstance(result["age_group"].dtype, pd.CategoricalDtype)
        assert isinstance(result["with_na"].dtype, pd.CategoricalDtype)

    def test_timedelta_substring_false_positives(self):
        """Strings that merely *contain* a digit-colon-digit substring or a
        ``P`` prefix (e.g., ``"v1:2"``, ``"foo1:2bar"``, ``"P31-40"``) must
        not be classified as timedelta — the prefilter is fully anchored."""
        df = pd.DataFrame(
            {
                "a": ["v1:2", "v3:4", "v5:6", "v7:8", "v9:0"] * 4,
                "b": ["foo1:2bar", "foo3:4bar", "foo5:6bar", "foo7:8bar", "foo9:0bar"] * 4,
                "c": ["P31-40", "P21-30", "P41-50", "P51-60", "P60-70"] * 4,
            }
        )
        result, schema = auto_convert_dtypes_pandas(df)
        for col in df.columns:
            assert schema[col] != "timedelta", f"'{col}' wrongly classified as timedelta (dtype={result[col].dtype})"
            assert not pd.api.types.is_timedelta64_dtype(result[col])

    def test_timedelta_compact_unit_forms(self):
        """Compact multi-unit forms that ``pd.to_timedelta`` accepts (e.g.,
        ``"2h30m"``, ``"1d2h"``, ``"1d2h30m"``) should still be recognized
        as timedelta despite having no whitespace between segments."""
        df = pd.DataFrame({"td": ["2h30m", "1d2h", "1d2h30m", "5h45m", "3d12h"] * 4})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["td"] == "timedelta", schema["td"]
        assert pd.api.types.is_timedelta64_dtype(result["td"])

    def test_timedelta_clock_one_digit_minutes_seconds(self):
        """Clock-format strings with 1-digit minutes/seconds (e.g., ``"1:2:3"``,
        ``"0:0:5"``, ``"12:3:4"``) are accepted by ``pd.to_timedelta`` and
        must not be falsely rejected by the prefilter."""
        df = pd.DataFrame({"td": ["1:2:3", "0:0:5", "12:3:4", "1:23:4", "0:0:0.123"] * 4})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["td"] == "timedelta", schema["td"]
        assert pd.api.types.is_timedelta64_dtype(result["td"])

    def test_timedelta_week_units(self):
        """Week units (``w``/``W``-suffixed plain forms and ``weeks``/``week``
        multi-letter forms) should be recognised. ``pd.to_timedelta`` accepts
        ``"1w"``/``"2 w"``/``"3W"`` as weeks; multi-letter ``"1 week"`` is
        coerced to ``NaT`` by pandas but the prefilter must still allow such
        strings through so the column is evaluated rather than wholesale
        rejected on the first letter mismatch."""
        df = pd.DataFrame({"td": ["1w", "2 w", "3w", "4w", "5w"] * 4})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["td"] == "timedelta", schema["td"]
        assert pd.api.types.is_timedelta64_dtype(result["td"])
        assert result["td"].iloc[0] == pd.Timedelta(days=7)
        assert result["td"].iloc[2] == pd.Timedelta(days=21)

    def test_sample_ratio_caps_inference_but_preserves_full_data(self):
        """``sample_ratio`` bounds the cost of the inference heuristics
        on large frames while still applying conversion to every row."""
        n = 50_000
        df = pd.DataFrame(
            {
                "ints": [str(i) for i in range(n)],
                "labels": ["a", "b", "c", "d"] * (n // 4),
            }
        )
        # n * 0.01 = 500 > 100, so the floor doesn't kick in and the
        # inference subset is exactly 500 rows.
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.01)
        assert schema == {"ints": "int", "labels": "category"}
        assert len(result) == n
        assert result["ints"].isna().sum() == 0
        assert np.array_equal(result["ints"].to_numpy(), np.arange(n))
        # Categorical conversion must cover the full column, not just the sample.
        assert set(result["labels"].cat.categories) == {"a", "b", "c", "d"}

    def test_sample_ratio_one_uses_full_dataframe(self):
        """``sample_ratio=1.0`` skips sampling so inference runs on the
        full DataFrame (matches the pre-floor behavior of the function)."""
        df = pd.DataFrame({"col": [str(i) for i in range(50)]})
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=1.0)
        assert schema["col"] == "int"
        assert len(result) == 50

    def test_sampled_int_vs_double_uses_full_data(self):
        """Sample-only int detection must NOT silently truncate decimals in the
        full column. With a small inference sample and a single decimal value
        deliberately placed outside the likely sample window, the column must
        still be classified as ``double`` so ``1.5`` is preserved."""
        n = 10_000
        vals = ["1", "2", "3", "4", "5"] * (n // 5)
        # Inject a decimal at an index unlikely to be picked by a 100-row sample.
        vals[7777] = "1.5"
        df = pd.DataFrame({"col": vals})
        # n * 0.01 = 100 (NOT > 100), floor kicks in: 100/n = 0.01 → 100 rows.
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.01)
        assert schema["col"] == "double", schema["col"]
        assert result["col"].iloc[7777] == 1.5

    def test_no_sampling_for_small_frames(self):
        """Frames with ``n_rows <= 100`` are always inferred in full
        regardless of ``sample_ratio`` — the floor formula clamps the
        effective ratio to 1.0 so existing behavior is preserved."""
        df = pd.DataFrame({"col": ["a", "b"] * 10})
        # Default sample_ratio=0.1, but n=20 → 100/20=5.0 → min(1, 5)=1.0 → full.
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["col"] == "category"
        assert len(result) == 20

    def test_invalid_sample_ratio_raises(self):
        """``sample_ratio`` outside ``(0, 1]`` is rejected up front."""
        df = pd.DataFrame({"col": ["a"]})
        with pytest.raises(ValueError, match="sample_ratio"):
            auto_convert_dtypes_pandas(df, sample_ratio=0)
        with pytest.raises(ValueError, match="sample_ratio"):
            auto_convert_dtypes_pandas(df, sample_ratio=1.5)
        with pytest.raises(ValueError, match="sample_ratio"):
            auto_convert_dtypes_pandas(df, sample_ratio=-0.1)

    def test_small_frame_uses_full_regardless_of_sample_ratio(self):
        """For ``n_rows <= 100`` the floor formula returns ``ratio_used=1.0``
        so even an aggressive ``sample_ratio=0.01`` still uses every row,
        and inference cannot silently fall back to a near-empty subset."""
        df = pd.DataFrame({"col": [str(i) for i in range(10)]})
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.01)
        assert schema["col"] == "int"
        assert len(result) == 10
        assert result["col"].isna().sum() == 0
        assert result["col"].tolist() == list(range(10))

    @pytest.mark.parametrize("n", [161, 167, 193, 194, 195, 199, 289, 305, 322])
    def test_sample_ratio_floor_never_below_100(self, monkeypatch, n):
        """The 100-row floor must hold even at frame sizes where the floor
        ratio ``100/n_full`` round-trips through float multiplication to a
        value just below 100 (e.g., ``161 * (100/161) == 99.999...``).
        """
        captured = {}
        original_sample = pd.DataFrame.sample

        def spy(self, *args, **kwargs):
            captured["n"] = kwargs.get("n")
            return original_sample(self, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "sample", spy)

        df = pd.DataFrame({"col": [str(i) for i in range(n)]})
        auto_convert_dtypes_pandas(df, sample_ratio=0.1)

        assert captured.get("n") == 100, f"n_full={n}: expected sample size 100, got {captured.get('n')}"

    def test_empty_dataframe_does_not_invent_types(self):
        """A truly empty DataFrame has ``n_inference == 0`` so threshold checks
        like ``>= 0 * convert_threshold`` would otherwise pass vacuously and
        coerce columns to arbitrary numeric/datetime types. With the guard,
        empty object columns fall through to the string/category fallback
        instead of being assigned a numeric/datetime/timedelta schema label."""
        df = pd.DataFrame({"col": pd.Series([], dtype=object)})
        result, schema = auto_convert_dtypes_pandas(df)
        assert len(result) == 0
        assert schema["col"] not in ("int", "double", "timestamp", "timedelta")

    def test_sampled_timedelta_masks_categorical_outliers_on_full(self):
        """When sampling marks a column as timedelta-like, the full-column
        conversion must apply the same regex prefilter so categorical values
        that survived only outside the inference sample (e.g., ``"31-40"``,
        ``"60+"``) are not silently coerced into nanosecond deltas."""
        n = 5_000
        # Sample picks rows uniformly with random_state=0; load the column
        # mostly with genuine timedeltas so the sample is dominated by them.
        vals = ["1 days"] * (n - 50) + ["31-40"] * 25 + ["60+"] * 25
        df = pd.DataFrame({"td": vals})
        # n * 0.04 = 200 > 100, so the inference sample is exactly 200 rows.
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.04)
        assert schema["td"] == "timedelta"
        # The category-like outliers must be NaT in the converted column,
        # not coerced to small nanosecond timedeltas.
        assert pd.api.types.is_timedelta64_dtype(result["td"])
        outlier_mask = pd.Series(vals).isin({"31-40", "60+"})
        assert result["td"][outlier_mask].isna().all()
        # And the genuine timedeltas must be preserved as 1 day each.
        good_mask = ~outlier_mask
        assert (result["td"][good_mask] == pd.Timedelta(days=1)).all()

    def test_unsampled_mixed_timedelta_column_does_not_miscoerce_outliers(self):
        """Even without sampling (i.e., the inference subset is the full
        column), a mixed column whose majority looks like timedeltas must
        still mask the categorical-shaped outliers before ``pd.to_timedelta``,
        so values like ``"31-40"`` do not get silently parsed into small
        nanosecond deltas."""
        # 80% genuine timedeltas, 20% categorical outliers — comfortably
        # above ``convert_threshold=0.6`` so the column IS classified as
        # timedelta. ``sample_ratio=1.0`` disables sampling so this
        # exercises the non-sampled branch.
        vals = ["2 hours"] * 80 + ["31-40"] * 10 + ["60+"] * 10
        df = pd.DataFrame({"td": vals})
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=1.0)
        assert schema["td"] == "timedelta"
        assert pd.api.types.is_timedelta64_dtype(result["td"])
        outlier_mask = pd.Series(vals).isin({"31-40", "60+"})
        assert result["td"][outlier_mask].isna().all()
        assert (result["td"][~outlier_mask] == pd.Timedelta(hours=2)).all()

    def test_sampled_category_decision_revalidated_on_full(self, monkeypatch):
        """When sampling, a small inference subset can have low cardinality
        even when the full column is high-cardinality. The category branch
        must re-validate ``unique_ratio`` on the full column so a genuinely
        unique-per-row column is not wrongly classified as ``category``.

        The inference sample is forced (via ``monkeypatch``) to be the head
        100 rows so the sample-side check sees 5/100=0.05 unique ratio
        (under the default ``category_threshold=0.3`` and below the bar
        for category) — exactly the scenario where the full re-validation
        must reject category and fall back to ``string``.
        """
        # Force ``df.sample`` to return the head N rows so the sample
        # deterministically sees only the duplicate head pattern. This
        # exercises the re-validation branch without relying on the
        # vagaries of random sampling at the new 100-row floor.
        monkeypatch.setattr(
            pd.DataFrame,
            "sample",
            lambda self, n=None, frac=None, random_state=None, **kwargs: self.head(
                n if n is not None else int(len(self) * (frac or 1.0))
            ),
        )
        # 1000 rows: head 100 are 5 repeated labels; tail 900 are unique.
        head = ["A", "B", "C", "D", "E"] * 20  # 100 rows, 5 unique
        tail = [f"unique_{i}" for i in range(900)]
        df = pd.DataFrame({"col": head + tail})
        # n=1000, sample_ratio=0.1 → 1000*0.1=100 (NOT > 100) → 100/1000=0.1 → 100 rows.
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.1)
        # Sample sees unique_ratio = 5/100 = 0.05 (under threshold), so the
        # sample-side check would classify as category. Full re-validation
        # sees (5+900)/1000 = 0.905 (over threshold) and rejects.
        assert schema["col"] != "category", schema["col"]
        assert len(result) == 1000


# ---------- get_random_dataframe ----------


class TestGetRandomDataframe:
    def test_default(self):
        df = get_random_dataframe()
        assert df.shape == (200, 14)
        assert "category" in df.columns

    def test_custom_params(self):
        df = get_random_dataframe(n_rows=50, ratio_none=0.0, seed=123)
        assert df.shape == (50, 14)


# ---------- get_output_from_log ----------


class TestGetOutputFromLog:
    def test_empty_log(self):
        """Cover get_output_from_log with no matching records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            fname = f.name
        try:
            result = get_output_from_log(fname, time_budget=100)
            assert len(result[0]) == 0  # search_time_list is empty
        finally:
            os.unlink(fname)


# ---------- scipy.sparse import coverage ----------


class TestSparseImport:
    def test_issparse_and_vstack_available(self):
        """Cover lines 20-22: scipy.sparse import."""
        from scipy.sparse import issparse, vstack

        m = csr_matrix(np.eye(3))
        assert issparse(m)
        result = vstack([m, m])
        assert result.shape == (6, 3)
