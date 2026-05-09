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

    def test_daily_frequency(self):
        """Cover lines 253-257: other freq (daily)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        X = pd.DataFrame({"ds": dates, "val": range(30)})
        result = add_time_idx_col(X)
        assert "time_idx" in result.columns


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
        """Cover line 726: sampling with sample_ratio < 1."""
        df = pd.DataFrame({"col": [str(i) for i in range(100)]})
        result, schema = auto_convert_dtypes_pandas(df, sample_ratio=0.5)
        assert "col" in schema

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

    def test_boolean_dtype_to_category(self):
        """Cover lines 740-743, 752, 784-790: BooleanDtype handling -> category."""
        df = pd.DataFrame(
            {"flag": pd.array([True, False, True, False, True, False, True, False, True, False], dtype="boolean")}
        )
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["flag"] == "category"

    def test_int_with_nan_becomes_double(self):
        """Cover line 761-763: int conversion failure falls back to double."""
        df = pd.DataFrame({"col": ["1", "2", "3.0", None, "5"]})
        result, schema = auto_convert_dtypes_pandas(df)
        assert schema["col"] in ("int", "double")


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
