"""Tests to improve coverage for time_series modules:
- ts_model.py
- ts_data.py
- sklearn.py
- tft.py
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _can_import(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers to build minimal TimeSeriesDataset instances
# ---------------------------------------------------------------------------


def _make_daily_df(n=60, n_targets=1, extra_float_col=False, extra_cat_col=False):
    """Return a DataFrame with a daily datetime column and target(s)."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {"date": dates}
    target_names = [f"target_{i}" for i in range(n_targets)]
    for t in target_names:
        data[t] = np.random.randn(n).cumsum()
    if extra_float_col:
        data["extra_float"] = np.random.randn(n)
    if extra_cat_col:
        data["extra_cat"] = np.random.choice(["a", "b"], size=n)
    return pd.DataFrame(data), target_names


def _make_ts_dataset(n=60, n_targets=1, test_len=10, extra_float_col=False, extra_cat_col=False):
    from flaml.automl.time_series.ts_data import TimeSeriesDataset

    df, target_names = _make_daily_df(n, n_targets, extra_float_col, extra_cat_col)
    train = df.iloc[:-test_len].reset_index(drop=True)
    test = df.iloc[-test_len:].reset_index(drop=True)
    return TimeSeriesDataset(train, "date", target_names, test_data=test)


# ===================================================================
# ts_data.py tests
# ===================================================================


class TestTimeSeriesDataset:
    """Cover properties and methods of TimeSeriesDataset."""

    def test_basic_construction_no_test(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        df, targets = _make_daily_df(30)
        ds = TimeSeriesDataset(df, "date", targets[0])
        assert ds.test_data is not None
        assert len(ds.test_data) == 0

    def test_all_data_with_test(self):
        ds = _make_ts_dataset(60, test_len=10)
        assert len(ds.all_data) == 60

    def test_all_data_without_test(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        df, targets = _make_daily_df(30)
        ds = TimeSeriesDataset(df, "date", targets[0])
        assert len(ds.all_data) == 30

    def test_regressors_property(self):
        ds = _make_ts_dataset(60, extra_float_col=True, extra_cat_col=True)
        regs = ds.regressors
        assert isinstance(regs, list)

    def test_end_date(self):
        ds = _make_ts_dataset(60, test_len=10)
        assert ds.end_date is not None

    def test_X_y_properties(self):
        ds = _make_ts_dataset(60, test_len=10)
        assert ds.X_train is not None
        assert len(ds.X_train.columns) > 0
        _ = ds.y_train
        _ = ds.X_val
        _ = ds.y_val
        _ = ds.y_all
        _ = ds.X_all

    def test_y_multivariate(self):
        ds = _make_ts_dataset(60, n_targets=2, test_len=10)
        y = ds._y(ds.train_data)
        assert isinstance(y, pd.DataFrame)
        assert y.shape[1] == 2

    def test_next_scale(self):
        ds = _make_ts_dataset(60)
        scale = ds.next_scale()
        # daily => 7
        assert scale == 7

    def test_known_features_to_floats(self):
        ds = _make_ts_dataset(60, extra_cat_col=True, test_len=10)
        train_feats = ds.known_features_to_floats(train=True)
        test_feats = ds.known_features_to_floats(train=False)
        assert train_feats.shape[0] == len(ds.train_data)
        assert test_feats.shape[0] == len(ds.test_data)

    def test_add_test_data(self):
        ds = _make_ts_dataset(60, test_len=10)
        new_test = ds.test_data.copy()
        new_ds = ds.add_test_data(new_test)
        assert new_ds.test_data is not None

    def test_to_dataframe(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        ds = _make_ts_dataset(60, test_len=10)
        X_val = ds.X_val
        y_val = ds.y_val
        result = TimeSeriesDataset.to_dataframe(X_val, y_val, ds.target_names, ds.time_col)
        assert isinstance(result, pd.DataFrame)

    def test_move_validation_boundary_positive(self):
        ds = _make_ts_dataset(60, test_len=10)
        new_ds = ds.move_validation_boundary(3)
        assert len(new_ds.train_data) == len(ds.train_data) + 3

    def test_move_validation_boundary_negative(self):
        ds = _make_ts_dataset(60, test_len=10)
        new_ds = ds.move_validation_boundary(-3)
        assert len(new_ds.train_data) == len(ds.train_data) - 3

    def test_move_validation_boundary_negative_no_test(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        df, targets = _make_daily_df(30)
        ds = TimeSeriesDataset(df, "date", targets[0])
        new_ds = ds.move_validation_boundary(-3)
        assert len(new_ds.test_data) == 3

    def test_move_validation_boundary_zero(self):
        ds = _make_ts_dataset(60, test_len=10)
        new_ds = ds.move_validation_boundary(0)
        assert len(new_ds.train_data) == len(ds.train_data)

    def test_cv_train_val_sets(self):
        ds = _make_ts_dataset(60, test_len=10)
        splits = list(ds.cv_train_val_sets(n_splits=3, val_length=5, step_size=5))
        assert len(splits) == 3

    def test_filter(self):
        ds = _make_ts_dataset(60, test_len=10)
        # filter with None
        same_ds = ds.filter(None)
        assert len(same_ds.train_data) == len(ds.train_data)

    def test_prettify_prediction_ndarray(self):
        ds = _make_ts_dataset(60, test_len=10)
        pred = np.random.randn(10)
        result = ds.prettify_prediction(pred)
        assert isinstance(result, pd.DataFrame)
        assert ds.time_col in result.columns

    def test_prettify_prediction_series(self):
        ds = _make_ts_dataset(60, test_len=10)
        pred = pd.Series(np.random.randn(10))
        result = ds.prettify_prediction(pred)
        assert isinstance(result, pd.DataFrame)

    def test_prettify_prediction_dataframe(self):
        ds = _make_ts_dataset(60, test_len=10)
        pred = pd.DataFrame({ds.target_names[0]: np.random.randn(10)})
        result = ds.prettify_prediction(pred)
        assert ds.time_col in result.columns

    def test_prettify_no_test_ndarray_raises(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        df, targets = _make_daily_df(30)
        ds = TimeSeriesDataset(df, "date", targets[0])
        with pytest.raises(ValueError, match="Can't enrich"):
            ds.prettify_prediction(np.array([1, 2, 3]))

    def test_prettify_no_test_series_raises(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        df, targets = _make_daily_df(30)
        ds = TimeSeriesDataset(df, "date", targets[0])
        with pytest.raises(NotImplementedError):
            ds.prettify_prediction(pd.Series([1, 2, 3]))

    def test_merge_prediction_with_target(self):
        ds = _make_ts_dataset(60, test_len=10)
        pred = np.random.randn(10)
        result = ds.merge_prediction_with_target(pred)
        assert isinstance(result, pd.DataFrame)


class TestEnrichDataframe:
    def test_enrich_with_fourier(self):
        from flaml.automl.time_series.ts_data import enrich_dataframe

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"time": dates, "value": np.random.randn(30)})
        result = enrich_dataframe(df, fourier_degree=2, fourier_time=True)
        assert result.shape[1] > 2

    def test_enrich_with_non_fourier_time(self):
        from flaml.automl.time_series.ts_data import enrich_dataframe

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"time": dates, "value": np.random.randn(30)})
        result = enrich_dataframe(df, fourier_degree=2, fourier_time=False)
        assert result.shape[1] > 2

    def test_enrich_remove_constants(self):
        from flaml.automl.time_series.ts_data import enrich_dataframe

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"time": dates, "value": np.random.randn(30)})
        result = enrich_dataframe(df, fourier_degree=2, remove_constants=True)
        assert result.shape[1] > 2

    def test_enrich_series_input(self):
        from flaml.automl.time_series.ts_data import enrich_dataframe

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        s = pd.Series(dates, name="time")
        result = enrich_dataframe(s, fourier_degree=2)
        assert isinstance(result, pd.DataFrame)


class TestEnrichDataset:
    def test_enrich_dataset(self):
        from flaml.automl.time_series.ts_data import enrich_dataset

        ds = _make_ts_dataset(60, test_len=10)
        result = enrich_dataset(ds, fourier_degree=2)
        assert len(result.train_data) == len(ds.train_data)


class TestDataTransformerTS:
    def test_fit_transform_numeric(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "num_feat": np.random.randn(30),
                "const_feat": [1.0] * 30,
            }
        )
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        X_out, y_out = transformer.fit_transform(df, y)
        assert "const_feat" not in X_out.columns

    def test_fit_transform_categorical(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "cat_feat": np.random.choice(["a", "b", "c"], size=30),
            }
        )
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        X_out, y_out = transformer.fit_transform(df.copy(), y)
        assert "cat_feat" in X_out.columns

    def test_fit_transform_with_drop_object_cols(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "uid_col": [f"uid_{i}" for i in range(30)],  # unique per row
                "const_obj": ["same"] * 30,
            }
        )
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        X_out, y_out = transformer.fit_transform(df.copy(), y)
        assert "uid_col" not in X_out.columns
        assert "const_obj" not in X_out.columns
        assert transformer._drop > 0

    def test_fit_transform_datetime_col(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "other_date": pd.date_range("2021-01-01", periods=30, freq="D"),
            }
        )
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        X_out, y_out = transformer.fit_transform(df.copy(), y)
        assert "other_date" in transformer.datetime_columns

    def test_transform_with_label_transformer(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates})
        y = pd.Series(np.random.choice(["cat", "dog", "fish"], size=30), name="target")
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y.copy())
        assert transformer.label_transformer is not None
        X_out, y_out = transformer.transform(df.copy(), y.copy())
        assert y_out is not None

    def test_transform_y_dataframe_with_label_transformer(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates})
        y = pd.DataFrame({"target": np.random.choice(["cat", "dog"], size=30)})
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y.copy())
        X_out, y_out = transformer.transform(df.copy(), y.copy())
        assert y_out is not None

    def test_transform_y_none(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates, "num": np.random.randn(30)})
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y)
        result = transformer.transform(df.copy(), y=None)
        assert isinstance(result, pd.DataFrame)

    def test_fit_y_dataframe(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates})
        y = pd.DataFrame({"target": np.random.randn(30)})
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y)
        assert transformer.label_transformer is None

    def test_fit_y_invalid_type_raises(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates})
        transformer = DataTransformerTS("date", "target")
        with pytest.raises(ValueError, match="y must be"):
            transformer.fit(df.copy(), "not_valid")

    def test_transform_y_invalid_type_raises(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates})
        y = pd.Series(np.random.choice(["a", "b"], size=30), name="target")
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y.copy())
        with pytest.raises(ValueError, match="y must be"):
            transformer.transform(df.copy(), y="invalid")

    def test_transform_category_dtype_with_nan(self):
        from flaml.automl.time_series.ts_data import DataTransformerTS

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        cat_vals = np.random.choice(["a", "b", "c"], size=30)
        df = pd.DataFrame(
            {
                "date": dates,
                "cat_feat": pd.Categorical(cat_vals),
            }
        )
        y = pd.Series(np.random.randn(30), name="target")
        transformer = DataTransformerTS("date", "target")
        transformer.fit(df.copy(), y)
        # set some NaN
        df2 = df.copy()
        df2.loc[0, "cat_feat"] = np.nan
        X_out, y_out = transformer.transform(df2, y)
        assert "__NAN__" in X_out["cat_feat"].cat.categories


class TestNormalizeTsData:
    def test_ndarray_1d(self):
        from flaml.automl.time_series.ts_data import normalize_ts_data

        X = np.array([1.0, 2.0, 3.0])
        result = normalize_ts_data(X, ["target"], "time")
        assert isinstance(result, pd.DataFrame)

    def test_ndarray_2d(self):
        from flaml.automl.time_series.ts_data import normalize_ts_data

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize_ts_data(X, ["target"], "time")
        assert isinstance(result, pd.DataFrame)

    def test_ndarray_with_y(self):
        from flaml.automl.time_series.ts_data import normalize_ts_data

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([10.0, 20.0])
        result = normalize_ts_data(X, ["target"], "time", y)
        assert "target" in result.columns

    def test_series_y(self):
        from flaml.automl.time_series.ts_data import normalize_ts_data

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        X = pd.DataFrame({"time": dates})
        y = pd.Series([1.0, 2.0, 3.0], name="target")
        result = normalize_ts_data(X, ["target"], "time", y)
        assert "target" in result.columns

    def test_ts_dataset_passthrough(self):
        from flaml.automl.time_series.ts_data import normalize_ts_data

        ds = _make_ts_dataset(30, test_len=5)
        result = normalize_ts_data(ds, ["t"], "time")
        assert result is ds


class TestCreateForwardFrame:
    def test_forward_frame(self):
        from flaml.automl.time_series.ts_data import create_forward_frame

        end_date = pd.Timestamp("2020-03-01")
        result = create_forward_frame("D", 5, end_date, "time")
        assert len(result) == 5
        assert "time" in result.columns


class TestFourierSeries:
    def test_fourier_series(self):
        from flaml.automl.time_series.ts_data import fourier_series

        feat = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        result = fourier_series(feat, "test")
        assert "test_sin" in result
        assert "test_cos" in result


# ===================================================================
# ts_model.py tests
# ===================================================================


class TestTimeSeriesEstimator:
    def test_init(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        est = TimeSeriesEstimator(task="ts_forecast")
        assert est.time_col is None

    def test_fit_sets_attributes(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        assert est.time_col == "date"
        assert est.frequency == "D"

    def test_enrich_int(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        result = est.enrich(5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_enrich_dataset(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        result = est.enrich(ds)
        assert hasattr(result, "train_data")

    def test_enrich_dataframe(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        result = est.enrich(ds.test_data)
        assert isinstance(result, pd.DataFrame)

    def test_score_with_metric(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        # Mock predict
        est.predict = lambda X, **kw: pd.Series(np.zeros(10), name="target_0")
        score = est.score(ds, ds.y_val, metric="mse")
        assert isinstance(score, float)

    def test_score_without_metric(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        ds = _make_ts_dataset(60, test_len=10)
        est = TimeSeriesEstimator(task="ts_forecast")
        est.fit(ds)
        est.predict = lambda X, **kw: pd.Series(ds.y_val.values, name="target_0")
        score = est.score(ds, ds.y_val)
        assert isinstance(score, float)

    def test_adjust_scale(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        scale, max_lags = TimeSeriesEstimator.adjust_scale(7, 100, 10)
        assert scale >= 2
        assert max_lags >= 2

    def test_adjust_scale_small_data(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        # Force scale reduction path
        scale, max_lags = TimeSeriesEstimator.adjust_scale(12, 30, 5)
        assert scale >= 2

    def test_top_level_params(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        params = TimeSeriesEstimator.top_level_params()
        assert "monthly_fourier_degree" in params

    def test_join(self):
        from flaml.automl.time_series.ts_model import TimeSeriesEstimator

        est = TimeSeriesEstimator(task="ts_forecast")
        X = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=5, freq="D")})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = est._join(X, y)
        assert isinstance(result, pd.DataFrame)


class TestARIMA:
    def test_init_missing_params_raises(self):
        from flaml.automl.time_series.ts_model import ARIMA

        with pytest.raises(ValueError, match="ARIMA initialized without required params"):
            ARIMA(task="ts_forecast")

    def test_search_space(self):
        from flaml.automl.task.generic_task import GenericTask
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10)
        task = GenericTask("ts_forecast")
        space = ARIMA.search_space(ds, task, 10)
        assert "p" in space
        assert "d" in space
        assert "q" in space

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_predict_dataset(self):
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        train_time = est.fit(ds, budget=10)
        assert train_time > 0
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_predict_with_regressors(self):
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10, extra_float_col=True)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        train_time = est.fit(ds, budget=10)
        assert train_time > 0
        preds = est.predict(ds)
        assert len(preds) == 10


class TestSARIMAX:
    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_predict_dataset(self):
        from flaml.automl.time_series.ts_model import SARIMAX

        ds = _make_ts_dataset(80, test_len=10)
        est = SARIMAX(task="ts_forecast", p=1, d=0, q=1, P=0, D=0, Q=0, s=7)
        train_time = est.fit(ds, budget=10)
        assert train_time > 0
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_predict_with_regressors(self):
        from flaml.automl.time_series.ts_model import SARIMAX

        ds = _make_ts_dataset(80, test_len=10, extra_float_col=True)
        est = SARIMAX(task="ts_forecast", p=1, d=0, q=0, P=0, D=0, Q=0, s=7)
        est.fit(ds, budget=10)
        preds = est.predict(ds)
        assert len(preds) == 10


class TestStatsModelsEstimator:
    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_predict_not_fit(self):
        from flaml.automl.time_series.ts_model import ARIMA

        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        est._model = None
        est.time_col = "date"
        est.frequency = "D"
        est.end_date = pd.Timestamp("2020-03-01")
        est.target_names = ["target"]
        est.regressors = []
        est.params = {"monthly_fourier_degree": 0, "fourier_time_features": False}
        preds = est.predict(5)
        assert len(preds) == 5
        assert np.all(preds == 1.0)

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_predict_with_empty_dataframe(self):
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        est.fit(ds, budget=10)
        empty_df = pd.DataFrame(columns=[est.time_col] + est.regressors)
        preds = est.predict(empty_df)
        assert len(preds) == 0

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_predict_dataset_empty_test(self):
        """Test fallback to train partition when test_data is empty."""
        from flaml.automl.time_series.ts_data import TimeSeriesDataset
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        est.fit(ds, budget=10)
        # Create dataset with empty test
        df, targets = _make_daily_df(50)
        ds_no_test = TimeSeriesDataset(df, "date", targets[0])
        preds = est.predict(ds_no_test)
        assert len(preds) == 50

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_predict_plain_dataframe(self):
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        est.fit(ds, budget=10)
        # predict with a plain dataframe
        test_df = ds.test_data[[ds.time_col]].copy()
        preds = est.predict(test_df)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_predict_with_regressors(self):
        from flaml.automl.time_series.ts_model import ARIMA

        ds = _make_ts_dataset(60, test_len=10, extra_float_col=True)
        est = ARIMA(task="ts_forecast", p=1, d=0, q=1)
        est.fit(ds, budget=10)
        preds = est.predict(ds)
        assert len(preds) == 10


class TestHoltWinters:
    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_predict(self):
        from flaml.automl.time_series.ts_model import HoltWinters

        ds = _make_ts_dataset(60, test_len=10)
        est = HoltWinters(
            task="ts_forecast", damped_trend=False, trend="add", seasonal=None, use_boxcox=False, seasonal_periods=7
        )
        est.fit(ds, ds.y_train, budget=10)
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_fit_small_data_no_seasonal(self):
        """Cover the branch where data is too small for seasonal_periods."""
        from flaml.automl.time_series.ts_model import HoltWinters

        ds = _make_ts_dataset(20, test_len=5)
        est = HoltWinters(
            task="ts_forecast",
            damped_trend=True,
            trend="add",
            seasonal="add",
            use_boxcox=False,
            seasonal_periods=12,
        )
        est.fit(ds, ds.y_train, budget=10)
        # seasonal should be overridden to None since data < 2 * seasonal_periods
        assert est.params["seasonal"] is None

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_multiplicative_with_zeros(self):
        from flaml.automl.time_series.ts_model import HoltWinters

        ds = _make_ts_dataset(60, test_len=10)
        # inject zeros
        ds.train_data.iloc[0, ds.train_data.columns.get_loc(ds.target_names[0])] = 0.0
        est = HoltWinters(
            task="ts_forecast",
            damped_trend=True,
            trend="mul",
            seasonal="mul",
            use_boxcox=False,
            seasonal_periods=7,
        )
        est.fit(ds, ds.y_train, budget=10)
        assert est.params["seasonal"] == "add"
        assert est.params["trend"] == "add"


class TestSimpleForecaster:
    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_seasonal_naive_fit_predict_int(self):
        from flaml.automl.time_series.ts_model import SeasonalNaive

        ds = _make_ts_dataset(60, test_len=10)
        est = SeasonalNaive(task="ts_forecast", season=3)
        est.fit(ds, budget=10)
        preds = est.predict(5)
        assert len(preds) == 5

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_naive_fit_predict_int(self):
        from flaml.automl.time_series.ts_model import Naive

        ds = _make_ts_dataset(60, test_len=10)
        est = Naive(task="ts_forecast")
        est.fit(ds, budget=10)
        preds = est.predict(5)
        assert len(preds) == 5

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_naive_predict_dataset(self):
        from flaml.automl.time_series.ts_model import Naive

        ds = _make_ts_dataset(60, test_len=10)
        est = Naive(task="ts_forecast")
        est.fit(ds, budget=10)
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_seasonal_average_fit(self):
        from flaml.automl.time_series.ts_model import SeasonalAverage

        ds = _make_ts_dataset(60, test_len=10)
        est = SeasonalAverage(task="ts_forecast", season=3)
        train_time = est.fit(ds, budget=10)
        assert train_time > 0

    @pytest.mark.skipif(
        not _can_import("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_average_fit(self):
        from flaml.automl.time_series.ts_model import Average

        ds = _make_ts_dataset(60, test_len=10)
        est = Average(task="ts_forecast")
        train_time = est.fit(ds, budget=10)
        assert train_time > 0


class TestProphet:
    @pytest.mark.skipif(
        not _can_import("prophet"),
        reason="prophet not installed",
    )
    def test_fit_predict_dataset(self):
        from flaml.automl.time_series.ts_model import Prophet as ProphetEst

        ds = _make_ts_dataset(60, test_len=10)
        est = ProphetEst(task="ts_forecast")
        train_time = est.fit(ds, budget=10)
        assert train_time > 0
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("prophet"),
        reason="prophet not installed",
    )
    def test_predict_empty_test_fallback(self):
        """Prophet predict with dataset whose test partition is empty falls back to train."""
        from flaml.automl.time_series.ts_data import TimeSeriesDataset
        from flaml.automl.time_series.ts_model import Prophet as ProphetEst

        ds = _make_ts_dataset(60, test_len=10)
        est = ProphetEst(task="ts_forecast")
        est.fit(ds, budget=10)
        df, targets = _make_daily_df(50)
        ds_no_test = TimeSeriesDataset(df, "date", targets[0])
        preds = est.predict(ds_no_test)
        assert len(preds) == 50

    @pytest.mark.skipif(
        not _can_import("prophet"),
        reason="prophet not installed",
    )
    def test_predict_not_fit(self):
        from flaml.automl.time_series.ts_model import Prophet as ProphetEst

        ds = _make_ts_dataset(60, test_len=10)
        est = ProphetEst(task="ts_forecast")
        est.fit(ds, budget=10)
        est._model = None
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("prophet"),
        reason="prophet not installed",
    )
    def test_predict_with_dataframe(self):
        from flaml.automl.time_series.ts_model import Prophet as ProphetEst

        ds = _make_ts_dataset(60, test_len=10)
        est = ProphetEst(task="ts_forecast")
        est.fit(ds, budget=10)
        # Predict with a plain dataframe
        test_df = ds.test_data[["date"]].copy()
        preds = est.predict(test_df)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("prophet"),
        reason="prophet not installed",
    )
    def test_fit_predict_with_regressors(self):
        from flaml.automl.time_series.ts_model import Prophet as ProphetEst

        ds = _make_ts_dataset(60, test_len=10, extra_float_col=True)
        est = ProphetEst(task="ts_forecast")
        est.fit(ds, budget=10)
        preds = est.predict(ds)
        assert len(preds) == 10


class TestTSSKLearn:
    @pytest.mark.skipif(
        not _can_import("lightgbm"),
        reason="lightgbm not installed",
    )
    def test_lgbm_ts_fit_predict(self):
        from flaml.automl.time_series.ts_model import LGBM_TS

        ds = _make_ts_dataset(80, test_len=10)
        est = LGBM_TS(task="ts_forecast", lags=5)
        train_time = est.fit(ds, budget=10, period=10)
        assert train_time > 0
        preds = est.predict(ds)
        assert len(preds) == 10

    @pytest.mark.skipif(
        not _can_import("lightgbm"),
        reason="lightgbm not installed",
    )
    def test_ts_sklearn_dataframe_path(self):
        from flaml.automl.time_series.ts_model import LGBM_TS

        dates = pd.date_range("2020-01-01", periods=80, freq="D")
        X_df = pd.DataFrame({"ds": dates, "feat1": np.random.randn(80)})
        y_series = pd.Series(np.random.randn(80).cumsum(), name="y")
        assert isinstance(X_df, pd.DataFrame) and isinstance(y_series, pd.Series)
        est = LGBM_TS(task="ts_forecast", lags=5)
        est.time_col = "ds"
        est.target_names = ["y"]
        est.regressors = ["feat1"]
        est.X_train = MagicMock()
        est.X_train.frequency = "D"
        est.X_train.end_date = dates[-1]
        # Test the DataFrame isinstance branch
        est.fit(
            _make_ts_dataset(80, test_len=10),
            budget=10,
            period=10,
        )

    @pytest.mark.skipif(
        not _can_import("lightgbm"),
        reason="lightgbm not installed",
    )
    def test_predict_not_fit(self):
        from flaml.automl.time_series.ts_model import LGBM_TS

        ds = _make_ts_dataset(80, test_len=10)
        est = LGBM_TS(task="ts_forecast", lags=5)
        est._model = None
        est.time_col = "date"
        est.target_names = ["target_0"]
        est.regressors = []
        est.frequency = "D"
        est.end_date = pd.Timestamp("2020-03-01")
        est.params = {"monthly_fourier_degree": 0, "fourier_time_features": False, "pca_features": False}
        preds = est.predict(ds)
        assert np.all(preds == 1.0)

    @pytest.mark.skipif(
        not _can_import("lightgbm"),
        reason="lightgbm not installed",
    )
    def test_predict_empty_test_fallback(self):
        from flaml.automl.time_series.ts_data import TimeSeriesDataset
        from flaml.automl.time_series.ts_model import LGBM_TS

        ds = _make_ts_dataset(80, test_len=10)
        est = LGBM_TS(task="ts_forecast", lags=5)
        est.fit(ds, budget=10, period=10)
        df, targets = _make_daily_df(70)
        ds_no_test = TimeSeriesDataset(df, "date", targets[0])
        preds = est.predict(ds_no_test)
        assert len(preds) == 70


# ===================================================================
# sklearn.py tests
# ===================================================================


class TestMakeLagFeatures:
    def test_basic(self):
        from flaml.automl.time_series.sklearn import make_lag_features

        X = pd.DataFrame({"feat": np.arange(20, dtype=float)})
        y = pd.Series(np.arange(20, dtype=float), name="target")
        result = make_lag_features(X, y, lags=3)
        assert len(result) == 17
        assert result.shape[1] > X.shape[1]


class TestSklearnWrapper:
    @pytest.mark.skipif(
        not _can_import("sklearn"),
        reason="sklearn not installed",
    )
    def test_fit_predict(self):
        from sklearn.linear_model import LinearRegression

        from flaml.automl.time_series.sklearn import SklearnWrapper

        X = pd.DataFrame({"feat": np.arange(60, dtype=float)})
        y = pd.Series(np.sin(np.arange(60, dtype=float) / 5), name="target")
        wrapper = SklearnWrapper(LinearRegression, horizon=5, lags=3)
        wrapper.fit(X, y)
        X_test = pd.DataFrame({"feat": np.arange(60, 65, dtype=float)})
        preds = wrapper.predict(X_test)
        assert len(preds) == 5

    @pytest.mark.skipif(
        not _can_import("sklearn"),
        reason="sklearn not installed",
    )
    def test_fit_with_pca(self):
        from sklearn.linear_model import LinearRegression

        from flaml.automl.time_series.sklearn import SklearnWrapper

        X = pd.DataFrame(
            {
                "feat1": np.arange(60, dtype=float),
                "feat2": np.random.randn(60),
            }
        )
        y = pd.Series(np.sin(np.arange(60, dtype=float) / 5), name="target")
        wrapper = SklearnWrapper(LinearRegression, horizon=3, lags=3, pca_features=True)
        wrapper.fit(X, y)
        X_test = pd.DataFrame(
            {
                "feat1": np.arange(60, 63, dtype=float),
                "feat2": np.random.randn(3),
            }
        )
        preds = wrapper.predict(X_test)
        assert len(preds) == 3

    @pytest.mark.skipif(
        not _can_import("sklearn"),
        reason="sklearn not installed",
    )
    def test_predict_longer_than_horizon(self):
        from sklearn.linear_model import LinearRegression

        from flaml.automl.time_series.sklearn import SklearnWrapper

        X = pd.DataFrame({"feat": np.arange(60, dtype=float)})
        y = pd.Series(np.sin(np.arange(60, dtype=float) / 5), name="target")
        wrapper = SklearnWrapper(LinearRegression, horizon=3, lags=2)
        wrapper.fit(X, y)
        # Predict more than horizon steps
        X_test = pd.DataFrame({"feat": np.arange(60, 68, dtype=float)})
        preds = wrapper.predict(X_test)
        assert len(preds) == 8

    @pytest.mark.skipif(
        not _can_import("sklearn"),
        reason="sklearn not installed",
    )
    def test_fit_with_is_retrain(self):
        from sklearn.linear_model import LinearRegression

        from flaml.automl.time_series.sklearn import SklearnWrapper

        X = pd.DataFrame({"feat": np.arange(60, dtype=float)})
        y = pd.Series(np.sin(np.arange(60, dtype=float) / 5), name="target")
        wrapper = SklearnWrapper(LinearRegression, horizon=3, lags=2)
        wrapper.fit(X, y, is_retrain=True)
        assert wrapper._X is not None

    @pytest.mark.skipif(
        not _can_import("sklearn"),
        reason="sklearn not installed",
    )
    def test_fit_short_series(self):
        from sklearn.linear_model import LinearRegression

        from flaml.automl.time_series.sklearn import SklearnWrapper

        X = pd.DataFrame({"feat": np.arange(5, dtype=float)})
        y = pd.Series(np.arange(5, dtype=float), name="target")
        wrapper = SklearnWrapper(LinearRegression, horizon=3, lags=2)
        wrapper.fit(X, y)


# ===================================================================
# tft.py tests
# ===================================================================


class TestTFT:
    @pytest.mark.skipif(
        not _can_import("pytorch_forecasting") or not _can_import("torch"),
        reason="pytorch_forecasting or torch not installed",
    )
    def test_search_space(self):
        from flaml.automl.time_series.tft import TemporalFusionTransformerEstimator

        space = TemporalFusionTransformerEstimator.search_space(None, None, 10)
        assert "gradient_clip_val" in space
        assert "hidden_size" in space
        assert "learning_rate" in space

    @pytest.mark.skipif(
        not _can_import("pytorch_forecasting") or not _can_import("torch"),
        reason="pytorch_forecasting or torch not installed",
    )
    def test_init(self):
        from flaml.automl.time_series.tft import TemporalFusionTransformerEstimator

        est = TemporalFusionTransformerEstimator(task="ts_forecast")
        assert est.time_col is None
