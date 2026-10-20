import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from flaml import AutoML
from flaml.automl.time_series.ts_data import TimeSeriesDataset, create_forward_frame


def test_prettify_prediction_generates_timestamps_without_test_data():
    train_data = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = TimeSeriesDataset(train_data, time_col="ds", target_names="y")
    expected_times = pd.date_range("2020-01-05", periods=2, freq="D")

    for y_pred in (
        pd.DataFrame({"y": [5.0, 6.0]}, index=[10, 11]),
        pd.Series([5.0, 6.0]),
        np.array([5.0, 6.0]),
    ):
        prediction = dataset.prettify_prediction(y_pred)
        assert isinstance(prediction, pd.DataFrame)
        pd.testing.assert_series_equal(prediction["ds"], pd.Series(expected_times, name="ds"), check_index=False)
        assert prediction["y"].tolist() == [5.0, 6.0]


def test_prettify_prediction_generates_monthly_timestamps_without_test_data():
    train_data = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = TimeSeriesDataset(train_data, time_col="ds", target_names="y")

    prediction = dataset.prettify_prediction(pd.DataFrame({"y": [5.0, 6.0]}))

    pd.testing.assert_series_equal(
        prediction["ds"],
        pd.Series(pd.date_range("2020-05-01", periods=2, freq="MS"), name="ds"),
        check_index=False,
    )
    assert prediction["y"].tolist() == [5.0, 6.0]


def test_create_forward_frame_uses_next_frequency_offset():
    # Pandas 3 uses QE-DEC while older supported versions use Q-DEC.
    quarter_end_freq = "QE-DEC"
    try:
        pd.tseries.frequencies.to_offset(quarter_end_freq)
    except ValueError:
        quarter_end_freq = "Q-DEC"

    weekly_frame = create_forward_frame("W-SUN", 2, pd.Timestamp("2020-01-05"), "ds")
    quarterly_frame = create_forward_frame(quarter_end_freq, 2, pd.Timestamp("2020-03-31"), "ds")

    pd.testing.assert_series_equal(
        weekly_frame["ds"], pd.Series(pd.date_range("2020-01-12", periods=2, freq="W-SUN"), name="ds")
    )
    pd.testing.assert_series_equal(
        quarterly_frame["ds"], pd.Series(pd.date_range("2020-06-30", periods=2, freq=quarter_end_freq), name="ds")
    )


@pytest.mark.parametrize("period", [1, 4])
@pytest.mark.parametrize("index_kind", ["default", "offset", "datetime"])
def test_cv_folds_match_time_series_split(period, index_kind):
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=24), "y": np.arange(24)})
    if index_kind == "offset":
        frame.index += 100
    elif index_kind == "datetime":
        frame.index = frame.ds
    original = frame.copy(deep=True)
    dataset = TimeSeriesDataset(frame, time_col="ds", target_names="y")
    folds = list(dataset.cv_train_val_sets(n_splits=3, val_length=period, step_size=period))
    reference = list(TimeSeriesSplit(n_splits=3, test_size=period).split(frame))

    assert len(folds) == len(reference)
    for fold, (train_indices, validation_indices) in zip(folds, reference):
        pd.testing.assert_frame_equal(fold.train_data, frame.iloc[train_indices])
        pd.testing.assert_frame_equal(fold.test_data, frame.iloc[validation_indices])
    pd.testing.assert_frame_equal(dataset.train_data, original)
    assert dataset.test_data.empty


@pytest.mark.parametrize("step_size, expected_starts", [(2, [16, 18, 20]), (7, [6, 13, 20])])
def test_cv_custom_step_sizes_end_at_latest_observation(step_size, expected_starts):
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=24), "y": np.arange(24)})
    dataset = TimeSeriesDataset(frame, time_col="ds", target_names="y")
    folds = list(dataset.cv_train_val_sets(n_splits=3, val_length=4, step_size=step_size))

    assert len(folds) == len(expected_starts)
    for fold, start in zip(folds, expected_starts):
        assert fold.test_data.y.tolist() == list(range(start, start + 4))
        assert fold.train_data.y.tolist() == list(range(start))
    assert folds[-1].test_data.ds.iloc[-1] == frame.ds.iloc[-1]


def test_automl_cv_scores_latest_observation():
    pytest.importorskip("statsmodels")
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=60), "y": np.zeros(60)})
    frame.loc[59, "y"] = 100.0
    automl = AutoML()
    automl.fit(
        dataframe=frame,
        label="y",
        task="ts_forecast",
        period=5,
        estimator_list=["avg"],
        eval_method="cv",
        n_splits=3,
        metric="mae",
        max_iter=2,
        time_budget=30,
        retrain_full=False,
        verbose=0,
    )

    # Every training fold contains only zeros. The final validation observation
    # must contribute its error to the mean across all three five-row folds.
    assert automl.best_loss == pytest.approx(100.0 / 15)


def test_add_time_idx_col_panel_data():
    from flaml.automl.data import add_time_idx_col

    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                ]
            ),
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    res = add_time_idx_col(df)
    assert "time_idx" in res.columns
    assert list(res["time_idx"]) == [0, 1, 2, 0, 1, 2]


def test_add_time_idx_col_edge_cases():
    from flaml.automl.data import add_time_idx_col

    # Single date
    df1 = pd.DataFrame({"ds": pd.to_datetime(["2021-01-01", "2021-01-01"])})
    res1 = add_time_idx_col(df1)
    assert list(res1["time_idx"]) == [0, 0]

    # Two dates
    df2 = pd.DataFrame({"ds": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01"])})
    res2 = add_time_idx_col(df2)
    assert list(res2["time_idx"]) == [0, 1, 0]


def test_add_time_idx_col_month_end_panel():
    from flaml.automl.data import TS_TIMESTAMP_COL, add_time_idx_col

    # Six month-ends from 2021-01-31 across panel groups (inferred as ME / M)
    dates = pd.to_datetime(
        [
            "2021-01-31",
            "2021-02-28",
            "2021-03-31",
            "2021-04-30",
            "2021-05-31",
            "2021-06-30",
        ]
    )
    df = pd.DataFrame(
        {
            TS_TIMESTAMP_COL: list(dates) + list(dates),
            "group": ["A"] * 6 + ["B"] * 6,
        }
    )
    res = add_time_idx_col(df)
    assert res["time_idx"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    assert np.issubdtype(res["time_idx"].dtype, np.integer)


def test_add_time_idx_col_dst_transition_panel():
    from flaml.automl.data import TS_TIMESTAMP_COL, add_time_idx_col

    # Six America/New_York local days crossing spring DST (2021-03-14: 23-hour day)
    dates = pd.date_range("2021-03-12", periods=6, freq="D", tz="America/New_York")
    df = pd.DataFrame(
        {
            TS_TIMESTAMP_COL: list(dates) + list(dates),
            "group": ["A"] * 6 + ["B"] * 6,
        }
    )
    res = add_time_idx_col(df)
    assert res["time_idx"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    assert np.issubdtype(res["time_idx"].dtype, np.integer)
