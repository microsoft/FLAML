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
