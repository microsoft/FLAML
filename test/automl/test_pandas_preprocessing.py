"""Keep tabular preprocessing usable with pandas deprecations as errors."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flaml.automl.data import DataTransformer, concat
from flaml.automl.task.generic_task import GenericTask

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


@pytest.mark.parametrize("series", [False, True])
def test_concat_resets_row_index_and_preserves_categories(series):
    left = pd.Series(pd.Categorical(["b", "a"], categories=["b", "a"]), index=[10, 20], name="category")
    right = pd.Series(pd.Categorical(["a"], categories=["b", "a"]), index=[40], name="category")
    if not series:
        left, right = left.to_frame(), right.to_frame()
    result = concat(left, right)
    column = result if series else result["category"]
    assert result.index.equals(pd.RangeIndex(3))
    assert column.tolist() == ["b", "a", "a"]
    assert list(column.cat.categories) == ["b", "a"]
    assert left.index.tolist() == [10, 20]
    assert right.index.tolist() == [40]


@pytest.mark.parametrize("dropped", [["constant"] * 4, [7] * 4, ["a", "b", "c", "d"]])
def test_transformer_drops_uninformative_columns(dropped):
    frame = pd.DataFrame({"drop": dropped, "value": [1.0, 2.0, 3.0, 4.0]}, index=[40, 30, 20, 10])
    original = frame.copy(deep=True)
    labels = np.arange(4, dtype=float)
    transformer = DataTransformer()
    fitted, fitted_labels = transformer.fit_transform(frame, labels, "regression")
    pd.testing.assert_frame_equal(fitted, frame[["value"]])
    pd.testing.assert_frame_equal(transformer.transform(frame), fitted)
    np.testing.assert_array_equal(fitted_labels, labels)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("legacy_accessor", [False, True])
def test_datetime_features_keep_existing_names_and_values(monkeypatch, legacy_accessor):
    if legacy_accessor:
        # Model the pandas 1.1 accessor surface; this is not a real old-pandas run.
        descriptor = pd.Series.__dict__["dt"]

        class LegacyDatetimeAccessor:
            def __init__(self, series):
                self.accessor = descriptor.__get__(series, pd.Series)

            def __getattr__(self, name):
                if name in ("day_of_week", "day_of_year"):
                    raise AttributeError(name)
                if name in ("dayofweek", "dayofyear"):
                    modern = {"dayofweek": "day_of_week", "dayofyear": "day_of_year"}[name]
                    return getattr(self.accessor, modern if hasattr(self.accessor, modern) else name)
                return getattr(self.accessor, name)

        def cache_accessor(series, accessor):
            series.__dict__["dt"] = accessor

        monkeypatch.setattr(pd.Series, "dt", property(LegacyDatetimeAccessor, cache_accessor))
    frame = pd.DataFrame({"when": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-03-01", "2024-03-02"])})
    transformer = DataTransformer()
    fitted, _ = transformer.fit_transform(frame, np.arange(4, dtype=float), "regression")
    assert fitted["dayofweek_when"].tolist() == [6, 0, 4, 5]
    assert fitted["dayofyear_when"].tolist() == [365, 1, 61, 62]
    assert "day_of_week_when" not in fitted and "day_of_year_when" not in fitted
    pd.testing.assert_frame_equal(transformer.transform(frame), fitted)
    predicted = transformer.transform(pd.DataFrame({"when": pd.to_datetime(["2024-12-31", "2025-01-01"])}))
    assert predicted.columns.equals(fitted.columns)
    assert predicted["dayofweek_when"].tolist() == [1, 2]
    assert predicted["dayofyear_when"].tolist() == [366, 1]


@pytest.mark.parametrize("dataframe", [False, True])
@pytest.mark.parametrize("weight_type", ["none", "array", "series"])
def test_prepare_data_keeps_weights_aligned_after_shuffle(dataframe, weight_type):
    index = pd.Index([50, 40, 30, 20, 10, 0])
    frame = pd.DataFrame({"row": np.arange(6)}, index=index)
    features = frame if dataframe else frame.to_numpy()
    labels = pd.Series(np.arange(6) + 100, index=index, name="target")
    weights = None if weight_type == "none" else np.arange(6, dtype=float) + 200
    if weight_type == "series":
        weights = pd.Series(weights, index=index, name="weight")
    state = SimpleNamespace(X_val=features, y_val=labels, groups=None, fit_kwargs={"sample_weight": weights})
    GenericTask("regression").prepare_data(
        state,
        features,
        labels,
        auto_augment=False,
        eval_method="cv",
        split_type="uniform",
        split_ratio=0.2,
        n_splits=2,
        data_is_df=dataframe,
        sample_weight_full=weights,
    )
    rows = state.X_train_all["row"].to_numpy() if dataframe else state.X_train_all[:, 0]
    np.testing.assert_array_equal(np.sort(rows), np.arange(6))
    np.testing.assert_array_equal(state.y_train_all, rows + 100)
    assert state.y_train_all.index.equals(pd.RangeIndex(6))
    if dataframe:
        assert state.X_train_all.index.equals(pd.RangeIndex(6))
    if weights is not None:
        np.testing.assert_array_equal(state.sample_weight_all, rows + 200)
        assert state.fit_kwargs["sample_weight"] is state.sample_weight_all
        if weight_type == "series":
            assert state.sample_weight_all.index.equals(pd.RangeIndex(6))
    assert labels.index.equals(index)
