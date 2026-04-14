# Auto-Create Timestamps in `prettify_prediction()` When `test_data` is None

## Why are these changes needed?

Currently, the `TimeSeriesDataset.prettify_prediction()` method in `flaml/automl/time_series/ts_data.py` throws a `NotImplementedError` when `test_data` is `None`.
This is frustrating for users who want to make predictions without providing explicit test data timestamps.

**This PR implements automatic timestamp generation** by:

1. Using the training data's end date as the starting point.
2. Generating future timestamps based on the inferred frequency.
3. Supporting `np.ndarray`, `pd.Series`, and `pd.DataFrame`.

## Checks

- [x] Pre-commit linting (black, ruff).
- [x] Added regression tests demonstrating the fix.
