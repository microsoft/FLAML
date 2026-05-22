# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
import json
import math
import os
import random
import re
import uuid
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Union

import numpy as np

from flaml.automl.logger import init_logger
from flaml.automl.spark import DataFrame, F, Series, T, pd, ps, psDataFrame, psSeries
from flaml.automl.training_log import training_log_reader

try:
    from pandas.api.types import is_datetime64_any_dtype
except ImportError:
    is_datetime64_any_dtype = None

try:
    from scipy.sparse import issparse, vstack
except ImportError:
    pass

if TYPE_CHECKING:
    from flaml.automl.task import Task


logger = init_logger(__name__)

TS_TIMESTAMP_COL = "ds"
TS_VALUE_COL = "y"


def load_openml_dataset(dataset_id, data_dir=None, random_state=0, dataset_format="dataframe"):
    """Load dataset from open ML.

    If the file is not cached locally, download it from open ML.

    Args:
        dataset_id: An integer of the dataset id in openml.
        data_dir: A string of the path to store and load the data.
        random_state: An integer of the random seed for splitting data.
        dataset_format: A string specifying the format of returned dataset. Default is 'dataframe'.
            Can choose from ['dataframe', 'array'].
            If 'dataframe', the returned dataset will be a Pandas DataFrame.
            If 'array', the returned dataset will be a NumPy array or a SciPy sparse matrix.

    Returns:
        X_train: Training data.
        X_test:  Test data.
        y_train: A series or array of labels for training data.
        y_test:  A series or array of labels for test data.
    """
    import pickle

    try:
        import openml
    except ImportError:
        openml = None
    from sklearn.model_selection import train_test_split

    filename = "openml_ds" + str(dataset_id) + ".pkl"
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        logger.info("load dataset from %s", filepath)
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("download dataset from openml")
        dataset = openml.datasets.get_dataset(dataset_id) if openml else None
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if dataset:
        logger.info("Dataset name: %s", dataset.name)
    try:
        X, y, *__ = dataset.get_data(target=dataset.default_target_attribute, dataset_format=dataset_format)
    except (ValueError, AttributeError, TypeError):
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(data_id=dataset_id, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    logger.info(
        "X_train.shape: %s, y_train.shape: %s;\nX_test.shape: %s, y_test.shape: %s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, X_test, y_train, y_test


def load_openml_task(task_id, data_dir):
    """Load task from open ML.

    Use the first fold of the task.
    If the file is not cached locally, download it from open ML.

    Args:
        task_id: An integer of the task id in openml.
        data_dir: A string of the path to store and load the data.

    Returns:
        X_train: A dataframe of training data.
        X_test:  A dataframe of test data.
        y_train: A series of labels for training data.
        y_test:  A series of labels for test data.
    """
    import pickle

    import openml

    task = openml.tasks.get_task(task_id)
    filename = "openml_task" + str(task_id) + ".pkl"
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        logger.info("load_openml_task: loading task_id=%s from cache filepath=%r", task_id, filepath)
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("load_openml_task: downloading task_id=%s from openml", task_id)
        dataset = task.get_dataset()
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    X, y, _, _ = dataset.get_data(task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0,
        fold=0,
        sample=0,
    )
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    logger.info(
        "load_openml_task: task_id=%s X_train.shape=%s y_train.shape=%s X_test.shape=%s y_test.shape=%s",
        task_id,
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, X_test, y_train, y_test


def get_output_from_log(filename, time_budget):
    """Get output from log file.

    Args:
        filename: A string of the log file name.
        time_budget: A float of the time budget in seconds.

    Returns:
        search_time_list: A list of the finished time of each logged iter.
        best_error_list: A list of the best validation error after each logged iter.
        error_list: A list of the validation error of each logged iter.
        config_list: A list of the estimator, sample size and config of each logged iter.
        logged_metric_list: A list of the logged metric of each logged iter.
    """

    best_config = None
    best_learner = None
    best_val_loss = float("+inf")

    search_time_list = []
    config_list = []
    best_error_list = []
    error_list = []
    logged_metric_list = []
    best_config_list = []
    with training_log_reader(filename) as reader:
        for record in reader.records():
            time_used = record.wall_clock_time
            val_loss = record.validation_loss
            config = record.config
            learner = record.learner.split("_")[0]
            sample_size = record.sample_size
            metric = record.logged_metric

            if time_used < time_budget and np.isfinite(val_loss):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = config
                    best_learner = learner
                    best_config_list.append(best_config)
                search_time_list.append(time_used)
                best_error_list.append(best_val_loss)
                logged_metric_list.append(metric)
                error_list.append(val_loss)
                config_list.append(
                    {
                        "Current Learner": learner,
                        "Current Sample": sample_size,
                        "Current Hyper-parameters": record.config,
                        "Best Learner": best_learner,
                        "Best Hyper-parameters": best_config,
                    }
                )

    return (
        search_time_list,
        best_error_list,
        error_list,
        config_list,
        logged_metric_list,
    )


def concat(X1, X2):
    """concatenate two matrices vertically."""
    if type(X1) != type(X2):
        if isinstance(X2, (psDataFrame, psSeries)):
            X1 = ps.from_pandas(pd.DataFrame(X1))
        elif isinstance(X1, (psDataFrame, psSeries)):
            X2 = ps.from_pandas(pd.DataFrame(X2))
        else:
            X1 = pd.DataFrame(X1)
            X2 = pd.DataFrame(X2)

    if isinstance(X1, (DataFrame, Series)):
        df = pd.concat([X1, X2], sort=False)
        df.reset_index(drop=True, inplace=True)
        if isinstance(X1, DataFrame):
            cat_columns = X1.select_dtypes(include="category").columns
            if len(cat_columns):
                df[cat_columns] = df[cat_columns].astype("category")
        return df
    if isinstance(X1, (psDataFrame, psSeries)):
        df = ps.concat([X1, X2], ignore_index=True)
        if isinstance(X1, psDataFrame):
            cat_columns = X1.select_dtypes(include="category").columns.values.tolist()
            if len(cat_columns):
                df[cat_columns] = df[cat_columns].astype("category")
        return df
    if issparse(X1):
        return vstack((X1, X2))
    else:
        return np.concatenate([X1, X2])


def _nat_aware_int_series(values: pd.Series, index: pd.Index) -> pd.Series:
    """Build a nullable ``Int64`` Series from int-coercible values aligned to
    ``index``; rows whose ``values`` slot is missing become ``pd.NA``.

    Centralizes the NaT-aware shape used by the MS / yearly / mode-of-deltas
    branches of ``add_time_idx_col`` so they all return the same nullable
    ``Int64`` dtype instead of mixing ``int64`` and ``float64`` (which is
    what plain ``.dt.year`` + ``NaT`` would produce).
    """
    not_na = values.notna()
    out = pd.Series(pd.NA, index=index, dtype="Int64")
    if not_na.any():
        out.loc[not_na] = values[not_na].astype("int64").values
    return out


def add_time_idx_col(X):
    ts = X[TS_TIMESTAMP_COL]
    # Object-dtype columns (and pandas ``StringDtype`` columns) that hold
    # Python ``datetime`` objects (or strings parseable by pandas) are
    # legitimately datetime-like inputs from callers' perspective; coerce
    # them up-front so they don't get rejected by the dtype-guard below.
    # ``errors='coerce'`` turns unparseable values into ``NaT``, so the
    # subsequent ``dropna()`` filters them out and the guard then makes
    # the final call based on the resulting dtype. We only attempt this
    # for string-like dtypes (``object`` or pandas ``StringDtype``) to
    # avoid round-tripping already-typed datetime/numeric columns (which
    # could lose precision).
    if ts.dtype == object or isinstance(ts.dtype, pd.StringDtype):
        # Snapshot whether the ORIGINAL column had any non-null entries
        # so the post-coerce check below can distinguish "all-NA input"
        # (legitimately empty) from "non-null input that ALL coerced
        # to NaT" (every value was unparseable garbage). Without this,
        # a column of unparseable strings would silently produce an
        # all-``pd.NA`` ``time_idx`` instead of failing loudly.
        had_nonnull_before = ts.notna().any()
        try:
            ts = pd.to_datetime(ts, errors="coerce")
        except (ValueError, TypeError):
            # Coercion itself can raise on exotic object contents
            # (e.g., mixed tz). Leave ``ts`` as-is so the dtype guard
            # below produces the clearer "convert with pd.to_datetime
            # first" message instead of an opaque conversion error.
            ts = X[TS_TIMESTAMP_COL]
        else:
            # Conversion succeeded structurally, but if it coerced
            # EVERY non-null input to ``NaT`` the column is unusable
            # for time_idx — raise the same dtype-guard error the
            # branch below would raise for a fully non-datetime dtype.
            if had_nonnull_before and not ts.notna().any():
                raise TypeError(
                    f"Column {TS_TIMESTAMP_COL!r} must be datetime-like to compute time_idx; "
                    "all non-null values were unparseable. Convert with pd.to_datetime first."
                )
            # Keep the returned DataFrame internally consistent: if we
            # successfully coerced ``ds`` to datetime for computing
            # ``time_idx``, write it back to the DataFrame too so any
            # downstream code (forecasting estimators, plotting, etc.)
            # that re-reads ``X[TS_TIMESTAMP_COL]`` sees the same
            # datetime dtype that ``time_idx`` was derived from. Use
            # ``X.loc[:, TS_TIMESTAMP_COL]`` instead of the bracket
            # assignment ``X[TS_TIMESTAMP_COL] = ts``: the latter can
            # trigger ``SettingWithCopyWarning`` when ``X`` is a view
            # of another DataFrame (e.g., a user passing
            # ``df.loc[mask]`` directly into ``add_time_idx_col``),
            # and on newer pandas may even silently fail to mutate the
            # underlying buffer. ``.loc`` performs an explicit label-
            # based assignment that pandas guarantees acts on ``X``
            # itself.
            X.loc[:, TS_TIMESTAMP_COL] = ts
    unique_dates = ts.dropna().drop_duplicates()
    # Require datetime-like input. Without this, a column of object strings
    # (or other non-datetime dtype) would silently pass ``pd.infer_freq``
    # (which raises ``TypeError`` we swallow below) and then fail with an
    # opaque ``TypeError`` deep inside ``.astype("int64")``. This guard
    # runs BEFORE ``sort_values`` so a column of mixed object dtypes
    # (e.g., tz-aware datetimes mixed with strings, or strings mixed
    # with ints) raises this clearer TypeError instead of the opaque
    # ``'<' not supported between instances of ...`` that
    # ``Series.sort_values`` would produce on unorderable mixed values.
    if len(unique_dates) and not (
        pd.api.types.is_datetime64_any_dtype(unique_dates) or isinstance(unique_dates.dtype, pd.DatetimeTZDtype)
    ):
        raise TypeError(
            f"Column {TS_TIMESTAMP_COL!r} must be datetime-like to compute time_idx; "
            f"got dtype {unique_dates.dtype!r}. Convert with pd.to_datetime first."
        )
    try:
        unique_dates = unique_dates.sort_values(ascending=True)
    except TypeError as exc:
        # ``sort_values`` can still raise ``TypeError`` on a datetime
        # Series in pathological mixed-tz situations that the dtype
        # guard above doesn't cover (e.g., an ``object`` Series whose
        # type-check we somehow bypassed). Surface the same clearer
        # guidance as the dtype guard above instead of letting the
        # low-level ordering error propagate.
        raise TypeError(
            f"Column {TS_TIMESTAMP_COL!r} must be datetime-like to compute time_idx; "
            "Convert with pd.to_datetime first."
        ) from exc
    try:
        # ``pd.infer_freq`` raises ``TypeError`` on tz-aware Series in older
        # pandas versions, which would otherwise misclassify a regular
        # monthly/yearly tz-aware series and force it down the mode-of-
        # deltas branch (variable month lengths cause drift). Localize to
        # tz-naive first (preserving the local timestamps the user chose)
        # so the calendar-frequency branches below still apply.
        if isinstance(unique_dates.dtype, pd.DatetimeTZDtype):
            freq_input = unique_dates.dt.tz_localize(None)
        else:
            freq_input = unique_dates
        freq = pd.infer_freq(freq_input)
    except (TypeError, ValueError):
        # pd.infer_freq needs >= 3 dates (ValueError) and older pandas
        # raises TypeError for tz-aware datetime Series — fall back to the
        # mode-of-deltas branch in either case.
        freq = None
    if freq in ("MS", "M", "ME", "BMS", "BM", "BME"):
        # Treat month-start and month-end frequencies (including the
        # business variants) identically: compute an integer
        # calendar-month index. Month lengths vary (28-31 days), so the
        # fixed mode-of-deltas branch below would drift over longer
        # ranges and fail to produce exactly one step per calendar
        # month. ``.dt.year * 12 + .dt.month`` would promote NaT rows to
        # ``NaN`` (float64), inconsistent with the mode-of-deltas branch
        # below that returns nullable Int64 with ``pd.NA`` for NaT.
        # Build the result via ``_nat_aware_int_series`` so all branches
        # behave identically. Use the local ``ts`` (which may have been
        # coerced from object dtype above) rather than re-reading
        # ``X[TS_TIMESTAMP_COL]`` — re-reading would silently revert to
        # the original object dtype and break ``ts.dt`` access.
        raw = ts.dt.year * 12 + ts.dt.month
        X["time_idx"] = _nat_aware_int_series(raw, X.index)
    elif freq is not None and freq.startswith(("Y", "A", "BY", "BA")):
        # Use the calendar year as the integer index for yearly
        # frequencies. ``pd.infer_freq`` returns several aliases that
        # all map to "one row per year":
        #   * ``Y-DEC`` / ``YE-DEC`` — year-end (year-aligned)
        #   * ``YS-JAN`` / ``AS-JAN`` — year-start
        #   * ``A-DEC`` — legacy year-end (still emitted by older pandas)
        #   * ``BA-DEC`` / ``BAS-JAN`` — business-year-end / -start
        #   * ``BY-DEC`` / ``BYS-JAN`` — newer business-year aliases
        # The first four all start with ``"Y"`` or ``"A"``; the
        # business variants start with ``"BY"`` or ``"BA"``. Without
        # the business-year prefixes those series fall through to the
        # mode-of-deltas branch and get nanosecond-elapsed indexes
        # instead of integer year indexes, which produces dense
        # multi-million step values that downstream forecasters
        # interpret as a completely different time axis.
        X["time_idx"] = _nat_aware_int_series(ts.dt.year, X.index)
    else:
        # Use the mode of time deltas as the step; handles irregular data
        # like business days with holidays removed.
        # Modern pandas (>= 2.0) emits ``FutureWarning`` for
        # ``Series.astype("int64")`` on datetime64[ns] dtypes and a
        # future release may remove that coercion entirely. Drop to a
        # numpy ndarray first and ``.view("int64")`` it — the canonical
        # zero-copy way to access the nanosecond integer representation
        # of a datetime64[ns] array. ``.view()`` is also valid on
        # tz-naive datetime64 ndarrays but undefined on tz-aware
        # ``DatetimeTZDtype`` Series, so normalize to tz-naive (via
        # ``tz_localize(None)`` below, which drops the timezone label
        # while preserving the LOCAL wall-clock time — NOT the UTC
        # instant; see the DST rationale in the per-branch comment)
        # before the conversion.
        unique_for_int = unique_dates
        if isinstance(unique_for_int.dtype, pd.DatetimeTZDtype):
            # Strip the timezone (preserve LOCAL wall-clock spacing) rather
            # than converting to UTC. Around DST transitions
            # ``tz_convert("UTC")`` turns local-midnight timestamps into
            # 23-hour or 25-hour gaps when measured in elapsed UTC
            # nanoseconds. Dividing those uneven deltas by a 24h mode
            # step then produces duplicate or skipped ``time_idx`` values
            # for the DST-affected days, silently corrupting the per-row
            # ordering. ``tz_localize(None)`` keeps every local
            # day-over-day delta at the same 86_400e9 ns, matching what
            # the user observes on the wall clock. This also aligns with
            # the earlier ``infer_freq`` input handling, which already
            # operates on wall-clock spacing.
            unique_for_int = unique_for_int.dt.tz_localize(None)
        timestamps_int64 = unique_for_int.to_numpy(dtype="datetime64[ns]", copy=False).view("int64")
        # Compute deltas via ``np.diff`` on the int64 numpy array rather
        # than ``timestamps.diff()``. ``Series.diff()`` produces float64
        # (the first element becomes ``NaN``), and ``float64`` can only
        # represent integers exactly up to 2**53; for nanosecond steps
        # above that (~104 days), the mode/step computation would lose
        # precision and shift the resulting ``time_idx`` by a fractional
        # amount. ``np.diff`` returns the differences as ``int64`` and
        # implicitly drops the would-be NaN (no leading sentinel), so
        # the explicit ``dropna()`` is no longer needed.
        #
        # ``np.diff`` on an ``int64`` array silently overflows when two
        # consecutive timestamps differ by more than ``2**63 - 1``
        # nanoseconds (~292 years), producing negative/garbled deltas
        # that then corrupt the ``step`` mode and every downstream
        # ``time_idx`` value. The ``datetime64[ns]`` representable
        # range itself is only ~584 years end-to-end, so this can
        # realistically only occur for a near-full-range pair, but
        # detect it via the overall span (computed in Python big-int
        # to avoid the same overflow when measuring) and route the
        # diff through ``astype(object)`` for exact Python-int
        # subtraction in that rare regime.
        if timestamps_int64.size:
            span_ns = int(timestamps_int64.max()) - int(timestamps_int64.min())
        else:
            span_ns = 0
        if span_ns > np.iinfo(np.int64).max:
            deltas_np = np.diff(timestamps_int64.astype(object))
        else:
            deltas_np = np.diff(timestamps_int64)
        # Use the local ``ts`` (coerced if applicable) rather than
        # ``X[TS_TIMESTAMP_COL]`` to keep the per-row index calculation
        # consistent with ``unique_for_int`` above.
        ts_for_int = ts
        if isinstance(ts_for_int.dtype, pd.DatetimeTZDtype):
            # Match the ``unique_for_int`` tz-handling above. Using
            # ``tz_localize(None)`` keeps per-row timestamps on the same
            # wall-clock axis as the mode-step computation, so the
            # ``(valid_int - origin) // step`` arithmetic stays
            # consistent across DST boundaries.
            ts_for_int = ts_for_int.dt.tz_localize(None)
        not_na = ts_for_int.notna()
        # Always assign nullable ``Int64`` regardless of NaT presence, so the
        # dtype of ``time_idx`` is identical across the empty-deltas,
        # multi-step, and MS/yearly branches. Otherwise the
        # ``deltas.empty & not_na.all()`` and ``deltas-present & not_na.all()``
        # paths would produce non-nullable ``int64``, breaking downstream
        # code that relies on a single dtype.
        time_idx = pd.Series(pd.NA, index=X.index, dtype="Int64")
        if deltas_np.size == 0:
            # No usable step (e.g., 0 or 1 unique non-null timestamp). Assign
            # 0 to rows with valid timestamps and pd.NA to rows whose
            # timestamp is NaT — consistent with the multi-step branch
            # below, which also returns nullable Int64 with pd.NA for NaT.
            if not_na.any():
                time_idx.loc[not_na] = 0
        else:
            # ``pd.Series(deltas_np).mode()`` preserves the int64 dtype
            # (no NaN to promote to float). If there are multiple
            # equally common deltas (multi-modal — e.g., data that
            # alternates between two regular intervals like a daily
            # schedule with weekend gaps, or a 9-to-5 trading calendar
            # with an extra overnight delta), ``.iloc[0]`` would
            # silently pick whichever appeared first in the sorted
            # ``mode()`` output. Picking an unexpectedly small step
            # then compresses ``time_idx`` because non-step rows get
            # forced into ``(delta // step)`` integer slots that
            # don't reflect the original cadence. Choose a
            # deterministic rule: pick the LARGEST mode. This
            # preserves the coarser cadence and matches what users
            # typically mean when they have, e.g., a daily series
            # that occasionally double-samples — the daily step is
            # the "natural" period, the smaller intra-day delta is
            # a fluke.
            #
            # Cast the scalar result back to a Python ``int`` so the
            # floor-division below operates on Python big-ints when
            # ``step`` exceeds 2**63 — avoids int64 overflow for
            # pathological huge deltas.
            modes = pd.Series(deltas_np).mode()
            step = int(modes.max() if len(modes) > 1 else modes.iloc[0])
            origin = int(timestamps_int64.min())
            if not_na.any():
                # Keep the per-row arithmetic in int64 too: subtract via
                # numpy then floor-divide, then wrap in a Series for the
                # ``time_idx.loc`` assignment. This avoids the float64
                # round-trip that ``Series.astype('int64') - origin``
                # might trigger under nullable-dtype edge cases. Use
                # ``.to_numpy(...).view("int64")`` for the same reason
                # as ``timestamps_int64`` above.
                #
                # ``(valid_int - origin)`` is computed in ``int64`` and
                # can overflow when the per-row span ``valid_int.max() -
                # origin`` exceeds ``np.iinfo(np.int64).max`` — mirroring
                # the ``np.diff`` overflow guard above. Casting ``step``
                # to ``int`` is necessary but not sufficient: the
                # numerator must also be in big-int form. Detect the
                # rare wide-span regime via Python-int subtraction and
                # route through ``astype(object)`` only when needed so
                # the common path stays vectorized int64.
                valid_int = ts_for_int[not_na].to_numpy(dtype="datetime64[ns]", copy=False).view("int64")
                if valid_int.size:
                    valid_span = int(valid_int.max()) - origin
                else:
                    valid_span = 0
                if valid_span > np.iinfo(np.int64).max:
                    # Compute the per-row indexes in Python big-int via
                    # the object-dtype path. The resulting ``valid_idx``
                    # is a numpy object array of Python ints — any of
                    # which could themselves exceed ``int64`` for
                    # pathological spans. Assigning such values into a
                    # nullable ``Int64`` Series raises ``OverflowError``
                    # during the implicit cast. Check whether the
                    # computed indices actually fit in ``int64`` before
                    # forcing the cast: if they do, materialize as an
                    # ``int64`` ndarray for an efficient Int64
                    # assignment; if they don't, fall back to object
                    # dtype for ``time_idx`` so the wide-span
                    # correctness fix in the comment above doesn't
                    # silently regress to an exception during
                    # assignment.
                    valid_idx_obj = (valid_int.astype(object) - origin) // step
                    int64_min = int(np.iinfo(np.int64).min)
                    int64_max = int(np.iinfo(np.int64).max)
                    if valid_idx_obj.size:
                        idx_min = int(min(valid_idx_obj))
                        idx_max = int(max(valid_idx_obj))
                    else:
                        idx_min = 0
                        idx_max = 0
                    if idx_min < int64_min or idx_max > int64_max:
                        time_idx = time_idx.astype(object)
                        time_idx.loc[not_na] = list(valid_idx_obj)
                    else:
                        time_idx.loc[not_na] = np.asarray(valid_idx_obj, dtype="int64")
                else:
                    valid_idx = (valid_int - origin) // step
                    time_idx.loc[not_na] = valid_idx
        X["time_idx"] = time_idx
    return X


class DataTransformer:
    """Transform input training data."""

    def fit_transform(self, X: Union[DataFrame, np.ndarray], y, task: Union[str, "Task"]):
        """Fit transformer and process the input training data according to the task type.

        Args:
            X: A numpy array or a pandas dataframe of training data.
            y: A numpy array or a pandas series of labels.
            task: An instance of type Task, or a str such as 'classification', 'regression'.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
            y: Processed numpy array or pandas series of labels.
        """
        if isinstance(task, str):
            from flaml.automl.task.factory import task_factory

            task = task_factory(task, X, y)

        if task.is_nlp():
            # if the mode is NLP, check the type of input, each column must be either string or
            # ids (input ids, token type id, attention mask, etc.)
            str_columns = []
            for column in X.columns:
                if isinstance(X[column].iloc[0], str):
                    str_columns.append(column)
            if len(str_columns) > 0:
                X[str_columns] = X[str_columns].astype("string")
            self._str_columns = str_columns
        elif isinstance(X, DataFrame):
            X = X.copy()
            n = X.shape[0]
            cat_columns, num_columns, datetime_columns = [], [], []
            drop = False
            if task.is_ts_forecast():
                X = X.rename(columns={X.columns[0]: TS_TIMESTAMP_COL})
                if task.is_ts_forecastpanel():
                    if "time_idx" not in X:
                        X = add_time_idx_col(X)
                ds_col = X.pop(TS_TIMESTAMP_COL)
                if isinstance(y, Series):
                    y = y.rename(TS_VALUE_COL)
            for column in X.columns:
                # sklearn\utils\validation.py needs int/float values
                if X[column].dtype.name in ("object", "category", "string", "str"):
                    if X[column].nunique() == 1 or X[column].nunique(dropna=True) == n - X[column].isnull().sum():
                        X.drop(columns=column, inplace=True)
                        drop = True
                    elif X[column].dtype.name == "category":
                        current_categories = X[column].cat.categories
                        if "__NAN__" not in current_categories:
                            X[column] = X[column].cat.add_categories("__NAN__").fillna("__NAN__")
                        cat_columns.append(column)
                    else:
                        X[column] = X[column].fillna("__NAN__")
                        cat_columns.append(column)
                elif X[column].nunique(dropna=True) < 2:
                    X.drop(columns=column, inplace=True)
                    drop = True
                else:  # datetime or numeric
                    if is_datetime64_any_dtype is not None and is_datetime64_any_dtype(X[column]):
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
                            if key not in X.columns and value.nunique(dropna=False) >= 2:
                                X[key] = value
                                num_columns.append(key)
                        X[column] = X[column].map(datetime.toordinal)
                        datetime_columns.append(column)
                        del tmp_dt
                    X[column] = X[column].fillna(np.nan)
                    num_columns.append(column)
            X = X[cat_columns + num_columns]
            if task.is_ts_forecast():
                X.insert(0, TS_TIMESTAMP_COL, ds_col)
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns]
                try:
                    is_int_cols = np.issubdtype(X_num.columns.dtype, np.integer)
                except TypeError:
                    is_int_cols = False
                if is_int_cols and (drop or min(X_num.columns) != 0 or max(X_num.columns) != X_num.shape[1] - 1):
                    X_num.columns = range(X_num.shape[1])
                    drop = True
                else:
                    drop = False
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer

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
        if task.is_classification() or not pd.api.types.is_numeric_dtype(y) and not task.is_nlg():
            if not task.is_token_classification():
                from sklearn.preprocessing import LabelEncoder

                self.label_transformer = LabelEncoder()
            else:
                from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

                self.label_transformer = LabelEncoderforTokenClassification()
            y = self.label_transformer.fit_transform(y)
        else:
            self.label_transformer = None
        self._task = task
        return X, y

    def __sklearn_is_fitted__(self):
        """sklearn 1.8 hook used by `check_is_fitted` (e.g. when this
        transformer is wrapped in a Pipeline). `_task` is the canonical
        attribute set by `fit_transform`."""
        return hasattr(self, "_task")

    def transform(self, X: Union[DataFrame, np.array]):
        """Process data using fit transformer.

        Args:
            X: A numpy array or a pandas dataframe of training data.

        Returns:
            X: Processed numpy array or pandas dataframe of training data.
        """
        X = X.copy()

        if self._task.is_nlp():
            # if the mode is NLP, check the type of input, each column must be either string or
            # ids (input ids, token type id, attention mask, etc.)
            if len(self._str_columns) > 0:
                X[self._str_columns] = X[self._str_columns].astype("string")
        elif isinstance(X, DataFrame):
            cat_columns, num_columns, datetime_columns = (
                self._cat_columns,
                self._num_columns,
                self._datetime_columns,
            )
            if self._task.is_ts_forecast():
                X = X.rename(columns={X.columns[0]: TS_TIMESTAMP_COL})
                ds_col = X.pop(TS_TIMESTAMP_COL)
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
            if self._task.is_ts_forecast():
                X.insert(0, TS_TIMESTAMP_COL, ds_col)
            for column in cat_columns:
                if X[column].dtype.name in ("object", "string", "str"):
                    X[column] = X[column].fillna("__NAN__")
                elif X[column].dtype.name == "category":
                    current_categories = X[column].cat.categories
                    if "__NAN__" not in current_categories:
                        X[column] = X[column].cat.add_categories("__NAN__").fillna("__NAN__")
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype("category")
            if num_columns:
                X_num = X[num_columns].fillna(np.nan)
                if self._drop:
                    X_num.columns = range(X_num.shape[1])
                X[num_columns] = self.transformer.transform(X_num)
        return X


def group_counts(groups):
    _, i, c = np.unique(groups, return_counts=True, return_index=True)
    return c[np.argsort(i)]


def get_random_dataframe(n_rows: int = 200, ratio_none: float = 0.1, seed: int = 42) -> DataFrame:
    """Generate a random pandas DataFrame with various data types for testing.
    This function creates a DataFrame with multiple column types including:
    - Timestamps
    - Integers
    - Floats
    - Categorical values
    - Booleans
    - Lists (tags)
    - Decimal strings
    - UUIDs
    - Binary data (as hex strings)
    - JSON blobs
    - Nullable text fields
    Parameters
    ----------
    n_rows : int, default=200
        Number of rows in the generated DataFrame
    ratio_none : float, default=0.1
        Probability of generating None values in applicable columns
    seed : int, default=42
        Random seed for reproducibility
    Returns
    -------
    pd.DataFrame
        A DataFrame with 14 columns of various data types
    Examples
    --------
    >>> df = get_random_dataframe(100, 0.05, 123)
    >>> df.shape
    (100, 14)
    >>> df.dtypes
    timestamp       datetime64[ns]
    id                       int64
    score                  float64
    status                  object
    flag                    object
    count                   object
    value                   object
    tags                    object
    rating                  object
    uuid                    object
    binary                  object
    json_blob               object
    category              category
    nullable_text           object
    dtype: object
    """

    np.random.seed(seed)
    random.seed(seed)

    def random_tags():
        tags = ["AI", "ML", "data", "robotics", "vision"]
        return random.sample(tags, k=random.randint(1, 3)) if random.random() > ratio_none else None

    def random_decimal():
        return (
            str(Decimal(random.uniform(1, 5)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            if random.random() > ratio_none
            else None
        )

    def random_json_blob():
        blob = {"a": random.randint(1, 10), "b": random.random()}
        return json.dumps(blob) if random.random() > ratio_none else None

    def random_binary():
        return bytes(random.randint(0, 255) for _ in range(4)).hex() if random.random() > ratio_none else None

    data = {
        "timestamp": [
            datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1000)) if np.random.rand() > ratio_none else None
            for _ in range(n_rows)
        ],
        "id": range(1, n_rows + 1),
        "score": np.random.uniform(0, 100, n_rows),
        "status": np.random.choice(
            ["active", "inactive", "pending", None],
            size=n_rows,
            p=[(1 - ratio_none) / 3, (1 - ratio_none) / 3, (1 - ratio_none) / 3, ratio_none],
        ),
        "flag": np.random.choice(
            [True, False, None], size=n_rows, p=[(1 - ratio_none) / 2, (1 - ratio_none) / 2, ratio_none]
        ),
        "count": [np.random.randint(0, 100) if np.random.rand() > ratio_none else None for _ in range(n_rows)],
        "value": [round(np.random.normal(50, 15), 2) if np.random.rand() > ratio_none else None for _ in range(n_rows)],
        "tags": [random_tags() for _ in range(n_rows)],
        "rating": [random_decimal() for _ in range(n_rows)],
        "uuid": [str(uuid.uuid4()) if np.random.rand() > ratio_none else None for _ in range(n_rows)],
        "binary": [random_binary() for _ in range(n_rows)],
        "json_blob": [random_json_blob() for _ in range(n_rows)],
        "category": pd.Categorical(
            np.random.choice(
                ["A", "B", "C", None],
                size=n_rows,
                p=[(1 - ratio_none) / 3, (1 - ratio_none) / 3, (1 - ratio_none) / 3, ratio_none],
            )
        ),
        "nullable_text": [random.choice(["Good", "Bad", "Average", None]) for _ in range(n_rows)],
    }

    return pd.DataFrame(data)


# Allowlist of Spark SQL types accepted by ``_spark_try_cast``. Strict
# matching prevents arbitrary tokens from being interpolated into the
# generated ``try_cast(... as <type>)`` expression by an upstream caller.
_SPARK_CAST_ALLOWED_TYPES = frozenset(
    {"int", "bigint", "long", "double", "float", "string", "boolean", "timestamp", "date"}
)

# Tolerance used by ``infer_pyspark_data_types`` to decide whether a sampled
# double column is effectively a column of whole numbers, AND by the matching
# cast at apply-time to decide whether an individual value should be coerced
# from double to int. Both sites MUST use the same constant: an inference
# tolerance looser than the cast tolerance would silently demote rows that
# passed inference (e.g., ``"1.0000005"`` is "whole enough" at 1e-6, but
# would be NULL'd out at 1e-9), turning the inferred ``int`` column into
# one with surprise nulls that did not originate from the source data.
# Picked at 1e-6 to absorb the float64 representation noise typical of
# decimal strings round-tripped through ``try_cast(... as double)`` without
# accepting values whose fractional part is clearly non-zero (e.g., ``0.1``).
_SPARK_INT_WHOLE_TOL = 1e-6

# Float64 mantissa is 52 bits, so integers in the range ``[-2**53, 2**53]``
# (53 bits including the implicit leading 1) are exactly representable.
# Values at or beyond ``2**53`` start losing the unit's place — e.g.,
# ``2**53 + 1`` rounds to ``2**53`` on conversion to ``float64``. Used by
# ``auto_convert_dtypes_pandas`` to gate precision-preserving Int64
# re-derivation in the ``pd.to_numeric`` fallback path.
_FLOAT64_SAFE_INT_CEILING = 2**53


def _spark_try_cast(col_name: str, sql_type: str):
    """Cast ``col_name`` to ``sql_type`` returning NULL for unparseable values.

    Wraps Spark SQL's ``try_cast`` (available since Spark 3.2) so the call does
    not raise ``NumberFormatException`` under ANSI mode, which is enabled by
    default in Spark 4.x. Embedded backticks in the column name are escaped by
    doubling so the generated expression remains valid. ``sql_type`` is
    checked against ``_SPARK_CAST_ALLOWED_TYPES`` so untrusted callers can't
    inject additional SQL tokens via the type string.
    """
    if sql_type not in _SPARK_CAST_ALLOWED_TYPES:
        raise ValueError(
            f"_spark_try_cast: unsupported sql_type {sql_type!r}; "
            f"expected one of {sorted(_SPARK_CAST_ALLOWED_TYPES)}"
        )
    escaped = col_name.replace("`", "``")
    return F.expr(f"try_cast(`{escaped}` as {sql_type})")


def _validate_na_values_list(na_values, *, arg_name: str = "na_values"):
    """Validate ``na_values`` and return it materialized as a ``list[str]``.

    Both ``auto_convert_dtypes_pandas`` and ``auto_convert_dtypes_spark``
    document ``na_values`` as a list of strings and later call
    ``str.lower`` / ``re.escape`` on each element. Three classes of bad
    input are caught here so failures happen at the public entry point
    instead of deep inside the per-column loop:

    1. ``str`` / ``bytes`` for ``na_values`` itself — iterating would
       silently treat it as a sequence of single characters (e.g.,
       ``na_values="NA"`` would behave like ``["N", "A"]``).
    2. Non-``str`` elements (``None``, numbers, ``bytes``, …) — would
       otherwise raise an opaque ``AttributeError`` from ``v.lower()``
       or ``TypeError`` from ``re.escape(v)``.
    3. One-shot iterables (generators / iterators) — would be consumed
       by the per-element check below and then behave as empty when
       the caller iterates them again to build the lowered list /
       regex. Materializing to ``list`` once here avoids that pitfall.

    The error message names the offending index and type so callers
    can immediately see the problem. Returns the materialized list so
    callers can re-bind the local name and use it safely downstream.
    """
    if isinstance(na_values, (str, bytes)):
        raise TypeError(
            f"{arg_name} must be a non-string iterable of str (e.g., list/tuple), "
            f"got {type(na_values).__name__}: {na_values!r}"
        )
    try:
        materialized = list(na_values)
    except TypeError as exc:
        raise TypeError(
            f"{arg_name} must be a non-string iterable of str (e.g., list/tuple), "
            f"got non-iterable {type(na_values).__name__}: {na_values!r}"
        ) from exc
    for i, v in enumerate(materialized):
        if not isinstance(v, str):
            raise TypeError(f"{arg_name}[{i}] must be a str, got {type(v).__name__}: {v!r}")
    return materialized


def _validate_skip_columns(skip_columns, *, arg_name: str = "skip_columns", require_str: bool = False):
    """Validate ``skip_columns`` and return it materialized as a ``list`` (or ``None``).

    ``set(skip_columns)`` would silently expand a bare ``str``/``bytes``
    into its characters (``set("abc") == {"a","b","c"}``), causing
    unrelated single-character columns to be skipped. Reject those
    upfront and materialize generators / iterators to ``list`` so callers
    can iterate them more than once without surprises (e.g., consuming
    a generator during validation and then having ``set(skip_columns)``
    see an empty iterable).

    By default, allow any *hashable* column labels — pandas DataFrames
    legitimately have non-string column labels (``int``, ``tuple``, …),
    and ``set()``-based membership only needs hashability. When
    ``require_str=True`` (Spark callers, where column names are always
    strings), enforce ``str`` elements as well. ``None`` is allowed and
    means "skip no columns"; ``None`` is returned unchanged so callers
    can keep their existing ``set(skip_columns) if skip_columns else
    set()`` pattern.
    """
    if skip_columns is None:
        return None
    if isinstance(skip_columns, (str, bytes)):
        raise TypeError(
            f"{arg_name} must be None or a non-string iterable of column labels, "
            f"got {type(skip_columns).__name__}: {skip_columns!r}"
        )
    try:
        materialized = list(skip_columns)
    except TypeError as exc:
        raise TypeError(
            f"{arg_name} must be None or a non-string iterable of column labels, "
            f"got non-iterable {type(skip_columns).__name__}: {skip_columns!r}"
        ) from exc
    for i, v in enumerate(materialized):
        if require_str:
            if not isinstance(v, str):
                raise TypeError(f"{arg_name}[{i}] must be a str, got {type(v).__name__}: {v!r}")
        else:
            try:
                hash(v)
            except TypeError as exc:
                raise TypeError(f"{arg_name}[{i}] must be hashable, got {type(v).__name__}: {v!r}") from exc
    return materialized


def auto_convert_dtypes_spark(
    df: psDataFrame,
    na_values: list = None,
    category_threshold: float = 0.3,
    convert_threshold: float = 0.6,
    sample_ratio: float = 0.1,
    skip_columns: list = None,
) -> tuple[psDataFrame, dict]:
    """Automatically convert data types in a PySpark DataFrame using heuristics.

    This function analyzes a sample of the DataFrame to infer appropriate data types
    and applies the conversions. It handles timestamps, numeric values, and
    categorical (low-cardinality) string fields. Already-typed boolean columns are
    coerced to ``int`` (Spark ``bool``→``int``) so downstream pipelines that expect
    numeric features keep working; boolean *detection* from string values
    (``"true"``/``"false"``/``"yes"``/``"no"``/``"0"``/``"1"``) is **not**
    performed — such columns are inferred as ``"category"`` (low cardinality) or
    ``"int"`` (``"0"``/``"1"`` strings cast through ``try_cast(... as int)``)
    depending on the values present.

    Args:
        df: A PySpark DataFrame to convert.
        na_values: List of strings to be considered as NA/NaN. Defaults to
            ['NA', 'na', 'NULL', 'null', ''].
        category_threshold: Maximum ratio of unique values to total values
            to consider a column categorical. Defaults to 0.3.
        convert_threshold: Minimum ratio of successfully converted values required
            to apply a type conversion. Defaults to 0.6.
        sample_ratio: Fraction of data to sample for type inference. Defaults to 0.1.
        skip_columns: Optional iterable of column names to leave untouched.
            Listed columns are not analyzed, not NA-normalized, and keep their
            original Spark dtype. They still appear in the returned schema
            dict with their existing dtype. Defaults to None (no skipping).

    Returns:
        tuple: (The DataFrame with converted types, A dictionary mapping column names to
                their inferred types as strings)

    Note:
        - 'category' in the schema dict is conceptual as PySpark doesn't have a true
            category type like pandas
        - The function uses sampling for efficiency with large datasets
    """
    if not (0 < sample_ratio <= 1):
        # Match the pandas-path validation so misuses are caught
        # consistently across both code paths. ``0`` or negative
        # values would silently bottom out on the 100-row floor in
        # the sampling math below, and values greater than 1 would
        # silently become a full sample — surprising callers either
        # way. Validate BEFORE ``df.count()`` so an invalid call
        # doesn't trigger a Spark job (full-input scan) only to
        # immediately fail.
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio!r}")
    n_rows = df.count()
    if na_values is None:
        na_values = ["NA", "na", "NULL", "null", ""]
    na_values = _validate_na_values_list(na_values)
    skip_columns = _validate_skip_columns(skip_columns, require_str=True)
    skip_set = set(skip_columns) if skip_columns else set()
    # Precompute the lowered NA values once; reused for every string column
    # below so we don't rebuild the list per column inside the Spark loop.
    na_values_lowered = [v.lower() for v in na_values]

    # Empty DataFrame: there's no data to sample, infer from, or normalize;
    # still apply the same dtype rules (notably boolean → int) so the
    # returned schema/dtypes are consistent with the non-empty path. Reaching
    # the main loop below would compute ``100 / n_rows`` and divide by zero.
    if n_rows == 0:
        original_dtypes = dict(df.dtypes)
        schema = {}
        for colname in df.columns:
            if colname in skip_set:
                schema[colname] = original_dtypes[colname]
            else:
                schema[colname] = "int" if original_dtypes[colname] == "boolean" else original_dtypes[colname]
        for colname, inferred_type in schema.items():
            if colname in skip_set:
                continue
            if inferred_type == "int":
                df = df.withColumn(colname, _spark_try_cast(colname, "int"))
        return df, schema

    # Normalize NA-like values (skip columns in ``skip_set``)
    for colname, coltype in df.dtypes:
        if coltype == "string" and colname not in skip_set:
            df = df.withColumn(
                colname,
                F.when(F.trim(F.lower(F.col(colname))).isin(na_values_lowered), None).otherwise(F.col(colname)),
            )

    schema = {}
    original_dtypes = dict(df.dtypes)
    # ``sample_ratio_to_use`` depends only on ``n_rows`` and ``sample_ratio``
    # (constant across all columns), so compute it once outside the loop
    # rather than re-deriving it per column. ``n_rows`` is guaranteed > 0
    # here because the empty-DataFrame early-return short-circuited above.
    sample_ratio_to_use = min(1.0, sample_ratio if n_rows * sample_ratio > 100 else 100 / n_rows)
    for colname in df.columns:
        # Honor skip_columns: record the existing dtype and move on.
        if colname in skip_set:
            schema[colname] = original_dtypes[colname]
            continue

        # Sample once at an appropriate ratio (constant across columns).
        col_sample = df.select(colname).sample(withReplacement=False, fraction=sample_ratio_to_use).dropna()

        inferred_type = "string"  # Default

        if col_sample.dtypes[0][1] != "string":
            existing = col_sample.dtypes[0][1]
            # Cast already-boolean columns to int (1/0/null). FLAML's
            # downstream estimators treat booleans as numeric features
            # anyway, and a bool dtype trips sklearn's early-conversion
            # path when mixed with string categoricals in a single
            # ColumnTransformer.
            if existing == "boolean":
                schema[colname] = "int"
            else:
                schema[colname] = existing
            continue

        # Consolidate ALL per-column metrics into a single ``.agg(...)``
        # so the entire inference for this column requires exactly one
        # Spark action (one job) instead of the historical 2-3 actions
        # (``col_sample.count()`` + ``numeric_sample.count()`` + one of
        # ``filter(...).count()`` / ``ts_col.filter(...).count()`` /
        # ``countDistinct().collect()``). For wide tables (hundreds of
        # columns) the per-action scheduler / task-setup overhead
        # dominates the actual work on a 10%-sample, so collapsing
        # them yields a large per-column speedup at no semantic cost
        # (the agg expressions encode exactly the same predicates
        # the prior per-action code used).
        #
        # ``_spark_try_cast`` returns NULL for malformed values rather
        # than raising under Spark ANSI mode; reusing the same cast
        # expression in multiple aggregate sums lets Catalyst dedupe
        # the projection in the physical plan. Filter both null AND
        # NaN values from the cast-to-double sample: Spark treats
        # ``Double.NaN`` as non-null (``isNotNull()`` returns True
        # for NaN), but a NaN double makes ``F.col("n") % 1``
        # evaluate to NaN — and the ``> _SPARK_INT_WHOLE_TOL``
        # predicate against NaN yields ``null`` (treated as
        # ``false`` by ``.filter``). The whole-number test would
        # therefore quietly count NaN rows as "whole", letting a
        # column with NaN-like values infer as ``int`` and then
        # silently null-out those rows on the subsequent
        # ``try_cast(... as int)``. Apply the same reasoning to
        # +/-Infinity values: an overflow-to-infinity ``try_cast``
        # (e.g., string ``"1e400"``) passes ``isNotNull()`` AND
        # ``~isnan()``, but ``inf % 1`` evaluates to ``null`` in
        # Spark, so the ``> tol`` predicate also yields ``null``
        # (false in filter), letting infinities be silently
        # counted as "whole" and skewing inference toward ``int``.
        # Add ``~isin(inf, -inf)`` to the predicate so whole-number
        # inference operates strictly on finite numerics.
        double_col = _spark_try_cast(colname, "double")
        is_finite_num = double_col.isNotNull() & ~F.isnan(double_col) & ~double_col.isin(float("inf"), float("-inf"))
        ts_parsed = _spark_try_cast(colname, "timestamp")
        metrics = col_sample.agg(
            F.count("*").alias("sample_count"),
            F.sum(F.when(is_finite_num, 1).otherwise(0)).alias("numeric_count"),
            F.sum(F.when(is_finite_num & (F.abs(double_col % 1) <= _SPARK_INT_WHOLE_TOL), 1).otherwise(0)).alias(
                "whole_count"
            ),
            F.sum(F.when(ts_parsed.isNotNull(), 1).otherwise(0)).alias("ts_count"),
            F.countDistinct(F.col(colname)).alias("distinct_count"),
        ).collect()[0]
        sample_count = metrics["sample_count"]
        numeric_count = metrics["numeric_count"] or 0
        whole_count = metrics["whole_count"] or 0
        ts_count = metrics["ts_count"] or 0
        distinct_count = metrics["distinct_count"] or 0

        if sample_count == 0:
            schema[colname] = "string"
            continue

        if numeric_count >= sample_count * convert_threshold:
            # ``all_whole`` mirrors the prior
            # ``numeric_sample.filter(... > tol).count() == 0`` test:
            # every finite numeric row had a (near-)zero fractional
            # part, i.e. ``whole_count`` reached parity with
            # ``numeric_count``.
            all_whole = whole_count == numeric_count
            inferred_type = "int" if all_whole else "double"

        # Check if timestamp BEFORE the low-cardinality check: a column of
        # string timestamps with limited diversity (e.g., daily timestamps
        # over a short window, or repeated event timestamps) would otherwise
        # be classified as ``category`` and never converted to ``timestamp``,
        # losing all temporal semantics for downstream feature engineering /
        # forecasting. The numeric check above stays first because a column
        # that parses as numeric AND as timestamp (rare, e.g., Unix epoch
        # seconds as digit-only strings) is almost always meant as numeric;
        # the timestamp check is what users actually want for strings like
        # ``"2024-01-15"``.
        elif ts_count >= sample_count * convert_threshold:
            inferred_type = "timestamp"

        # Check low-cardinality (category-like)
        elif sample_count > 0 and distinct_count / sample_count <= category_threshold:
            inferred_type = "category"  # Will just be string, but marked as such

        schema[colname] = inferred_type

    # Apply inferred schema
    for colname, inferred_type in schema.items():
        if colname in skip_set:
            continue
        # Use ``try_cast`` for numeric conversions so any rows that survived
        # the sampling-based inference but are not actually castable become
        # NULL rather than raising under Spark ANSI mode.
        if inferred_type == "int":
            # Preserve whole-number string values like ``"1.0"`` /
            # ``"2.000"`` which pass the inference-time
            # ``try_cast(... as double)`` test but become NULL under
            # ``try_cast(... as int)`` in Spark ANSI mode. Strategy:
            #
            # 1. Try ``try_cast(... as int)`` directly (works for
            #    plain integer literals like ``"42"``).
            # 2. If that's NULL, try ``try_cast(... as double)`` and
            #    keep the double's int value when the fractional part
            #    is effectively zero. Use the SAME ``_SPARK_INT_WHOLE_TOL``
            #    tolerance as the inference branch above so a value that
            #    passed inference as "whole enough" also passes the
            #    cast — otherwise a column inferred as ``int`` would
            #    silently grow surprise nulls from rows whose
            #    fractional part is between the (cast) tolerance and
            #    the (inference) tolerance. The final int conversion
            #    goes through ``try_cast`` so an out-of-range double
            #    (e.g., ``"9.9e20"``) returns NULL instead of raising
            #    ``ArithmeticException`` / ``CastException`` under
            #    Spark ANSI mode.
            # Build the ``try_cast(... as double)`` SQL fragment once
            # and reference the same string everywhere we need it: the
            # whole-number check, the rounding source, and the rounded
            # int conversion. Catalyst's common-subexpression elimination
            # then dedups the resulting subtrees in the generated logical
            # plan, but keeping the source-level SQL fragment in a single
            # variable also makes the intent explicit: there is exactly
            # one "parse this column as double" operation, and downstream
            # expressions consume its result.
            escaped = colname.replace("`", "``")
            as_double_sql = f"try_cast(`{escaped}` as double)"
            as_int = _spark_try_cast(colname, "int")
            as_double = F.expr(as_double_sql)
            rounded_int = F.expr(f"try_cast(round({as_double_sql}) as int)")
            # The inference step at the top of ``auto_convert_dtypes_spark``
            # tests ``F.abs(col % 1) > _SPARK_INT_WHOLE_TOL`` to count
            # NON-whole values — meaning a value with ``abs(x % 1)``
            # exactly EQUAL to the tolerance is classified as whole at
            # inference (because the strict ``>`` predicate rejects it).
            # The apply-time predicate below MUST use the same expression
            # AND the same boundary rule (``<=``) so a row that the
            # inference would classify as non-whole (e.g., ``1.9999995``,
            # which has ``abs(x % 1) ≈ 0.9999995 > tol``) is also
            # rejected here — rather than being silently rounded to ``2``
            # via the previous ``abs(x - round(x))`` distance test, which
            # only checks proximity to the NEAREST integer and disagrees
            # with the inference predicate on values close to the upper
            # boundary of a unit interval. A naive ``<`` here would also
            # create an asymmetry where inference accepts a value at the
            # exact tolerance boundary but application rejects it; the
            # ``<=`` preserves boundary symmetry.
            whole_from_double = F.when(
                as_double.isNotNull() & (F.abs(as_double % F.lit(1)) <= F.lit(_SPARK_INT_WHOLE_TOL)),
                rounded_int,
            )
            df = df.withColumn(colname, F.coalesce(as_int, whole_from_double))
        elif inferred_type == "double":
            df = df.withColumn(colname, _spark_try_cast(colname, "double"))
        elif inferred_type == "timestamp":
            df = df.withColumn(colname, _spark_try_cast(colname, "timestamp"))
        elif inferred_type == "category":
            df = df.withColumn(colname, F.col(colname).cast(T.StringType()))  # Marked conceptually

        # otherwise keep as string (or original type)

    return df, schema


# Regex used by ``auto_convert_dtypes_pandas`` to decide whether a string value
# plausibly represents a timedelta before delegating to ``pd.to_timedelta``.
# ``pd.to_timedelta`` is overly permissive: it interprets bare integers as
# nanoseconds and silently extracts a leading signed number from strings such
# as "31-40" or "60+", which would otherwise mis-classify categorical labels
# as timedeltas. The pattern is matched with ``str.fullmatch`` so the entire
# string must match — substrings like ``"v1:2"`` or ``"foo1:2bar"`` cannot
# accidentally satisfy it. A value is accepted only if it matches one of:
#   * an ISO 8601 duration like ``P1DT2H`` / ``PT30S`` (must contain at least
#     one valid ``<number><Y|M|W|D|H|S>`` token, so ``"P31-40"`` is rejected),
#   * one or more ``<number><unit>`` segments — possibly compact like
#     ``"2h30m"`` or ``"1d2h"``, possibly followed by a ``HH:MM:SS`` clock,
#   * a bare ``HH:MM:SS[.fff]`` clock segment.
# Recognized units: ``days``, ``weeks``, ``hours``, ``hrs``, ``minutes``,
# ``mins``, ``min``, ``seconds``, ``secs``, ``sec``, ``milliseconds``,
# ``microseconds``, ``nanoseconds``, ``ms``, ``us``, ``µs``, ``ns``, or
# single-letter ``d``/``h``/``m``/``s``/``w``.
_TIMEDELTA_UNIT = (
    r"days?|weeks?|hours?|hrs?|minutes?|mins?|min|seconds?|secs?|sec"
    r"|milliseconds?|microseconds?|nanoseconds?"
    r"|ms|us|µs|ns"
    r"|[dhmsw]"
)
_TIMEDELTA_LIKE_PATTERN = re.compile(
    r"\s*[+-]?(?:"
    # ISO 8601 duration. Requires a date part with valid letters [WD] and/or
    # a time part introduced by ``T`` with valid letters [HMS]. We exclude
    # years (``Y``) and the date-part ``M`` (months) here because
    # ``pd.to_timedelta`` does not support them as unambiguous timedeltas —
    # accepting them would inflate the timedelta-like count during
    # thresholding and then fail (or become NaT) during the actual
    # ``pd.to_timedelta`` conversion. The time-part ``M`` (minutes) is still
    # accepted because it appears AFTER the ``T`` separator. Strings like
    # ``"P31-40"`` are still rejected because ``-`` is not a valid ISO
    # unit letter.
    r"P(?:(?:\d+(?:\.\d+)?[WD])+(?:T(?:\d+(?:\.\d+)?[HMS])+)?" r"|T(?:\d+(?:\.\d+)?[HMS])+)" r"|"
    # One or more ``<number><unit>`` segments, optionally followed by a clock
    # segment. Whitespace and a single comma between segments are tolerated.
    # The clock allows 1- or 2-digit minutes/seconds (``"1:2:3"``, ``"0:0:5"``)
    # to mirror what ``pd.to_timedelta`` actually parses.
    rf"(?:\d+(?:\.\d+)?\s*(?:{_TIMEDELTA_UNIT})\s*,?\s*)+" r"(?:\d{1,4}:\d{1,2}:\d{1,2}(?:\.\d+)?)?" r"|"
    # Bare ``HH:MM:SS[.fff]`` clock (pandas requires the seconds segment).
    r"\d{1,4}:\d{1,2}:\d{1,2}(?:\.\d+)?" r")\s*",
    re.IGNORECASE,
)


def auto_convert_dtypes_pandas(
    df: DataFrame,
    na_values: list = None,
    category_threshold: float = 0.3,
    convert_threshold: float = 0.6,
    sample_ratio: float = 0.1,
    skip_columns: list = None,
) -> tuple[DataFrame, dict]:
    """Automatically convert data types in a pandas DataFrame using heuristics.

    This function analyzes the DataFrame to infer appropriate data types
    and applies the conversions. It handles timestamps, timedeltas, numeric values,
    and categorical fields.

    For large DataFrames the type-inference heuristics (numeric/datetime/
    timedelta/category checks) are run on a sampled subset to bound their cost.
    The actual conversions are still applied to every row, and each conversion
    is re-validated against the full column before the resulting column is
    written back, so a non-representative sample cannot silently corrupt data.

    Args:
        df: A pandas DataFrame to convert.
        na_values: List of strings to be considered as NA/NaN. Defaults to
            ['NA', 'na', 'NULL', 'null', ''].
        category_threshold: Maximum ratio of unique values to total values
            to consider a column categorical. Defaults to 0.3.
        convert_threshold: Minimum ratio of successfully converted values required
            to apply a type conversion. Defaults to 0.6.
        sample_ratio: Fraction of rows to use for type inference. The conversions
            themselves are still applied to all rows. Must be in ``(0, 1]``.
            Defaults to 0.1, matching ``auto_convert_dtypes_spark``. The effective
            ratio is ``min(1.0, sample_ratio if n_rows * sample_ratio > 100 else
            100 / n_rows)`` so small frames (``n_rows <= 100``) are always
            inferred in full and larger frames see at least 100 rows. Pass
            ``sample_ratio=1.0`` to force full-frame inference (the previous
            default behavior).
        skip_columns: Optional iterable of column names to leave untouched.
            Listed columns are not analyzed, not NA-normalized, and keep their
            original dtype and values in the returned DataFrame. They still
            appear in the returned schema dict with their existing dtype.
            Defaults to None (no skipping).

    Returns:
        tuple: (The DataFrame with converted types, A dictionary mapping column names to
                their inferred types as strings)
    """
    if not (0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio!r}")

    if na_values is None:
        na_values = ["NA", "na", "NULL", "null", ""]
    na_values = _validate_na_values_list(na_values)
    skip_columns = _validate_skip_columns(skip_columns)
    skip_set = set(skip_columns) if skip_columns else set()
    has_blank = "" in na_values
    # Remove the empty string separately; whitespace-only strings should
    # only be treated as NA when the caller includes ``""`` in
    # ``na_values`` — matching the Spark path, which only normalizes
    # whitespace-only values to NULL when ``""`` is in ``na_values``.
    vals = [re.escape(v) for v in na_values if v != ""]
    # Build inner alternation group. Match case-insensitively so callers
    # supplying ``["NA"]`` also normalize lowercase ``"na"`` — consistent
    # with the Spark path which lowercases values for comparison.
    inner = "|".join(vals) if vals else ""
    if inner:
        if has_blank:
            # Whitespace-only OR any of the explicit NA tokens (possibly
            # padded by whitespace) match.
            pattern = re.compile(rf"^(?:\s*$|\s*(?:{inner})\s*)$", re.IGNORECASE)
        else:
            # Only the explicit NA tokens (with optional surrounding
            # whitespace) match; whitespace-only strings are kept as-is.
            pattern = re.compile(rf"^\s*(?:{inner})\s*$", re.IGNORECASE)
    elif has_blank:
        pattern = re.compile(r"^\s*$")
    else:
        # No NA tokens at all — use a pattern that never matches.
        pattern = re.compile(r"(?!)")

    # ``df.copy()`` (shallow-data-but-distinct-frame) is used here instead of
    # ``df.convert_dtypes()`` because EVERY column in the loop below
    # explicitly overwrites ``df_converted[col]`` along one of the
    # inference branches (Int64 fast path, numeric, timedelta, datetime,
    # category, string fallback, ``except`` fallback). The
    # ``convert_dtypes()`` call would do a full-frame nullable-dtype
    # rewrite up front, then every column is overwritten anyway — pure
    # wasted work (and memory) proportional to the input size, with no
    # observable behavior change. ``df.copy()`` preserves the
    # frame-level structure (column order, index) without the per-column
    # dtype-inference pass.
    df_converted = df.copy()
    schema = {}

    n_full = len(df)

    # Apply the same floor formula as ``auto_convert_dtypes_spark``: use at
    # least ~100 rows for inference when the frame is large enough, and never
    # exceed ``sample_ratio=1.0`` (i.e., always inferred in full for small
    # frames). Empty frames skip the formula to avoid division by zero. Use
    # ceil + clamping so the documented "at least 100 rows" floor is not
    # violated by floating-point rounding (e.g., 99.999... -> 99).
    if n_full > 0:
        sample_ratio_to_use = min(1.0, sample_ratio if n_full * sample_ratio > 100 else 100 / n_full)
        target_n = max(1, min(n_full, math.ceil(n_full * sample_ratio_to_use)))
    else:
        sample_ratio_to_use = 1.0
        target_n = 0
    if 0 < target_n < n_full:
        inference_df = df.sample(n=target_n, random_state=0)
    else:
        inference_df = df

    sampled = inference_df is not df

    for col in df.columns:
        # Honor skip_columns: preserve the original column verbatim.
        if col in skip_set:
            df_converted[col] = df[col]
            schema[col] = str(df[col].dtype)
            continue

        full_series = df[col]
        inf_series = inference_df[col] if sampled else full_series

        # Replace NA-like values for any string-like dtype (``object`` *or*
        # pandas ``StringDtype``). Without this, ``StringDtype`` columns
        # would keep literal ``"NA"`` / ``"null"`` strings and skew the
        # downstream nunique / category-threshold checks.
        is_string_like = full_series.dtype == object or isinstance(full_series.dtype, pd.StringDtype)
        if is_string_like and not isinstance(full_series.dtype, pd.CategoricalDtype):
            if isinstance(full_series.dtype, pd.StringDtype):
                # ``StringDtype`` is guaranteed to contain ``str`` (or
                # NA) values, so ``.str.match`` is safe without an
                # ``isinstance(v, str)`` pre-filter. ``fillna(False)``
                # collapses any NA returned by ``str.match`` (for NA
                # cells) so the result is a plain bool mask.
                #
                # Use ``pd.NA`` (the canonical missing-value sentinel
                # for ``StringDtype``) rather than ``np.nan`` when
                # masking out NA-like tokens — assigning ``np.nan``
                # into a ``StringDtype`` Series coerces it to
                # ``object`` dtype, silently dropping the
                # ``StringDtype`` extension dtype that downstream
                # callers may rely on. ``pd.NA`` preserves
                # ``StringDtype`` through the ``.where`` call so the
                # column stays on the same extension dtype it
                # entered with.
                full_mask = full_series.str.match(pattern).fillna(False)
                full_cleaned = full_series.where(~full_mask, pd.NA)
                if sampled:
                    inf_mask = inf_series.str.match(pattern).fillna(False)
                    inf_cleaned = inf_series.where(~inf_mask, pd.NA)
                else:
                    inf_cleaned = full_cleaned
            else:
                # Object dtype may contain arbitrary non-string objects
                # (lists, dicts, custom classes, numbers). The previous
                # ``full_series.astype(str).str.match(pattern)`` forced
                # ``str()`` on every element, which (a) can raise when
                # an element's ``__str__`` itself raises, and (b)
                # applies the NA-token regex to representations like
                # ``"[1, 2]"`` / ``"<MyObj 0x...>"`` that have no
                # business being treated as NA tokens. Restrict the
                # match to elements that are ALREADY ``str``: use
                # ``isinstance`` masking and only apply ``.str.match``
                # to the string subset. Non-string elements stay
                # unmasked (and hence un-NA-replaced), which is the
                # correct behavior — they aren't string NA tokens
                # regardless of their repr.
                is_str_full = full_series.map(lambda v: isinstance(v, str))
                full_mask = pd.Series(False, index=full_series.index)
                if is_str_full.any():
                    full_mask.loc[is_str_full] = full_series.loc[is_str_full].str.match(pattern).fillna(False).values
                full_cleaned = full_series.where(~full_mask, np.nan)
                if sampled:
                    is_str_inf = inf_series.map(lambda v: isinstance(v, str))
                    inf_mask = pd.Series(False, index=inf_series.index)
                    if is_str_inf.any():
                        inf_mask.loc[is_str_inf] = inf_series.loc[is_str_inf].str.match(pattern).fillna(False).values
                    inf_cleaned = inf_series.where(~inf_mask, np.nan)
                else:
                    inf_cleaned = full_cleaned
        else:
            full_cleaned = full_series
            inf_cleaned = inf_series

        # Skip conversion if already non-object dtype. Boolean columns
        # (``bool`` numpy or ``BooleanDtype`` pandas extension) are cast to
        # ``Int64`` (nullable int) instead of being left as-is, because
        # ``is_bool_dtype``-True columns trip sklearn's pandas
        # early-conversion path (e.g. ``SimpleImputer`` inside a
        # ``ColumnTransformer`` that also handles string-typed categoricals
        # fails with "Cannot cast string dtype to float64"). FLAML's own
        # ``DataTransformer`` routes booleans down its numeric branch
        # anyway, so the int representation is consistent end-to-end and
        # preserves ``pd.NA``.
        if pd.api.types.is_bool_dtype(full_cleaned.dtype) and not isinstance(full_cleaned.dtype, pd.CategoricalDtype):
            df_converted[col] = full_series.astype("Int64")
            schema[col] = "int"
            continue
        if not isinstance(full_cleaned.dtype, pd.StringDtype) and full_cleaned.dtype != "object":
            # Keep the original data type for non-object dtypes, but record
            # a canonical schema label so the schema dict doesn't mix raw
            # dtype strings (e.g. ``"datetime64[ns]"``, ``"Int64"``) with
            # the inferred labels (``"timestamp"``, ``"int"``) used by the
            # conversion paths below.
            df_converted[col] = full_series
            dtype = full_cleaned.dtype
            if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.DatetimeTZDtype):
                schema[col] = "timestamp"
            elif pd.api.types.is_timedelta64_dtype(dtype):
                schema[col] = "timedelta"
            elif pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                schema[col] = "int"
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = "double"
            elif isinstance(dtype, pd.CategoricalDtype):
                schema[col] = "category"
            else:
                schema[col] = str(dtype)
            continue

        # Compute the non-null counts ONCE per column for the threshold
        # checks below. Using the non-null count as the denominator
        # (instead of the raw row count ``n_full`` / ``n_inference``)
        # matches the Spark backend, which calls ``col_sample.dropna()``
        # before its own threshold check. Without this, a column whose
        # non-null values are entirely parseable can still be misclassified
        # as ``string`` purely because of a high fraction of missing
        # values — e.g., 60% NA + 100% of the remaining 40% parseable
        # would yield ``0.4 >= 0.6`` (False) under the old denominator,
        # versus ``1.0 >= 0.6`` (True) under the non-null denominator,
        # matching user intent and matching Spark. Columns with zero
        # non-null values fall through to the ``string`` default.
        n_full_nonnull = full_cleaned.notna().sum()
        n_inf_nonnull = inf_cleaned.notna().sum() if sampled else n_full_nonnull

        # Try integer parsing FIRST without a ``float64`` intermediate.
        # ``pd.to_numeric(..., errors="coerce")`` produces ``float64``
        # whenever NA values are present, and casting a ``float64``
        # series to pandas' nullable ``Int64`` silently loses precision
        # for integers > ``2**53`` (the float64 mantissa limit) — a
        # surveyed customer DataFrame with 19-digit IDs would round to
        # the nearest representable double and back. Match a strict
        # integer shape (optional sign + digits, with an optional
        # ``.0+`` tail so ``"3.0"`` still counts as int), strip the
        # trailing zeros, and parse directly to ``Int64`` via
        # ``pd.array(..., dtype="Int64")`` so arbitrarily-large valid
        # integers round-trip exactly. Only fall through to the
        # ``pd.to_numeric`` / ``double`` path when the column doesn't
        # match the integer shape.
        int_parsed = False
        int_overflow = False
        try:
            s_str = full_cleaned.astype("string")
            int_like = s_str.str.fullmatch(r"\s*[+-]?\d+(?:\.0+)?\s*").fillna(False)
            # Require EVERY non-null value to be integer-shaped before
            # taking the direct-Int64 fast path. The existing
            # ``pd.to_numeric`` path below uses ``(numeric.dropna() % 1
            # == 0).all()`` to gate the int branch, which means a
            # single decimal value (e.g., ``"1.5"`` among "1", "2",
            # "3") flips the inferred type to ``double`` to preserve
            # the fractional value. A threshold check here (e.g.,
            # ``>= n_full_nonnull * convert_threshold``) would let a
            # 60%-int column infer as ``int`` and silently null out
            # the fractional rows, contradicting the documented
            # all-or-nothing int-vs-double rule and breaking the
            # corresponding regression test.
            if n_full_nonnull > 0 and int_like.sum() == n_full_nonnull:
                try:
                    ints = s_str.where(int_like).str.strip().str.replace(r"\.0+$", "", regex=True)
                    # ``pd.array(dtype="Int64")`` returns an ``IntegerArray``
                    # which exposes ``isna()`` but not ``notna()``. Wrap in
                    # a ``Series`` so the subsequent threshold check uses
                    # the same ``notna()`` API as the ``numeric_full`` path
                    # below.
                    int_full = pd.Series(pd.array(ints, dtype="Int64"), index=full_cleaned.index)
                except (OverflowError, TypeError, ValueError):
                    # The column was confirmed integer-shaped by the
                    # ``int_like`` regex (every non-null value matches
                    # ``\s*[+-]?\d+(?:\.0+)?\s*``), yet
                    # ``pd.array(dtype="Int64")`` still rejected at least
                    # one element:
                    #   * ``OverflowError`` — value outside the int64
                    #     range (e.g., a 20+ digit ID column).
                    #   * ``TypeError`` / ``ValueError`` — defensive
                    #     catch for any element the regex accepted but
                    #     pandas' ``Int64`` parser declined (unlikely
                    #     after the regex prefilter, but treat it
                    #     symmetrically with overflow so a future
                    #     pandas/numpy stricter parser doesn't silently
                    #     flip the column to ``double``).
                    # In either case, falling through to ``pd.to_numeric``
                    # → ``float64`` → ``astype('double')`` would round
                    # large integers above ``2**53`` (~16 decimal
                    # digits) and silently corrupt ID values. Mark for
                    # the string-preservation branch below instead so
                    # the exact original digits round-trip.
                    int_overflow = True
                else:
                    if int_full.notna().sum() >= n_full_nonnull * convert_threshold:
                        df_converted[col] = int_full
                        schema[col] = "int"
                        int_parsed = True
        except MemoryError:
            # Process-wide memory exhaustion is NOT a per-column
            # stringification failure — masking it as "not int-like"
            # and continuing the conversion would either retry the
            # allocation in a downstream branch (same OOM, possibly
            # with a less informative traceback) or silently truncate
            # the conversion mid-frame. Re-raise so the caller sees
            # the actual condition. ``MemoryError`` IS a subclass of
            # ``Exception`` so without this explicit branch the
            # broad ``except Exception`` below would swallow it.
            raise
        except Exception:  # noqa: BLE001
            # ``astype("string")`` / ``str.fullmatch`` can invoke per-
            # element ``__str__`` on object-dtype columns. A single
            # element whose ``__str__`` itself raises (potentially
            # with a non-TypeError exception — e.g. a custom class
            # whose ``__str__`` raises ``RuntimeError`` /
            # ``AttributeError`` / a library-specific subclass) would
            # otherwise abort the WHOLE column's dtype conversion.
            # Catch ``Exception`` broadly here and treat the result
            # as "not int-like" so the offending column falls through
            # to the other inference branches (numeric / timedelta /
            # datetime / category) instead of crashing the whole
            # auto_convert_dtypes_pandas call.
            #
            # Int64-construction failures (``OverflowError`` /
            # ``TypeError`` / ``ValueError``) are caught by the INNER
            # ``try/except`` above so they still trigger the
            # ``int_overflow`` string-preservation branch — only
            # genuine stringification errors land here.
            pass
        if int_parsed:
            continue
        if int_overflow:
            # Column is uniformly integer-shaped but holds at least one
            # value beyond int64 range. Skip numeric / timedelta /
            # datetime / category inference and persist as ``string``
            # so the original lossless digit sequence survives. Going
            # through ``pd.to_numeric`` here would coerce to
            # ``float64`` and silently round the value (e.g.,
            # ``"123456789012345678901"`` → ``1.2345678901234568e+20``),
            # which is exactly the data-corruption risk the
            # direct-Int64 fast path was added to prevent — falling
            # through would defeat the entire fast-path mitigation.
            # ``category`` is also skipped: huge-ID columns are
            # typically high-cardinality so the category branch would
            # land at ``string`` anyway, and even in the rare low-
            # cardinality case the safer choice for out-of-range
            # integers is to preserve the verbatim string rather
            # than introduce a category-code encoding that obscures
            # the original value.
            df_converted[col] = full_cleaned.astype("string")
            schema[col] = "string"
            continue

        # Try numeric (int or float). The integer-vs-double decision is made
        # from the FULL converted series so a sample that happened to miss
        # decimal values can't cause silent truncation via ``astype("int")``.
        # Validate against the FULL column even when sampling is enabled:
        # gating the full check on the sample passing first (the pre-iter-63
        # behavior) meant an unlucky sample could miss enough numeric-looking
        # values to drop below ``convert_threshold`` and prevent a numeric
        # conversion that would have met the threshold on the full data —
        # a real risk now that ``sample_ratio`` defaults to ``0.1``.
        # ``pd.to_numeric`` itself can raise ``TypeError`` when the column
        # contains non-scalar objects (lists, dicts, custom objects) that
        # ``errors='coerce'`` can't reduce. Treat that as "not numeric" so
        # the function can keep going and try datetime / timedelta /
        # category / string instead of aborting the whole DataFrame.
        try:
            numeric_full = pd.to_numeric(full_cleaned, errors="coerce")
        except (TypeError, ValueError):
            numeric_full = None
        if (
            numeric_full is not None
            and n_full_nonnull > 0
            and numeric_full.notna().sum() >= n_full_nonnull * convert_threshold
        ):
            if (numeric_full.dropna() % 1 == 0).all():
                # ``pd.to_numeric(..., errors='coerce')`` on string/object
                # input routes through ``float64`` (NA support requires a
                # nullable container, and the non-Int64 fast path lands
                # here precisely because some values failed the strict
                # integer regex above and were coerced to NaN). For
                # mixed columns where the surviving numeric values
                # include integers above ``2**53``, the float64
                # intermediate has already silently rounded them — e.g.,
                # the string ``"9007199254740993"`` (2**53+1) becomes
                # ``9.007199254740992e15`` (== 2**53) in float64 and the
                # subsequent ``astype("Int64")`` writes back the rounded
                # value. This is exactly the data-corruption mode the
                # direct-Int64 fast path above was added to prevent, but
                # the fast path is bypassed for mixed columns. Detect
                # the precision-loss case here by checking whether the
                # max absolute float64 value reaches the safe-integer
                # ceiling, and when it does, re-derive the Int64 values
                # directly from ``full_cleaned`` via Python ``int(...)``
                # (arbitrary precision) so the original digits round-trip
                # exactly. Use ``2**53`` (exclusive) as the threshold:
                # values strictly below it are exactly representable in
                # float64; values at or above need precision-preserving
                # re-derivation.
                nf_nonnull = numeric_full.dropna()
                needs_precision_preserve = (not nf_nonnull.empty) and (
                    nf_nonnull.abs() >= _FLOAT64_SAFE_INT_CEILING
                ).any()
                if needs_precision_preserve:
                    try:
                        # Re-parse each source value via Python ``int``
                        # so values above 2**53 keep full precision. NA
                        # markers (and source values that ``pd.to_numeric``
                        # coerced to NaN — the < convert_threshold tail
                        # we tolerate) map to ``pd.NA`` so the Int64 array
                        # stays index-aligned with the source. A source
                        # value like ``"1.0"`` would not match strict
                        # ``int(...)``; route those through the float64
                        # value only when the float is within the safe
                        # range and is exactly an integer.
                        #
                        # For non-string numeric sources (``float``,
                        # ``np.floating``, ``Decimal``, ``np.integer``),
                        # the safe-integer ceiling check does NOT apply
                        # to the value itself — those types ARE the
                        # source representation, so whatever value the
                        # caller stored (even one already truncated by
                        # an earlier float64 cast) IS the truth and
                        # writing it back as Int64 preserves the
                        # observed semantics. The safe-integer ceiling
                        # guard exists specifically to prevent
                        # rewriting a STRING source's high-precision
                        # digits as a rounded float; it should not drop
                        # valid float-source rows (e.g.,
                        # ``pd.Series([9007199254740992.0])`` is a
                        # perfectly representable IEEE-754 value and
                        # the user explicitly stored it as a float —
                        # there's nothing more precise to recover).
                        def _precise_int_or_na(src, parsed_float):
                            if pd.isna(parsed_float):
                                return pd.NA
                            if isinstance(src, str):
                                s = src.strip()
                                try:
                                    return int(s)
                                except (TypeError, ValueError):
                                    # String like ``"1.0"`` / ``"1e3"`` —
                                    # fall through to the float-based
                                    # check below (with the safe-integer
                                    # guard, because the string MAY have
                                    # encoded more digits than float64
                                    # can represent and we can't tell
                                    # from the float alone).
                                    try:
                                        f = float(s)
                                    except (TypeError, ValueError, OverflowError):
                                        return pd.NA
                                    if math.isfinite(f) and abs(f) < _FLOAT64_SAFE_INT_CEILING and f == int(f):
                                        return int(f)
                                    return pd.NA
                            # ``bool`` is an ``int`` subclass; reject so
                            # ``True``/``False`` don't silently become
                            # ``1``/``0`` in a column inferred as
                            # ``int``. Matches the symmetric
                            # ``bool``-rejection in the fabric helpers.
                            if isinstance(src, bool):
                                return pd.NA
                            # Integer sources (``int``, ``np.integer``)
                            # are already exact — no float64
                            # intermediate to lose precision through.
                            if isinstance(src, (int, np.integer)):
                                try:
                                    return int(src)
                                except (TypeError, ValueError, OverflowError):
                                    return pd.NA
                            # Other numeric sources (``float``,
                            # ``np.floating``, ``Decimal``, anything
                            # exposing ``__int__``): the source IS the
                            # precision ceiling — there's no
                            # higher-precision representation to
                            # recover. Accept any finite whole-number
                            # value WITHOUT the safe-integer ceiling
                            # guard so that, e.g.,
                            # ``pd.Series([9007199254740992.0])``
                            # round-trips intact instead of dropping
                            # to NA.
                            try:
                                f = float(src)
                            except (TypeError, ValueError, OverflowError):
                                return pd.NA
                            if math.isfinite(f) and f == int(f):
                                try:
                                    return int(src)
                                except (TypeError, ValueError, OverflowError):
                                    try:
                                        return int(f)
                                    except (TypeError, ValueError, OverflowError):
                                        return pd.NA
                            return pd.NA

                        precise_ints = [
                            _precise_int_or_na(src, parsed) for src, parsed in zip(full_cleaned, numeric_full)
                        ]
                        df_converted[col] = pd.Series(pd.array(precise_ints, dtype="Int64"), index=full_cleaned.index)
                        schema[col] = "int"
                        continue
                    except MemoryError:
                        # Process-wide OOM during precision-preserving
                        # parse — re-raise rather than masking as
                        # ``not int-like`` (would either retry the
                        # allocation downstream or silently truncate).
                        raise
                    except (TypeError, ValueError, OverflowError):
                        # Fall through to the float64-via-Int64 path
                        # below. The float64 path may STILL lose
                        # precision but at least produces SOMETHING;
                        # the alternative (aborting the whole column
                        # conversion) would be worse.
                        pass
                try:
                    # ``"Int64"`` is pandas' nullable integer dtype, which
                    # preserves NA semantics. ``"int"`` (numpy int64) is
                    # NOT nullable and raises on series containing NaN,
                    # silently routing integer columns with NAs into the
                    # ``double`` fallback below. (Reached only when the
                    # direct-Int64 fast path above didn't apply, e.g.,
                    # numeric source ``Series[float64]`` whose values
                    # happen to all be whole numbers — small enough to
                    # round-trip through float64 without precision loss.)
                    df_converted[col] = numeric_full.astype("Int64")
                    schema[col] = "int"
                    continue
                except (TypeError, ValueError, OverflowError):
                    # Narrowly scoped: only swallow the dtype-conversion
                    # errors ``astype("Int64")`` can legitimately raise
                    # (e.g., out-of-range values). Programmer errors
                    # like ``AttributeError`` / ``NameError`` propagate.
                    pass
            df_converted[col] = numeric_full.astype("double")
            schema[col] = "double"
            continue

        # Try timedelta FIRST, before datetime, so bare time-only values
        # like ``"01:02:03"`` (which ``pd.to_datetime`` would silently
        # parse as today + the given clock and classify as ``timestamp``)
        # are routed to the timedelta branch where they semantically
        # belong. The strict ``_TIMEDELTA_LIKE_PATTERN`` ensures date
        # strings like ``"2024-01-15"`` or full datetime strings like
        # ``"2024-01-15 01:02:03"`` do NOT match (no recognized timedelta
        # unit / shape), so genuine timestamps still fall through to the
        # datetime branch below unchanged.
        try:
            non_na = inf_cleaned.dropna()
            if len(non_na) > 0:
                # Only probe timedelta patterns on string-like scalar
                # values. Object columns can contain non-scalar values
                # (lists, dicts, custom classes); casting those to
                # ``str`` produces representations like ``"[1, 2]"`` or
                # ``"{'a': 1}"`` that can accidentally inflate the
                # match count and route fundamentally non-scalar data
                # into ``pd.to_timedelta`` — which would either raise
                # or silently coerce to nonsense deltas.
                #
                # Fast-path: ``pd.api.types.infer_dtype`` runs a single
                # C-level pass over the values to classify the column's
                # element type. When the answer is ``"string"`` every
                # value is already a ``str``, so the expensive Python-
                # level ``.map(isinstance, str)`` filter (and the
                # ``full_cleaned`` filter below) can be skipped entirely.
                # The ``.map`` fallback only runs for genuinely mixed
                # object columns, where it's the only correct option.
                inf_kind = pd.api.types.infer_dtype(non_na, skipna=True)
                if inf_kind == "string":
                    looks_like_td = non_na.str.fullmatch(_TIMEDELTA_LIKE_PATTERN).fillna(False).sum()
                    inf_all_str = True
                else:
                    is_str_inf = non_na.map(lambda v: isinstance(v, str))
                    if is_str_inf.any():
                        looks_like_td = non_na[is_str_inf].str.fullmatch(_TIMEDELTA_LIKE_PATTERN).fillna(False).sum()
                    else:
                        looks_like_td = 0
                    inf_all_str = False

                # Always re-validate against the FULL column when sampling is
                # in effect: a small fixed-size sample (~100 rows) can
                # spuriously fail the sample-level threshold for a column
                # whose full distribution actually does meet
                # ``convert_threshold`` (e.g., sample happened to draw the
                # non-timedelta rows). The early
                # ``full_matches.sum() >= n_full_nonnull * convert_threshold``
                # gate below cheaply rejects columns that can't possibly
                # pass the full check, so the only extra work in the
                # "sample failed but we try anyway" branch is a single
                # ``str.fullmatch`` on the full column — bounded by the
                # column length and C-vectorized in pandas.
                sample_pass = n_inf_nonnull > 0 and looks_like_td >= n_inf_nonnull * convert_threshold
                if sample_pass or (sampled and n_full_nonnull > 0):
                    # Always mask non-matching values to NaN on the FULL
                    # column before delegating to ``pd.to_timedelta``.
                    # Otherwise a mixed column (genuine timedeltas plus a
                    # few categorical labels like "31-40", "60+") would
                    # still have those labels silently coerced into small
                    # nanosecond deltas. This is the same hazard the
                    # column-level prefilter prevents, applied at the
                    # value level so it covers both the sampled and
                    # non-sampled inference paths.
                    #
                    # Same fast-path applies to the full column: when the
                    # sample was 100% strings AND the full column is the
                    # same series (un-sampled path) OR the full column's
                    # ``infer_dtype`` also reports ``"string"``, skip the
                    # ``.map(isinstance, str)`` Python-level iteration.
                    if inf_all_str and (not sampled or pd.api.types.infer_dtype(full_cleaned, skipna=True) == "string"):
                        full_matches = (
                            full_cleaned.str.fullmatch(_TIMEDELTA_LIKE_PATTERN).fillna(False)
                            # ``fullmatch`` returns NA for NA inputs; the
                            # outer ``fillna(False)`` already converts
                            # those, but ``where`` below uses the mask
                            # as a boolean indexer so it must be a
                            # plain bool Series — coerce defensively.
                            .astype(bool)
                        )
                    else:
                        is_str_full = full_cleaned.map(lambda v: isinstance(v, str))
                        full_matches = pd.Series(False, index=full_cleaned.index)
                        if is_str_full.any():
                            full_matches.loc[is_str_full] = (
                                full_cleaned.loc[is_str_full]
                                .str.fullmatch(_TIMEDELTA_LIKE_PATTERN)
                                .fillna(False)
                                .values
                            )
                    # Cheap pattern-level pre-check on the full column:
                    # short-circuit before the expensive ``pd.to_timedelta``
                    # when the full column itself can't possibly satisfy
                    # ``convert_threshold``. This both bounds the cost of
                    # the "sample failed but we try anyway" branch and
                    # avoids handing a clearly-unsuitable column to
                    # ``pd.to_timedelta``.
                    if n_full_nonnull > 0 and full_matches.sum() >= n_full_nonnull * convert_threshold:
                        timedelta_full = pd.to_timedelta(full_cleaned.where(full_matches, np.nan), errors="coerce")
                        if timedelta_full.notna().sum() >= n_full_nonnull * convert_threshold:
                            df_converted[col] = timedelta_full
                            schema[col] = "timedelta"
                            continue
        except (TypeError, ValueError):
            pass

        # Try datetime. ``pd.to_datetime`` can raise ``TypeError`` for
        # non-scalar objects similar to ``pd.to_numeric``; treat the same way.
        # When sampling is in effect, ALSO validate against the full
        # column even if the sample-level threshold isn't met — same
        # rationale as the timedelta branch above (small samples can
        # spuriously fail for columns whose full distribution actually
        # parses as datetime). The full-column ``pd.to_datetime`` call
        # is the expensive step but only runs in the "sample failed but
        # we try anyway" branch when ``sampled and n_full_nonnull > 0``;
        # unsampled callers go through the same ``datetime_inf`` path
        # they did before.
        datetime_full = None
        try:
            datetime_inf = pd.to_datetime(inf_cleaned, errors="coerce")
        except (TypeError, ValueError):
            datetime_inf = None

        if (
            datetime_inf is not None
            and n_inf_nonnull > 0
            and datetime_inf.notna().sum() >= n_inf_nonnull * convert_threshold
        ):
            try:
                datetime_full = pd.to_datetime(full_cleaned, errors="coerce") if sampled else datetime_inf
            except (TypeError, ValueError):
                datetime_full = None
        elif sampled and n_full_nonnull > 0:
            try:
                datetime_full = pd.to_datetime(full_cleaned, errors="coerce")
            except (TypeError, ValueError):
                datetime_full = None

        if (
            datetime_full is not None
            and n_full_nonnull > 0
            and datetime_full.notna().sum() >= n_full_nonnull * convert_threshold
        ):
            df_converted[col] = datetime_full
            schema[col] = "timestamp"
            continue

        # Try category. The unique-ratio is estimated from the inference sample
        # but the categorical column is materialized from the full data. When
        # sampling is in effect the decision is ALWAYS re-validated on the full
        # column (in both the "sample says yes" AND "sample says no" branches):
        # with a small fixed-size sample (~100 rows) a truly low-cardinality
        # column can look high-cardinality in the sample (e.g., 60 distinct
        # values in 100 rows = 0.6 > category_threshold, but the same 60
        # distinct values in 10000 rows = 0.006) and would otherwise be
        # incorrectly cast to string. Symmetrically a small sample with few
        # unique values cannot wrongly classify a genuinely high-cardinality
        # column as ``category``.
        #
        # Use the NON-NULL row count as the denominator (``n_full_nonnull`` /
        # ``n_inf_nonnull``) so the unique-ratio matches the Spark path
        # (which computes ``countDistinct(col) / sample_count`` after
        # ``dropna()``). Using ``n_full`` here would let a high-cardinality
        # but mostly-null column slip below ``category_threshold`` purely
        # because the null rows inflate the denominator, producing
        # categorical/Spark-string skew on identical data.
        try:
            if sampled:
                denom = n_full_nonnull
            else:
                denom = n_inf_nonnull
            if denom > 0:
                if sampled:
                    unique_ratio_full = full_cleaned.nunique(dropna=True) / denom
                else:
                    unique_ratio_full = inf_cleaned.nunique(dropna=True) / denom
            else:
                unique_ratio_full = 1.0
            if unique_ratio_full <= category_threshold:
                df_converted[col] = full_cleaned.astype("category")
                schema[col] = "category"
                continue
        except (TypeError, ValueError, ArithmeticError):
            # Narrow set: ``nunique`` / division / ``astype("category")``
            # may legitimately fail with these for exotic value
            # combinations; anything else (e.g., AttributeError from a
            # malformed object) is a programming bug and should surface.
            pass
        # Final fallback: cast to pandas ``StringDtype``. This call
        # invokes per-element ``__str__`` on object-dtype columns, so
        # an element whose ``__str__`` itself raises (a custom class
        # raising e.g. ``RuntimeError`` / ``AttributeError``) would
        # otherwise abort the WHOLE ``auto_convert_dtypes_pandas`` call
        # at the very last step — even though earlier branches
        # (direct-Int64 fast path, NA normalization) already
        # defensively guarded against the same hazard. Wrap in a broad
        # ``except Exception`` and fall back to leaving the column as
        # the original object dtype so a single misbehaving cell can't
        # corrupt the entire conversion for the rest of the frame.
        try:
            df_converted[col] = full_cleaned.astype("string")
            schema[col] = "string"
        except MemoryError:
            # Same rationale as the direct-Int64 fast path's MemoryError
            # branch: process-wide OOM is NOT a per-column ``__str__``
            # failure. Silently falling back to the original object
            # dtype would discard the genuine memory-exhaustion signal
            # and leave the caller with an unexpected mix of converted
            # and unconverted columns (depending on which column happened
            # to trigger the allocation that finally pushed the process
            # over the limit). Re-raise so the actual condition is
            # visible. ``MemoryError`` is a subclass of ``Exception`` so
            # this explicit branch is required to escape the broad
            # ``except Exception`` below.
            raise
        except Exception:  # noqa: BLE001
            df_converted[col] = full_series
            schema[col] = str(full_series.dtype)

    return df_converted, schema
