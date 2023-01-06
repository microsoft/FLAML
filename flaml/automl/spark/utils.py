import logging
import os
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import issparse

from flaml.automl.data import (
    concat,
    CLASSIFICATION,
    TOKENCLASSIFICATION,
    TS_FORECAST,
    TS_FORECASTREGRESSION,
    TS_FORECASTPANEL,
    TS_TIMESTAMP_COL,
    REGRESSION,
    _is_nlp_task,
    NLG_TASKS,
)

logger = logging.getLogger(__name__)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)
logger.propagate = False
try:
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame
    import pyspark.pandas as ps
    from pyspark.util import VersionUtils
    import pyspark.sql.functions as f
    import pyspark

    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)
except ImportError:
    msg = """use_spark=True requires installation of PySpark. Please run pip install flaml[spark]
    and check [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
    for more details about installing Spark."""
    raise ImportError(msg)


def to_pandas_on_spark(
    df: Union[pd.DataFrame, DataFrame, pd.Series, ps.DataFrame, ps.Series],
    index_col: Optional[str] = None,
) -> Union[ps.DataFrame, ps.Series]:
    """Convert pandas or pyspark dataframe/series to pandas_on_Spark dataframe/series.

    Args:
        df: pandas.DataFrame/series or pyspark dataframe | The input dataframe/series.
        index_col: str, optional | The column name to use as index, default None.

    Returns:
        pyspark.pandas.DataFrame/Series: The converted pandas-on-Spark dataframe/series.

    ```python
    import pandas as pd
    from flaml.automl.spark.utils import to_pandas_on_spark

    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    psdf = to_pandas_on_spark(pdf)
    print(psdf)

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pdf)
    psdf = to_pandas_on_spark(sdf)
    print(psdf)

    pds = pd.Series([1, 2, 3])
    pss = to_pandas_on_spark(pds)
    print(pss)
    ```
    """
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return ps.from_pandas(df)
    elif isinstance(df, DataFrame):
        if _spark_major_minor_version[1] < 3:
            return df.to_pandas_on_spark(index_col=index_col)
        else:
            return df.pandas_api(index_col=index_col)
    elif isinstance(df, (ps.DataFrame, ps.Series)):
        return df
    else:
        raise TypeError(
            f"{type(df)} is not one of pandas.DataFrame, pandas.Series and pyspark.sql.DataFrame"
        )


def train_test_split_pyspark(
    df: Union[DataFrame, ps.DataFrame],
    startify_column: str = "",
    unique_col: str = None,
    test_fraction: float = 0.2,
    seed: int = 1234,
    to_pandas_spark: bool = True,
) -> Tuple[Union[DataFrame, ps.DataFrame], Union[DataFrame, ps.DataFrame]]:
    """Split a pyspark dataframe into train and test dataframes.

    Args:
        df: pyspark.sql.DataFrame | The input dataframe.
        startify_column: str | The column name to stratify the split.
        unique_col: str | The column name to use as unique identifier.
        test_fraction: float | The fraction of the test data.
        seed: int | The random seed.
        to_pandas_spark: bool | Whether to convert the output to pandas_on_spark.

    Returns:
        pyspark.sql.DataFrame/pandas_on_spark DataFrame | The train dataframe.
        pyspark.sql.DataFrame/pandas_on_spark DataFrame | The test dataframe.
    """
    if isinstance(df, ps.DataFrame):
        df = df.to_spark(index_col="_tmp_unique_col")
        unique_col = "_tmp_unique_col"

    if not unique_col:
        unique_col = "_tmp_unique_col"
        df = df.withColumn(unique_col, f.monotonically_increasing_id())

    if startify_column:
        # Test data
        test_fraction_dict = (
            df.select(startify_column)
            .distinct()
            .withColumn("fraction", f.lit(test_fraction))
            .rdd.collectAsMap()
        )
        df_test = df.stat.sampleBy(startify_column, test_fraction_dict, seed)
        # Train data
        df_train = df.subtract(df_test)
    else:
        df_train, df_test = df.randomSplit([1 - test_fraction, test_fraction], seed)

    if unique_col == "_tmp_unique_col":
        df_train = df_train.drop(unique_col)
        df_test = df_test.drop(unique_col)
    if to_pandas_spark:
        return (to_pandas_on_spark(df_train), to_pandas_on_spark(df_test))
    return (df_train, df_test)


def unique_pandas_on_spark(psds: Union[ps.Series, ps.DataFrame]):
    """Get the unique values and counts of a pandas_on_spark series."""
    if isinstance(psds, ps.DataFrame):
        psds = psds.iloc[:, 0]
    _tmp = psds.value_counts().to_pandas()
    label_set = _tmp.index.values
    counts = _tmp.values
    return label_set, counts


def len_labels(y, return_labels=False):
    """Get the number of unique labels in y."""
    if not isinstance(y, (ps.DataFrame, ps.Series)):
        labels = np.unique(y)
    else:
        labels = y.unique()
    if return_labels:
        return len(labels), labels
    return len(labels)


class SearchState:
    def search_space(self):
        pass

    def estimated_cost4improvement(self):
        pass

    def valid_starting_point_one_dim(self, value_one_dim, domain_one_dim):
        pass

    def valid_starting_point(self, starting_point, search_space):
        pass

    def update(self, result, time_used):
        pass

    def get_hist_config_sig(self, sample_size, config):
        pass

    def est_retrain_time(self, retrain_sample_size):
        pass


class AutoMLState:
    def _prepare_sample_train_data(self, sample_size):
        pass

    def _compute_with_config_base(config_w_resource, state, estimator, is_report=True):
        pass

    def sanitize(cls, config: dict) -> dict:
        pass


class AutoML(BaseEstimator):
    def score(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        estimator = getattr(self, "_trained_estimator", None)
        if estimator is None:
            logger.warning(
                "No estimator is trained. Please run fit with enough budget."
            )
            return None
        X = self._preprocess(X)
        if self._label_transformer:
            y = self._label_transformer.transform(y)
        return estimator.score(X, y, **kwargs)

    def predict(
        self,
        X: Union[np.array, pd.DataFrame, List[str], List[List[str]]],
        **pred_kwargs,
    ):
        estimator = getattr(self, "_trained_estimator", None)
        if estimator is None:
            logger.warning(
                "No estimator is trained. Please run fit with enough budget."
            )
            return None
        X = self._preprocess(X)
        y_pred = estimator.predict(X, **pred_kwargs)
        if (
            isinstance(y_pred, np.ndarray)
            and y_pred.ndim > 1
            and isinstance(y_pred, np.ndarray)
        ):
            y_pred = y_pred.flatten()
        if self._label_transformer:
            return self._label_transformer.inverse_transform(
                pd.Series(y_pred.astype(int))
            )
        else:
            return y_pred

    def predict_proba(self, X, **pred_kwargs):
        estimator = getattr(self, "_trained_estimator", None)
        if estimator is None:
            logger.warning(
                "No estimator is trained. Please run fit with enough budget."
            )
            return None
        X = self._preprocess(X)
        proba = self._trained_estimator.predict_proba(X, **pred_kwargs)
        return proba

    def _preprocess(self, X):
        if isinstance(X, List):
            try:
                if isinstance(X[0], List):
                    X = [x for x in zip(*X)]
                X = pd.DataFrame(
                    dict(
                        [
                            (self._transformer._str_columns[idx], X[idx])
                            if isinstance(X[0], List)
                            else (self._transformer._str_columns[idx], [X[idx]])
                            for idx in range(len(X))
                        ]
                    )
                )
            except IndexError:
                raise IndexError(
                    "Test data contains more columns than training data, exiting"
                )
        elif isinstance(X, int):
            return X
        elif issparse(X):
            X = X.tocsr()
        if self._state.task in TS_FORECAST:
            X = pd.DataFrame(X)
        if self._transformer:
            X = self._transformer.transform(X)
        return X

    def _validate_ts_data():
        pass

    def _validate_data():
        pass

    def _prepare_data(self, eval_method, split_ratio, n_splits):
        pass


# ml.py
def metric_loss_score():
    pass


def _eval_estimator():
    pass


def get_val_loss():
    pass


def default_cv_score_agg_func(val_loss_folds, log_metrics_folds):
    pass


def evaluate_model_CV():
    pass


def compute_estimator():
    pass
