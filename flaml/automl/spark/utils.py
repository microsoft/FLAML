import logging
import os
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np

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
        if _spark_major_minor_version[0] == 3 and _spark_major_minor_version[1] < 3:
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


def unique_pandas_on_spark(
    psds: Union[ps.Series, ps.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unique values and counts of a pandas_on_spark series."""
    if isinstance(psds, ps.DataFrame):
        psds = psds.iloc[:, 0]
    _tmp = psds.value_counts().to_pandas()
    label_set = _tmp.index.values
    counts = _tmp.values
    return label_set, counts


def len_labels(
    y: Union[ps.Series, np.ndarray], return_labels=False
) -> Union[int, Optional[np.ndarray]]:
    """Get the number of unique labels in y."""
    if not isinstance(y, (ps.DataFrame, ps.Series)):
        labels = np.unique(y)
    else:
        labels = y.unique() if isinstance(y, ps.Series) else y.iloc[:, 0].unique()
    if return_labels:
        return len(labels), labels
    return len(labels)


def unique_value_first_index(
    y: Union[pd.Series, ps.Series, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unique values and indices of a pandas series,
    pandas_on_spark series or numpy array."""
    if isinstance(y, ps.Series):
        y_unique = y.drop_duplicates().sort_index()
        label_set = y_unique.values
        first_index = y_unique.index.values
    else:
        label_set, first_index = np.unique(y, return_index=True)
    return label_set, first_index


def iloc_pandas_on_spark(
    psdf: Union[ps.DataFrame, ps.Series],
    index: Union[int, slice, list],
    index_col: Optional[str] = "index",
) -> Union[ps.DataFrame, ps.Series]:
    """Get the rows of a pandas_on_spark dataframe/series by index."""
    if isinstance(index, (int, slice)):
        if isinstance(psdf, ps.Series):
            return psdf.iloc[index]
        else:
            return psdf.iloc[index, :]
    elif isinstance(index, list):
        if isinstance(psdf, ps.Series):
            sdf = psdf.to_frame().to_spark(index_col=index_col)
        else:
            if index_col not in psdf.columns:
                sdf = psdf.to_spark(index_col=index_col)
            else:
                sdf = psdf.to_spark()
        sdfiloc = sdf.filter(f.col(index_col).isin(index))
        psdfiloc = to_pandas_on_spark(sdfiloc)
        if isinstance(psdf, ps.Series):
            psdfiloc = psdfiloc[psdfiloc.columns.drop(index_col)[0]]
        elif index_col not in psdf.columns:
            psdfiloc = psdfiloc.drop(columns=[index_col])
        return psdfiloc
    else:
        raise TypeError(
            f"{type(index)} is not one of int, slice and list for pandas_on_spark iloc"
        )
