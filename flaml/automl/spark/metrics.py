import json
from typing import Union

import numpy as np
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    MultilabelClassificationEvaluator,
    RankingEvaluator,
    RegressionEvaluator,
)

from flaml.automl.spark import F, T, ps, psDataFrame, psSeries, sparkDataFrame


def ps_group_counts(groups: Union[psSeries, np.ndarray]) -> np.ndarray:
    if isinstance(groups, np.ndarray):
        _, i, c = np.unique(groups, return_counts=True, return_index=True)
    else:
        i = groups.drop_duplicates().index.values
        c = groups.value_counts().sort_index().to_numpy()
    return c[np.argsort(i)].tolist()


def _compute_label_from_probability(df, probability_col, prediction_col):
    # array_max finds the maximum value in the 'probability' array
    # array_position finds the index of the maximum value in the 'probability' array
    max_index_expr = F.expr(f"array_position({probability_col}, array_max({probability_col}))-1")
    # Create a new column 'prediction' based on the maximum probability value
    df = df.withColumn(prediction_col, max_index_expr.cast("double"))
    return df


def string_to_array(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []


string_to_array_udf = F.udf(string_to_array, T.ArrayType(T.DoubleType()))


def spark_metric_loss_score(
    metric_name: str,
    y_predict: psSeries,
    y_true: psSeries,
    sample_weight: psSeries = None,
    groups: psSeries = None,
) -> float:
    """
    Compute the loss score of a metric for spark models.

    Args:
        metric_name: str | the name of the metric.
        y_predict: psSeries | the predicted values.
        y_true: psSeries | the true values.
        sample_weight: psSeries | the sample weights. Default: None.
        groups: psSeries | the group of each row. Default: None.

    Returns:
        float | the loss score. A lower value indicates a better model.
    """
    import warnings

    warnings.filterwarnings("ignore")

    label_col = "label"
    prediction_col = "prediction"
    kwargs = {}

    y_predict.name = prediction_col
    y_true.name = label_col
    df = y_predict.to_frame().join(y_true)
    if sample_weight is not None:
        sample_weight.name = "weight"
        df = df.join(sample_weight)
        kwargs = {"weightCol": "weight"}

    df = df.to_spark()

    metric_name = metric_name.lower()
    min_mode_metrics = ["log_loss", "rmse", "mse", "mae"]

    if metric_name == "rmse":
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "mse":
        evaluator = RegressionEvaluator(
            metricName="mse",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "mae":
        evaluator = RegressionEvaluator(
            metricName="mae",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "r2":
        evaluator = RegressionEvaluator(
            metricName="r2",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "var":
        evaluator = RegressionEvaluator(
            metricName="var",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "roc_auc":
        evaluator = BinaryClassificationEvaluator(
            metricName="areaUnderROC",
            labelCol=label_col,
            rawPredictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "pr_auc":
        evaluator = BinaryClassificationEvaluator(
            metricName="areaUnderPR",
            labelCol=label_col,
            rawPredictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "accuracy":
        evaluator = MulticlassClassificationEvaluator(
            metricName="accuracy",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "log_loss":
        # For log_loss, prediction_col should be probability, and we need to convert it to label
        # handle data like "{'type': '1', 'values': '[1, 2, 3]'}"
        # Fix cannot resolve "array_max(prediction)" due to data type mismatch: Parameter 1 requires the "ARRAY" type,
        # however "prediction" has the type "STRUCT<type: TINYINT, size: INT, indices: ARRAY<INT>, values: ARRAY<DOUBLE>>"
        df = df.withColumn(prediction_col, df[prediction_col].cast(T.StringType()))
        df = df.withColumn(prediction_col, string_to_array_udf(df[prediction_col]))
        df = _compute_label_from_probability(df, prediction_col, prediction_col + "_label")
        evaluator = MulticlassClassificationEvaluator(
            metricName="logLoss",
            labelCol=label_col,
            predictionCol=prediction_col + "_label",
            probabilityCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "f1":
        evaluator = MulticlassClassificationEvaluator(
            metricName="f1",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "micro_f1":
        evaluator = MultilabelClassificationEvaluator(
            metricName="microF1Measure",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "macro_f1":
        evaluator = MultilabelClassificationEvaluator(
            metricName="f1MeasureByLabel",
            labelCol=label_col,
            predictionCol=prediction_col,
            **kwargs,
        )
    elif metric_name == "ap":
        evaluator = RankingEvaluator(
            metricName="meanAveragePrecision",
            labelCol=label_col,
            predictionCol=prediction_col,
        )
    elif "ndcg" in metric_name:
        # RankingEvaluator compares arrays of ranked and relevant item ids, not
        # prediction scores and graded relevance labels, so rank the documents of
        # each query by score here, the same way as sklearn_metric_loss_score.
        # Each query is scored on the executors, only the mean reaches the driver.
        k = int(metric_name.split("@", 1)[-1]) if "@" in metric_name else None
        group_col = "group"
        if groups is None:
            # all rows are one query
            df = df.withColumn(group_col, F.lit(0))
        else:
            # match the groups to the rows by index, since they can cover more rows
            # than y_true, e.g. the training loss of a cross-validation fold
            if not isinstance(groups, psSeries):
                groups = ps.Series(groups)
            df = y_predict.to_frame().join(y_true).join(groups.rename(group_col)).to_spark()

        def query_ndcg(pdf):
            import pandas as pd
            from sklearn.metrics import ndcg_score

            if len(pdf) == 1:
                # a query with one document is always ranked perfectly
                score = 1.0
            else:
                score = ndcg_score([pdf[label_col].to_numpy()], [pdf[prediction_col].to_numpy()], k=k)
            return pd.DataFrame({"ndcg": [score]})

        ndcg = df.groupBy(group_col).applyInPandas(query_ndcg, schema="ndcg double")
        return 1 - ndcg.agg(F.avg("ndcg")).first()[0]
    else:
        raise ValueError(f"Unknown metric name: {metric_name} for spark models.")

    return evaluator.evaluate(df) if metric_name in min_mode_metrics else 1 - evaluator.evaluate(df)
