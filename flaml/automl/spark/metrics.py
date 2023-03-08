import logging
import os

logger = logging.getLogger(__name__)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)
logger.propagate = False
try:
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
    from pyspark.sql import DataFrame
    import pyspark.pandas as ps
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,
        RegressionEvaluator,
        MulticlassClassificationEvaluator,
        MultilabelClassificationEvaluator,
        RankingEvaluator,
    )
except ImportError:
    msg = """use_spark=True requires installation of PySpark. Please run pip install flaml[spark]
    and check [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
    for more details about installing Spark."""
    raise ImportError(msg)


def spark_metric_loss_score(
    metric_name: str,
    y_predict: ps.Series,
    y_true: ps.Series,
    sample_weight: ps.Series = None,
) -> float:
    """
    Compute the loss score of a metric for spark models.

    Args:
        metric_name: str | the name of the metric.
        y_predict: ps.Series | the predicted values.
        y_true: ps.Series | the true values.
        sample_weight: ps.Series | the sample weights. Default: None.

    Returns:
        float | the loss score.
    """
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

    logger.debug(f"metric_name: {metric_name}")
    logger.debug(f"df: {df.shape}\n{df.head(10)}")

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
        evaluator = MulticlassClassificationEvaluator(
            metricName="logLoss",
            labelCol=label_col,
            predictionCol=prediction_col,
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
        if "@" in metric_name:
            evaluator = RankingEvaluator(
                metricName="ndcgAt",
                labelCol=label_col,
                predictionCol=prediction_col,
                k=int(metric_name.split("@", 1)[-1]),
            )
        else:
            evaluator = RankingEvaluator(
                metricName="ndcgAt", labelCol=label_col, predictionCol=prediction_col
            )
    else:
        raise ValueError(f"Unknown metric name: {metric_name} for spark models.")

    return (
        evaluator.evaluate(df)
        if metric_name in min_mode_metrics
        else 1 - evaluator.evaluate(df)
    )
