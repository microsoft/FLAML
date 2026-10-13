import sys
from unittest.mock import Mock

import numpy as np
import pytest
from rouge_score.scoring import AggregateScore, Score

from flaml.automl.ml import metric_loss_score


@pytest.mark.parametrize("metric_name", ["rouge1", "rouge2"])
@pytest.mark.parametrize("score", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("score_format", ["float", "numpy_float", "aggregate"])
@pytest.mark.parametrize("loader", ["evaluate", "datasets"])
def test_rouge_loss_supports_current_and_legacy_scores(monkeypatch, metric_name, score, score_format, loader):
    if score_format == "aggregate":
        value = Score(precision=score, recall=score, fmeasure=score)
        result = AggregateScore(low=value, mid=value, high=value)
    elif score_format == "numpy_float":
        result = np.float64(score)
    else:
        result = score

    metric = Mock()
    metric.compute.return_value = {metric_name: result}
    load = Mock(return_value=metric)
    if loader == "evaluate":
        import evaluate

        monkeypatch.setattr(evaluate, "load", load)
    else:
        import datasets

        monkeypatch.setitem(sys.modules, "evaluate", None)
        monkeypatch.setattr(datasets, "load_metric", load, raising=False)

    predictions = ["a short summary"]
    references = ["a reference summary"]
    loss = metric_loss_score(metric_name, predictions, references)

    assert loss == pytest.approx(1 - score)
    load.assert_called_once_with("rouge", trust_remote_code=True)
    metric.compute.assert_called_once_with(predictions=predictions, references=references)


def test_non_rouge_metric_keeps_its_loss_direction(monkeypatch):
    import evaluate

    metric = Mock()
    metric.compute.return_value = {"wer": 0.25}
    monkeypatch.setattr(evaluate, "load", Mock(return_value=metric))

    assert metric_loss_score("wer", ["a prediction"], ["a reference"]) == pytest.approx(0.25)


def test_missing_huggingface_dependency_keeps_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(ValueError, match=r"\[hf\] is not installed"):
        metric_loss_score("rouge1", ["a prediction"], ["a reference"])


def test_missing_metric_keeps_helpful_error(monkeypatch):
    import datasets

    monkeypatch.setitem(sys.modules, "evaluate", None)
    monkeypatch.setattr(datasets, "load_metric", Mock(side_effect=FileNotFoundError), raising=False)

    with pytest.raises(ValueError, match="neither an sklearn metric nor a huggingface metric"):
        metric_loss_score("rouge1", ["a prediction"], ["a reference"])
