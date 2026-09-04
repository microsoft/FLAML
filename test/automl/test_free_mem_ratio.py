import numpy as np
from sklearn.datasets import make_classification

from flaml import AutoML
from flaml.automl.task.generic_task import GenericTask


def test_free_mem_ratio_reaches_the_search():
    """`free_mem_ratio` must be forwarded to the per-trial evaluation.

    `compute_estimator` accepted the setting and then passed a hard-coded 0
    on to the holdout/CV evaluation, so the value only took effect during the
    final retrain and was ignored for every trial of the search.
    """
    received = []
    original = GenericTask.evaluate_model_CV

    def recording_evaluate_model_CV(self, *args, **kwargs):
        received.append(kwargs.get("free_mem_ratio"))
        return original(self, *args, **kwargs)

    GenericTask.evaluate_model_CV = recording_evaluate_model_CV
    try:
        X, y = make_classification(n_samples=120, n_features=6, random_state=0)
        AutoML().fit(
            X,
            y,
            task="classification",
            time_budget=3,
            estimator_list=["lgbm"],
            eval_method="cv",
            n_splits=2,
            free_mem_ratio=0.25,
            verbose=0,
        )
    finally:
        GenericTask.evaluate_model_CV = original

    assert received, "evaluate_model_CV was never called"
    assert set(received) == {0.25}, f"free_mem_ratio not forwarded: {sorted(set(received))}"
