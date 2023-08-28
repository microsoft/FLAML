import sys
import pytest
import requests
from utils import get_toy_data_summarization, get_automl_settings
import os
import shutil


@pytest.mark.skipif(
    sys.platform == "darwin" or sys.version < "3.7",
    reason="do not run on mac os or py<3.7",
)
def test_hf_ms():
    from flaml import AutoML

    X_train, y_train, X_val, y_val, X_test = get_toy_data_summarization()

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 20,
        "task": "summarization",
        "metric": "rouge1",
        "log_file_name": "seqclass.log",
        "use_ray": False,
        "estimator_list": ["transformer_ms"],
    }

    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **automl_settings
        )
        automl.score(X_val, y_val, **{"metric": "accuracy"})
        automl.pickle("automl.pkl")
    except requests.exceptions.HTTPError:
        return


if __name__ == "__main__":
    test_hf_ms()
