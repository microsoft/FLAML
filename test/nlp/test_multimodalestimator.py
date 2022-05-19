from flaml import AutoML
import pandas as pd
import numpy as np
import os
import sys
import platform
import pickle
from sklearn.model_selection import train_test_split
os.environ["AUTOGLUON_TEXT_TRAIN_WITHOUT_GPU"] = "1"


def test_multimodalestimator():
    if sys.version < "3.7":
        # do not test on python3.6
        return
    train_data = {
        "sentence1": [
            "Mary had a little lamb.",
            "Its fleece was white as snow."
        ],
        "numerical1": [1, 2],
        "label": [1, 2],
    }

    valid_data = {
        "sentence1": [
            "Mary had a little lamb.",
            "Its fleece was white as snow."
        ],
        "numerical1": [1, 2],
        "label": [1, 2],
    }
    train_dataset = pd.DataFrame(train_data)
    valid_dataset = pd.DataFrame(valid_data)

    feature_columns = ["sentence1", "numerical1"]
    metric = "r2"
    automl = AutoML()
    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 30,
        "task": "mm-regression",
        "metric": "r2",
        "seed": 123,
    }

    automl_settings["ag_args"] = {
        "output_dir": "test/ag_output/",
        "hf_model_path": "google/electra-small-discriminator"
    }

    automl.fit(
        X_train=train_dataset[feature_columns],
        y_train=train_dataset["label"],
        X_val=valid_dataset[feature_columns],
        y_val=valid_dataset["label"],
        eval_method="holdout",
        auto_augment=False,
        **automl_settings
    )
    automl.pickle("automl.pkl")
    with open("automl.pkl", "rb") as f:
        automl = pickle.load(f)
    print("Try to run inference on validation set")
    score = automl.score(valid_dataset[feature_columns], valid_dataset["label"])
    print(f"Inference on validation set complete, {metric}: {score}")
