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
    elif platform.system() == "Windows":
        # do not test on windows with py3.8
        return

    seed = 123
    metric = "r2"
    train_data = {
        "sentence1": [
            "Amrozi accused his brother of deliberately distorting his evidence.",
        ],
        "numerical1": [1],
        "label": [1],
    }

    valid_data = {
        "sentence1": [
            "They had published an advertisement on the Internet on June 10.",
        ],
        "numerical1": [1],
        "label": [1],
    }
    train_dataset = pd.DataFrame(train_data)
    valid_dataset = pd.DataFrame(valid_data)
    
    feature_columns = ["sentence1", "numerical1"]
    automl = AutoML()
    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 10,
        "task": "mm-regression",
        "metric": "r2",
        "seed": seed,
    }

    automl_settings["ag_args"] = {
        "output_dir": "test/ag_output/",
        "backend": "mxnet",
        "text_backbone": "electra_small",
        "multimodal_fusion_strategy": "fuse_late",
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
