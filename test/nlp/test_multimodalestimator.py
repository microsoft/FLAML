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
    metric = "accuracy"
    train_data = {
        "sentence1": [
            'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
            "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
            "They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .",
            "Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .",
        ],
        "numerical1": [1, 2, 3, 4],
        "categorical1": ["a", "b", "a", "b"],
        "label": [5, 4, 3, 2],
    }
    train_dataset = pd.DataFrame(train_data)
    train_dataset, valid_dataset = train_test_split(train_dataset,
                                    test_size=0.2,
                                    random_state=np.random.RandomState(seed))
    
    feature_columns = ["sentence1", "numerical1", "categorical1"]
    automl = AutoML()
    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 20,
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
