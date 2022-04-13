from flaml import AutoML
import pandas as pd
import gc
import numpy as np
import os
import sys
import platform
import pickle
from sklearn.model_selection import train_test_split
os.environ["AUTOGLUON_TEXT_TRAIN_WITHOUT_GPU"] = "1"

def default_holdout_frac(num_train_rows, hyperparameter_tune=False):
    """
    Returns default holdout_frac used in fit().
    Between row count 5,000 and 25,000 keep 0.1 holdout_frac, as we want to grow validation set to a stable 2500 examples.
    Ref: https://github.com/awslabs/autogluon/blob/master/core/src/autogluon/core/utils/utils.py#L243
    """
    if num_train_rows < 5000:
        holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
    else:
        holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

    if hyperparameter_tune:
        holdout_frac = min(0.2, holdout_frac * 2)  # to allocate more validation data for HPO to avoid overfitting

    return holdout_frac

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
            "The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .",
            "Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .",
            "The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .",
            "The DVD-CCA then appealed to the state Supreme Court .",
            "Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .",
            "PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .",
            "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .",
            "The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .",
        ],
        "sentence2": [
            'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
            "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .",
            "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .",
            "Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .",
            "PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .",
            "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .",
            "The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .",
            "The DVD CCA appealed that decision to the U.S. Supreme Court .",
            "The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .",
            "The DVD-CCA then appealed to the state Supreme Court .",  
            "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
            "They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .",
        ],
        "numerical1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "categorical1": ["a", "b", "a", "a", "a", "b", "a", "a", "a", "b", "a", "a"],
        "label": [1, 0, 2, 0, 1, 2, 0, 1, 1, 2, 0, 1],
    }
    train_dataset = pd.DataFrame(train_data)

    # FORCE THE SAME TRAIN-VALID SPLIT IN & OUT THE PREDICTOR
    holdout_frac = default_holdout_frac(len(train_dataset), False)

    _, valid_dataset = train_test_split(train_dataset,
                                    test_size=holdout_frac,
                                    random_state=np.random.RandomState(seed))
    
    feature_columns = ["sentence1", "sentence2", "numerical1", "categorical1"]

    automl = AutoML()
    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 10,
        "task": "classification",
        "metric": "accuracy",
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
        estimator_list=["multimodal"],
        **automl_settings
    )
    automl.pickle("automl.pkl")
    with open("automl.pkl", "rb") as f:
        automl = pickle.load(f)
    print("Try to run inference on validation set")
    score = automl.score(valid_dataset[feature_columns], valid_dataset["label"])
    print(f"Inference on validation set complete, {metric}: {score}")
    del automl
    gc.collect()

if __name__ == "__main__":
    test_multimodalestimator()