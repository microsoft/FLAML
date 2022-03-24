from flaml import AutoML
import pandas as pd
import requests
import gc
import numpy as np
import os
import sys
import platform
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

    test_data = {
            "sentence1": [
                "That compared with $ 35.18 million , or 24 cents per share , in the year-ago period .",
                "Shares of Genentech , a much larger company with several products on the market , rose more than 2 percent .",
                "Legislation making it harder for consumers to erase their debts in bankruptcy court won overwhelming House approval in March .",
                "The Nasdaq composite index increased 10.73 , or 0.7 percent , to 1,514.77 .",
            ],
            "sentence2": [
                "Earnings were affected by a non-recurring $ 8 million tax benefit in the year-ago period .",
                "Shares of Xoma fell 16 percent in early trade , while shares of Genentech , a much larger company with several products on the market , were up 2 percent .",
                "Legislation making it harder for consumers to erase their debts in bankruptcy court won speedy , House approval in March and was endorsed by the White House .",
                "The Nasdaq Composite index , full of technology stocks , was lately up around 18 points .",
            ],
            "numerical1": [3, 4, 5, 6],
            "categorical1": ["b", "a", "a", "b"],
            "label": [0, 1, 1, 2],
    }
    test_dataset = pd.DataFrame(test_data)

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
        "time_budget": 50,
        "task": "classification",
        "metric": "accuracy",
    }

    automl_settings["ag_args"] = {
        "output_dir": "test/ag/output/",
        "backend": "mxnet",
        "text_backbone": "electra_base",
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

    print("Try to run inference on test set")
    score = automl.model.estimator.evaluate(test_dataset)
    print(f"Inference on test set complete, {metric}: {score}")
    del automl
    gc.collect()
