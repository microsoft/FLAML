def test_classification_head():
    from flaml import AutoML
    import requests
    import pandas as pd

    try:
        train_data = {
            "text": [
                "i didnt feel humiliated",
                "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake",
                "im grabbing a minute to post i feel greedy wrong",
                "i am ever feeling nostalgic about the fireplace i will know that it is still on the property",
            ],
            "label": [0, 0, 3, 2],
        }
        train_dataset = pd.DataFrame(train_data)

        dev_data = {
            "text": [
                "i am feeling grouchy",
                "ive been feeling a little burdened lately wasnt sure why that was",
                "ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny",
                "i feel as confused about life as a teenager or as jaded as a year old man",
            ],
            "label": [3, 0, 5, 4],
        }
        dev_dataset = pd.DataFrame(dev_data)

    except requests.exceptions.ConnectionError:
        return

    custom_sent_keys = ["text"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 5,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
