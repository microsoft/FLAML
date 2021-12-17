import os
import pytest


@pytest.mark.skipif(os.name == "posix", reason="do not run on mac os")
def test_hf_data():
    from flaml import AutoML

    from datasets import load_dataset

    # train_dataset = (
    #     load_dataset("swag","regular", split="train[:1%]").to_pandas().iloc[0:4]
    # )
    # dev_dataset = (
    #     load_dataset("swag","regular", split="train[1%:2%]").to_pandas().iloc[0:4]
    # )
    # test_dataset = (
    #     load_dataset("swag","regular", split="test[1%:2%]").to_pandas().iloc[0:4]
    # )

    train_dataset = (
        load_dataset("swag", "regular", split="train[:3]").to_pandas()
    )
    dev_dataset = (
        load_dataset("swag", "regular", split="train[1%:2%]").to_pandas().iloc[0:4]
    )
    test_dataset = (
        load_dataset("swag", "regular", split="test[1%:2%]").to_pandas().iloc[0:4]
    )

    custom_sent_keys = ["sent1", "sent2", "ending0", "ending1", "ending2", "ending3",
                        "gold-source", "video-id", "startphrase", "fold-ind"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 5,
        "task": "multichoice-classification",
        "metric": "accuracy",
        "log_file_name": "seqclass.log",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "bert-base-uncased",
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 3,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
    automl = AutoML()
    automl.retrain_from_log(
        X_train=X_train,
        y_train=y_train,
        train_full=True,
        record_id=0,
        **automl_settings
    )

    automl.predict(X_test)
    automl.predict(["test test", "test test"])
    automl.predict(
        [
            ["test test", "test test"],
            ["test test", "test test"],
            ["test test", "test test"],
        ]
    )

    automl.predict_proba(X_test)
    print(automl.classes_)


def _test_custom_data():
    from flaml import AutoML

    import pandas as pd

    train_dataset = pd.read_csv("data/input/train.tsv", delimiter="\t", quoting=3)
    dev_dataset = pd.read_csv("data/input/dev.tsv", delimiter="\t", quoting=3)
    test_dataset = pd.read_csv("data/input/test.tsv", delimiter="\t", quoting=3)

    custom_sent_keys = ["#1 String", "#2 String"]
    label_key = "Quality"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 10,
        "time_budget": 300,
        "task": "multichoice-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "roberta-base",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 3,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
    automl.predict(X_test)
    automl.predict(["test test"])
    automl.predict(
        [
            ["test test", "test test"],
            ["test test", "test test"],
            ["test test", "test test"],
        ]
    )


if __name__ == "__main__":
    test_hf_data()
