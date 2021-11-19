import os
import pytest


def _test_ray():
    try:
        import ray
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = (
        load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:100]
    )
    dev_dataset = (
        load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:100]
    )
    test_dataset = (
        load_dataset("glue", "mrpc", split="test").to_pandas().iloc[0:100]
    )

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    def toy_metric(
        X_test,
        y_test,
        estimator,
        labels,
        X_train,
        y_train,
        weight_test=None,
        weight_train=None,
        config=None,
        groups_test=None,
        groups_train=None,
    ):
        return 0, {
            "test_loss": 0,
            "train_loss": 0,
            "pred_time": 0,
        }

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 30,
        "task": "seq-classification",
        "metric": toy_metric,
        "log_file_name": "seqclass.log",
        "use_ray": False
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
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 1,
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
    _test_ray()