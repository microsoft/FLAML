import sys
import pytest
from pandas import DataFrame

# @pytest
def test_distilling_hf():
    try:
        import ray
    except ImportError:
        return
    from flaml.automl import AutoML
    import requests
    
    # TODO: change dataset
    from datasets import load_dataset
    try:
        train_dataset = (
            load_dataset("glue", "mrpc", split="train[:1%]").to_pandas().iloc[0:4]
        )
        dev_dataset = (
            load_dataset("glue", "mrpc", split="train[1%:2%]").to_pandas().iloc[0:4]
        )
        test_dataset = (
            load_dataset("glue", "mrpc", split="test[1%:2%]").to_pandas().iloc[0:4]
        )
    except requests.exceptions.ConnectionError:
        return

    custom_sent_keys = ["sentence1", "sentence2"]
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
        "time_budget": 10,
        "task": "seq-classification",
        "metric": "accuracy",
        "starting_points": {"transformer": {"num_train_epochs": 1}},
        "use_ray": True,
        "estimator_list": ['distilling'],
        # "teacher_type": "bert",
        # "student_type": "distilbert",
    }

    automl_settings["custom_hpo_args"] = {
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
        "student_type": "distilbert",
        "student_name_or_path": "distilbert-base-uncased",
        "model_path": "distilbert-base-uncased",
        "teacher_type":"bert",
        "teacher_name_or_path":"bert-base-uncased",
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
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


def test_distilling_emo():
    try:
        import ray
    except ImportError:
        return
    from flaml.automl import AutoML
    import requests

    # TODO: change dataset
    from datasets import load_dataset
    try:
        train_dataset = (
            load_dataset("emotion", split="train[:1%]").to_pandas().iloc[0:10]
        )
        dev_dataset = (
            load_dataset("emotion", split="train[1%:2%]").to_pandas().iloc[0:5]
        )
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
        "time_budget": 10,
        "task": "seq-classification",
        "metric": "accuracy",
        "starting_points": {"transformer": {"num_train_epochs": 1}},
        "use_ray": True,
        "estimator_list": ['distilling'],
        # "teacher_type": "bert",
        # "student_type": "distilbert",
    }

    automl_settings["custom_hpo_args"] = {
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
        "student_type": "distilbert",
        "student_name_or_path": "distilbert-base-uncased",
        "model_path": "distilbert-base-uncased",
        "teacher_type": "bert",
        "teacher_name_or_path": "bert-base-uncased",
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
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

if __name__ == "__main__":
    test_distilling_hf()
    test_distilling_emo()