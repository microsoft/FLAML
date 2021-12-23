import sys
import pytest

# @pytest
def test_distilling():
    try:
        import ray
    except ImportError:
        return
    from flaml.automl import AutoML
    import requests
    from datasets import load_dataset

    try:
        # TODO: change dataset
        train_dataset = (
            load_dataset("glue", "sst2", split="train[:1%]").to_pandas().iloc[:20]
        )
        dev_dataset = (
            load_dataset("glue", "sst2", split="train[1%:2%]").to_pandas().iloc[:20]
        )
    except requests.exceptions.ConnectionError:
        return
    
    custom_sent_keys = ["sentence"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 5,
        "task": "question-answering",
        "metric": "f1",
        "starting_points": {"transformer": {"num_train_epochs": 1}},
        "use_ray": True,
        # "estimator_list": ['distilling'],
        "teacher_type": "bert",
        "student_type": "distilbert",
    }

    automl_settings["custom_hpo_args"] = {
        # TODO: modify the model_path
        "model_path": "google/electra-small-discriminator", # TODO:replace the path
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        dataset='squad', **automl_settings
    )


if __name__ == "__main__":
    test_distilbert()