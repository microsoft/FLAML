import os
import pytest


@pytest.mark.skipif(os.name == "posix", reason="do not run on mac os")
def test_ray():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:4]
    dev_dataset = load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:4]

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 1,
        "max_iter": 2,
        "time_budget": 50,
        "task": "seq-classification",
        "metric": "accuracy",
        "log_file_name": "seqclass.log",
        "use_ray": True,
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )


if __name__ == "__main__":
    test_ray()
