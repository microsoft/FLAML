import sys
import pytest


@pytest.mark.skipif(sys.platform == "darwin", reason="do not run on mac os")
def test_tokenclass():
    from flaml import AutoML
    import requests
    from datasets import load_dataset

    try:
        train_data = (
            load_dataset("conll2003", split="train[:1%]").to_pandas().iloc[0:10]
        )
        dev_data = (
            load_dataset("conll2003", split="train[1%:2%]").to_pandas().iloc[0:10]
        )
    except requests.exceptions.ConnectionError:
        return

    sent_keys = ["tokens"]
    labels = "ner_tags"

    X_train = train_data[sent_keys]
    y_train = train_data[labels]

    X_val = dev_data[sent_keys]
    y_val = dev_data[labels]

    automl = AutoML()

    settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 500,
        "task": "token-classification",
        "metric": "seqeval",
    }

    settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings
    )


if __name__ == "__main__":
    test_tokenclass()