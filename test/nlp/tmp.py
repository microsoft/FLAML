import sys
import pytest
import pickle
import shutil
import requests


@pytest.mark.skipif(sys.platform == "darwin", reason="do not run on mac os")
def test_hf_data():
    from flaml import AutoML
    import pandas as pd
    from datasets import load_dataset

    train_dataset = load_dataset("glue", "sst2", split="train").to_pandas()
    dev_dataset = load_dataset("glue", "sst2", split="validation").to_pandas()
    test_dataset = load_dataset("glue", "sst2", split="test").to_pandas()

    custom_sent_keys = ["sentence"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 1,
        "max_iter": 2,
        "time_budget": 500,
        "task": "seq-classification",
        "metric": "accuracy",
        "log_file_name": "seqclass.log",
        "n_concurrent_trials": 1,
        "sample": False
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 1,
        "fp16": True,
    }

    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **automl_settings
        )
    except requests.exceptions.HTTPError:
        return

    automl = AutoML()
    automl.retrain_from_log(
        X_train=X_train,
        y_train=y_train,
        train_full=True,
        record_id=0,
        **automl_settings
    )
    with open("automl.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    with open("automl.pkl", "rb") as f:
        automl = pickle.load(f)
    shutil.rmtree("test/data/output/")
    from flaml.data import get_output_from_log
    time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
        get_output_from_log(filename=automl_settings['log_file_name'], time_budget=240)





if __name__ == "__main__":
    test_hf_data()
