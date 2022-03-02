import sys
import pytest
import pickle
import shutil
import requests


@pytest.mark.skipif(sys.platform == "darwin", reason="do not run on mac os")
def _test_hf_data():
    from flaml import AutoML
    import pandas as pd
    from datasets import load_dataset
    import ray

    ray.init()

    train_dataset = load_dataset("swag", split="train").to_pandas().iloc[0:100]
    dev_dataset = load_dataset("swag", split="validation").to_pandas().iloc[0:100]
    test_dataset = load_dataset("swag", split="test").to_pandas().iloc[0:100]

    custom_sent_keys = [
        "sent1",
        "sent2",
        "ending0",
        "ending1",
        "ending2",
        "ending3",
        "gold-source",
        "video-id",
        "startphrase",
        "fold-ind",
    ]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "time_budget": 500,  # setting the time budget
        "task": "multichoice-classification",  # setting the task as multiplechoice-classification
        "hf_args": {
            "output_dir": "data/output/",  # setting the output directory
            "ckpt_per_epoch": 1,  # setting the number of checkoints per epoch
        },
        "gpu_per_trial": 1,  # set to 0 if no GPU is available
        "log_file_name": "seqclass.log",  # set the file to save the log for HPO
        "log_type": "all",
        # the log type for checkpoints: all if keeping all checkpoints, best if only keeping the best checkpoints                        # the batch size for validation (inference)
        "use_ray": True,  # set whether to use Ray
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

    automl.predict(X_test, **automl_settings)

    # automl = AutoML()
    # automl.retrain_from_log(
    #     X_train=X_train,
    #     y_train=y_train,
    #     train_full=True,
    #     record_id=0,
    #     **automl_settings
    # )
    # with open("automl.pkl", "wb") as f:
    #     pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    # with open("automl.pkl", "rb") as f:
    #     automl = pickle.load(f)
    # shutil.rmtree("test/data/output/")
    # from flaml.data import get_output_from_log
    #
    # (
    #     time_history,
    #     best_valid_loss_history,
    #     valid_loss_history,
    #     config_history,
    #     metric_history,
    # ) = get_output_from_log(filename=automl_settings["log_file_name"], time_budget=240)


if __name__ == "__main__":
    _test_hf_data()
