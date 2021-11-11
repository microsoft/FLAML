# import os
#
# os.environ["WANDB_DISABLED"] = "true"


def test_hf_data():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train[:1%]").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="validation[:1%]").to_pandas()

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 40,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )


def _test_classification_head():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset(
        "glue", "mnli", split="validation_matched[:2%]"
    ).to_pandas()
    dev_dataset = load_dataset(
        "glue", "mnli", split="validation_matched[2%:3%]"
    ).to_pandas()

    custom_sent_keys = ["premise", "hypothesis"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 40,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )


def _test_custom_data():
    try:
        import ray
    except ImportError:
        return
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


def _test_rspt():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train[:1%]").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="train[1%:2%]").to_pandas()
    test_dataset = load_dataset("glue", "mrpc", split="train[2%:3%]").to_pandas()

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
        "time_budget": 40,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
        "fp16": False,
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


def _test_cv():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train[:1%]").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="validation[:1%]").to_pandas()
    test_dataset = load_dataset("glue", "mrpc", split="test[:1%]").to_pandas()

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
        "time_budget": 40,
        "task": "seq-classification",
        "metric": "accuracy",
        "eval_method": "cv",
        "n_splits": 2,
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
        "fp16": False,
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


def test_load_args():
    import subprocess
    import sys

    subprocess.call(
        [sys.executable, "load_args.py", "--output_dir", "data/"], shell=True
    )
