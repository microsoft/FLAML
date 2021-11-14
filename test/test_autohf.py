def _test_hf_data():
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
        "time_budget": 20,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )
    automl.retrain_from_log(
        log_file_name="flaml.log",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_full=True,
        record_id=0,
        **automl_settings
    )


def _test_no_train():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train[:1%]").to_pandas()

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 20,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
        "fp16": False,
    }

    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)


def _test_multigpu():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
        "gpu_per_trial": 2,
        "max_iter": 3,
        "time_budget": 20,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
        "fp16": False,
    }
    import transformers

    transformers.logging.set_verbosity_error()

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        use_ray=True,
        verbose=4,
        **automl_settings
    )
    automl.retrain_from_log(
        log_file_name="flaml.log",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_full=True,
        record_id=0,
        **automl_settings
    )


def test_classification_head():
    try:
        import ray
    except ImportError:
        return
    from flaml import AutoML

    from datasets import load_dataset

    train_dataset = load_dataset(
        "glue", "mnli", split="validation_matched[:1%]"
    ).to_pandas()
    dev_dataset = load_dataset(
        "glue", "mnli", split="validation_matched[1%:2%]"
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
        "time_budget": 30,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
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
        "time_budget": 20,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
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
        "time_budget": 20,
        "task": "seq-classification",
        "metric": "accuracy",
        "eval_method": "cv",
        "n_splits": 2,
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 20,
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


def _test_load_args():
    import subprocess
    import sys

    subprocess.call(
        [sys.executable, "load_args.py", "--output_dir", "data/"], shell=True
    )


if __name__ == "__main__":
    _test_multigpu()
