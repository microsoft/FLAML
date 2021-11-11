def test_hf_data():
    import ray
    from flaml import AutoML

    ray.init(local_mode=False)

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="validation").to_pandas()

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 1,
        "max_iter": 10,
        "time_budget": 300,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )


def test_custom_data():
    import ray
    from flaml import AutoML

    ray.init(local_mode=True)

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
        "gpu_per_trial": 1,
        "max_iter": 10,
        "time_budget": 300,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
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


def test_rspt():
    import ray
    from flaml import AutoML

    ray.init(local_mode=False)

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train[:80%]").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="train[80%:90%]").to_pandas()
    test_dataset = load_dataset("glue", "mrpc", split="train[90%:]").to_pandas()

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 1,
        "max_iter": 10,
        "time_budget": 300,
        "task": "seq-classification",
        "metric": "accuracy",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
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


def test_cv():
    import ray
    from flaml import AutoML

    ray.init(local_mode=False)

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="validation").to_pandas()
    test_dataset = load_dataset("glue", "mrpc", split="test").to_pandas()

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    X_test = test_dataset[custom_sent_keys]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 1,
        "max_iter": 10,
        "time_budget": 300,
        "task": "seq-classification",
        "metric": "accuracy",
        "eval_method": "cv",
        "n_splits": 2,
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 10,
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
    test_cv()
