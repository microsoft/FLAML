from flaml.nlp.autotransformers import AutoTransformers

autohf_settings = {
    "model_path": "google/electra-small-discriminator",
    "output_dir": "data/output/",
    "resources_per_trial": {"cpu": 1, "gpu": 1},
    "sample_num": -1,
    "time_budget": 300,
}

# def case1():
#     autohf = AutoTransformers()
#     # resplit_mode="ori"
#     analysis = autohf.fit(dataset_config=
#                           {
#                              "path": "glue",
#                              "name": "mrpc",
#                              "mapping": ["train", "validation"]
#                           },
#                           **autohf_settings)
#
#     analysis = autohf.fit()
#     if analysis.validation_metric is not None:
#         predictions = autohf.predict(dataset_config=["glue", "mrpc"],
#                                      fold="test")
#
# def case3():
#     autohf = AutoTransformers()
#     # resplit_mode="rspt"
#     analysis = autohf.fit(dataset_config=
#                           {
#                              "path": "glue",
#                              "name": "mrpc",
#                              "mapping": ["train"]
#                           },
#                           **autohf_settings)
#
#     analysis = autohf.fit()
#     if analysis.validation_metric is not None:
#         predictions = autohf.predict(dataset_config=["glue", "mrpc"],
#                                      fold="test")
#
# def case4():
#     autohf = AutoTransformers()
#     # resplit_mode="ori"
#     analysis = autohf.fit(dataset_config=
#                           {
#                               "path": "glue",
#                               "name": "mrpc",
#                               "mapping": {"train": "icecream",
#                                           "validation": "milk"}
#                           },
#                           **autohf_settings)
#
#     analysis = autohf.fit()
#     if analysis.validation_metric is not None:
#         predictions = autohf.predict(dataset_config=["glue", "mrpc"],
#                                      fold="test")
#
#


def case1():
    import ray
    from flaml import AutoML

    ray.init(local_mode=False)

    from datasets import load_dataset

    train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas()
    dev_dataset = load_dataset("glue", "mrpc", split="validation").to_pandas()

    custom_sent_keys = ["#1 String", "#2 String"]
    label_key = "Quality"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "resources_per_trial": {"cpu": 1, "gpu": 1},
        "metric": "accuracy",
        "task": "seq-classification",
        "time_budget": 300,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )


def case2():
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


#
# def case6():
#     # only of use to myself
#     autohf = AutoTransformers()
#     autohf_settings = {
#         "model_path": "google/electra-small-discriminator",
#         "output_dir": "data/output/",
#         "resources_per_trial": {"cpu": 1, "gpu": 1},
#         "sample_num": -1,
#         "time_budget": 300,
#     }
#
#     train_dataset = load_dataset("glue", "mrpc", split="train")
#     validation_dataset = load_dataset("glue", "mrpc", split="validation")
#     test_dataset = load_dataset("glue", "mrpc", split="test")
#
#     analysis = autohf.fit(dataset_config={
#                             "path": "glue",
#                             "name": "mrpc",
#                             "mapping": {"train": train_dataset},
#                             "is_resplit": True
#                            },
#                           custom_sentence_keys=get_sentence_keys("glue-mrpc"),
#                           **autohf_settings)
#
#     if analysis.validation_metric is not None:
#         predictions = autohf.predict(
#             datasets={"test": test_dataset},
#             custom_sentence_keys=get_sentence_keys("glue-mrpc"))
#
# def case7():
#
#
# def case5():
#     autohf = AutoTransformers()
#
#     autohf_settings = {
#         "model_path": "google/electra-small-discriminator",
#         "output_dir": "data/output/",
#         "resources_per_trial": {"cpu": 1, "gpu": 1},
#         "sample_num": -1,
#         "resplit_mode": "rspt",
#         "time_budget": 300,
#     }
#
#     if analysis.validation_metric is not None:
#         predictions = autohf.predict(dataset_config=["glue", "mrpc"])

if __name__ == "__main__":
    case2()
