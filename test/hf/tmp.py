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


def case2():
    import ray
    from flaml import AutoML

    ray.init(local_mode=True)
    automl = AutoML()

    # for custom dataset_config:
    #    if only train exist, resplit_mode must be rspt but users will not see it
    #    if only train and validation exist, resplit_mode is ori and users will not see it
    #    remove resplit_mode for user

    automl_settings = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "resources_per_trial": {"cpu": 1, "gpu": 1},
        "sample_num": -1,
        "time_budget": 10,
        "metric_name": "accuracy",
        "metric_mode_name": "max",
        "task": "seq-classification",
        "label_name": "Quality",
    }

    automl.fit(
        dataset_config={
            "path": "csv",
            "data_files": {
                "train": "data/input/train.tsv",
                "validation": "data/input/dev.tsv",
            },
            "delimiter": "\t",
            "quoting": 3,
        },
        custom_sentence_keys=("#1 String", "#2 String"),
        **automl_settings
    )

    # predict input can only be one sentence (pair) or a list of sentence (pairs) or huggingface Dataset
    # predictions = automl.predict(input_text=["the sun rises ", "the sun set"])
    # predictions = autohf.predict(dataset_config=
    #                  {"path": "csv",
    #                   "data_files": {"test": "data/input/test.tsv"},
    #                   "delimiter": "\t",
    #                   "quoting": 3
    #                   })
    automl.output_prediction()
    # predictions = autohf.predict(
    #     dataset_config={"path": "csv",
    #         "data_files":
    #          {"test": "data/input/test.tsv"},
    #         "delimiter": "\\t"},
    #     custom_sentence_keys=("#1 String", "#2 String"))


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
