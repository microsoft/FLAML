from flaml.nlp.autotransformers import AutoTransformers

autohf = AutoTransformers()
preparedata_setting = {
    "dataset_subdataset_name": "glue:mrpc",
    "pretrained_model_size": ["google/electra-small-discriminator", "small"],
    "load_config_mode": "args",
    "server_name": "tmdev",
    "data_root_path": "data/",
    "max_seq_length": 128,
    "resplit_portion": {
        "source": ["train", "validation"],
        "train": [0, 0.001],
        "validation": [0.001, 0.002],
        "test": [0.002, 0.003],
    },
}
autohf.prepare_data(**preparedata_setting)

autohf_settings = {
    "resources_per_trial": {"cpu": 1, "gpu": 1},
    "num_samples": -1,
    "time_budget": 300,
}

validation_metric, analysis = autohf.fit(**autohf_settings)
if validation_metric is not None:
    predictions, test_metric = autohf.predict()
