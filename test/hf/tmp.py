from flaml.nlp.autotransformers import AutoTransformers

autohf = AutoTransformers()

autohf_settings = {
    "dataset_config": ["glue", "mrpc"],
    "model_path": "google/electra-small-discriminator",
    "output_dir": "data/",
    "resources_per_trial": {"cpu": 1, "gpu": 1},
    "num_samples": -1,
    "load_config_mode": "args",
    "space_mode": "gnr_test",
    "time_budget": 60,
}

validation_metric, analysis = autohf.fit(**autohf_settings)
if validation_metric is not None:
    predictions, test_metric = autohf.predict()
