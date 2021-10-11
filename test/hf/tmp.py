from flaml.nlp.autotransformers import AutoTransformers
import transformers
import ray

ray.shutdown()
ray.init(local_mode=False)

autohf = AutoTransformers()

this_hpo_space = {
    "learning_rate": {"l": 3e-5, "u": 1.5e-4, "space": "log"},
    "warmup_ratio": {"l": 0, "u": 0.2, "space": "linear"},
    "num_train_epochs": [3],
    "per_device_train_batch_size": [16, 32, 64],
    "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
    "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
    "hidden_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
}

autohf_settings = {
    "dataset_config": ["glue", "mrpc"],
    "model_path": "google/electra-base-discriminator",
    "output_dir": "data/",
    "resources_per_trial": {"gpu": 1, "cpu": 1},
    "num_samples": -1,
    "time_budget": 100,
    "ckpt_per_epoch": 5,
    "fp16": True,
    "algo_mode": "hpo",  # set the search algorithm mode to hpo
    "algo_name": "rs",
    "space_mode": "cus",  # customized search space (this_hpo_space)
    "custom_search_space": this_hpo_space,
    "transformers_verbose": transformers.logging.ERROR,
}
validation_metric, analysis = autohf.fit(**autohf_settings)
if validation_metric is not None:
    predictions, test_metric = autohf.predict()
