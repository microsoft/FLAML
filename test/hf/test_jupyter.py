from flaml.nlp.autotransformers import AutoTransformers

def _test_jupyter():
    autohf = AutoTransformers()
    preparedata_setting = {
        "dataset_subdataset_name": "glue:mrpc",
        "pretrained_model_size": "google/electra-base-discriminator:base",
        "data_root_path": "data/",
        "max_seq_length": 128,
    }
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": -1,
                       "time_budget": 600,  # unlimited time budget
                       "ckpt_per_epoch": 5,
                       "fp16": True,
                       "algo_mode": "hpo",  # set the search algorithm to grid search
                       "algo_name": "rs",
                       "space_mode": "cus",  # set the search space to the recommended grid space
                       "hpo_space": {
                           "learning_rate": {"l": 3e-5, "u": 1.5e-5, "space": "log"},
                           "warmup_ratio": {"l": 0, "u": 0.2, "space": "linear"},
                           "num_train_epochs": [3],
                           "per_device_train_batch_size": [16, 32, 64],
                           "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
                           "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                           "hidden_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"}
                       }
                       }
    validation_metric, analysis = autohf.fit(**autohf_settings, )

if __name__ == "__main__":
    _test_jupyter()
