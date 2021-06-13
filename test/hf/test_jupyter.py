from flaml.nlp.autotransformers import AutoTransformers

autohf = AutoTransformers()
preparedata_setting = {
    "dataset_subdataset_name": "glue:mrpc",
    "pretrained_model_size": "google/electra-base-discriminator:base",
    "data_root_path": "data/",
    "max_seq_length": 128,
}
autohf.prepare_data(**preparedata_setting)

autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                   "num_samples": 1,
                   "time_budget": 100000,  # unlimited time budget
                   "ckpt_per_epoch": 5,
                   "fp16": True,
                   "algo_mode": "grid",  # set the search algorithm to grid search
                   "space_mode": "grid", # set the search space to the recommended grid space
                   }
validation_metric, analysis = autohf.fit(**autohf_settings, )
