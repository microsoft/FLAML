from flaml.nlp import AutoTransformers

autohf = AutoTransformers()
preparedata_setting = {
        "dataset_subdataset_name": "glue:mrpc",
        "pretrained_model_size": "google/electra-base-discriminator:base",
        "data_root_path": "data/",
        "max_seq_length": 128,
        }
autohf.prepare_data(**preparedata_setting)

#autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
#                   "num_samples": 1,
#                   "time_budget": 100000,  # unlimited time budget
#                   "ckpt_per_epoch": 5,
#                   "fp16": True,
#                   "algo_mode": "grid",  # set the search algorithm to grid search
#                   "space_mode": "grid", # set the search space to the recommended grid space
#                   }
#validation_metric, analysis = autohf.fit(**autohf_settings,)
#
#predictions, test_metric = autohf.predict()
#from flaml.nlp.result_analysis.azure_utils import AzureUtils
#
#print(autohf.jobid_config)
#
#azure_utils = AzureUtils(root_log_path="logs_test/", autohf=autohf)
#azure_utils.write_autohf_output(valid_metric=validation_metric,
#                                predictions=predictions,
#                                duration= autohf.last_run_duration)
#print(validation_metric)


hpo_space_min = {
               "learning_rate": {"l": 3e-5, "u": 1.5e-4, "space": "log"},
               "warmup_ratio": [0.1],
               "num_train_epochs": [3],
               "per_device_train_batch_size": [16, 32, 64],
               "weight_decay": [0.0],
               "attention_probs_dropout_prob": [0.1],
               "hidden_dropout_prob": [0.1],
            }

def tune_hpo(time_budget, hpo_space_full):
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": -1,
                       "time_budget": time_budget,  # unlimited time budget
                       "ckpt_per_epoch": 5,
                       "fp16": True,
                       "algo_mode": "hpo",  # set the search algorithm to grid search
                       "algo_name": "rs",
                       "space_mode": "cus", # set the search space to the recommended grid space
                       "hpo_space": hpo_space_full
                       }
    validation_metric, analysis = autohf.fit(**autohf_settings,)
    predictions, test_metric = autohf.predict()
    from flaml.nlp.result_analysis.azure_utils import AzureUtils
    azure_utils = AzureUtils(root_log_path="logs_test/", autohf=autohf)
    azure_utils.write_autohf_output(valid_metric=validation_metric,
                                    predictions=predictions,
                                    duration= autohf.last_run_duration)
    print(validation_metric)

tune_hpo(500, hpo_space_min)
