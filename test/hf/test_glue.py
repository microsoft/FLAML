'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import os
import shutil

from flaml.nlp import AutoTransformers
from flaml.nlp import AzureUtils, JobID
from flaml.nlp.result_analysis.wandb_utils import WandbUtils
from flaml.nlp.utils import load_console_args

global azure_log_path
global azure_key

glue_time_budget_mapping = {
    "wnli": {
        "electra": 420,
        "roberta": 660,
    },
    "rte": {
        "electra": 1000,
        "roberta": 720,
    },
    "mrpc": {
        "electra": 420,
        "roberta": 720,
    },
    "cola": {
        "electra": 420,
        "roberta": 1200
    },
    "stsb": {
        "electra": 1200,
        "roberta": 1000,
    },
    "sst2": {
        "electra": 1200,
        "roberta": 7800,
    },
    "qnli": {
        "electra": 1800,
        "roberta": 7800,
    },
    "qqp": {
        "electra": 7800,
    },
    "mnli": {
        "electra": 6600
    }
}

def get_preparedata_setting(args, jobid_config, wandb_utils):
    preparedata_setting = {
        "server_name": args.server_name,
        "data_root_path": args.data_root_dir,
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "wandb_utils": wandb_utils
        }
    if jobid_config.dat[0] == "glue" and jobid_config.subdat == "mnli":
        preparedata_setting["fold_name"] = ['train', 'validation_matched', 'test_matched']
    return preparedata_setting

def get_autohf_settings(jobid_config):
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": 100000 if jobid_config.mod != "grid" else 1,
                       "time_budget": 100000 if jobid_config.mod == "grid" else glue_time_budget_mapping[jobid_config.subdat][jobid_config.pre],
                       "ckpt_per_epoch": 0.001 # if jobid_config.subdat in ("rte", "mrpc", "cola", "stsb", "wnli") else 10,
                      }
    autohf_settings["hpo_space"] = get_search_space(jobid_config.mod, jobid_config.subdat, jobid_config.pre)
    return autohf_settings

def get_search_space(algo_mode, subdataset, model_name):
    if model_name == "electra":
        return {
            "learning_rate": {"l": 2.99e-5, "u": 1.51e-4, "space": "log"},
            "warmup_ratio": {"l": 0, "u": 0.2, "space": "linear"},
            "attention_dropout": {"l": 0, "u": 0.2, "space": "linear"},
            "hidden_dropout": {"l": 0, "u": 0.2, "space": "linear"},
            "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
            "per_device_train_batch_size": [16, 32, 64],
            "num_train_epochs": [10] if subdataset == "rte" else [3],
            "adam_epsilon": [1e-6]
        }
    elif model_name == "roberta":
        return {
            "learning_rate": {"l": 0.99e-5, "u": 3.01e-5, "space": "linear"},
            "warmup_ratio": {"l": 0, "u": 0.12, "space": "linear"},
            "attention_dropout": {"l": 0, "u": 0.2, "space": "linear"},
            "hidden_dropout": {"l": 0, "u": 0.2, "space": "linear"},
            "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
            "per_device_train_batch_size": [16, 32, 64],
            "num_train_epochs": [10],
            "adam_epsilon": [1e-6]
        }

def rm_home_result():
    from os.path import expanduser
    home = expanduser("~")
    if os.path.exists(home + "/ray_results/"):
        shutil.rmtree(home + "/ray_results/")

def _test_hpo(args,
              jobid_config,
              autohf,
              wandb_utils,
              azure_utils = None,
              ):
    try:
        if not azure_utils:
            azure_utils = AzureUtils("logs_acl", args, jobid_config, autohf)
        preparedata_setting = get_preparedata_setting(args, jobid_config, wandb_utils)
        autohf.prepare_data(**preparedata_setting)

        autohf_settings = get_autohf_settings(jobid_config)
        validation_metric, analysis = autohf.fit(**autohf_settings,)

        predictions, test_metric = autohf.predict()
        if test_metric:
            validation_metric.update({"test": test_metric})

        json_log = azure_utils.extract_log_from_analysis(analysis)
        azure_utils.write_autohf_output(json_log = json_log,
                                        valid_metric = validation_metric,
                                        predictions = predictions,
                                        duration = autohf.last_run_duration)

    except AssertionError as err:
        azure_utils.write_exception()
    rm_home_result()

if __name__ == "__main__":
    args = load_console_args()

    for subdat in ["wnli", "rte", "mrpc", "cola", "stsb", "sst2", "qnli", "mnli"]:
        args.dataset_subdataset_name = "glue:" + subdat
        jobid_config = JobID(args)
        autohf = AutoTransformers()
        wandb_utils = WandbUtils(is_wandb_on = False, console_args=args, jobid_config=jobid_config)
        wandb_utils.set_wandb_per_run()
        _test_hpo(args, jobid_config, autohf, wandb_utils)
