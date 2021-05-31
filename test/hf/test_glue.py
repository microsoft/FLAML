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
        "electra": 100,
        "roberta": 500,
    },
    "rte": {
        "electra": 1800,
        "roberta": 3000,
    },
    "mrpc": {
        "electra": 600,
        "roberta": 3000,
    },
    "cola": {
        "electra": 600,
        "roberta": 1700
    },
    "stsb": {
        "electra": 800,
        "roberta": 3000,
    },
    "sst2": {
        "electra": 2000,
        "roberta": 8000,
    },
    "qnli": {
        "electra": 3000,
    },
    "mnli": {
        "electra": 10000
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

def set_space_resource(round_idx, jobid_config):
    if round_idx == 0:
        return "full", 1
    else:
        if jobid_config.pre == "electra":
            return "full", 1
        else:
            return "full", 1

def get_autohf_settings(console_args, jobid_config):
    space_mode, time_as_grid = set_space_resource(console_args.round_idx, jobid_config)
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": 100000 if jobid_config.mod != "grid" else 1,
                       "time_budget": 100000 if jobid_config.mod == "grid"
                       else 4 * glue_time_budget_mapping[jobid_config.subdat][jobid_config.pre],
                       "ckpt_per_epoch": 5
                      }
    autohf_settings["hpo_space"] = get_search_space(space_mode, jobid_config.subdat, jobid_config.pre)
    return autohf_settings

def get_search_space(space_mode, subdataset, model_name):
    if model_name == "electra":
        if space_mode == "full":
            return {
                "learning_rate": {"l": 2.99e-5, "u": 1.51e-4, "space": "log"},
                "warmup_ratio": {"l": 0, "u": 0.2, "space": "linear"},
                "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "hidden_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": {"l": 9.5, "u": 10.5, "space": "linear"} if subdataset in ("rte", "stsb") else {"l": 2.5, "u": 3.5, "space": "linear"},
                "adam_epsilon": [1e-6]
            }
        elif space_mode == "fixhalf":
            return {
                "learning_rate": {"l": 2.99e-5, "u": 1.51e-4, "space": "log"},
                "warmup_ratio": [0.1],
                "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "hidden_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": [10] if subdataset in ("rte", "stsb") else [3],
                "adam_epsilon": [1e-6]
            }
        elif space_mode == "fixall":
            return {
                "learning_rate": {"l": 2.99e-5, "u": 1.51e-4, "space": "log"},
                "warmup_ratio": [0.1],
                "attention_probs_dropout_prob": [0.1],
                "hidden_dropout_prob": [0.1],
                "weight_decay": [0.0],
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": [10] if subdataset in ("rte", "stsb") else [3],
                "adam_epsilon": [1e-6]
            }
    elif model_name == "roberta":
        if space_mode == "full":
            return {
                "learning_rate": {"l": 0.99e-5, "u": 3.01e-5, "space": "linear"},
                "warmup_ratio": {"l": 0, "u": 0.12, "space": "linear"},
                "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "hidden_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": [10],
            }
        elif space_mode == "fixhalf":
            return {
                "learning_rate": {"l": 0.99e-5, "u": 3.01e-5, "space": "linear"},
                "warmup_ratio": [0.06],
                "attention_probs_dropout_prob": {"l": 0, "u": 0.2, "space": "linear"},
                "hidden_dropout_prob": [0.1],
                "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": [10],
            }
        elif space_mode == "fixall":
            return {
                "learning_rate": {"l": 0.99e-5, "u": 3.01e-5, "space": "linear"},
                "warmup_ratio": [0.06],
                "attention_probs_dropout_prob": [0.1],
                "hidden_dropout_prob": [0.1],
                "weight_decay": [0.1],
                "per_device_train_batch_size": [16, 32, 64],
                "num_train_epochs": [10],
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
            azure_utils = AzureUtils(root_log_path= args.root_log_path,
                                     console_args= args,
                                     jobid=jobid_config,
                                     autohf=autohf)
        preparedata_setting = get_preparedata_setting(args, jobid_config, wandb_utils)
        autohf.prepare_data(**preparedata_setting)

        autohf_settings = get_autohf_settings(args, jobid_config)
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
    jobid_config = JobID(args)
    autohf = AutoTransformers()
    wandb_utils = WandbUtils(is_wandb_on = True, console_args=args, jobid_config=jobid_config)
    wandb_utils.set_wandb_per_run()
    _test_hpo(args, jobid_config, autohf, wandb_utils)
