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

def get_resplit_portion(jobid_config):
    if jobid_config.dat == ["glue"] and jobid_config.subdat in {"mnli", "qqp"}:
        return {"source": ["train", "validation"], "train": [0, 0.25], "validation": [0.25, 0.275], "test": [0.275, 0.3]}
    elif jobid_config.dat[0] in {"imdb", "dbpedia_14", "yelp_review_full"}:
        return {"source": ["train", "test"], "train": [0, 0.05], "validation": [0.05, 0.055], "test": [0.055, 0.06]}
    else:
        return {"source": ["train", "validation"], "train": [0, 0.8], "validation": [0.8, 0.9], "test": [0.9, 1.0]}

def get_preparedata_setting(args, jobid_config, wandb_utils):
    preparedata_setting = {
        "server_name": args.server_name,
        "data_root_path": args.data_root_dir,
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "wandb_utils": wandb_utils
        }
    if jobid_config.spt == 'rspt':
        preparedata_setting["resplit_portion"] = get_resplit_portion(jobid_config)
    if ("albert" == jobid_config.pre and jobid_config.dat == ["squad"]) or \
        ("funnel" in jobid_config.pre and jobid_config.dat[0] in {"imdb", "yelp_review_full", "yelp_polarity", "amazon_polarity", "amazon_review_multi"}):
        preparedata_setting["max_seq_length"] = 512
    if jobid_config.dat[0] == "glue" and jobid_config.subdat == "mnli":
        preparedata_setting["fold_name"] = ['train', 'validation_matched', 'test_matched']
    return preparedata_setting

def get_autohf_settings(args, **custom_args):
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": args.sample_num,
                       "time_budget": args.time_budget,
                       "ckpt_per_epoch": 1,
                      }
    for other_attr in ["ds_config", "rep_id"]:
        if hasattr(args, other_attr):
            autohf_settings[other_attr] = getattr(args, other_attr)
        else:
            autohf_settings[other_attr] = None
    if len(custom_args) > 0:
        autohf_settings.update(custom_args)
    return autohf_settings

def rm_home_result():
    from os.path import expanduser
    home = expanduser("~")
    if os.path.exists(home + "/ray_results/"):
        shutil.rmtree(home + "/ray_results/")

def search_base_and_search_around_best(args, jobid_config, autohf, wandb_utils):
    import copy, re
    args_small = copy.deepcopy(args)
    args_small.algo_mode = "list"
    args_small.sample_num = 10000
    args_small.time_budget = 3600
    jobid_config_small = JobID(args_small)
    jobid_config_small.presz = "small"
    jobid_config_small.pre_full = re.sub("(xlarge|large|intermediate)", "small", jobid_config_small.pre_full)
    azure_utils_small = AzureUtils(args_small, jobid_config_small, autohf)
    preparedata_setting = get_preparedata_setting(args, jobid_config, wandb_utils)
    autohf.prepare_data(**preparedata_setting)
    autohf.set_metric()

    best_config = azure_utils_small.get_ranked_configs_from_azure_file(autohf.metric_mode_name)[0]

    args_large = copy.deepcopy(args)
    args_large.time_budget = args.time_budget - 3600
    args_large.sample_num = 100000
    args_large.algo_name = "cfo"
    args_large.search_alg_args_mode = "cus"
    args_large.space_mode = "cus"
    jobid_config_large = JobID(args_large)
    jobid_config_large.presz = jobid_config.presz
    jobid_config_large.pre_full = jobid_config.pre_full
    azure_utils_large = AzureUtils(args_large, jobid_config_large, autohf)
    best_config_neighbor = {key: [best_config[key]] for key in best_config.keys()}
    best_config_neighbor["learning_rate"] = {
        "l": best_config_neighbor["learning_rate"][0] / 500,
        "u": best_config_neighbor["learning_rate"][0],
        "space": "log"}
    best_config_neighbor["num_train_epochs"] = {
        "l": 3,
        "u": 10,
        "space": "linear"
    }
    _test_hpo(args_large,
              jobid_config_large,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=
        get_autohf_settings(args_large,
                **{"hpo_space": best_config_neighbor,
                   "points_to_evaluate": [best_config]}))

def search_base_and_evaluate_large(args, jobid_config, autohf, wandb_utils):
    import copy, re
    args_small = copy.deepcopy(args)
    args_small.sample_num = 10000
    args_small.time_budget = 3600
    jobid_config_small = JobID(args_small)
    jobid_config_small.presz = "small"
    jobid_config_small.pre_full = re.sub("(xlarge|large|intermediate)", "small", jobid_config_small.pre_full)
    azure_utils_small = AzureUtils(args_small, jobid_config_small, autohf)
    _test_hpo(args_small, jobid_config_small, autohf, wandb_utils, azure_utils_small)

    ranked_all_small_configs = azure_utils_small.get_ranked_configs_from_azure_file(autohf.metric_mode_name)

    args_large = copy.deepcopy(args)
    args_large.time_budget = 100000
    args_large.sample_num = int(len(ranked_all_small_configs) / 2)
    args_large.search_alg_args_mode = "cus"
    jobid_config_large = JobID(args_large)
    jobid_config_large.presz = jobid_config.presz
    jobid_config_large.pre_full = jobid_config.pre_full
    azure_utils_large = AzureUtils(args_large, jobid_config_large, autohf)
    _test_hpo(args_large,
              jobid_config_large,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=
              get_autohf_settings(args_large,
                **{"points_to_evaluate": ranked_all_small_configs}))

def _test_hpo(args,
              jobid_config,
              autohf,
              wandb_utils,
              azure_utils = None,
              autohf_settings = None,
              ):
    try:
        if not azure_utils:
            azure_utils = AzureUtils(args, jobid_config, autohf)
        preparedata_setting = get_preparedata_setting(args, jobid_config, wandb_utils)
        autohf.prepare_data(**preparedata_setting)

        analysis = validation_metric = test_metric = None
        if not autohf_settings:
            autohf_settings = get_autohf_settings(args)
        if args.algo_mode != "hfhpo":
            validation_metric, analysis = autohf.fit(**autohf_settings,)
        else:
            autohf.fit_hf(**autohf_settings)

        if jobid_config.spt == "ori":
            predictions, test_metric = autohf.predict()
            if validation_metric:
                test_metric.update({"validation": validation_metric})
        else:
            predictions = None
            if validation_metric:
                test_metric = {"validation": validation_metric}

        if analysis is not None:
            json_log = azure_utils.extract_log_from_analysis(analysis)
        else:
            json_log = None
        azure_utils.write_autohf_output(json_log = json_log,
                                        test_metric= test_metric,
                                        predictions = predictions)

    except AssertionError as err:
        azure_utils.write_exception()
    rm_home_result()

if __name__ == "__main__":
    args = load_console_args()

    jobid_config = JobID(args)
    autohf = AutoTransformers()
    wandb_utils = WandbUtils(args, jobid_config)
    wandb_utils.set_wandb_per_run()

    if args.algo_mode in ("hpo", "hfhpo", "grid", "gridbert"):
        _test_hpo(args, jobid_config, autohf, wandb_utils)
    elif args.algo_mode == "bestnn":
        search_base_and_search_around_best(args, jobid_config, autohf, wandb_utils)
    else:
        search_base_and_evaluate_large(args, jobid_config, autohf, wandb_utils)