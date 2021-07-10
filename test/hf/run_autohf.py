'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
import os
import shutil
import multiprocessing
import torch

from flaml.nlp import AutoTransformers
from flaml.nlp import AzureUtils, JobID
from flaml.nlp.result_analysis.wandb_utils import WandbUtils
from flaml.nlp.utils import load_dft_args

global azure_log_path
global azure_key


def get_resplit_portion(jobid_config):
    if jobid_config.dat == ["glue"] and jobid_config.subdat in {"mnli", "qqp"}:
        return {"source": ["train", "validation"], "train": [0, 0.25], "validation": [0.25, 0.275],
                "test": [0.275, 0.3]}
    elif jobid_config.dat[0] in {"imdb", "dbpedia_14", "yelp_review_full"}:
        return {"source": ["train", "test"], "train": [0, 0.05], "validation": [0.05, 0.055], "test": [0.055, 0.06]}
    else:
        return {"source": ["train", "validation"], "train": [0, 0.8], "validation": [0.8, 0.9], "test": [0.9, 1.0]}


def get_preparedata_setting(console_args, jobid_config, wandb_utils, **custom_args):
    preparedata_setting = {
        "server_name": console_args.server_name,
        "data_root_path": console_args.data_root_dir,
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "wandb_utils": wandb_utils
    }
    if jobid_config.spt in ('rspt', 'cv'):
        preparedata_setting["resplit_portion"] = get_resplit_portion(jobid_config)
    if ("albert" == jobid_config.pre and jobid_config.dat == ["squad"]) or \
            ("funnel" in jobid_config.pre and jobid_config.dat[0] in {"imdb", "yelp_review_full", "yelp_polarity",
                                                                      "amazon_polarity", "amazon_review_multi"}):
        preparedata_setting["max_seq_length"] = 512
    if jobid_config.dat[0] == "glue" and jobid_config.subdat == "mnli":
        preparedata_setting["fold_name"] = ['train', 'validation_matched', 'test_matched']
    preparedata_setting.update(custom_args)
    return preparedata_setting


def get_autohf_settings(console_args, **custom_args):
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "num_samples": console_args.sample_num,
                       "time_budget": console_args.time_budget,
                       "ckpt_per_epoch": 1,
                       }
    for other_attr in ["ds_config", "rep_id"]:
        if hasattr(console_args, other_attr):
            autohf_settings[other_attr] = getattr(console_args, other_attr)
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


def get_best_base_config(console_args, jobid_config, autohf, wandb_utils):
    import copy
    import re
    args_small = copy.deepcopy(console_args)
    args_small.algo_name = "optuna"
    args_small.search_alg_args_mode = "dft"
    args_small.algo_mode = "hpo"
    args_small.space_mode = "uni"
    args_small.pruner = "None"

    if "funnel" not in args_small.pretrained_model_size:
        args_small.algo_mode = "hpo"
    else:
        args_small.algo_mode = "list"
    args_small.sample_num = 10000
    args_small.time_budget = 3600
    args_small.rep_id = 0
    jobid_config_small = JobID(args_small)
    if jobid_config_small.pre == "deberta":
        jobid_config_small.presz = "base"
    else:
        jobid_config_small.presz = "small"
    jobid_config_small.pre_full = re.sub("(xlarge|large|intermediate)", jobid_config_small.presz,
                                         jobid_config_small.pre_full)
    azure_utils_small = AzureUtils(root_log_path=console_args.root_log_path, azure_key_path=console_args.key_path, autohf=autohf)
    preparedata_setting = get_preparedata_setting(console_args, jobid_config, wandb_utils)
    autohf.prepare_data(**preparedata_setting)
    autohf.set_metric()

    best_config = azure_utils_small.get_config_and_score_from_partial_jobid(
        args_small.root_log_path,
        jobid_config_small)[0].get_best_config()
    return best_config


def search_base_and_search_lower_lr(console_args, jobid_config, autohf, wandb_utils):
    best_config = get_best_base_config(console_args, jobid_config, autohf, wandb_utils)

    import copy
    args_large = copy.deepcopy(console_args)
    args_large.time_budget = console_args.time_budget - 3600
    args_large.sample_num = 100000
    args_large.algo_name = console_args.algo_name
    args_large.search_alg_args_mode = "cus"
    args_large.space_mode = "buni"
    args_large.pruner = "None"
    jobid_config_large = JobID(args_large)
    jobid_config_large.presz = jobid_config.presz
    jobid_config_large.pre_full = jobid_config.pre_full
    azure_utils_large = AzureUtils(root_log_path=console_args.root_log_path, azure_key_path=console_args.key_path, autohf=autohf)

    _test_hpo(args_large,
              jobid_config_large,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=get_autohf_settings(args_large, **{"points_to_evaluate": [best_config], "bound":
                                                  {"learning_rate": {"u": best_config["learning_rate"]}}}))


def search_base_and_search_around_best(console_args, jobid_config, autohf, wandb_utils):
    console_args.algo_name = "bs"
    console_args.search_alg_args_mode = "dft"
    console_args.spa = "uni"
    console_args.pru = "None"
    best_config = get_best_base_config(console_args, jobid_config, autohf, wandb_utils)

    import copy
    args_large = copy.deepcopy(console_args)
    args_large.time_budget = console_args.time_budget - 3600
    args_large.sample_num = 100000
    args_large.algo_name = "cfo"
    args_large.search_alg_args_mode = "cus"
    args_large.space_mode = "uni"
    jobid_config_large = JobID(args_large)
    jobid_config_large.presz = jobid_config.presz
    jobid_config_large.pre_full = jobid_config.pre_full
    azure_utils_large = AzureUtils(root_log_path=console_args.root_log_path, azure_key_path=console_args.key_path, autohf=autohf)

    _test_hpo(args_large,
              jobid_config_large,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=get_autohf_settings(args_large, **{"points_to_evaluate": [best_config]}))


def evaluate_configs(autohf, console_args, ranked_all_configs):
    import copy
    this_args = copy.deepcopy(console_args)
    this_args.time_budget = 100000
    this_args.sample_num = int(len(ranked_all_configs))
    this_args.search_alg_args_mode = "cus"
    jobid_config = JobID(this_args)
    azure_utils_large = AzureUtils(
                         root_log_path=console_args.root_log_path,
                         azure_key_path=console_args.key_path, autohf=autohf)
    _test_hpo(this_args,
              jobid_config,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=get_autohf_settings(this_args, **{"points_to_evaluate": ranked_all_configs}))

def evaluate_configs_cv(autohf, console_args):
    import copy
    from run_analysis import get_exhaustive_sweep_result

    partial_jobid_config = JobID(console_args)
    setattr(partial_jobid_config, "var1", set(console_args.learning_rate))
    setattr(partial_jobid_config, "var2", set(console_args.weight_decay))
    top1_score, top1_config = get_exhaustive_sweep_result(console_args, "logs_seed/", partial_jobid_config)
    # top1_config = {"learning_rate": 1e-5, "per_device_train_batch_size": 2,
    #                "num_train_epochs": 0.01, "warmup_ratio": 0.1, "weight_decay": 0.0}
    this_args = copy.deepcopy(console_args)
    autohf.jobid_config = partial_jobid_config
    azure_utils_large = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path, autohf=autohf)
    custom_args = {
        "foldnum": 5
    }
    _test_hpo(this_args,
              partial_jobid_config,
              autohf,
              wandb_utils,
              azure_utils_large,
              autohf_settings=get_autohf_settings(this_args, **{"points_to_evaluate": [top1_config]}),
              **custom_args)


def convert_config_to_different_size(origin_config, mode):
    import re
    import copy
    if mode == "small":
        new_config = copy.deepcopy(origin_config)
        if new_config.pre == "funnel":
            new_config.mod = "list"
        else:
            new_config.mod = "hpo"
        if new_config.pre == "funnel":
            new_config.presz = "small"
        else:
            new_config.presz = "base"
        new_config.pre_full = re.sub("(xlarge|large|intermediate)", new_config.presz, origin_config.pre_full)
    elif mode == "large":
        new_config = copy.deepcopy(origin_config)
        new_config.mod = "hpo"
        if new_config.pre == "funnel":
            new_config.presz = "xlarge"
            new_config.pre_full = re.sub("(small)", "xlarge", origin_config.pre_full)
        else:
            new_config.presz = "large"
            new_config.pre_full = re.sub("(small)", "large", origin_config.pre_full)

    return new_config


def add_dict_item_to_list(this_list, this_dict):
    is_exist = len([x for x in this_list if x == this_dict]) > 0
    if not is_exist:
        this_list.append(this_dict)
    return this_list

def train_cv(batch_dict):
    idx = batch_dict["idx"]
    autohf.train_dataset = batch_dict["train"]
    autohf.eval_dataset = batch_dict["eval"]
    autohf_settings = batch_dict["autohf_settings"]
    azure_utils = batch_dict["azure_utils"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 4)
    validation_metric, analysis = autohf.fit(**autohf_settings)
    # json.dump(validation_metric, open("tmp_" + str(idx) + ".json", "w"))
    # azure_utils.write_autohf_output(valid_metric=validation_metric,
    #                                 local_file_path=)
    return idx, validation_metric

def _test_hpo(console_args,
              jobid_config,
              autohf,
              wandb_utils,
              azure_utils=None,
              autohf_settings=None,
              **custom_args
              ):
    preparedata_setting = get_preparedata_setting(console_args, jobid_config, wandb_utils, **custom_args)
    autohf.prepare_data(**preparedata_setting)

    analysis = validation_metric = None
    if not autohf_settings:
        autohf_settings = get_autohf_settings(console_args, **custom_args)

    if jobid_config.spt != "cv":
        if console_args.algo_mode != "hfhpo":
            validation_metric, analysis = autohf.fit(**autohf_settings)
        else:
            autohf.fit_hf(**autohf_settings)
        predictions, test_metric = autohf.predict()
        if test_metric:
            validation_metric.update({"test": test_metric})

        if not azure_utils:
            azure_utils = AzureUtils(root_log_path=console_args.root_log_path,
                                     azure_key_path=console_args.key_path,
                                     autohf=autohf)

        if analysis is not None:
            configscore_list = azure_utils.extract_configscore_list_from_analysis(analysis)
        else:
            configscore_list = None
        azure_utils.write_autohf_output(configscore_list=configscore_list,
                                        valid_metric=validation_metric,
                                        predictions=predictions,
                                        duration=autohf.last_run_duration)
    else:
        import multiprocessing
        import json
        cv_k = len(autohf.train_datasets)
        batches = [{"idx": i, "train": autohf.train_datasets[i],
                    "eval": autohf.eval_datasets[i],
                    "autohf_settings": autohf_settings,
                    "azure_utils": azure_utils} for i in range(cv_k)]
        validation_metrics = []
        # with multiprocessing.Pool(processes=5) as p:
        #     for idx, validation_metric in p.imap_unordered(train_cv, batches):
        #         validation_metrics.append(validation_metric)
        for idx in range(len(batches)):
            idx, validation_metric = train_cv(batches[idx])
            validation_metrics.append(validation_metric)
        azure_utils.write_autohf_output(valid_metric=validation_metrics)

    rm_home_result()


def _exhaustive_sweep(console_args,
                      jobid_config,
                      autohf,
                      wandb_utils,
                      azure_utils=None,
                      autohf_settings=None, ):
    from flaml.nlp.hpo.hpo_searchspace import AutoHPOSearchSpace
    console_args.space_mode = jobid_config.spa = "cus"
    console_args.algo_mode = jobid_config.mod = "grid"
    console_args.algo_name = jobid_config.alg = "grid"

    gridunion_space = AutoHPOSearchSpace.from_model_and_dataset_name(
        "uni",
        jobid_config.pre,
        jobid_config.presz,
        jobid_config.dat,
        jobid_config.subdat
    )

    gridunion_space["learning_rate"] = [console_args.learning_rate]
    gridunion_space["weight_decay"] = [console_args.weight_decay]
    _test_hpo(console_args, jobid_config, autohf, wandb_utils, azure_utils,
              autohf_settings,
              root_log_path=console_args.root_log_path,
              **{"hpo_space": gridunion_space})


if __name__ == "__main__":
    console_args = load_dft_args()

    jobid_config = JobID(console_args)
    autohf = AutoTransformers()
    wandb_utils = WandbUtils(is_wandb_on=False, wandb_key_path=console_args.key_path, jobid_config=jobid_config)
    wandb_utils.set_wandb_per_run()

    #_test_hpo(console_args, jobid_config, autohf, wandb_utils)

    # search_base_and_search_lower_lr(console_args, jobid_config, autohf, wandb_utils)

    # evaluate_small_best_configs_on_large(console_args, autohf)

    # evaluate_large_best_configs_on_small(console_args, autohf)

    #_exhaustive_sweep(console_args, jobid_config, autohf, wandb_utils)


    evaluate_configs_cv(autohf, console_args)