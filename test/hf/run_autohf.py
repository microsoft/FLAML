'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

import wandb

from flaml.nlp.autotransformers import AutoTransformers

# setting wandb key
wandb_key = "7553d982a2247ca8324ec648bd302678105e1058"

dataset_names = [["yelp_review_full"]]
subdataset_names = [None]

pretrained_models = ["funnel-transformer/small", "google/electra-small-discriminator", "google/electra-base-discriminator"] #, "bert-base-uncased"]

search_algos = ["BlendSearch"]
scheduler_names = ["None"]

hpo_searchspace_modes = ["hpo_space_gridunion_other", "hpo_space_gridunion_other_large"]
search_algo_args_modes = ["default", "default"]
num_sample_time_budget_mode, time_as_grid = ("times_grid_time_budget", 4.0)

def get_full_name(autohf, is_grid, hpo_searchspace_mode = None):
    if is_grid == False:
        return autohf.full_dataset_name.lower() + "_" + autohf.model_type.lower() + "_" + \
        autohf.model_size_type.lower() + "_" + autohf.search_algo_name.lower() \
        + "_" + autohf.scheduler_name.lower() + "_" \
        + "_" + hpo_searchspace_mode.lower() + "_" + autohf.path_utils.group_hash_id
    else:
        return autohf.full_dataset_name.lower() + "_" + autohf.model_type.lower() + "_" + \
               autohf.model_size_type.lower() + "_" + autohf.search_algo_name.lower() \
               + "_" + autohf.scheduler_name.lower() + "_" + autohf.path_utils.group_hash_id

def get_resplit_portion(this_dataset_name, this_subset_name):
    if this_dataset_name == ["glue"] and this_subset_name in {"mnli", "qqp"}:
        return {"source": ["train", "validation"], "train": [0, 0.25], "validation": [0.25, 0.275], "test": [0.275, 0.3]}
    elif this_dataset_name[0] in {"imdb", "dbpedia_14", "yelp_review_full"}:
        return {"source": ["train", "test"], "train": [0, 0.05], "validation": [0.05, 0.055], "test": [0.055, 0.06]}
    else:
        return {"source": ["train", "validation"], "train": [0, 0.8], "validation": [0.8, 0.9], "test": [0.9, 1.0]}

def get_preparedata_setting(args, this_dataset_name, this_subset_name, each_pretrained_model):
    preparedata_setting = {
        "dataset_config": {"task": "text-classification",
                           "dataset_name": this_dataset_name,
                           "subdataset_name": this_subset_name,
                           },
        "resplit_portion": get_resplit_portion(this_dataset_name, this_subset_name),
        "model_name": each_pretrained_model,
        "server_name": args.server_name,
        "split_mode": "resplit",
        "ckpt_path": "../../../data/checkpoint/",
        "result_path": "../../../data/result/",
        "log_path": "../../../data/result/",
        "max_seq_length": 512,
        }
    if this_dataset_name[0] == "glue" and this_subset_name and this_subset_name == "mnli":
        preparedata_setting["dataset_config"]["fold_name"] = ['train', 'validation_matched', 'test_matched']
    return preparedata_setting

def get_autohf_settings_grid():
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                           "wandb_key": wandb_key,
                           "search_algo_name": "grid_search",
                           "scheduler_name": "None",
                           "ckpt_per_epoch": 1,
                           "custom_metric_name": "accuracy",
                           "custom_metric_mode_name": "max",
                           }
    return autohf_settings

def get_autohf_settings(this_search_algo, this_scheduler_name, hpo_searchspace_mode, this_time_as_grid, search_algo_args_mode = None):
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                       "wandb_key": wandb_key,
                       "search_algo_name": this_search_algo,
                       "scheduler_name": this_scheduler_name,
                       "ckpt_per_epoch": 1,
                       "search_algo_args_mode": search_algo_args_mode,
                      }
    autohf_settings["hpo_searchspace_mode"] = hpo_searchspace_mode
    autohf_settings["num_sample_time_budget_mode"] = num_sample_time_budget_mode
    autohf_settings["time_as_grid"] = this_time_as_grid
    return autohf_settings

def get_autohf_settings_enumeratehp():
    autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                           "wandb_key": wandb_key,
                           "search_algo_name": "grid_search_enumerate",
                           "scheduler_name": "None",
                           "ckpt_per_epoch": 1,
                           "hp_to_fix": ("warmup_ratio", 0.05),
                           "hp_to_tune": ("learning_rate", [1e-5 * x for x in range(1, 11)]),
                            "hpo_searchspace_mode": "enumerate_onehp",
                           }
    return autohf_settings

def flush_and_upload(fout, args):
    fout.flush()
    api = wandb.Api()
    runs = api.runs("liususan/upload_file")
    runs[0].upload_file(os.path.abspath("log_" + args.server_name + ".log"))

def output_predict(args, test_dataset, autohf, fout, save_file_name):
    if test_dataset:
        predictions, output_metric = autohf.predict(test_dataset)
        fout.write("test " + (autohf.metric_name) + ":" + json.dumps(output_metric) + "\n\n")
        flush_and_upload(fout, args)
        if autohf.split_mode == "origin":
            autohf.output_prediction(predictions,
                                     output_prediction_path="../../../data/result/",
                                     output_dir_name=save_file_name)

def rm_home_result():
    from os.path import expanduser
    home = expanduser("~")
    if os.path.exists(home + "/ray_results/"):
        shutil.rmtree(home + "/ray_results/")

def write_exception(args, save_file_name, fout):
    fout.write(save_file_name + ":\n")
    fout.write("timestamp:" + str(str(datetime.datetime.now()))  + ":\n")
    fout.write("failed, no checkpoint found\n")
    flush_and_upload(fout, args)

def write_regular(autohf, args, validation_metric, save_file_name, fout):
    fout.write(save_file_name + ":\n")
    fout.write("timestamp:" + str(str(datetime.datetime.now())) + ":\n")
    fout.write("validation " + (autohf.metric_name) + ":" + json.dumps(validation_metric) + "\n")
    fout.write("duration:" + str(autohf.last_run_duration) + "\n")
    flush_and_upload(fout, args)

def _test_grid(args, fout, autohf):
    for data_idx in range(len(dataset_names)):
        this_dataset_name = dataset_names[data_idx]
        this_subset_name = subdataset_names[data_idx]

        for model_idx in range(0, len(pretrained_models)):
            each_pretrained_model = pretrained_models[model_idx]

            preparedata_setting = get_preparedata_setting(args, this_dataset_name, this_subset_name, each_pretrained_model)
            train_dataset, eval_dataset, test_dataset = \
            autohf.prepare_data(**preparedata_setting)
            autohf_settings = get_autohf_settings_grid()

            try:
                validation_metric, analysis = autohf.fit(train_dataset,
                           eval_dataset,
                           **autohf_settings,)
            except AssertionError as err:
                raise err

            save_file_name = get_full_name(autohf, is_grid=True)
            write_regular(autohf, args, validation_metric, save_file_name, fout)
            output_predict(args, test_dataset, autohf, fout, save_file_name)
            rm_home_result()

def _test_hpo(args, fout, autohf):
    for data_idx in range(len(dataset_names)):
        this_dataset_name = dataset_names[data_idx]
        this_subset_name = subdataset_names[data_idx]

        for algo_idx in range(0, len(search_algos)):
            this_search_algo = search_algos[algo_idx]
            for model_idx in range(0, 1): #len(pretrained_models)):
                each_pretrained_model = pretrained_models[model_idx]

                this_scheduler_name = scheduler_names[algo_idx]
                for space_idx in range(0, len(hpo_searchspace_modes)):
                    hpo_searchspace_mode = hpo_searchspace_modes[space_idx]
                    search_algo_args_mode = search_algo_args_modes[space_idx]
                    preparedata_setting = get_preparedata_setting(args, this_dataset_name, this_subset_name,
                                                                  each_pretrained_model)

                    train_dataset, eval_dataset, test_dataset = \
                        autohf.prepare_data(**preparedata_setting)
                    autohf_settings = get_autohf_settings(this_search_algo, this_scheduler_name, hpo_searchspace_mode, time_as_grid, search_algo_args_mode)

                    try:
                        validation_metric, analysis = autohf.fit(train_dataset,
                                   eval_dataset,
                                   **autohf_settings,)
                    except AssertionError:
                        save_file_name = get_full_name(autohf, is_grid=True)
                        write_exception(args, save_file_name, fout)
                        continue

                    save_file_name = get_full_name(autohf, is_grid=True)
                    write_regular(autohf, args, validation_metric, save_file_name, fout)
                    output_predict(args, test_dataset, autohf, fout, save_file_name)
                    rm_home_result()

    fout.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True, choices=["tmdev", "dgx"])
    arg_parser.add_argument('--algo', type=str, help='hpo or grid search', required=True, choices=["grid", "hpo"])
    args = arg_parser.parse_args()

    fout = open("log_" + args.server_name + ".log", "a")
    if args.algo == "grid":
        _test_grid(args, fout, autohf = AutoTransformers())
    else:
        _test_hpo(args, fout, autohf = AutoTransformers())
    fout.close()
