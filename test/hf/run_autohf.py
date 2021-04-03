'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
import json
import os
import shutil
import sys
import time

from flaml.nlp.autotransformers import AutoTransformers

dataset_to_task_mapping = {
    "glue": "text-classification",
    "squad": "question-answering",
}

def _test_grid():
    # setting wandb key
    wandb_key = "7553d982a2247ca8324ec648bd302678105e1058"
    # setting server name
    server_name = "tmdev"

    autohf = AutoTransformers()

    dataset_names = [["glue"]]
    subdataset_names = ["mnli"]

    pretrained_models = ["google/electra-small-discriminator", "google/electra-base-discriminator"]

    search_algos = ["grid_search"]
    scheduler_names = ["None"]

    fout = open("log.log", "a")

    for data_idx in range(len(dataset_names)):
        this_dataset_name = dataset_names[data_idx]
        this_subset_name = subdataset_names[data_idx]

        for model_idx in range(0, len(pretrained_models)):
            each_pretrained_model = pretrained_models[model_idx]

            for algo_idx in range(0, len(search_algos)):
                this_search_algo = search_algos[algo_idx]
                this_scheduler_name = scheduler_names[algo_idx]

                preparedata_setting = {
                    "dataset_config": {"task": dataset_to_task_mapping[this_dataset_name[0]],
                                       "dataset_name": this_dataset_name,
                                       "subdataset_name": this_subset_name,
                                       },
                    "resplit_portion": {"train": [0, 0.25], "dev": [0.25, 0.275], "test": [0.275, 0.3]},
                    "model_name": each_pretrained_model,
                    "server_name": server_name,
                    "split_mode": "resplit",
                    "ckpt_path": "../../../data/checkpoint/",
                    "result_path": "../../../data/result/",
                    "log_path": "../../../data/result/",
                    "max_seq_length": 128,
                }
                if this_dataset_name[0] == "glue" and this_subset_name and this_subset_name == "mnli":
                    preparedata_setting["dataset_config"]["fold_name"] = ['train', 'validation_matched', 'test_matched']
                train_dataset, eval_dataset, test_dataset = \
                autohf.prepare_data(**preparedata_setting)

                autohf_settings = {"resources_per_trial": {"gpu": 1, "cpu": 1},
                                       "wandb_key": wandb_key,
                                       "search_algo_name": this_search_algo,
                                       "scheduler_name": this_scheduler_name,
                                       "ckpt_per_epoch": 1,
                                       }

                try:
                    validation_metric = autohf.fit(train_dataset,
                               eval_dataset,
                               **autohf_settings,)
                except AssertionError:
                    save_file_name = autohf.full_dataset_name + "_" + autohf.model_type + "_" + autohf.search_algo_name \
                                     + "_" + autohf.scheduler_name + "_" + autohf.path_utils.group_hash_id
                    fout.write(save_file_name + ":\n")
                    fout.write("timestamp:" + str(time.time()))
                    fout.write("failed, no checkpoint found\n")
                    fout.flush()
                    continue

                save_file_name = autohf.full_dataset_name.lower() + "_" + autohf.model_type.lower() + "_" \
                                 + autohf.search_algo_name.lower() + "_" + autohf.scheduler_name.lower() + "_" + autohf.path_utils.group_hash_id
                fout.write(save_file_name + ":\n")
                fout.write("validation " + (autohf.metric_name) + ":" + json.dumps(validation_metric) + "\n")
                fout.write("duration:" + str(autohf.last_run_duration) + "\n")
                fout.flush()

                if test_dataset:
                    predictions, output_metric = autohf.predict(test_dataset)
                    fout.write("test " + (autohf.metric_name) + ":" + json.dumps(output_metric) + "\n")
                    fout.flush()
                    if autohf.split_mode == "origin":
                        autohf.output_prediction(predictions,
                                             output_prediction_path="../../../data/result/",
                                             output_dir_name=save_file_name)

                if os.path.exists("/home/xliu127/ray_results/"):
                    shutil.rmtree("/home/xliu127/ray_results/")

    fout.close()

def _test_hpo():
    # setting wandb key
    wandb_key = "7553d982a2247ca8324ec648bd302678105e1058"
    # setting server name
    server_name = "tmdev"

    autohf = AutoTransformers()

    dataset_names = [["glue"]]
    subdataset_names = ["mnli"]

    pretrained_models = ["google/electra-small-discriminator", "google/electra-base-discriminator"]

    search_algos = ["BlendSearch"]
    scheduler_names = ["None"]

    hpo_searchspace_modes = ["hpo_space_generic", "hpo_space_gridunion_continuous", "hpo_space_gridunion"]
    num_sample_time_budget_mode, time_as_grid = ("times_grid_time_budget", 4.0)

    fout = open("log.log", "a")

    for data_idx in range(len(dataset_names)):
        this_dataset_name = dataset_names[data_idx]
        this_subset_name = subdataset_names[data_idx]

        for model_idx in range(0, len(pretrained_models)):
            each_pretrained_model = pretrained_models[model_idx]

            for algo_idx in range(0, len(search_algos)):
                this_search_algo = search_algos[algo_idx]
                this_scheduler_name = scheduler_names[algo_idx]

                for space_idx in range(0, len(hpo_searchspace_modes)):
                    hpo_searchspace_mode = hpo_searchspace_modes[space_idx]

                    for rep in range(2):
                        if rep == 0:
                            resources_dict = {"gpu": 1, "cpu": 1}
                        else:
                            resources_dict = {"gpu": 4, "cpu": 4}

                        preparedata_setting = {
                            "dataset_config": {"task": dataset_to_task_mapping[this_dataset_name[0]],
                                               "dataset_name": this_dataset_name,
                                               "subdataset_name": this_subset_name,
                                               },
                            "resplit_portion": {"train": [0, 0.25], "dev": [0.25, 0.275], "test": [0.275, 0.3]},
                            "model_name": each_pretrained_model,
                            "server_name": server_name,
                            "split_mode": "resplit",
                            "ckpt_path": "../../../data/checkpoint/",
                            "result_path": "../../../data/result/",
                            "log_path": "../../../data/result/",
                            "max_seq_length": 128,
                        }
                        if this_dataset_name[0] == "glue" and this_subset_name and this_subset_name == "mnli":
                            preparedata_setting["dataset_config"]["fold_name"] = ['train', 'validation_matched', 'test_matched']
                        train_dataset, eval_dataset, test_dataset = \
                        autohf.prepare_data(**preparedata_setting)

                        autohf_settings = {"resources_per_trial": resources_dict,
                                               "wandb_key": wandb_key,
                                               "search_algo_name": this_search_algo,
                                               "scheduler_name": this_scheduler_name,
                                               "ckpt_per_epoch": 1,
                                               }
                        autohf_settings["hpo_searchspace_mode"] = hpo_searchspace_mode
                        autohf_settings["num_sample_time_budget_mode"] = num_sample_time_budget_mode
                        autohf_settings["time_as_grid"] = time_as_grid

                        try:
                            validation_metric = autohf.fit(train_dataset,
                                       eval_dataset,
                                       **autohf_settings,)
                        except AssertionError:
                            save_file_name = autohf.full_dataset_name + "_" + autohf.model_type + "_" + autohf.search_algo_name \
                                             + "_" + autohf.scheduler_name + "_" + autohf.path_utils.group_hash_id
                            fout.write(save_file_name + ":\n")
                            fout.write("timestamp:" + str(time.time()))
                            fout.write("failed, no checkpoint found\n")
                            fout.flush()
                            continue

                        save_file_name = autohf.full_dataset_name.lower() + "_" + autohf.model_type.lower() + "_" \
                                         + autohf.search_algo_name.lower() + "_" + autohf.scheduler_name.lower() + "_" + autohf.path_utils.group_hash_id
                        fout.write(save_file_name + ":\n")
                        fout.write("validation " + (autohf.metric_name) + ":" + json.dumps(validation_metric) + "\n")
                        fout.write("duration:" + str(autohf.last_run_duration) + "\n")
                        fout.flush()

                        if test_dataset:
                            predictions, output_metric = autohf.predict(test_dataset)
                            fout.write("test " + (autohf.metric_name) + ":" + json.dumps(output_metric) + "\n")
                            fout.flush()
                            if autohf.split_mode == "origin":
                                autohf.output_prediction(predictions,
                                                     output_prediction_path="../../../data/result/",
                                                     output_dir_name=save_file_name)

                        if os.path.exists("/home/xliu127/ray_results/"):
                            shutil.rmtree("/home/xliu127/ray_results/")

    fout.close()

if __name__ == "__main__":
    _test_hpo()
