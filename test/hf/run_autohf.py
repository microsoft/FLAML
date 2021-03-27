'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
import json
import os
import shutil
import time

from flaml.nlp.autohf import AutoHuggingFace

dataset_to_task_mapping = {
    "glue": "text-classification",
    "squad": "question-answering",
}

def _test_electra():
    # setting wandb key
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    dataset_names = [["squad"]]
    subdataset_names = [None]

    pretrained_models = ["bert-base-uncased", "google/electra-base-discriminator"]

    search_algos = ["RandomSearch", "BlendSearch"]
    scheduler_names = ["None", "None"]
    time_limits = [4000, 4000]

    fout = open("log.log", "w")

    for data_idx in range(len(dataset_names)):
        this_dataset_name = dataset_names[data_idx]
        this_subset_name = subdataset_names[data_idx]
        for each_pretrained_model in pretrained_models:

            for algo_idx in range(len(search_algos)):
                this_search_algo = search_algos[algo_idx]
                this_scheduler_name = scheduler_names[algo_idx]
                preparedata_setting = {
                    "dataset_config": {"task": dataset_to_task_mapping[this_dataset_name[0]],
                                       "dataset_name": this_dataset_name,
                                       "subdataset_name": this_subset_name},
                    "model_name": each_pretrained_model,
                    "split_mode": "origin",
                    "ckpt_path": "../../../data/checkpoint/",
                    "result_path": "../../../data/result/",
                    "log_path": "../../../data/result/",
                    "max_seq_length": 128,
                }
                train_dataset, eval_dataset, test_dataset =\
                    autohf.prepare_data(**preparedata_setting)

                autohf_settings = {"resources_per_trial": {"gpu": 4, "cpu": 4},
                                   "wandb_key": wandb_key,
                                   "num_samples": 4 if this_search_algo != "grid_search" else 1,
                                   "time_budget": time_limits[algo_idx],
                                   "search_algo_name": this_search_algo,
                                   "scheduler_name": this_scheduler_name
                                   }

                try:
                    validation_metric = autohf.fit(train_dataset,
                               eval_dataset,
                               **autohf_settings,)
                except AssertionError:
                    save_file_name = autohf.full_dataset_name + "_" + autohf.model_type + "_" + autohf.search_algo_name + "_" + autohf.scheduler_name + "_" + autohf.path_utils.group_hash_id
                    fout.write(save_file_name + ":\n")
                    fout.write("failed, no checkpoint found\n")
                    fout.flush()
                    continue

                if this_search_algo == "grid_search":
                    this_grid_search_time = autohf.last_run_duration

                save_file_name = autohf.full_dataset_name + "_" + autohf.model_type + "_" + autohf.search_algo_name + "_" + autohf.scheduler_name + "_" + autohf.path_utils.group_hash_id
                if test_dataset:
                    predictions = autohf.predict(test_dataset)
                    autohf.output_prediction(predictions,
                                             output_prediction_path="../../../data/result/",
                                             output_dir_name=save_file_name)

                fout.write(save_file_name + ":\n")
                fout.write((autohf.metric_name) + ":" + json.dumps(validation_metric) + "\n")
                fout.write("duration:" + str(autohf.last_run_duration) + "\n\n")
                fout.flush()

                if os.path.exists("/home/xliu127/ray_results/"):
                    shutil.rmtree("/home/xliu127/ray_results/")

        fout.close()

if __name__ == "__main__":
    _test_electra()
