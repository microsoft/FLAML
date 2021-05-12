import re
import pandas
from ..dataset.metric_auto import get_default_and_alternative_metric
import subprocess
from .utils import init_blob_client
import pathlib
from scipy.stats import kendalltau
import numpy as np

def get_config_to_score(dataset_name, subdataset_name, this_group_name):
    import wandb
    api = wandb.Api()
    runs = api.runs('liususan/' + dataset_name + "_" + subdataset_name, filters={"group": this_group_name})
    config2score = []
    for idx in range(0, len(runs)):
        run = runs[idx]
        this_hp_config_str = "_".join([str(key) + "=" + str(run.summary[key])
            for key in sorted(run.summary.keys())
            if key in ("weight_decay", "learning_rate", "num_train_epochs", "per_device_train_batch_size", "warmup_ratio")])
        default_metric, _, _, _ = get_default_and_alternative_metric(dataset_name, subdataset_name)
        try:
            this_eval_acc = run.summary['eval/' + default_metric]
            config2score.append((this_hp_config_str, this_eval_acc, run.name))
        except KeyError:
            pass
    return config2score

def convert_list_to_pd_frame(list1):
    pd1 = pandas.DataFrame(index=[x for x in range(len(list1))], columns=[x for x in range(2)])
    for x in range(len(list1)):
        pd1.loc[x] = [list1[x][0], list1[x][1]]
    return pd1

def analysis_model_size(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, split_modes):
    """ analyze the following hypothesis: the ranking orders of hyperparameters
        are exactly the same on small/base and large models, therefore we can
        fine tune large models by fine tuning small models and apply on large models"""
    subprocess.run(["wandb", "login", "--relogin", args.wandb_key])
    small_model_id = 3
    large_model_id = 5

    space_id = 1
    algo_id = 2

    resplit_id = 0

    for run_idx in range(len(dataset_names)):
        all_runs = []
        task_name = dataset_names[run_idx][0] + "_" + subdataset_names[run_idx]
        all_blobs = task2blobs[resplit_id][task_name]
        configs = []
        configs2scores = []
        for model_id in [small_model_id, large_model_id]:
            config2scores = {}
            for rep_id in range(3):
                this_blob_file = all_blobs[model_id][algo_id][space_id][rep_id]
                blob_client = init_blob_client(args.azure_key, this_blob_file)
                pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", this_blob_file).group("parent_path")).mkdir(
                    parents=True, exist_ok=True)
                with open(this_blob_file, "wb") as fout:
                    fout.write(blob_client.download_blob().readall())
                with open(this_blob_file, "r") as fin:
                    this_group_name = fin.readline().rstrip(":\n")
                    for (config, score, trial_name) in get_config_to_score(dataset_names[run_idx][0], subdataset_names[run_idx], this_group_name):
                        config2scores.setdefault(config, [])
                        config2scores[config].append(score)
            configs.append(sorted(config2scores.keys(), key = lambda x: np.mean(config2scores[x]), reverse= True))
            configs2scores.append(config2scores)
        intersection_config = set(configs[0]).intersection(set(configs[1]))
        list1 = [(x, str(np.mean(configs2scores[0][x]))) for x in configs[0] if x in intersection_config]
        list2 = [(x, str(np.mean(configs2scores[1][x]))) for x in configs[1] if x in intersection_config]

        pd1 = convert_list_to_pd_frame(list1)
        pd2 = convert_list_to_pd_frame(list2)

        pd1.to_csv("small_model.csv")
        pd2.to_csv("large_model.csv")

        similarity = kendalltau(list1, list2)
        stop = 0