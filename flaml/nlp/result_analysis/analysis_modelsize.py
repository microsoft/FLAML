import re, wandb

from flaml.nlp.dataset.metric_auto import get_default_and_alternative_metric

from flaml.nlp.result_analysis.utils import get_all_runs, init_blob_client
import pathlib
from scipy.stats import kendalltau
import numpy as np

api = wandb.Api()

def get_config_to_score(dataset_name, subdataset_name, this_group_name):
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

def analysis_model_size(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, split_modes):
    """ analyze the following hypothesis: the ranking orders of hyperparameters
        are exactly the same on small/base and large models, therefore we can
        fine tune large models by fine tuning small models and apply on large models"""

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
        for model_id in [small_model_id, large_model_id]:
            config2scores = {}
            for rep_id in range(3):
                this_config2score = {}
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
                        config2scores[config].append((score, trial_name))
                        this_config2score.setdefault(config, [])
                        this_config2score[config].append((score, trial_name))
                        if len(this_config2score[config]) > 1:
                            stop = 0
            configs.append(sorted(config2scores.keys(), key = lambda x: np.mean(config2scores[x]), reverse= True))
        intersection_config = set(configs[0]).intersection(set(configs[1]))
        similarity = kendalltau([x for x in configs[0] if x in intersection_config],
                                [x for x in configs[1] if x in intersection_config])
