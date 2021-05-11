import re

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
        for task_id in task2blobs.keys():
            all_blobs = task2blobs[resplit_id][task_name]
            for model_id in [small_model_id, large_model_id]:
                for rep_id in range(3):
                    this_blob_file = all_blobs[model_id][algo_id][space_id][rep_id]
