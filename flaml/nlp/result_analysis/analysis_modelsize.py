# import re
# import pandas
# from ..dataset.metric_auto import get_default_and_alternative_metric
# import subprocess
# from .utils import init_blob_client
# import pathlib
# from scipy.stats import kendalltau
# import numpy as np
#
# small_model_id = 7
# large_model_id = 8
# model_type = "funnel"
#
# space_id = 1
# all_algo_ids = [0, 2, 4]
# key_list = ["weight_decay", "learning_rate", "num_train_epochs", "per_device_train_batch_size", "warmup_ratio", "adam_epsilon", "hidden_dropout_prob", "attention_probs_dropout_prob"]
#
#
#
# def convert_list_to_pd_frame(list1):
#     pd1 = pandas.DataFrame(index=[x for x in range(len(list1))], columns=[x for x in range(2)])
#     for x in range(len(list1)):
#         pd1.loc[x] = [list1[x][0], list1[x][1]]
#     return pd1
#
# def analysis_topsmalltoplarge(algo, rep_id, task_name, configs_str_list, configstr2scores, smaller_size, larger_size):
#     small_list = configs_str_list[0]
#     large_list = configs_str_list[1]
#     small_dict = configstr2scores[0]
#     large_dict = configstr2scores[1]
#
#     start_idx = 0
#     while start_idx < len(large_list) and large_list[start_idx] not in small_list:
#         start_idx += 1
#
#     if start_idx >= len(large_list):
#         print("{}: small and large has no overlap".format(task_name))
#         return
#
#     small_position = small_list.index(large_list[start_idx])
#     gap = np.mean(large_dict[large_list[0]]) - np.mean(large_dict[large_list[start_idx]])
#
#     print("{}'s {}-th rep, {}: {}'s {}-th value is in {}'s {}-th position, gap = {}".format(algo, rep_id, task_name, larger_size, start_idx, smaller_size, small_position, gap))
#
# def get_config2score(args, tab, this_dataset_name, this_subdataset_name, model_size, algo_lowerbound, algo_upperbound, rep_lowerbound, rep_upperbound, all_blobs, model_id):
#     config2scores = {}
#     for algo_idx in range(algo_lowerbound, algo_upperbound):  # len(all_algo_ids)):
#         algo_id = all_algo_ids[algo_idx]
#         for rep_id in range(rep_lowerbound, rep_upperbound):
#             this_blob_file = all_blobs[model_id][algo_id][space_id][rep_id]
#             blob_client = init_blob_client(args.azure_key, this_blob_file)
#             pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", this_blob_file).group("parent_path")).mkdir(
#                 parents=True, exist_ok=True)
#             with open(this_blob_file, "wb") as fout:
#                 fout.write(blob_client.download_blob().readall())
#             with open(this_blob_file, "r") as fin:
#                 wandb_group_name = fin.readline().rstrip(":\n")
#                 if wandb_group_name == "glue_cola_funnel_base_optuna_none_hpo_space_gridunion_other_1edp44is":
#                     stop = 0
#                 for (configs, default_metric, score, trial_name) in get_config_to_score(this_dataset_name,
#                                                                                         this_subdataset_name,
#                                                                                         wandb_group_name):
#                     try:
#                         this_hp_configs = [str(key) + "=" + str(configs[key]) for key in key_list]
#                         config2scores.setdefault("_".join(this_hp_configs), [])
#                         config2scores["_".join(this_hp_configs)].append(score)
#                         this_row = {key: configs[key] for key in key_list}
#                         this_row["eval_score"] = score
#                         this_row["dataset"] = this_dataset_name
#                         this_row["model"] = model_size
#                         this_row["rep_id"] = rep_id
#                         tab = tab.append(this_row, ignore_index=True)
#                     except KeyError:
#                         pass
#     return tab, config2scores
#
#
# def analysis_model_size(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, split_modes):
#     """ analyze the following hypothesis: the ranking orders of hyperparameters
#         are exactly the same on small/base and large models, therefore we can
#         fine tune large models by fine tuning small models and apply on large models"""
#     subprocess.run(["wandb", "login", "--relogin", args.wandb_key])
#
#
#     resplit_id = 0
#
#
#     tab = pandas.DataFrame(columns= key_list + ["eval_score", "dataset", "model", "rep_id"])
#     modelsize2id = {"small": small_model_id, "large": large_model_id}
#
#     configs_str_list = [None, None]
#     configstr2scores = [None, None]
#
#     is_large_first = True
#
#     for small_algo_lowerbound in range(1, 2):
#         for small_rep_lowerbound in range(0, 1):
#             for run_idx in range(3):
#                 all_runs = []
#                 task_name = dataset_names[run_idx][0] + "_" + subdataset_names[run_idx]
#                 all_blobs = task2blobs[resplit_id][task_name]
#
#                 for model_size_idx in range(2):
#                     if model_size_idx == 0:
#                         model_size = "small"
#                     else:
#                         model_size = "large"
#                     if model_size == "small":
#                         algo_lowerbound, algo_upperbound = small_algo_lowerbound, small_algo_lowerbound + 1
#                         rep_lowerbound, rep_upperbound = small_rep_lowerbound, small_rep_lowerbound + 1
#                     else:
#                         algo_lowerbound, algo_upperbound = 1,2
#                         rep_lowerbound, rep_upperbound = 0,3
#                     model_id = modelsize2id[model_size]
#                     if model_size == "small":
#                         tab, config2scores = get_config2score(args, tab, dataset_names[run_idx][0], subdataset_names[run_idx], model_size, algo_lowerbound, algo_upperbound, rep_lowerbound, rep_upperbound, all_blobs, model_id)
#                         configs_str_list[model_size_idx] = sorted(config2scores.keys(), key = lambda x: np.mean(config2scores[x]), reverse= True)
#                         configstr2scores[model_size_idx] = config2scores
#                     elif model_size == "large":
#                         if is_large_first:
#                             tab, config2scores = get_config2score(args, tab, dataset_names[run_idx][0],
#                                                                   subdataset_names[run_idx], model_size,
#                                                                   algo_lowerbound, algo_upperbound, rep_lowerbound,
#                                                                   rep_upperbound, all_blobs, model_id)
#                             configs_str_list[model_size_idx] = sorted(config2scores.keys(),
#                                                                       key=lambda x: np.mean(config2scores[x]),
#                                                                       reverse=True)
#                             configstr2scores[model_size_idx] = config2scores
#                             is_large_first = False
#                 analysis_topsmalltoplarge(search_algos[all_algo_ids[small_algo_lowerbound]], small_rep_lowerbound, task_name, configs_str_list, configstr2scores, pretrained_models[small_model_id][1], pretrained_models[large_model_id][1])
#             print("\n")
#         print("\n")
#         stop = 0
#
#     tab.to_csv(model_type + "_smallvlarge.csv")