'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
import argparse
from flaml.nlp.result_analysis.azure_utils import JobID


def extract_sorted_config_list(dataset2configscorelist, topk):
    dataset2topkconfigs = {}
    for dataset, configscorelist in dataset2configscorelist.items():
        all_configscorelist = []
        for scorelist in configscorelist:
            for item in scorelist:
                if item[0] not in [x[0] for x in all_configscorelist]:
                    all_configscorelist.append(item)
        sorted_all_configscorelist = sorted(all_configscorelist, key=lambda x: x[1], reverse=True)
        topk_configs = []

        for each_hp in ("learning_rate", "num_train_epochs", "per_device_train_batch_size", "warmup_ratio",
                        "weight_decay", "adam_epsilon"):
            topk_configs.append((each_hp, [sorted_all_configscorelist[x][0][each_hp] for x in range(topk)]))
        topk_configs.append(("perf", [sorted_all_configscorelist[x][1] for x in range(topk)]))

        dataset2topkconfigs[dataset] = topk_configs
    return dataset2topkconfigs


def dict2tuple(this_dict):
    tuple_list = []
    for key in sorted(this_dict.keys()):
        tuple_list.append(this_dict[key])
    return tuple(tuple_list)


def merge_configscore_list(small_dataset2configscorelist):
    dataset2merged_configscorelist = {}
    for (dataset, each_configscore_list) in small_dataset2configscorelist.items():
        merged_configscore_list = {}
        for rep_id in range(len(each_configscore_list)):
            for each_configscore_entry in each_configscore_list[rep_id]:
                exists = False
                for configscore in merged_configscore_list.keys():
                    if configscore[0] == each_configscore_entry[0]:
                        exists = True
                        break
                if exists is False:
                    merged_configscore_list[dict2tuple(each_configscore_entry[0])] = each_configscore_entry[1]
        dataset2merged_configscorelist[dataset] = merged_configscore_list
    return dataset2merged_configscorelist


def print_all_configs(console_args, partial_jobid_config):
    from flaml.nlp.result_analysis.azure_utils import AzureUtils, JobID
    azure_utils = AzureUtils(console_args=console_args)
    matched_config_score_lists = \
        azure_utils.get_config_and_score_from_partial_jobid(
            console_args.azure_root_log_path,
            partial_jobid_config)

    for each_configscore_list in matched_config_score_lists:
        for (config_dict, score, time_stamp) in each_configscore_list._config_score_list:
            print_config(config_dict)
            print(score)
            print()


def print_config(config_dict):
    for key in sorted(config_dict.keys()):
        if key in ("attention_probs_dropout_prob", "hidden_dropout_prob", "seed"):
            continue
        if key == "per_device_train_batch_size":
            short_key = "batch_size"
        elif key == "num_train_epochs":
            short_key = "epochs"
        else:
            short_key = key
        print(short_key, config_dict[key])


def compare_small_vs_large(console_args):
    from flaml.nlp.result_analysis.azure_utils import AzureUtils, JobID
    azure_utils = AzureUtils(console_args=console_args)

    partial_jobid_config = JobID()
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "uni"
    partial_jobid_config.presz = "base"

    small_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_jobid(
        console_args.azure_root_log_path,
        partial_jobid_config)

    small_mergedconfiglist = merge_configscore_list(small_dataset2configscorelist)

    partial_jobid_config = JobID()
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "uni"
    partial_jobid_config.presz = "large"

    large_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_jobid(
        console_args.azure_root_log_path,
        partial_jobid_config)

    large_mergedconfiglist = merge_configscore_list(large_dataset2configscorelist)

    for (each_dataset, merged_small_configlist) in small_mergedconfiglist.items():
        merged_large_configlist = large_mergedconfiglist[each_dataset]
        print(each_dataset)
        print()
        for (each_tuple, large_score) in sorted(merged_large_configlist.items(), key=lambda x: x[1], reverse=True):
            # small_score = merged_small_configlist[each_tuple]
            is_in_onlysmall = each_tuple in small_mergedconfiglist[each_dataset]
            for each_val in each_tuple:
                print(each_val, end=", ")
            print(large_score, is_in_onlysmall, sep=",")
        print()
        for (each_tuple, small_score) in \
                sorted(small_mergedconfiglist[each_dataset].items(), key=lambda x: x[1], reverse=True):
            is_in_large = each_tuple in large_mergedconfiglist[each_dataset]
            for each_val in each_tuple:
                print(each_val, end=", ")
            print(small_score, is_in_large, sep=",")


def print_sorted_configs(console_args,
                         sort_method):
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils
    jobid_config = JobID()
    jobid_config.mod = "bestnn"
    jobid_config.spa = "buni"
    jobid_config.alg = "bs"
    jobid_config.pre = "funnel"
    jobid_config.presz = "xlarge"

    subdat_list = ["rte", "mrpc", "cola"]

    for each_rep in range(3):
        jobid_config.rep = console_args.rep_id = each_rep
        jobid_config.subdat = subdat_list[each_rep]
        azure_utils = AzureUtils(console_args=console_args)

        matched_config_score_lists = \
            azure_utils.get_config_and_score_from_partial_jobid(
                console_args.azure_root_log_path,
                jobid_config)
        configscorelist = matched_config_score_lists[0]._config_score_list
        count = 0
        for (each_config, each_score, time_stamp) in configscorelist.sorted(sort_method):
            print(count)
            print(each_score)
            print_config(each_config)
            print()
            count += 1


def analyze_exhaustive_sweep(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils, ConfigScoreList
    partial_jobid_config = JobID()
    partial_jobid_config.mod = "grid"
    partial_jobid_config.pre = "electra"
    partial_jobid_config.presz = "base"

    azure_utils = AzureUtils(root_log_path= console_args.azure_root_log_path, console_args=console_args)

    for subdat in ["mrpc"]:
        partial_jobid_config.subdat = subdat
        matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
            "logs_seed/",
            partial_jobid_config)
        ConfigScoreList([x for config_score_list in matched_config_score_lists
                         for x in config_score_list._config_score_list])


def create_partial_config_bestnn():
    jobid_config = JobID()
    # funnel xlarge
    # jobid_config.mod = "bestnn"
    jobid_config.spa = "uni"
    # jobid_config.arg = "cus"
    # jobid_config.alg = "cfo"
    jobid_config.pre = "funnel"
    jobid_config.presz = "xlarge"
    # funnel small
    # jobid_config.mod = "list"
    # jobid_config.pre = "funnel"
    # jobid_config.presz = "small"
    # jobid_config.rep = 0

    # # deberta large
    # jobid_config.mod = "bestnn"
    # jobid_config.spa = "uni"
    # jobid_config.arg = "cus"
    # jobid_config.alg = "cfo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "large"

    # # deberta base
    # jobid_config.mod = "hpo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "base"
    # jobid_config.rep = 0

    # # deberta large
    # jobid_config.mod = "hpo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "large"

    return jobid_config


def create_partial_config_list():
    jobid_config = JobID()
    jobid_config.mod = "list"
    jobid_config.spa = "uni"
    jobid_config.presz = "xlarge"
    return jobid_config


def create_partial_config_hpo():
    jobid_config = JobID()
    jobid_config.mod = "bestnn"
    jobid_config.spa = "buni"
    jobid_config.pre = "deberta"
    jobid_config.presz = "large"
    jobid_config.alg = "cfo"
    jobid_config.pru = "None"

    return jobid_config


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--key_path', type=str, help='key path', required=False, default="../../")
    arg_parser.add_argument('--azure_root_log_path', type=str,
                            help='root log path of blob storage', required=True, default="logs_azure/")
    args = arg_parser.parse_args()

    partial_config_large = create_partial_config_hpo()
    analyze_exhaustive_sweep(console_args=args)
