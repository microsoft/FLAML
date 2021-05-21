
def extract_ranked_config_score(console_args, partial_config_dict):
    from .azure_utils import AzureUtils
    import numpy as np
    azure_utils = AzureUtils(console_args)

    for method, each_partial_config in partial_config_dict.items():
        dataset2configscorelist = azure_utils.get_config_and_score_from_partial_config(each_partial_config, ["dat", "subdat"], method)
        for each_dataset, configscorelist in dataset2configscorelist.items():
            for config_idx in range(len(configscorelist)):
                avg_scores = configscorelist[config_idx][0][1]
                top_config = configscorelist[config_idx][0][0]
                # print(method + "," + str(each_dataset) + ",rep=" + str(config_idx))
                # print("avg score :" + str(avg_scores))
                # print(''.join(['{0}={1}\n'.format(key, top_config[key]) for key in sorted(top_config.keys())]))

def extract_sorted_config_list(dataset2configscorelist, topk):
    dataset2topkconfigs = {}
    for dataset, configscorelist in dataset2configscorelist.items():
        all_configscorelist = []
        for scorelist in configscorelist:
            for item in scorelist:
                if item[0] not in [x[0] for x in all_configscorelist]:
                    all_configscorelist.append(item)
        sorted_all_configscorelist = sorted(all_configscorelist, key = lambda x:x[1], reverse = True)
        topk_configs = []

        for each_hp in ("learning_rate", "num_train_epochs", "per_device_train_batch_size", "warmup_ratio", "weight_decay", "adam_epsilon"):
            topk_configs.append((each_hp, [sorted_all_configscorelist[x][0][each_hp] for x in range(topk)]))
        topk_configs.append(("perf", [sorted_all_configscorelist[x][1] for x in range(topk)]))

        dataset2topkconfigs[dataset] = topk_configs
    return dataset2topkconfigs

def compare_small_vs_large(console_args):
    from .azure_utils import AzureUtils, JobID
    azure_utils = AzureUtils(console_args)

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "list"
    partial_jobid_config.spa = "uni"
    partial_jobid_config.presz = "small"

    small_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_config(partial_jobid_config,
                                                                                   ["dat", "subdat"], "list")
    small_dataset2topkconfig = extract_sorted_config_list(small_dataset2configscorelist, 10)

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "uni"

    large_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_config(partial_jobid_config,
                                                                                         ["dat", "subdat"], "hpo")

    large_dataset2topkconfig = extract_sorted_config_list(large_dataset2configscorelist, 10)

    for each_dataset in small_dataset2topkconfig.keys():
        print(each_dataset)
        print(small_dataset2topkconfig[each_dataset])
        print(large_dataset2topkconfig[each_dataset])