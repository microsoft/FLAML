from dataclasses import dataclass, field
import os, json
import pathlib,argparse

def load_console_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"], default = None)
    arg_parser.add_argument('--algo_mode', type=str, help='hpo or grid search', required=True,
                            choices=["grid", "gridbert", "hpo", "hfhpo", "list"], default = None)
    arg_parser.add_argument('--data_root_dir', type=str, help='data dir', required=True)
    arg_parser.add_argument('--dataset_subdataset_name', type=str, help='dataset and subdataset name',
                            required=False, default = None)
    arg_parser.add_argument('--space_mode', type=str, help='space mode', required=False, choices = ["gnr", "uni"], default = None)
    arg_parser.add_argument('--search_alg_args_mode', type=str, help = 'search algorithm args mode', required = False, choices = ["dft", "exp", "cus"])
    arg_parser.add_argument('--algo_name', type=str, help='algorithm', required=False, choices = ["bs", "optuna", "cfo"], default = None)
    arg_parser.add_argument('--pruner', type=str, help='pruner', required=False, choices=["asha", "None"], default = None)
    arg_parser.add_argument('--pretrained_model_size', type=str, help='pretrained model', required=False,
                        choices=["xlnet-base-cased:base", "albert-large-v1:small", "distilbert-base-uncased:base",
                                 "microsoft/deberta-base:base","funnel-transformer/small-base:base", "microsoft/deberta-large:large",
                                 "funnel-transformer/large-base:large", "funnel-transformer/intermediate-base:intermediate", "funnel-transformer/xlarge-base:xlarge"], default = None)
    arg_parser.add_argument('--sample_num', type=int, help='sample num', required=False, default = None)
    arg_parser.add_argument('--time_budget', type=int, help='time budget', required=False, default = None)
    arg_parser.add_argument('--rep_id', type=int, help='rep id', required=False, default = None)
    arg_parser.add_argument('--azure_key', type=str, help='azure key', required=False, default = None)
    arg_parser.add_argument('--resplit_mode', type=str, help='resplit mode', required=True, choices = ["rspt", "ori"], default = None)
    arg_parser.add_argument('--ds_config', type=str, help='deep speed config file path', required = False, default = None)
    arg_parser.add_argument('--yml_file', type=str, help='yml file path', required=True, default = None)
    arg_parser.add_argument('--key_path', type=str, help='path for key.json', required=True, default=None)
    return arg_parser.parse_args()

def get_wandb_azure_key(key_path):
    key_json = json.load(open(os.path.join(key_path, "key.json"), "r"))
    wandb_key = key_json["wandb_key"]
    azure_key = key_json["azure_key"]
    azure_container_name = key_json["container_name"]
    return wandb_key, azure_key, azure_container_name

def merge_dicts(dict1, dict2):
    for key2 in dict2.keys():
        if key2 in dict1:
            dict1_vals = set(dict1[key2])
            dict2_vals = set(dict2[key2])
            dict1[key2] = list(dict1_vals.union(dict2_vals))
        else:
            dict1[key2] = dict2[key2]
    return dict1

def _check_dict_keys_overlaps(dict1: dict, dict2: dict):
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    return len(dict1_keys.intersection(dict2_keys)) > 0

def _variable_override_default_alternative(logger, obj_ref, var_name, default_value, all_values, overriding_value=None):
    """
        Setting the value of var. If overriding_value is specified, var is set to overriding_value;
        If overriding_value is not specified, var is set to default_value meanwhile showing all_values
    """
    assert isinstance(all_values, list)
    if overriding_value:
        setattr(obj_ref, var_name, overriding_value)
        logger.warning("The value for {} is specified as {}".format(var_name, overriding_value))
    else:
        setattr(obj_ref, var_name, default_value)
        logger.warning("The value for {} is not specified, setting it to the default value {}. "
                       "Alternatively, you can set it to {}".format(var_name, default_value, ",".join(all_values)))

@dataclass
class PathUtils:
    """
    This is the class for maintaining the paths (checkpoints, results) in AutoTransformers.

    Args:
        hpo_output_dir:
            A string variable, the root directory for outputing data
        dataset_name:
            A list, the first element in the list is the name of the dataset, e.g., ["glue"]
            If the dataset contains a second component, the list should contain a second element, e.g., ["openbookqa", "main"]
        subdataset_name:
            The sub dataset name, e.g., "qnli", not required
        model_name:
            The huggingface name for loading the huggingface from huggingface.co/models, e.g., "google/electra-base-discriminator"
    """
    hpo_ckpt_path: str = field(metadata={"help": "the directory for hpo output"})
    hpo_result_path: str = field(metadata={"help": "the directory for hpo result"})
    hpo_log_path: str = field(metadata={"help": "the directory for log"})
    hpo_config_path: str = field(metadata={"help": "the directory for log"})

    log_dir_per_run: str = field(metadata={"help": "log directory for each run."})
    result_dir_per_run: str = field(metadata={"help": "result directory for each run."})
    ckpt_dir_per_run: str = field(metadata={"help": "checkpoint directory for each run."})
    ckpt_dir_per_trial: str = field(metadata={"help": "checkpoint directory for each trial."})

    def __init__(self,
                 jobid_config,
                 hpo_data_root_path,
                 ):
        self.jobid_config = jobid_config
        self.hpo_data_root_path = hpo_data_root_path
        self.hpo_ckpt_path = os.path.join(hpo_data_root_path, "checkpoint")
        self.hpo_result_path = os.path.join(hpo_data_root_path, "result")
        self.hpo_log_path = self.hpo_result_path

    @staticmethod
    def init_and_make_one_dir(dir_path):
        assert dir_path
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    def make_dir_per_run(self):
        jobid_str = self.jobid_config.to_jobid_string()
        self.ckpt_dir_per_run = os.path.join(self.hpo_ckpt_path, jobid_str)
        PathUtils.init_and_make_one_dir(self.ckpt_dir_per_run)

        self.result_dir_per_run = os.path.join(self.hpo_result_path, jobid_str)
        PathUtils.init_and_make_one_dir(self.result_dir_per_run)

        self.log_dir_per_run = os.path.join(self.hpo_log_path, jobid_str)
        PathUtils.init_and_make_one_dir(self.log_dir_per_run)

    def make_dir_per_trial(self, trial_id):
        jobid_str = self.jobid_config.to_jobid_string()
        ckpt_dir_per_run = os.path.join(self.hpo_ckpt_path, jobid_str)
        self.ckpt_dir_per_trial = os.path.join(ckpt_dir_per_run, jobid_str, trial_id)
        PathUtils.init_and_make_one_dir(self.ckpt_dir_per_trial)

