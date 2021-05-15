from dataclasses import dataclass, field
import os, json
import pathlib

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
        group_hash_id:
            The group name in wandb
        model_name:
            The huggingface name for loading the huggingface from huggingface.co/models, e.g., "google/electra-base-discriminator"
    """
    hpo_ckpt_path: str = field(metadata={"help": "the directory for hpo output"})
    hpo_result_path: str = field(metadata={"help": "the directory for hpo result"})
    hpo_log_path: str = field(metadata={"help": "the directory for log"})
    hpo_config_path: str = field(metadata={"help": "the directory for log"})

    dataset_name: str = field(metadata={"help": "dataset name"})
    subdataset_name: str = field(metadata={"help": "sub dataset name"})
    _search_algo_name: str = field(metadata={"help": "The hpo method."})

    _group_hash_id: str = field(metadata={"help": "hash code for the hpo run"})

    model_name: str = field(metadata={"help": "huggingface name."})
    model_size_type: str = field(metadata={"help": "model size type."})
    _folder_name: str = field(metadata={"help": "folder name."})

    _log_dir_per_run: str = field(metadata={"help": "log directory for each run."})
    _result_dir_per_run: str = field(metadata={"help": "result directory for each run."})
    _ckpt_dir_per_run: str = field(metadata={"help": "checkpoint directory for each run."})
    _ckpt_dir_per_trial: str = field(metadata={"help": "checkpoint directory for each trial."})

    def __init__(self,
                 hpo_data_root_path,
                 dataset_name,
                 subdataset_name,
                 model_name,
                 model_size_type,
                 ):
        self.hpo_data_root_path = hpo_data_root_path
        self.hpo_ckpt_path = os.path.join(hpo_data_root_path, "checkpoint")
        self.hpo_result_path = os.path.join(hpo_data_root_path, "result")
        self.hpo_log_path = self.hpo_result_path
        self.dataset_name = dataset_name
        self.subdataset_name = subdataset_name
        self.model_name = model_name
        self.model_size_type = model_size_type

    def set_folder_name(self, autohf_ref):
        self._folder_name = autohf_ref.model_type.lower() + "_" + autohf_ref.split_mode.lower()
        if hasattr(autohf_ref, "search_algo_name"):
            self._folder_name = autohf_ref.search_algo_name.lower() + "_" + self._folder_name
        if hasattr(autohf_ref, "scheduler_name"):
            self._folder_name =  autohf_ref.scheduler_name.lower() + "_" + self._folder_name

    @property
    def folder_name(self):
        return self._folder_name

    @property
    def group_hash_id(self):
        return self._group_hash_id

    @property
    def model_checkpoint(self):
        return self.model_name # os.path.join(self.hpo_output_dir, "huggingface", self.model_name)

    @property
    def dataset_dir_name(self):
        assert self.dataset_name, "dataset name is required"
        data_dir_name = "_".join(self.dataset_name)
        if self.subdataset_name:
            data_dir_name = data_dir_name + "/" + self.subdataset_name
        return data_dir_name

    @property
    def _ckpt_root_dir_abs(self):
        assert self.hpo_ckpt_path, "output directory is required"
        checkpoint_root_dir_abs = os.path.join(self.hpo_ckpt_path + "/" + self.dataset_dir_name + "/")
        return checkpoint_root_dir_abs

    @property
    def _result_root_dir_abs(self):
        assert self.hpo_result_path, "output directory is required"
        results_root_dir_abs = os.path.join(self.hpo_result_path + "/" + self.dataset_dir_name + "/")
        return results_root_dir_abs

    @property
    def _log_root_dir_abs(self):
        assert self.hpo_log_path, "output directory is required"
        log_root_dir_abs = os.path.join(self.hpo_log_path + "/" + self.dataset_dir_name + "/")
        return log_root_dir_abs

    @property
    def log_dir_per_run(self):
        return self._log_dir_per_run

    @property
    def result_dir_per_run(self):
        return self._result_dir_per_run

    @property
    def ckpt_dir_per_run(self):
        return self._ckpt_dir_per_run

    @property
    def ckpt_dir_per_trial(self):
        return self._ckpt_dir_per_trial

    @property
    def search_algo_name(self):
        """
        Get the search algorithm name
        """
        return self._search_algo_name

    @staticmethod
    def init_and_make_one_dir(dir_path):
        assert dir_path
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    def init_and_make_dirs(self):
        PathUtils.init_and_make_one_dir(self._ckpt_root_dir_abs)
        PathUtils.init_and_make_one_dir(self._result_root_dir_abs)
        PathUtils.init_and_make_one_dir(self._log_root_dir_abs)

    def make_dir_per_run(self):
        assert self._folder_name and self._group_hash_id
        self._ckpt_dir_per_run = os.path.join(self._ckpt_root_dir_abs, self._folder_name, self._group_hash_id)
        PathUtils.init_and_make_one_dir(self._ckpt_dir_per_run)

        self._result_dir_per_run = os.path.join(self._result_root_dir_abs, self._folder_name, self._group_hash_id)
        PathUtils.init_and_make_one_dir(self._result_dir_per_run)

        self._log_dir_per_run = os.path.join(self._log_root_dir_abs, self._folder_name, self._group_hash_id)
        PathUtils.init_and_make_one_dir(self._log_dir_per_run)

    def make_dir_per_trial(self, trial_id):
        assert self._folder_name and self._group_hash_id
        self._ckpt_dir_per_trial = os.path.join(self._ckpt_root_dir_abs, self._folder_name, self._group_hash_id,
                                                trial_id)
        PathUtils.init_and_make_one_dir(self._ckpt_dir_per_trial)

    @search_algo_name.setter
    def search_algo_name(self, value):
        self._search_algo_name = value

    @group_hash_id.setter
    def group_hash_id(self, value):
        self._group_hash_id = value
