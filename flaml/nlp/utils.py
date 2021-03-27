from dataclasses import dataclass, field
import os, json
import pathlib


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
    This is the class for maintaining the paths (checkpoints, results) in AutoHuggingFace.

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
    _hpo_ckpt_path: str = field(default=None, metadata={"help": "the directory for hpo output"})
    _hpo_result_path: str = field(default=None, metadata={"help": "the directory for hpo result"})
    _hpo_log_path: str = field(default=None, metadata={"help": "the directory for log"})

    _dataset_name: str = field(default=None, metadata={"help": "dataset name"})
    _subdataset_name: str = field(default=None, metadata={"help": "sub dataset name"})
    _search_algo_name: str = field(default=None, metadata={"help": "The hpo method."})

    _group_hash_id: str = field(default=None, metadata={"help": "hash code for the hpo run"})

    _model_name: str = field(default= None, metadata={"help": "huggingface name."})
    _folder_name: str = field(default=None, metadata={"help": "folder name."})

    _log_dir_per_run: str = field(default= None, metadata={"help": "log directory for each run."})
    _result_dir_per_run: str = field(default=None, metadata={"help": "result directory for each run."})
    _ckpt_dir_per_run: str = field(default=None, metadata={"help": "checkpoint directory for each run."})
    _ckpt_dir_per_trial: str = field(default=None, metadata={"help": "checkpoint directory for each trial."})

    def __init__(self,
                 hpo_ckpt_path,
                 hpo_result_path,
                 hpo_log_path,
                 dataset_name,
                 subdataset_name,
                 model_name,
                 ):
        self._hpo_ckpt_path = hpo_ckpt_path
        self._hpo_result_path = hpo_result_path
        self._hpo_log_path = hpo_log_path
        self._dataset_name = dataset_name
        self._subdataset_name = subdataset_name
        self._model_name = model_name

    def set_folder_name(self, autohf_ref):
        self._folder_name = autohf_ref.search_algo_name.lower() + "_" + autohf_ref.scheduler_name.lower() + "_" \
                            + autohf_ref.model_type.lower() + "_" + autohf_ref.split_mode.lower()

    @property
    def folder_name(self):
        return self._folder_name

    @property
    def group_hash_id(self):
        return self._group_hash_id

    @property
    def model_checkpoint(self):
        return self._model_name # os.path.join(self.hpo_output_dir, "huggingface", self.model_name)

    @property
    def dataset_dir_name(self):
        assert self._dataset_name, "dataset name is required"
        data_dir_name = "_".join(self._dataset_name)
        if self._subdataset_name:
            data_dir_name = data_dir_name + "/" + self._subdataset_name
        return data_dir_name

    @property
    def _ckpt_root_dir_abs(self):
        assert self._hpo_ckpt_path, "output directory is required"
        checkpoint_root_dir_abs = os.path.join(self._hpo_ckpt_path + "/" + self.dataset_dir_name + "/")
        return checkpoint_root_dir_abs

    @property
    def _result_root_dir_abs(self):
        assert self._hpo_result_path, "output directory is required"
        results_root_dir_abs = os.path.join(self._hpo_result_path + "/" + self.dataset_dir_name + "/")
        return results_root_dir_abs

    @property
    def _log_root_dir_abs(self):
        assert self._hpo_log_path, "output directory is required"
        log_root_dir_abs = os.path.join(self._hpo_log_path + "/" + self.dataset_dir_name + "/")
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
