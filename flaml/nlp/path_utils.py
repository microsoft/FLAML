from dataclasses import dataclass, field
import os, json
import pathlib

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
            The model name for loading the model from huggingface.co/models, e.g., "google/electra-base-discriminator"
    """

    hpo_output_dir: str = field(default="", metadata={"help": "the directory for hpo output"})

    dataset_name: str = field(default="glue", metadata={"help": "dataset name"})
    subdataset_name: str = field(default="glue", metadata={"help": "sub dataset name"})

    group_hash_id: str = field(default=None, metadata={"help": "hash code for the hpo run"})

    model_name: str = field(default= "google/grid-base-discriminator", metadata={"help": "model name."})

    def set_folder_name(self, search_algo, scheduler_name, model_type, submit_mode):
        if not scheduler_name:
            scheduler_name = "none"
        self.folder_name = search_algo.lower() + "_" + scheduler_name + "_" + model_type + "_" + submit_mode
        return self.folder_name

    @property
    def model_checkpoint(self):
        return self.model_name # os.path.join(self.hpo_output_dir, "model", self.model_name)

    @property
    def dataset_dir_name(self):
        assert self.dataset_name, "dataset name is required"
        data_dir_name = "_".join(self.dataset_name)
        if self.subdataset_name:
            data_dir_name = data_dir_name + "/" + self.subdataset_name
        return data_dir_name

    @property
    def _ckpt_root_dir_abs(self):
        assert self.hpo_output_dir, "output directory is required"
        output_dir = os.path.join(self.hpo_output_dir + "output/" + self.dataset_dir_name + "/")
        checkpoint_root_dir_abs = os.path.join(output_dir, "checkpoints/")
        return checkpoint_root_dir_abs

    @property
    def _result_root_dir_abs(self):
        assert self.hpo_output_dir, "output directory is required"
        output_dir = os.path.join(self.hpo_output_dir + "output/" + self.dataset_dir_name + "/")
        results_root_dir_abs = os.path.join(output_dir, "result/")
        return results_root_dir_abs

    def init_and_make_one_dir(self, dir_path):
        assert dir_path
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    def init_and_make_dirs(self):
        self.init_and_make_one_dir(self._ckpt_root_dir_abs)
        self.init_and_make_one_dir(self._result_root_dir_abs)

    def make_dir_per_run(self):
        assert self.folder_name and self.group_hash_id
        self._ckpt_dir_per_run = os.path.join(self._ckpt_root_dir_abs, self.folder_name, self.group_hash_id)
        self.init_and_make_one_dir(self._ckpt_dir_per_run)

        self._result_dir_per_run = os.path.join(self._result_root_dir_abs, self.folder_name, self.group_hash_id)
        self.init_and_make_one_dir(self._result_dir_per_run)

    def make_dir_per_trial(self, trial_id):
        assert self.folder_name and self.group_hash_id
        self._ckpt_dir_per_trial = os.path.join(self._ckpt_root_dir_abs, self.folder_name, self.group_hash_id, trial_id)
        self.init_and_make_one_dir(self._ckpt_dir_per_trial)
