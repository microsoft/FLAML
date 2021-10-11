import argparse
import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional, List


def dataset_config_format_check(dataset_config):
    from datasets import load_dataset

    assert isinstance(dataset_config, dict) or isinstance(dataset_config, list), (
        "dataset_name must either be a dict or list, see the example in the"
        "documentation of HPOArgs::dataset_name"
    )
    if isinstance(dataset_config, dict):
        args_for_load_dataset = load_dataset.__code__.co_varnames
        for each_key in dataset_config.keys():
            assert (
                each_key in args_for_load_dataset
            ), "If HPOArgs::dataset_name is a dict, each key must be from load_dataset's argument list"


def points_to_evaluate_format_check(points_to_evaluate_dict):
    assert isinstance(points_to_evaluate_dict, dict), (
        "points_to_evaluate must be a dict,"
        "see the example in the documentation of HPOArgs::points_to_evaluate"
    )
    return points_to_evaluate_dict


def custom_search_space_format_check(custom_search_space_dict):
    assert isinstance(custom_search_space_dict, dict), (
        "custom_search_space must be a dict,"
        "see the example in the documentation of HPOArgs::custom_search_space"
    )
    return custom_search_space_dict


def dft_arg_for_dataset():
    return {
        "path": "csv",
        "datafiles": [
            "data/input/train.csv",
            "data/input/validation.csv",
            "data/input/test.csv",
        ],
    }


def dft_arg_for_fold_names():
    return ["train", "validation", "test"]


def dft_arg_for_resources_per_trial():
    return {"cpu": 1, "gpu": 1}


@dataclass
class HPOArgs:
    """The HPO setting

    Args:
        output_dir (:obj:`str`):
            data root directory for outputing the log, etc.
        sample_num (:obj:`int`, `optional`, defaults to :obj:`-1`):
            An integer, the upper bound of trial number for HPO.
        time_budget (:obj:`int`, `optional`, defaults to :obj:`-1`):
            An integer, the upper bound of time budget for HPO
        points_to_evaluate (:obj:`dict`, `optional`, defaults to :obj:`{}`):
            A dict, the first HPO configuration to evaluate for the HPO algorithm. If not set,
            depending on the HPO algorithm, HPO will search for the default initial config (bs
            , CFO), or use a random confi as the first config (Optuna)
        custom_search_space (:obj:`dict`, `optional`, defaults to :obj:`{}`):
            A dict, the custom search space the HPO algorithm. If not set, HPO will use
            the default search space, i.e., :
                output_config = {
                    "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
                    "num_train_epochs": {"l": 1.0, "u": 10.0, "space": "log"},
                    "per_device_train_batch_size": [4, 8, 16, 32],
                    "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
                    "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
                    "adam_epsilon": {"l": 1e-8, "u": 1e-6, "space": "linear"},
                    "seed": list(range(40, 45)),
                }
        dataset_config (:obj:`list`, defaults to :obj:`[]`):
            A dict, the input dataset configuration. This configuration follows the same
            format as HuggingFace's datasets.load_dataset, e.g.:
                | dataset_config = ["glue", "mrpc"]
                | dataset_config = {"path": "glue", "name": "mrpc"}
                | dataset_config = {"csv", "data_files":
                                    ["data/output/train.csv",
                                    "data/output/validation.csv",
                                    "data/output/test.csv"]
        model_path (:obj:`str`, `optional`, defaults to :obj:`facebook/muppet-roberta-base`):
            A string, the path of the language model file, either a path from huggingface
            model card huggingface.co/models, or a local path for the model
        resources_per_trial (:obj:`dict`, `optional`, defaults to :obj:`{"cpu": 1, "gpu": 1}`):
            A dict for specifying the resource used by each trial
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            A bool, whether to use FP16
        ray_verbose (:obj:`1`, `optional`, defaults to :obj:`1`):
            An int, the verbose level of Ray tune
        transformers_verbose (:obj:`1`, `optional`,  defaults to :obj:`1`):
            An int, the verbose level of Transformers
        custom_metric_name (:obj:`str`, `optional`, defaults to :obj:`accuracy`):
            A string, the custom metric name
        custom_metric_mode_name (:obj:`str`, `optional`, defaults to :obj:`max`):
            A string, the custom metric mode name
        task (:obj:`str`, `optional`, defaults to :obj:`seq-classification`):
            A string, the task in for HPO fine-tuning, e.g., seq-classification

    The following arguments are for developer mode only, user can ignore these arguments:

    Args:

        space_mode (:obj:`str`, `optional`, defaults to :obj:`gnr`):
            A string, the search space mode for HPO, e.g.:

                | grid: using the recommended space by one pre-trained language model,
                e.g., the grid space recommended by bert. If search_alg_args_mode is set to grid:
                and grid_space_model_type is specified in custom_hpo_args (can be different from
                the model specified in the HPO setting), grid search will
                search within the space of the model of grid_space_model_type. Otherwise, grid search
                will search within the space of the model in the HPO setting.
                | gnr: the generic continuous space, i.e., {
                        "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
                        "num_train_epochs": {"l": 0.001, "u": 0.1, "space": "log"},
                        "per_device_train_batch_size": [1],
                        "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
                        "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
                        "adam_epsilon": {"l": 1e-8, "u": 1e-6, "space": "linear"},
                        "seed": list(range(40, 45)),
                    }
                | uni: grid union space
                | uni_test: test space for grid union
                | cus: customized search space
                | gnr_test: test space for generic space

    """

    # show this in doc str
    output_dir: str = field(
        default="data/output/", metadata={"help": "data dir", "required": True}
    )

    # show this in docs str
    sample_num: Optional[int] = field(
        default=-1, metadata={"help": "sample number for HPO"}
    )

    # show this in doc str
    time_budget: Optional[int] = field(
        default=None, metadata={"help": "time budget for HPO"}
    )

    # show this in doc str, check format
    points_to_evaluate: Optional[points_to_evaluate_format_check] = field(
        default=None, metadata={"help": "init config for HPO to evaluate"}
    )

    # show this in doc str, check format
    custom_search_space: Optional[custom_search_space_format_check] = field(
        default=None, metadata={"help": "user customized search space"}
    )

    # user cannot only be able to use hf dataset
    dataset_config: Optional[dataset_config_format_check] = field(
        default_factory=dft_arg_for_dataset,
        metadata={
            "help": "dataset config, which is either a dict or a list, the dict or list must"
            "be consistent with the argument in datasets.load_dataset, "
            "See the documentation for HPOArgs::dataset_config"
        },
    )

    # show this in doc str
    model_path: str = field(
        default="facebook/muppet-roberta-base",
        metadata={"help": "model path model for HPO"},
    )

    # show this in doc str
    resources_per_trial: dict = field(
        default_factory=dft_arg_for_resources_per_trial,
        metadata={"help": "resources per trial"},
    )

    # show this in doc str
    fp16: str = field(default=True, metadata={"help": "whether to use the FP16 mode"})

    # show this in doc str
    ray_verbose: int = field(default=1, metadata={"help": "ray verbose level"})

    # show this in doc str
    transformers_verbose: int = field(
        default=10, metadata={"help": "transformers verbose level"}
    )

    # show this in doc str
    custom_metric_name: str = field(
        default=None, metadata={"help": "custom metric name"}
    )

    # show this in doc str
    custom_metric_mode_name: str = field(
        default=None, metadata={"help": "custom metric mode name"}
    )

    # # show this in doc str
    task: str = field(
        default="seq-classification",
        metadata={"help": "NLP task specified by user (user mode)"},
    )

    # the arguments below are used for developer mode only, do not include them in doc str
    grid_space_model_type: str = field(
        default=None,
        metadata={
            "help": "which model's grid configuration to use"
            "for grid search. Only set this argument when "
            "algo_mode=grid "
        },
    )
    max_seq_length: int = field(default=128, metadata={"help": "max seq length"})

    key_path: str = field(
        default=None,
        metadata={
            "help": "path for storing key.json which contains"
            "the container key for Azure"
        },
    )
    root_log_path: str = field(
        default=None, metadata={"help": "root log path for logs on Azure"}
    )
    source_fold: List[str] = field(
        default=None,
        metadata={"help": "the source folds for the data when resplit_mode='rspt'"},
    )
    fold_names: List[str] = field(
        default_factory=dft_arg_for_fold_names,
        metadata={"help": "the fold names for the data when resplit_mode='rspt'"},
    )

    split_portion: List[str] = field(
        default=None, metadata={"help": "the resplit portion when resplit_mode='rspt'"}
    )

    algo_mode: str = field(
        default="hpo",
        metadata={
            "help": "hpo or grid search",
            "choices": ["grid", "hpo", "hfhpo", "gridcv", "hpocv", "eval"],
        },
    )

    space_mode: str = field(
        default="gnr",
        metadata={
            "help": "space mode",
            "choices": ["grid", "gnr", "uni", "uni_test", "cus", "gnr_test"],
        },
    )
    search_alg_args_mode: str = field(
        default="dft",
        metadata={
            "help": "search algorithm args mode",
            "choices": ["dft", "exp", "cus"],
        },
    )
    algo_name: str = field(
        default="bs",
        metadata={
            "help": "algorithm",
            "choices": ["bs", "optuna", "cfo", "rs", "grid"],
        },
    )
    pruner: str = field(
        default="asha", metadata={"help": "pruner for HPO", "choices": ["asha", "None"]}
    )

    rep_id: int = field(default=0, metadata={"help": "rep id in HPO experiment"})

    resplit_mode: str = field(
        default="ori",
        metadata={
            "help": "mode for splitting the data",
            "choices": ["rspt", "ori", "cv", "cvrspt"],
        },
    )
    seed_data: int = field(
        default=101,
        metadata={"help": "seed for data shuffling when resplit_mode='rspt'"},
    )

    seed_bs: int = field(default=20, metadata={"help": "the seed for blend search"})

    seed_transformers: int = field(
        default=42, metadata={"help": "seed for HuggingFace transformers class"}
    )

    ckpt_per_epoch: int = field(default=1, metadata={"help": "checkpoint per epoch"})

    keep_checkpoints_num: int = field(
        default=1, metadata={"help": "number of checkpoints to keep"}
    )

    cv_k: int = field(default=-1, metadata={"help": "cv fold number"})

    is_wandb_on: bool = field(
        default=False, metadata={"help": "whether to use the wandb mode for logging"}
    )

    model_size: str = field(default="base", metadata={"help": "size of the model path"})

    def load_args(self, mode="args", **custom_hpo_args):
        from dataclasses import fields

        if mode == "args":
            return self._load_custom_hpo_args(custom_hpo_args)

        arg_parser = argparse.ArgumentParser()
        for each_field in fields(HPOArgs):
            print(each_field)
            arg_parser.add_argument(
                "--" + each_field.name,
                type=each_field.type,
                help=each_field.metadata["help"],
                required=each_field.metadata["required"]
                if "required" in each_field.metadata
                else False,
                choices=each_field.metadata["choices"]
                if "choices" in each_field.metadata
                else None,
                default=each_field.default,
            )
        console_args, unknown = arg_parser.parse_known_args()
        return console_args

    def _load_custom_hpo_args(self, custom_hpo_args):
        dft_args = self
        for key, val in custom_hpo_args.items():
            setattr(dft_args, key, val)

        return dft_args

    @staticmethod
    def _get_default_config():
        return {
            "dataset_config": ["glue", "mrpc"],
            "algo_mode": "hpo",
            "space_mode": "gnr",
            "search_alg_args_mode": "dft",
            "algo_name": "bs",
            "pruner": "None",
            "model_path": "albert-base-v1",
            "model_size": "small",
            "resplit_mode": "rspt",
            "rep_id": 0,
            "seed_data": 101,
            "seed_transformers": 42,
            "seed_bs": 20,
        }

    @staticmethod
    def _get_unittest_config():
        return {
            "dataset_config": ["glue", "mrpc"],
            "algo_mode": "hpo",
            "space_mode": "gnr_test",
            "search_alg_args_mode": "cus",
            "algo_name": "bs",
            "pruner": "None",
            "model_path": "albert-base-v1",
            "model_size": "small",
            "resplit_mode": "rspt",
            "rep_id": 0,
            "seed_data": 101,
            "seed_transformers": 42,
            "seed_bs": 20,
        }


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


def _variable_override_default_alternative(
    obj_ref, var_name, default_value, all_values, overriding_value=None
):
    """
    Setting the value of var. If overriding_value is specified, var is set to overriding_value;
    If overriding_value is not specified, var is set to default_value meanwhile showing all_values
    """
    assert isinstance(all_values, list)
    if overriding_value:
        setattr(obj_ref, var_name, overriding_value)
        print("The value for {} is specified as {}".format(var_name, overriding_value))
    else:
        setattr(obj_ref, var_name, default_value)
        print(
            "The value for {} is not specified, setting it to the default value {}. "
            "Alternatively, you can set it to {}".format(
                var_name, default_value, ",".join(all_values)
            )
        )


@dataclass
class PathUtils:
    hpo_ckpt_path: str = field(metadata={"help": "the directory for hpo output"})
    hpo_result_path: str = field(metadata={"help": "the directory for hpo result"})
    hpo_log_path: str = field(metadata={"help": "the directory for log"})
    hpo_config_path: str = field(metadata={"help": "the directory for log"})

    log_dir_per_run: str = field(metadata={"help": "log directory for each run."})
    result_dir_per_run: str = field(metadata={"help": "result directory for each run."})
    ckpt_dir_per_run: str = field(
        metadata={"help": "checkpoint directory for each run."}
    )
    ckpt_dir_per_trial: str = field(
        metadata={"help": "checkpoint directory for each trial."}
    )

    def __init__(
        self,
        jobid_config,
        hpo_output_dir,
    ):
        self.jobid_config = jobid_config
        self.hpo_output_dir = hpo_output_dir
        self.hpo_ckpt_path = os.path.join(hpo_output_dir, "checkpoint")
        self.hpo_result_path = os.path.join(hpo_output_dir, "result")
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
