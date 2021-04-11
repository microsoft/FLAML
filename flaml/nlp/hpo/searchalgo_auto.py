import itertools
from collections import OrderedDict

import ray
from ray.tune.suggest.optuna import OptunaSearch

from flaml import CFO
from flaml import BlendSearch
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

SEARCH_ALGO_MAPPING = OrderedDict(
    [
        ("Optuna", OptunaSearch),
        ("CFO", CFO),
        ("BlendSearch", BlendSearch),
        ("Dragonfly", DragonflySearch),
        ("SkOpt", SkOptSearch),
        ("Nevergrad", NevergradSearch),
        ("HyperOpt", HyperOptSearch),
        ("grid_search", None),
        ("grid_search_enumerate", None),
        ("grid_search_bert", None),
        ("RandomSearch", None)
    ]
)


class AutoSearchAlgorithm:
    """
    This is a generic huggingface class that will be instantiated as one of the huggingface classes of the library
    ---with the search algorithm
    ---when created with the when created with the
    :meth:`~transformers.AutoSearchAlgorithm.from_method_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoSearchAlgorithm is designed to be instantiated "
            "using the `AutoSearchAlgorithm.from_method_name(method_name)` methods."
        )

    @classmethod
    def from_method_name(cls, search_algo_name, search_algo_args_mode, grid_search_space, hpo_search_space, **custom_hpo_args):
        assert search_algo_args_mode in {"default", "grid", "custom"}
        if search_algo_name in SEARCH_ALGO_MAPPING.keys():
            try:
                default_search_algo_kwargs = DEFAULT_SEARCH_ALGO_ARGS_MAPPING[search_algo_name](hpo_search_space = hpo_search_space)
                if search_algo_args_mode == "default":
                    search_algo_args = default_search_algo_kwargs
                elif search_algo_args_mode == "grid":
                    search_algo_args = {"points_to_evaluate": AutoSearchAlgorithm.grid2list(grid_search_space)}
                else:
                    search_algo_args = custom_hpo_args

                algo = SEARCH_ALGO_MAPPING[search_algo_name]()
                if algo:
                    allowed_arguments = algo.__init__.__code__.co_varnames
                    search_algo_args = {key: search_algo_args[key] for key in search_algo_args.keys() if key in allowed_arguments}

                return SEARCH_ALGO_MAPPING[search_algo_name](**search_algo_args)
            except:
                return None
        raise ValueError(
            "Unrecognized method {} for this kind of AutoSearchAlgorithm: {}.\n"
            "Method name should be one of {}.".format(
                search_algo_name, cls.__name__, ", ".join(c.__name__ for c in SEARCH_ALGO_MAPPING.keys())
            )
        )

    @staticmethod
    def grid2list(grid_config):
        key_val_list = [[(key, each_val) for each_val in val_list['grid_search']] for (key, val_list) in grid_config.items()]
        config_list = [dict(x) for x in itertools.product(*key_val_list)]
        return config_list

def get_search_algo_args_optuna(hpo_search_space = None):
    return {}

def default_search_algo_args_cfo(hpo_search_space = None):
    return {}

def default_search_algo_args_bs(hpo_search_space = None):
    if isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Categorical):
        min_epoch = min(hpo_search_space["num_train_epochs"].categories)
    else:
        assert isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Float)
        min_epoch = hpo_search_space["num_train_epochs"].lower
    default_search_algo_args = {
        "points_to_evaluate": [{
            "num_train_epochs": max(1, min_epoch),
            "per_device_train_batch_size": max(hpo_search_space["per_device_train_batch_size"].categories),
        }]}
    return default_search_algo_args

def default_search_algo_args_skopt(hpo_search_space = None):
    return {}

def default_search_algo_args_dragonfly(hpo_search_space = None):
    return {}

def default_search_algo_args_nevergrad(hpo_search_space = None):
    return {}

def default_search_algo_args_hyperopt(hpo_search_space = None):
    return {}

def default_search_algo_args_grid_search(hpo_search_space = None):
    return {}

def default_search_algo_args_random_search(hpo_search_space = None):
    return {}

DEFAULT_SEARCH_ALGO_ARGS_MAPPING = OrderedDict(
        [
            ("Optuna", get_search_algo_args_optuna),
            ("CFO", default_search_algo_args_cfo),
            ("BlendSearch", default_search_algo_args_bs),
            ("Dragonfly", default_search_algo_args_dragonfly),
            ("SkOpt", default_search_algo_args_skopt),
            ("Nevergrad", default_search_algo_args_nevergrad),
            ("HyperOpt", default_search_algo_args_hyperopt),
            ("grid_search", default_search_algo_args_grid_search),
            ("RandomSearch", default_search_algo_args_random_search)
        ]
    )
