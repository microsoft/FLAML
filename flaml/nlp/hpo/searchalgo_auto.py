import itertools
from collections import OrderedDict

import ray
from ray.tune.suggest.optuna import OptunaSearch
from flaml import CFO, BlendSearch

SEARCH_ALGO_MAPPING = OrderedDict(
    [
        ("optuna", OptunaSearch),
        ("cfo", CFO),
        ("bs", BlendSearch),
        ("grid", None),
        ("gridbert", None),
        ("rs", None)
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
    def from_method_name(cls, search_algo_name, search_algo_args_mode, hpo_search_space, **custom_hpo_args):
        if not search_algo_name:
            search_algo_name = "grid"
        if search_algo_name in SEARCH_ALGO_MAPPING.keys():
            try:
                algo = SEARCH_ALGO_MAPPING[search_algo_name]()
                this_search_algo_kwargs = allowed_custom_args = None
                if algo:
                    allowed_arguments = algo.__init__.__code__.co_varnames
                    allowed_custom_args = {key: custom_hpo_args[key] for key in custom_hpo_args.keys() if
                                        key in allowed_arguments}

                if search_algo_args_mode == "dft":
                    this_search_algo_kwargs = DEFAULT_SEARCH_ALGO_ARGS_MAPPING[search_algo_name](
                        "dft", hpo_search_space = hpo_search_space, **allowed_custom_args)
                elif search_algo_args_mode == "cus":
                    this_search_algo_kwargs = DEFAULT_SEARCH_ALGO_ARGS_MAPPING[search_algo_name](
                        "cus", hpo_search_space=hpo_search_space, **allowed_custom_args)

                return SEARCH_ALGO_MAPPING[search_algo_name](**this_search_algo_kwargs)
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
        key_val_list = [[(key, each_val) for each_val in val_list['grid_search']]
                        for (key, val_list) in grid_config.items()]
        config_list = [dict(x) for x in itertools.product(*key_val_list)]
        return config_list

def get_search_algo_args_optuna(search_args_mode, hpo_search_space = None, **custom_hpo_args):
    return {}

def default_search_algo_args_bs(search_args_mode, hpo_search_space = None, **custom_hpo_args):
    if isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Categorical):
        min_epoch = min(hpo_search_space["num_train_epochs"].categories)
    else:
        assert isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Float)
        min_epoch = hpo_search_space["num_train_epochs"].lower
    default_search_algo_args = {
        "low_cost_partial_config": {
            "num_train_epochs": min_epoch,
            "per_device_train_batch_size": max(hpo_search_space["per_device_train_batch_size"].categories),
        },
    }
    if search_args_mode == "cus":
        default_search_algo_args.update(custom_hpo_args)
    return default_search_algo_args

def experiment_search_algo_args_bs(hpo_search_space = None):
    if isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Categorical):
        min_epoch = min(hpo_search_space["num_train_epochs"].categories)
    else:
        assert isinstance(hpo_search_space["num_train_epochs"], ray.tune.sample.Float)
        min_epoch = hpo_search_space["num_train_epochs"].lower
    default_search_algo_args = {
        "low_cost_partial_config": {
            "num_train_epochs": max(1, min_epoch),
        },
    }
    return default_search_algo_args

def default_search_algo_args_skopt(hpo_search_space = None):
    return {}

def default_search_algo_args_dragonfly(hpo_search_space = None):
    return {}

def default_search_algo_args_nevergrad(hpo_search_space = None):
    return {}

def default_search_algo_args_hyperopt(hpo_search_space = None):
    return {}

def default_search_algo_args_grid_search(search_args_mode, hpo_search_space = None, **custom_hpo_args):
    return {}

def default_search_algo_args_random_search(search_args_mode, hpo_search_space = None, **custom_hpo_args):
    return {}

DEFAULT_SEARCH_ALGO_ARGS_MAPPING = OrderedDict(
        [
            ("optuna", get_search_algo_args_optuna),
            ("cfo", default_search_algo_args_bs),
            ("bs", default_search_algo_args_bs),
            ("grid", default_search_algo_args_grid_search),
            ("gridbert", default_search_algo_args_random_search)
        ]
    )
