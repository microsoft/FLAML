from collections import OrderedDict

from ray.tune.suggest.optuna import OptunaSearch
from transformers.configuration_auto import replace_list_option_in_docstrings

from flaml import CFO
from flaml import BlendSearch
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.zoopt import ZOOptSearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

HPO_METHOD_MAPPING = OrderedDict(
    [
        ("Optuna", OptunaSearch),
        ("CFO", CFO),
        ("BlendSearch", BlendSearch),
        ("Dragonfly", DragonflySearch),
        ("SkOpt", SkOptSearch),
        ("Nevergrad", NevergradSearch),
        ("ZOOpt", ZOOptSearch),
        ("Ax", AxSearch),
        ("HyperOpt", HyperOptSearch),
    ]
)

class AutoSearchAlgorithm:
    def __init__(self):
        raise EnvironmentError(
            "AutoClassificationHead is designed to be instantiated "
            "using the `AutoHPO.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_config_and_method_name(cls, method_name, **kwargs):
        if method_name in HPO_METHOD_MAPPING.keys():
            return HPO_METHOD_MAPPING[method_name](**kwargs)
        raise ValueError(
            "Unrecognized method {} for this kind of AutoHPO: {}.\n"
            "Method name should be one of {}.".format(
                method_name, cls.__name__, ", ".join(c.__name__ for c in HPO_METHOD_MAPPING.keys())
            )
        )