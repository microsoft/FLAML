from collections import OrderedDict
from ray.tune.suggest.optuna import OptunaSearch

from flaml import CFO
from flaml import BlendSearch
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

HPO_METHOD_MAPPING = OrderedDict(
    [
        ("Optuna", OptunaSearch),
        ("CFO", CFO),
        ("BlendSearch", BlendSearch),
        ("Dragonfly", DragonflySearch),
        ("SkOpt", SkOptSearch),
        ("Nevergrad", NevergradSearch),
        ("HyperOpt", HyperOptSearch),
        ("grid_search", None),
        ("RandomSearch", None)
    ]
)


class AutoSearchAlgorithm:
    """
    This is a generic model class that will be instantiated as one of the model classes of the library
    ---with the search algorithm
    ---when created with the when created with the
    :meth:`~transformers.AutoSearchAlgorithm.from_config_and_method_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoSearchAlgorithm is designed to be instantiated "
            "using the `AutoHPO.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_config_and_method_name(cls, method_name, **kwargs):
        if method_name in HPO_METHOD_MAPPING.keys():
            try:
                return HPO_METHOD_MAPPING[method_name](**kwargs)
            except:
                return None
        raise ValueError(
            "Unrecognized method {} for this kind of AutoSearchAlgorithm: {}.\n"
            "Method name should be one of {}.".format(
                method_name, cls.__name__, ", ".join(c.__name__ for c in HPO_METHOD_MAPPING.keys())
            )
        )
