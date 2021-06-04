from collections import OrderedDict
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, MedianStoppingRule

SCHEDULER_MAPPING = OrderedDict(
    [
        ("None", None),
        ("asha", ASHAScheduler),
        ("hb", HyperBandScheduler),
    ]
)


class AutoScheduler:
    """
    This is a generic huggingface class that will be instantiated as one of the huggingface classes of the library
    ---with the tune scheduler
    ---when created with the when created with the
    :meth:`~transformers.AutoScheduler.from_scheduler_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoScheduler is designed to be instantiated "
            "using the `AutoScheduler.from_scheduler_name(method_name)` methods."
        )

    @classmethod
    def from_scheduler_name(cls, scheduler_name, **kwargs):
        if scheduler_name in SCHEDULER_MAPPING.keys():
            try:
                return SCHEDULER_MAPPING[scheduler_name](**kwargs)
            except:
                return None
        raise ValueError(
            "Unrecognized scheduler {} for this kind of AutoScheduler: {}.\n"
            "Scheduler name should be one of {}.".format(
                scheduler_name, cls.__name__, ", ".join(c.__name__ for c in SCHEDULER_MAPPING.keys())
            )
        )
