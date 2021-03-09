

class AutoClassificationHead:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with the
    architecture used for pretraining this model---when created with the when created with the
    :meth:`~transformers.AutoModelForPreTraining.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForPreTraining.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoClassificationHead is designed to be instantiated "
            "using the `AutoClassificationHead.from_config(config)` methods."
        )