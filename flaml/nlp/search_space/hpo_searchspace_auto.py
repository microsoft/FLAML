from collections import OrderedDict
from flaml.nlp.search_space.hpo.electra import electra_glue_hpo

HPO_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        (("electra", "glue"), electra_glue_hpo),
    ]
)


class AutoHPOSearchSpace:
    """
    This is a generic model class that will be instantiated as one of the model classes of the library
    ---with the search space for hpo
    ---when created with the when created with the
    :meth:`~transformers.AutoHPOSearchSpace.from_model_and_dataset_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoGridSearchSpace is designed to be instantiated "
            "using the `AutoGridSearchSpace.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(cls, model_name, dataset_name):
        if (model_name, dataset_name) in HPO_SEARCH_SPACE_MAPPING.keys():
            try:
                return HPO_SEARCH_SPACE_MAPPING[(model_name, dataset_name)]
            except:
                return None
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                model_name, dataset_name, cls.__name__, ", ".join(c.__name__ for c in HPO_SEARCH_SPACE_MAPPING.keys())
            )
        )
