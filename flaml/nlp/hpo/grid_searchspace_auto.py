from collections import OrderedDict

from flaml.nlp.hpo.get_grid_search_space import \
    (get_electra_space,
     get_bert_space,
     get_mobilebert_space,
     get_roberta_space,
     )

GRID_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        (("electra"), get_electra_space),
        (("bert"), get_bert_space),
        (("mobilebert"), get_mobilebert_space),
        (("roberta"), get_roberta_space),
    ]
)


class AutoGridSearchSpace:
    """
    This is a generic huggingface class that will be instantiated as one of the huggingface classes of the library
    ---with the search space for grid search
    ---when created with the when created with the
    :meth:`~transformers.AutoGridSearchSpace.from_model_and_dataset_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoGridSearchSpace is designed to be instantiated "
            "using the `AutoGridSearchSpace.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(cls, model_type, model_size_type, dataset_name, subdataset_name = None):
        if model_type in GRID_SEARCH_SPACE_MAPPING.keys():
            try:
                return GRID_SEARCH_SPACE_MAPPING[model_type](model_size_type, dataset_name, subdataset_name)
            except:
                return None
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                model_type, dataset_name, cls.__name__, ", ".join(c.__name__ for c in GRID_SEARCH_SPACE_MAPPING.keys())
            )
        )
