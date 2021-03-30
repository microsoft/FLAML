from collections import OrderedDict

from .get_grid_search_space import \
    (get_electra_space,
     get_bert_space,
     get_mobilebert_space,
     get_roberta_space,
     )

GRID_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        (("electra"), get_electra_space),
        (("bert"), get_bert_space),
        #(("mobilebert"), get_mobilebert_space),
        (("roberta"), get_roberta_space),
    ]
)

time_budget_grid_electra_base_glue_tmdev = {
    "qnli": 1678.1,
    "mnli": 6630.17,
}

time_budget_grid_electra_base_glue_dgx = {
    "qqp": 5682.09,
}

time_budget_grid_bert_base_glue_tmdev = {
    "qnli": 1777.34,
    "mnli": 5799,
}

time_budget_grid_bert_base_glue_dgx = {
    "qqp": 5625.18,
}

GRID_SEARCH_TIME_BUDGET_LOOKUP_TABLE = OrderedDict(
    [
        (("electra", "base", "glue", "tmdev"), time_budget_grid_electra_base_glue_tmdev),
        (("electra", "base", "glue", "dgx"), time_budget_grid_electra_base_glue_dgx),
        (("bert", "base", "glue", "tmdev"), time_budget_grid_bert_base_glue_tmdev),
        (("bert", "base", "glue", "dgx"), time_budget_grid_bert_base_glue_dgx),
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

    @classmethod
    def get_grid_time_budget(cls, logger, model_type, model_size_type, dataset_name, server_name, subdataset_name = None):
        if (model_type, model_size_type, dataset_name, subdataset_name) in GRID_SEARCH_TIME_BUDGET_LOOKUP_TABLE.keys():
            try:
                return GRID_SEARCH_TIME_BUDGET_LOOKUP_TABLE[(model_type, model_size_type, dataset_name, server_name)][subdataset_name]
            except Exception as err:
                logger.error("the time budget for this setting does not exist in the look up table")
                raise err
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                model_type, dataset_name, cls.__name__, ", ".join(c.__name__ for c in GRID_SEARCH_SPACE_MAPPING.keys())
            )
        )

    @staticmethod
    def get_trial_number_in_space(grid_config):
        trial_num = 1
        for each_hp in grid_config.keys():
            trial_num = trial_num * len(grid_config[each_hp])
        return trial_num
