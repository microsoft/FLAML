from collections import OrderedDict

from .get_grid_search_space import \
    (get_electra_space,
     get_bert_space,
     get_roberta_space,
     get_funnel_space,
     get_deberta_space,
     get_albert_space
     )

GRID_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        ("electra", get_electra_space),
        ("bert", get_bert_space),
        ("roberta", get_roberta_space),
        # ("funnel", get_funnel_space),
        # ("deberta", get_deberta_space),
        # ("albert", get_albert_space),
    ]
)

time_budget_grid_electra_base_glue = {
    "rte": 100,
    "cola": 300,
    "sst2": 900,
    "mrpc": 40,
    "qnli": 1456.84,
    "mnli": 1500,
}

time_budget_grid_electra_small_glue = {
    "rte": 60,
    "cola": 100,
    "sst2": 600,
    "mrpc": 30,
    "qnli": 885.58,
    "mnli": 1030.21,
}

time_budget_grid_bert_base_glue = {
    "rte": 30,
    "cola": 300,
    "sst2": 900,
    "mrpc": 40,
    "qnli": 1381.97,
    "mnli": 1512.72,
}

time_budget_grid_deberta_base_glue = {
    "rte": 120,
    "mrpc": 140,
    "cola": 339.85,
    "sst2": 1214.32,
}

time_budget_grid_roberta_base_glue = {
    "rte": 100,
    "mrpc": 120,
    "cola": 278.58,
    "sst2": 909.17,
}

time_budget_grid_funnel_small_yelp_review_full = 1864
time_budget_grid_electra_small_yelp_review_full = 733.82
time_budget_grid_electra_base_yelp_review_full = 1921.91


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
                search_space_union, search_space_unique = GRID_SEARCH_SPACE_MAPPING[model_type](model_size_type, dataset_name, subdataset_name)
                return search_space_union, search_space_unique
            except:
                raise ValueError("{}, {}, {}, {} Return empty".format(model_type, model_size_type, dataset_name, str(subdataset_name)))
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                model_type, dataset_name, cls.__name__, ", ".join(c.__name__ for c in GRID_SEARCH_SPACE_MAPPING.keys())
            )
        )

    @classmethod
    def get_grid_time_budget(cls, logger, model_type, model_size_type, dataset_name, server_name, subdataset_name = None):
        try:
            grid_lookup_table_name = "time_budget_grid_" + model_type + "_" + model_size_type + "_" + dataset_name
            grid_lookup_table_vale = globals()[grid_lookup_table_name]
            if subdataset_name:
                return grid_lookup_table_vale[subdataset_name]
            else:
                return grid_lookup_table_vale
        except KeyError:
            raise ValueError(
                "Unrecognized method {},{},{},{} for this kind of AutoGridSearchSpace: {}".format(
                    model_type, model_size_type, dataset_name,server_name, cls.__name__
                )
            )

    @staticmethod
    def get_trial_number_in_space(grid_config):
        trial_num = 1
        for each_hp in grid_config.keys():
            trial_num = trial_num * len(grid_config[each_hp])
        return trial_num
