from collections import OrderedDict

from flaml.nlp.search_space.grid.electra import electra_glue_grid

GRID_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        (("electra", "glue"), electra_glue_grid),
    ]
)

class AutoGridSearchSpace:
    def __init__(self):
        raise EnvironmentError(
            "AutoGridSearchSpace is designed to be instantiated "
            "using the `AutoGridSearchSpace.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(cls, model_name, dataset_name):
        if (model_name, dataset_name) in GRID_SEARCH_SPACE_MAPPING.keys():
            try:
                return GRID_SEARCH_SPACE_MAPPING[(model_name, dataset_name)]
            except:
                return None
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                model_name, dataset_name, cls.__name__, ", ".join(c.__name__ for c in GRID_SEARCH_SPACE_MAPPING.keys())
            )
        )