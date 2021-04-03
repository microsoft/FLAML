from collections import OrderedDict

from ray import tune
from transformers import TrainingArguments

from flaml.nlp.hpo.grid_searchspace_auto import GRID_SEARCH_SPACE_MAPPING, AutoGridSearchSpace

hp_type_mapping = {"learning_rate": [tune.sample.Float, "log"],
                   "num_train_epochs": [tune.sample.Float, "linear"],
                   "per_device_train_batch_size": tune.sample.Categorical,
                   "weight_decay": [tune.sample.Float, "linear"],
                   "warmup_ratio": [tune.sample.Float, "linear"]}

def hpo_space_gridunion_continuous(logger, model_type, model_size_type, dataset_name, subdataset_name = None):
    gridunion_space = hpo_space_gridunion(logger, model_type, model_size_type, dataset_name, subdataset_name)
    gridunion_space_continuous = {}
    for each_hp in hp_type_mapping.keys():
        if hp_type_mapping[each_hp] == tune.sample.Categorical:
            gridunion_space_continuous[each_hp] = gridunion_space[each_hp]
        else:
            assert type(hp_type_mapping[each_hp]) == list
            gridunion_space_continuous[each_hp] = {"l": min(gridunion_space[each_hp]), "u": max(gridunion_space[each_hp]), "space": hp_type_mapping[each_hp][1]}
    return gridunion_space_continuous

def hpo_space_gridunion(logger, model_type, model_size_type, dataset_name, subdataset_name = None):
    output_config = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    for each_hp in hp_type_mapping.keys():
        output_config[each_hp] = []

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        each_grid_search_config = AutoGridSearchSpace.from_model_and_dataset_name(each_model_type, model_size_type, dataset_name, subdataset_name)
        for each_hp in hp_type_mapping.keys():
            try:
                output_config[each_hp] = list(set(output_config[each_hp] + each_grid_search_config[each_hp]))
            except TypeError:
                logger.warning("Grid config in {} not specified, skipping".format(each_model_type))
                pass
            except KeyError:
                training_args = TrainingArguments(output_dir=".")
                try:
                    default_hp_value = getattr(training_args, each_hp)
                    output_config[each_hp] = list(set(output_config[each_hp] + [default_hp_value]))
                except AttributeError as err:
                    logger.error("Wrong hyperparameter {}, not specified in transformers.TrainingArguments".format(each_hp))
                    raise err
                pass

    return output_config

def hpo_space_generic(logger, model_type, model_size_type, dataset_name, subdataset_name = None):
    config_json = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    output_config = {}

    for each_hp in config_json.keys():
        if each_hp == "learning_rate":
            if len(config_json[each_hp]) > 1:
                output_config[each_hp] = {"l": 1e-6, "u": 1e-3, "space": "log"}
            else:
                output_config[each_hp] = config_json[each_hp]
        elif each_hp == "num_train_epochs":
            output_config[each_hp] = {"l": 1.0, "u": 10.0, "space": "log"}
        elif each_hp == "per_device_train_batch_size":
            output_config[each_hp] = [4, 8, 16, 32, 48, 64]
        elif each_hp == "warmup_ratio":
            output_config[each_hp] = {"l": 0.0, "u": 0.3, "space": "linear"}
        elif each_hp == "weight_decay":
            output_config[each_hp] = {"l": 0.0, "u": 0.3, "space": "linear"}
        else:
            output_config[each_hp] = config_json[each_hp]

    return output_config

HPO_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        ("hpo_space_gridunion", hpo_space_gridunion),
        ("hpo_space_generic", hpo_space_generic),
        ("hpo_space_gridunion_continuous", hpo_space_gridunion_continuous)
    ]
)

class AutoHPOSearchSpace:
    """
    This is a generic huggingface class that will be instantiated as one of the huggingface classes of the library
    ---with the search space for grid search
    ---when created with the when created with the
    :meth:`~transformers.AutoHPOSearchSpace.from_model_and_dataset_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoHPOSearchSpace is designed to be instantiated "
            "using the `AutoHPOSearchSpace.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(cls, logger, hpo_searchspace_name, model_type, model_size_type, dataset_name, subdataset_name = None):
        if hpo_searchspace_name in HPO_SEARCH_SPACE_MAPPING.keys():
            try:
                return HPO_SEARCH_SPACE_MAPPING[hpo_searchspace_name](logger, model_type, model_size_type, dataset_name, subdataset_name)
            except:
                return None
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoHPOSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                hpo_searchspace_name, dataset_name, cls.__name__, ", ".join(c.__name__ for c in HPO_SEARCH_SPACE_MAPPING.keys())
            )
        )
