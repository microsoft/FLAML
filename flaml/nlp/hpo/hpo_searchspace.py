from collections import OrderedDict
from transformers import TrainingArguments

from flaml.nlp.hpo.grid_searchspace_auto import GRID_SEARCH_SPACE_MAPPING, AutoGridSearchSpace

def lr_epoch_bs_gridunion(logger, model_type, model_size_type, dataset_name, subdataset_name = None):
    output_config = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)

    hps_to_union = {"learning_rate", "num_train_epochs", "per_device_train_batch_size", "weight_decay", "warmup_ratio"}

    for each_hp in hps_to_union:
        output_config[each_hp] = []

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        each_grid_search_config = AutoGridSearchSpace.from_model_and_dataset_name(each_model_type, model_size_type, dataset_name, subdataset_name)
        for each_hp in hps_to_union:
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

def lr_epoch_bs_generic(logger, model_type, model_size_type, dataset_name, subdataset_name = None):
    config_json = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    output_config = {}

    for each_hp in config_json.keys():
        if each_hp == "learning_rate":
            if len(config_json[each_hp]) > 1:
                output_config[each_hp] = {"l": 1e-6, "u": 1e-3, "space": "log"}
            else:
                output_config[each_hp] = config_json[each_hp]
        elif each_hp == "num_train_epochs":
            output_config[each_hp] = {"l": 0.01, "u": 10.0, "space": "linear"}
        elif each_hp == "per_device_train_batch_size":
            output_config[each_hp] = [4, 8, 16, 32, 48, 64]
        else:
            output_config[each_hp] = config_json[each_hp]

    return output_config

HPO_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        (("lr_epoch_bs_gridunion"), lr_epoch_bs_gridunion),
        (("lr_epoch_bs_generic"), lr_epoch_bs_generic),
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

