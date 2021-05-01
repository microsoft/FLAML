from collections import OrderedDict

from ..huggingface.trainer import TrainerForAutoTransformers
from ray import tune
from transformers import TrainingArguments

from .grid_searchspace_auto import GRID_SEARCH_SPACE_MAPPING, AutoGridSearchSpace

hp_type_mapping = {"learning_rate": [tune.sample.Float, "log"],
                   "num_train_epochs": [tune.sample.Float, "linear"],
                   "per_device_train_batch_size": tune.sample.Categorical,
                   "weight_decay": [tune.sample.Float, "linear"],
                   "warmup_ratio": [tune.sample.Float, "linear"],
                   }

def hpo_space_gridunion_continuous(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    gridunion_space = hpo_space_gridunion(logger, model_type, model_size_type, dataset_name, subdataset_name)
    gridunion_space_continuous = {}
    for each_hp in hp_type_mapping.keys():
        if hp_type_mapping[each_hp] == tune.sample.Categorical:
            gridunion_space_continuous[each_hp] = gridunion_space[each_hp]
        else:
            assert type(hp_type_mapping[each_hp]) == list
            gridunion_space_continuous[each_hp] = {"l": min(gridunion_space[each_hp]), "u": max(gridunion_space[each_hp]), "space": hp_type_mapping[each_hp][1]}
    return gridunion_space_continuous

def hpo_space_gridunion_other_large(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    output_config = {}
    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        if each_model_type == model_type: continue
        each_grid_search_config, _ = AutoGridSearchSpace.from_model_and_dataset_name(each_model_type, model_size_type, dataset_name, subdataset_name)
        from ..utils import merge_dicts
        output_config = merge_dicts(output_config, each_grid_search_config)
        default_values = {}
        training_args = TrainingArguments(output_dir=".")
        for each_hp in output_config.keys():
            try:
                default_values[each_hp] = [getattr(training_args, each_hp)]
            except AttributeError:
                pass
        output_config = merge_dicts(output_config, default_values)

    # for each_hp in hp_type_mapping.keys():
    #     if each_hp == "warmup_ratio":
    #         output_config[each_hp] = [x for x in output_config[each_hp] if x != 0]

    return output_config

def hpo_space_gridunion_other(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    output_config = {}
    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        #if each_model_type == model_type: continue
        this_config = AutoGridSearchSpace.from_model_and_dataset_name(each_model_type, model_size_type, dataset_name, subdataset_name)
        import pdb; pdb.set_trace()
        from ..utils import merge_dicts
        output_config = merge_dicts(output_config, this_config)
        default_values = {}
        training_args = TrainingArguments(output_dir=".")
        for each_hp in output_config.keys():
            try:
                default_values[each_hp] = [getattr(training_args, each_hp)]
            except AttributeError:
                pass
        output_config = merge_dicts(output_config, default_values)

    # for each_hp in output_config.keys():
    #     if each_hp == "warmup_ratio":
    #         output_config[each_hp] = [x for x in output_config[each_hp] if x != 0]
    #     if each_hp == "max_steps":
    #         output_config[each_hp] = [x for x in output_config[each_hp] if x != -1]

    return output_config

def hpo_space_gridunion(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    _, output_config = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    for each_hp in hp_type_mapping.keys():
        output_config[each_hp] = []

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        _, each_grid_search_config = AutoGridSearchSpace.from_model_and_dataset_name(each_model_type, model_size_type, dataset_name, subdataset_name)
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
    for each_hp in hp_type_mapping.keys():
        if each_hp == "warmup_ratio":
            output_config[each_hp] = [x for x in output_config[each_hp] if x != 0]

    output_config["learning_rate"] = list(set(output_config["learning_rate"] + [3e-5, 5e-5, 1e-4, 1.5e-4]))
    return output_config

def enumerate_onehp(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    _, electra_config = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    try:
        hp_to_fix, hp_to_fix_value = custom_hpo_args["hp_to_fix"]
        hp_to_tune, hp_to_tune_grids = custom_hpo_args["hp_to_tune"]
        assert type(hp_to_fix_value) in {int, float, bool}
    except Exception as err:
        logger.log("When hpo_searchspace_mode = enumerate_onehp must specify both hp_to_fix and hp_to_tune in custom_hpo_args. "
                   "hp_to_fix must be a tuple containing the hp and a scaler, hp_to_tun must be a tuple containing the hp and "
                   "a list. ")
        raise err
    electra_config[hp_to_fix] = [hp_to_fix_value]
    electra_config[hp_to_tune] = hp_to_tune_grids

    electra_config = TrainerForAutoTransformers.resolve_hp_conflict(electra_config)

    return electra_config

def hpo_space_generic(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    output_config = {
        "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
        "num_train_epochs": {"l": 1.0, "u": 10.0, "space": "log"},
        "per_device_train_batch_size": [4, 8, 16, 32, 48, 64],
        "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
        "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"}
    }
    return output_config

def hpo_space_generic_grid(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    output_config = {
        "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1.5e-4],
        "num_train_epochs": [3, 10],
        "per_device_train_batch_size": [16, 32],
        "warmup_ratio": [0, 0.06, 0.1],
        "weight_decay": [0, 0.1]
    }
    return output_config

def hpo_space_small(logger, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
    _, config_json = AutoGridSearchSpace.from_model_and_dataset_name(model_type, model_size_type, dataset_name, subdataset_name)
    output_config = {}

    for each_hp in config_json.keys():
        if each_hp == "learning_rate":
            if len(config_json[each_hp]) > 1:
                output_config[each_hp] = {"l":3e-5, "u": 1.5e-4, "space": "log"}
            else:
                output_config[each_hp] = config_json[each_hp]
        elif each_hp == "num_train_epochs":
            output_config[each_hp] = {"l": 2.0, "u": 4.0, "space": "linear"}
        elif each_hp == "per_device_train_batch_size":
            output_config[each_hp] = [16, 32, 64]
        elif each_hp == "warmup_ratio":
            output_config[each_hp] = {"l": 0.0, "u": 0.2, "space": "linear"}
        elif each_hp == "weight_decay":
            output_config[each_hp] = {"l": 0.0, "u": 0.3, "space": "linear"}
        else:
            output_config[each_hp] = config_json[each_hp]

    return output_config

HPO_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        ("hpo_space_gridunion_other_large", hpo_space_gridunion_other_large),
        ("hpo_space_gridunion_other", hpo_space_gridunion_other),
        ("hpo_space_gridunion", hpo_space_gridunion),
        ("hpo_space_generic", hpo_space_generic),
        ("hpo_space_small", hpo_space_small),
        ("hpo_space_gridunion_continuous", hpo_space_gridunion_continuous),
        ("enumerate_onehp", enumerate_onehp)
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
    def from_model_and_dataset_name(cls, logger, hpo_searchspace_name, model_type, model_size_type, dataset_name, subdataset_name = None, **custom_hpo_args):
        if hpo_searchspace_name in HPO_SEARCH_SPACE_MAPPING.keys():
            try:
                hpo_space = HPO_SEARCH_SPACE_MAPPING[hpo_searchspace_name](logger, model_type, model_size_type, dataset_name, subdataset_name, **custom_hpo_args)
                if "warmup_steps" in hpo_space:
                    hpo_space["warmup_ratio"] = hpo_space["warmup_ratio"] + hpo_space["warmup_steps"]
                    del hpo_space["warmup_steps"]
                if "max_steps" in hpo_space:
                    hpo_space["num_train_epochs"] = hpo_space["num_train_epochs"] + hpo_space["max_steps"]
                    del hpo_space["max_steps"]
                return hpo_space
            except:
                return None
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoHPOSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                hpo_searchspace_name, dataset_name, cls.__name__, ", ".join(c.__name__ for c in HPO_SEARCH_SPACE_MAPPING.keys())
            )
        )
