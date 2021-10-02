from collections import OrderedDict
from .grid_searchspace_auto import AutoGridSearchSpace


def hpo_space_custom(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    """
    The 5 arguments here cannot be deleted, they need to be kept consistent with
    other functions in HPO_SEARCH_SPACE_MAPPING
    """
    assert "hpo_space" in custom_hpo_args
    custom_search_space = custom_hpo_args["hpo_space"]
    return custom_search_space


def hpo_space_gridunion(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    output_config = {}
    for each_model_type in ["bert", "roberta", "electra"]:
        # if each_model_type == model_type: continue
        this_config = AutoGridSearchSpace.from_model_and_dataset_name(
            each_model_type, model_size_type, dataset_name_list, subdataset_name, "hpo"
        )
        from ..utils import merge_dicts

        output_config = merge_dicts(output_config, this_config)
        default_values = {}
        """
        adding the default configuration from transformers/training_args.py into hpo space
        """
        from transformers import TrainingArguments

        training_args = TrainingArguments(output_dir=".")
        for each_hp in output_config.keys():
            try:
                default_values[each_hp] = [getattr(training_args, each_hp)]
            except AttributeError:
                print("training args do not contain {}, passed".format(each_hp))
                pass

        output_config = merge_dicts(output_config, default_values)

    return output_config


def hpo_space_gridunion_smoke_test(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    return {
        "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
        "num_train_epochs": [0.01],
        "per_device_train_batch_size": [2],
        "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
        "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
    }


def hpo_space_generic(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    output_config = {
        "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
        "num_train_epochs": {"l": 1.0, "u": 10.0, "space": "log"},
        "per_device_train_batch_size": [4, 8, 16, 32],
        "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
        "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
        "adam_epsilon": {"l": 1e-8, "u": 1e-6, "space": "linear"},
        "seed": [x for x in range(40, 45)],
    }
    return output_config


def hpo_space_generic_test(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    output_config = {
        "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
        "num_train_epochs": {"l": 0.001, "u": 10.0, "space": "log"},
        "per_device_train_batch_size": [1, 2, 4, 8, 16, 32, 64],
        "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
        "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
        "adam_epsilon": {"l": 1e-8, "u": 1e-6, "space": "linear"},
        "seed": [x for x in range(40, 45)],
    }
    return output_config


def hpo_space_grid(
    model_type=None,
    model_size_type=None,
    dataset_name_list: list = None,
    subdataset_name=None,
    algo_mode=None,
    **custom_hpo_args
):
    return AutoGridSearchSpace.from_model_and_dataset_name(
        model_type, model_size_type, dataset_name_list, subdataset_name, algo_mode
    )


HPO_SEARCH_SPACE_MAPPING = OrderedDict(
    [
        ("grid", hpo_space_grid),
        ("uni", hpo_space_gridunion),
        ("gnr", hpo_space_generic),
        ("gnr_test", hpo_space_generic_test),
        ("uni_test", hpo_space_gridunion_smoke_test),
        ("cus", hpo_space_custom),
    ]
)


class AutoHPOSearchSpace:
    """
    This is a class for getting the hpo search space based on the search space mode
    (a string variable) instantiated as one of the HPO search spaces of the library when
    created with the `~flaml.nlp.hpo.AutoHPOSearchSpace.from_model_and_dataset_name` method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoHPOSearchSpace is designed to be instantiated "
            "using the `AutoHPOSearchSpace.from_config_and_method_name(cls, hpo_searchspace_name,"
            "model_type,model_size_type,dataset_name,subdataset_name=None,**custom_hpo_args)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(
        cls,
        hpo_searchspace_mode,
        model_type,
        model_size_type,
        dataset_name_list: list = None,
        subdataset_name=None,
        algo_mode=None,
        **custom_hpo_args
    ):
        """
        Instantiate one of the classes for getting the hpo search space from the search space name, model type,
        model size type, dataset name and sub dataset name

        Args:

            hpo_searchspace_mode:
                A string variable which is the mode of the hpo search space, it must be chosen from the following options:
                    - uni: the union of BERT, RoBERTa and Electra's grid configs
                    - grid: the recommended grid config of the LM specified in jobconfig.pre
                    - gnr: the generic continuous search space
                    - uni_test: the search space for smoke test
                    - cus: user customized search space, specified in the "hpo_space" argument in AutoTransformers.fit

            model_type:
                A string variable which is the type of the model, e.g., "electra"

            model_size_type:
                A string variable which is the type of the model size, e.g., "small"

            dataset_name:
                A string variable which is the dataset name, e.g., "glue"

            subdataset_name:
                A string variable which is the sub dataset name,e.g., "rte"

            custom_hpo_args:
                Any additional keyword argument to be used for the function for the HPO search space

        Example:
            >>> AutoHPOSearchSpace.from_model_and_dataset_name("uni", "electra", "small", ["glue"], "rte", "hpo")
        """

        if hpo_searchspace_mode in HPO_SEARCH_SPACE_MAPPING.keys():
            hpo_space = HPO_SEARCH_SPACE_MAPPING[hpo_searchspace_mode](
                model_type,
                model_size_type,
                dataset_name_list,
                subdataset_name,
                algo_mode,
                **custom_hpo_args
            )
            return hpo_space
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoHPOSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                hpo_searchspace_mode,
                dataset_name_list,
                cls.__name__,
                ", ".join(HPO_SEARCH_SPACE_MAPPING.keys()),
            )
        )
