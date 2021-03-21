from collections import OrderedDict

from transformers import ElectraConfig
from transformers import RobertaConfig

from transformers.models.auto.configuration_auto import replace_list_option_in_docstrings
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

MODEL_CLASSIFICATION_HEAD_MAPPING = OrderedDict(
    [
        (ElectraConfig, ElectraClassificationHead),
        (RobertaConfig, RobertaClassificationHead),
    ]
)


class AutoSeqClassificationHead:
    """
    This is a generic model class that will be instantiated as one of the model classes of the library---with a head for sequence classification
    ---when created with the when created with the
    :meth:`~transformers.AutoSeqClassificationHead.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoSeqClassificationHead is designed to be instantiated "
            "using the `AutoSeqClassificationHead.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_CLASSIFICATION_HEAD_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a head for sequence classification---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use :meth:`~transformers.AutoSeqClassificationHead.from_pretrained` to load the model
            weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::
        """
        if type(config) in MODEL_CLASSIFICATION_HEAD_MAPPING.keys():
            return MODEL_CLASSIFICATION_HEAD_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_CLASSIFICATION_HEAD_MAPPING.keys())
            )
        )

model_type_list = [
    "bert",
    "mobilebert",
    "electra"
]
