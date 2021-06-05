from collections import OrderedDict

from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

MODEL_CLASSIFICATION_HEAD_MAPPING = OrderedDict(
    [
        ("electra", ElectraClassificationHead),
        ("roberta", RobertaClassificationHead),
    ]
)


class AutoSeqClassificationHead:
    """
    This is a generic huggingface class that will be instantiated as one
    of the huggingface classes of the library---with a head for sequence classification
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
    def from_model_type_and_config(cls, model_type, config):
        """
        Instantiates one of the huggingface classes of the library---with
        a head for sequence classification---from a configuration.

        Note:
            Loading a huggingface from its configuration file does **not** load the huggingface weights.
            It only affects the huggingface's configuration. Use :meth:`~transformers.AutoSeqClassificationHead
            .from_pretrained` to load the huggingface
            weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The huggingface class to instantiate is selected based on the configuration class:

                List options

        Examples::
        """
        if model_type in MODEL_CLASSIFICATION_HEAD_MAPPING.keys():
            return MODEL_CLASSIFICATION_HEAD_MAPPING[model_type](config)
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
