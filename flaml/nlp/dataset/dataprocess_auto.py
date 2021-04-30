from collections import OrderedDict
from functools import partial

from transformers import AutoTokenizer
from flaml.nlp.dataset.sentence_keys_auto import get_sentence_keys

def tokenize_gule(examples,
                  this_tokenizer,
                  dataset_name,
                  subdataset_name = None,
                  **kwargs):
    sentence_keys = get_sentence_keys(dataset_name, subdataset_name)

    if len(sentence_keys) > 1:
        sentence1_key, sentence2_key = sentence_keys[0], sentence_keys[1]
    else:
        sentence1_key = sentence_keys[0]
        sentence2_key = None

    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )
    assert "max_seq_length" in kwargs, "max_seq_length must be provided for glue"
    return this_tokenizer(*args, padding="max_length", max_length= kwargs["max_seq_length"], truncation=True)

TOKENIZER_MAPPING = OrderedDict(
    [
        (("glue", "rte"), tokenize_gule),
        (("glue", "mrpc"), tokenize_gule),
        (("glue", "cola"), tokenize_gule),
        (("glue", "wnli"), tokenize_gule),
        (("glue", "stsb"), tokenize_gule),
        (("glue", "sst2"), tokenize_gule),
        (("glue", "mnli"), tokenize_gule),
        (("glue", "qqp"), tokenize_gule),
        (("glue", "qnli"), tokenize_gule),
    ]
)

class AutoToEncoded:
    """
    This is a generic huggingface class that will be instantiated as one of the huggingface classes of the library
    ---with the search space for grid search
    ---when created with the when created with the
    :meth:`~transformers.AutoTokenizer.from_model_and_dataset_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoGridSearchSpace is designed to be instantiated "
            "using the `AutoGridSearchSpace.from_config_and_method_name(method_name)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(cls, data_raw, model_checkpoint_path, dataset_name, subdataset_name = None, **kwargs):
        if (dataset_name, subdataset_name) in TOKENIZER_MAPPING.keys():
            try:
                this_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, use_fast=True)
                token_func = TOKENIZER_MAPPING[(dataset_name, subdataset_name)]
                return data_raw.map(
                    partial(token_func, this_tokenizer, dataset_name,subdataset_name,**kwargs), batched=True)
            except:
                raise ValueError("{}, {}, Return empty".format(dataset_name, subdataset_name))
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                dataset_name, subdataset_name, cls.__name__, ", ".join(c.__name__ for c in TOKENIZER_MAPPING.keys())
            )
        )