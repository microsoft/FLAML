import copy
import re
from collections import OrderedDict
from functools import partial

from transformers import AutoTokenizer
from flaml.nlp.dataset.sentence_keys_auto import get_sentence_keys
from bisect import bisect
import numpy as np

def inserting_sepp(sent, start, end, this_tokenizer):
    return sent[:start].rstrip() \
        + " " + this_tokenizer.sep_token + " " \
        + sent[start:end] \
        + " " + this_tokenizer.sep_token + " " \
        + sent[end:].lstrip()

def tokenize_superglue_copa(this_example,
                           this_tokenizer,
                           dataset_name,
                           subdataset_name=None,
                           **kwargs):

    return None

def tokenize_superglue_wic_gpt2(this_example,
                               this_tokenizer,
                               dataset_name,
                               subdataset_name=None,
                               **kwargs):
    return None

def tokenize_superglue_wic(this_example,
                           this_tokenizer,
                           dataset_name,
                           subdataset_name=None,
                           **kwargs
                           ):
    sent1, sent2 = this_example["sentence1"], this_example["sentence2"]
    start1, end1 = this_example["start1"], this_example["end1"]
    start2, end2 = this_example["start2"], this_example["end2"]
    altered_sent1 = inserting_sepp(sent1, start1, end1, this_tokenizer)
    altered_sent2 = inserting_sepp(sent2, start2, end2, this_tokenizer)
    input_ids_sepp = this_tokenizer(*(altered_sent1, altered_sent2), padding="max_length", max_length= 1024, truncation=True)["input_ids"]
    data_pair = (sent1, sent2)
    assert "max_seq_length" in kwargs, "max_seq_length must be provided for glue"
    this_data = this_tokenizer(*data_pair, padding="max_length", max_length=kwargs["max_seq_length"], truncation=True)
    input_ids = this_data["input_ids"]
    which_sepp = 0
    span_start_end = [[100000, 100000], [100000, 100000]]
    ptr_sepp = ptr_nosepp = 0
    try:
        padding_direction = this_tokenizer.padding_side
        if padding_direction == "left":
            padding_id = input_ids_sepp[0]
            while input_ids_sepp[ptr_sepp] == padding_id:
                ptr_sepp += 1
            while input_ids[ptr_nosepp] == padding_id:
                ptr_nosepp += 1
    except:
        pass
    sep_id = this_tokenizer.convert_tokens_to_ids([this_tokenizer.sep_token])[0]
    while ptr_sepp < len(input_ids_sepp) and ptr_nosepp < len(input_ids) and \
        input_ids_sepp[ptr_sepp] != 0 and input_ids[ptr_nosepp] != 0:
        if input_ids_sepp[ptr_sepp] == input_ids[ptr_nosepp]:
            ptr_sepp += 1; ptr_nosepp += 1
        else:
            if not (input_ids_sepp[ptr_sepp] == sep_id or this_tokenizer.convert_ids_to_tokens([input_ids_sepp[ptr_sepp]])[0] in ('▁', '_')):
                break
            if input_ids_sepp[ptr_sepp] == sep_id:
                span_start_end[int(which_sepp / 2)][which_sepp % 2] = ptr_nosepp
                which_sepp += 1
                ptr_sepp += 1
            else:
                ptr_sepp += 1
    max_word_span = 16
    word_indices = []
    for idx1 in range(2):
        if span_start_end[idx1][1] < kwargs["max_seq_length"]:
            first_span = [x for x in range(span_start_end[idx1][0], span_start_end[idx1][1]) if x < kwargs["max_seq_length"]] \
                        + [0] * (max_word_span - span_start_end[idx1][1] + span_start_end[idx1][0])
            word_indices.append(first_span)
    this_data["word_spans"] = word_indices
    return this_data

def tokenize_gule(this_example,
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

    data_pair = (
        (this_example[sentence1_key],) if sentence2_key is None else (
            this_example[sentence1_key], this_example[sentence2_key])
    )
    assert "max_seq_length" in kwargs, "max_seq_length must be provided for glue"
    return this_tokenizer(*data_pair, padding="max_length", max_length= kwargs["max_seq_length"], truncation=True)

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
        (("super_glue", "wic"), tokenize_superglue_wic),
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
                    partial(token_func,
                            this_tokenizer=this_tokenizer,
                            dataset_name = dataset_name,
                            subdataset_name = subdataset_name,
                            **kwargs), batched=False)
            except:
                raise ValueError("{}, {}, Return empty".format(dataset_name, subdataset_name))
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                dataset_name, subdataset_name, cls.__name__, ", ".join(c.__name__ for c in TOKENIZER_MAPPING.keys())
            )
        )