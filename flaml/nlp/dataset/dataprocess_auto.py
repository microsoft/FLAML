from functools import partial
from collections import OrderedDict

def get_mapping_func_squad(logger, **kwargs):
    try:
        glue_sentence_keys = kwargs["sentence_keys"]
        return partial(tokenize_glue, sentence_keys=glue_sentence_keys)
    except KeyError as err:
        logger.error("for glue, you must specify 'sentence_keys'")
        raise err

def get_mapping_func_glue(logger, **kwargs):
    try:
        glue_sentence_keys = kwargs["sentence_keys"]
        return partial(tokenize_glue, sentence_keys=glue_sentence_keys)
    except KeyError as err:
        logger.error("for glue, you must specify 'sentence_keys'")
        raise err

def tokenize_glue(max_seq_length,
                  this_tokenizer,
                  examples,
                  sentence_keys):
        if len(sentence_keys) > 1:
            sentence1_key, sentence2_key = sentence_keys[0], sentence_keys[1]
        else:
            sentence1_key = sentence_keys[0]
            sentence2_key = None

        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        return this_tokenizer(*args, padding="max_length", max_length= max_seq_length, truncation=True)

MAPPING_FUNC_MAPPING = OrderedDict(
    [
        ("squad", get_mapping_func_squad),
        ("glue", get_mapping_func_glue),
    ]
)