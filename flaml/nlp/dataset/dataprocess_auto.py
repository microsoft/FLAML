from functools import partial

def get_mapping_func_glue():
    return partial(self._tokenize, sentence_keys= sentence_keys)

def _tokenize(self,
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
        return self._tokenizer(*args, padding="max_length", max_length=self._max_seq_length, truncation=True)

MAPPING_FUNC_MAPPING = OrderedDict(
    [
        ("squad", metric_mode_mapping_squad),
        ("glue", metric_mode_mapping_glue),
        ("super_glue", metric_mode_mapping_super_glue),
    ]
)