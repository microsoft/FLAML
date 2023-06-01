import json
from flaml import oai
import regex as re
from itertools import compress
import time
import logging

logger = logging.getLogger(__name__)

def generate_adversarial_examples(data, verif_func, eval_func, num_examples=5, **config):
    base_prompt = """ 
    # Instructions
    - Generate adversarial versions of the examples in the following task.
    - Make sure that the input would result in the same target as specified. 
    - Make sure that the inputs are of the same types that are specified in the examples. 
    - Generate parsable json with double quotes. 
    - Do not replace integers with words.
    <|start|>(example)
    {example}
    <|end|>
    <|start|>(answer)
    """

    # base_settings = {
    #     "max_tokens": 64,
    #     "temperature": 1,
    #     "top_p": 1,
    #     "n": 5,
    #     "model": "gpt-4",
    # }
    max_iter = 10
    iteration = 0
    adv_examples = []

    def group_check(candidate): # replace with loss function
        verif = verif_func(candidate)
        cand_test = eval_func(candidate)
        return verif and not cand_test

    while len(adv_examples) < num_examples and iteration < max_iter:
        # query = base_settings
        # query["prompt"] = base_prompt.format(examples=str(data))
        # time.sleep(62)
        response = oai.Completion.create({"example": str(data)}, prompt=base_prompt, **config)
        resp_candidates = re.findall(r"(?={).*(?<=})", oai.Completion.extract_text(response)[0])
        adv_candidates = list(map(eval, resp_candidates))
        eval_candidates = list(map(group_check, adv_candidates))
        valid_candidates = list(compress(adv_candidates, eval_candidates))
        if len(valid_candidates) > 0:
            adv_examples.append(valid_candidates)
            iteration = 0
        else:
            iteration += 1

    return adv_examples


# base_prompt = """
    # <|meta_start|>
    # # Introduction
    # - You are an adversarial example generation assistant
    # - Your goal is to generate more complex versions of the examples in the following task. 
    # - Make sure that the input would result in the same target as specified. 
    # - Make sure that the inputs are of the same types that are specified in the examples. 
    # - Generate parsable json with double quotes. 
    # - Do not replace integers with words.
    # <|meta_end|>
    # <|start|>(example)
    # {examples}
    # <|end|>
    # <|start|>(answer)
    # """