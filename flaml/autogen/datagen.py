import json
from flaml import oai
import regex as re
from itertools import compress
import time
import logging

logger = logging.getLogger(__name__)

def generate_adversarial_examples(data, test_func, eval_func, num_examples=5, **config):
    base_prompt = """ 
# Instructions
- Generate a complex version of the example in the following task.
- Make sure that the inputs are of the same types that are specified in the examples. 
- Maintain the same format as the input examples, but feel free to be creative within that.
- Generate a json with double quotes. 
- Do not replace integers with words.
- For mathematical examples use programmatic syntax. For example, use '*' instead of 'x' for multiplication
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
        eval_cands = eval_func(candidate)
        test_cands = test_func(candidate, eval_cands)
        return (test_cands == 0)

    ii = 0
    while len(adv_examples) < num_examples and iteration < max_iter:
        # query = base_settings
        # query["prompt"] = base_prompt.format(examples=str(data))
        print(f"iteration={iteration}")
        sample = data[ii % len(data)]
        response = oai.Completion.create({"example": sample}, prompt=base_prompt, **config)
        resp_candidates = re.findall(r"(?={).*(?<=})", oai.Completion.extract_text(response)[0])
        if len(resp_candidates) > 0:
            adv_candidates = list(map(eval, resp_candidates))
            time.sleep(30)
            eval_candidates = list(map(group_check, adv_candidates))
            valid_candidates = list(compress(adv_candidates, eval_candidates))
            if len(valid_candidates) > 0:
                adv_examples.append(valid_candidates)
                iteration = 0
            else:
                iteration += 1
        time.sleep(30)
        ii += 1

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