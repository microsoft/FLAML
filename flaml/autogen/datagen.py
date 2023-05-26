import regex as re
import numpy as np
from flaml import oai


def generate_adversarial_examples(data, verif_func, eval_func, num_examples=5, reduction=np.mean, **config):
    base_prompt = "Generate more complex versions of the input following examples. Make sure that the testing the input would result in the same target as specified. Make sure that the inputs are of the same types that are specified in the examples. Do not replace integers with words.\nexamples:{examples}"

    # base_settings = {
    #     "max_tokens": 64,
    #     "temperature": 1,
    #     "top_p": 1,
    #     "n": 5,
    #     "model": "gpt-4",
    # }
    max_iter = 10
    iter = 0
    adv_examples = []
    while len(adv_examples) < num_examples and iter < max_iter:
        # query = base_settings
        # query["prompt"] = base_prompt.format(examples=str(data))
        response = oai.Completion.create({"examples": str(data)}, prompt=base_prompt, **config)
        resp = oai.Completion.extract_text(response)[0]
        adv_candidates = re.findall(r"(?={).*(?<=})", resp)
        for cand in adv_candidates:
            candidate = eval(cand)
            cand_verification = verif_func(candidate)
            cand_test = eval_func(candidate, **config)
            if cand_verification and not cand_test:
                adv_examples.append(candidate)

    input_data_metric = reduction(eval_func(data, **config))
    adv_metric = reduction(eval_func(adv_examples, **config))

    return adv_examples, (input_data_metric - adv_metric)
