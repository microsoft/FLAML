import json
from flaml import oai


def generate_adversarial_examples(data, verif_func, eval_func, num_examples=5, **config):
    base_prompt = "Generate more complex versions of the following input examples. Make sure that the input would result in the same target as specified. Make sure that the inputs are of the same types that are specified in the examples. Generate parsable json with double quotes. Do not replace integers with words.\nexamples:{examples}"

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
    while len(adv_examples) < num_examples and iteration < max_iter:
        # query = base_settings
        # query["prompt"] = base_prompt.format(examples=str(data))
        response = oai.Completion.create({"examples": str(data)}, prompt=base_prompt, **config)
        resp = oai.Completion.extract_text(response)[0]
        adv_candidates = json.loads(resp.strip().replace("'", '"'))  # re.findall(r"(?={).*(?<=})", resp)
        for cand in adv_candidates:
            candidate = cand
            cand_verification = verif_func(candidate)
            cand_test = eval_func(candidate)
            if cand_verification and not cand_test:
                adv_examples.append(candidate)
        iteration += 1
    return adv_examples
