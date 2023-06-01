from flaml import oai
from flaml.autogen.datagen import generate_adversarial_examples
import re
import logging
import hydra
import wikipedia

KEY_LOC = "test/autogen"
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def test_adv_gen(cfg):
    try:
        import openai
    except ImportError:
        return

    # config_list_adv = oai.config_list_gpt4_gpt35(KEY_LOC)
    config_list_adv = oai.config_list_openai_aoai(KEY_LOC)
    config_list_adv[0].update(cfg.openai.adv)
    config_list_eval = oai.config_list_openai_aoai(KEY_LOC)
    config_list_eval[0].update(cfg.openai.eval)

    test_cases = [# SimpleArith(config_list=config_list_eval)
        WikipediaQGen(config_list=config_list_eval)
        ]

    for case in test_cases:
        adv_examples = generate_adversarial_examples(
            data=case.input_examples,
            verif_func=case.verif_func,
            eval_func=case.test_func,
            num_examples=5,
            # reduction=np.mean,
            config_list=config_list_adv,
        )
        print(adv_examples)

class SimpleArith:
    input_examples = [
        {"input": "1 + 4 =", "target": "5"},
        {"input": "4 + 9 =", "target": "13"},
        {"input": "8 + 3 =", "target": "11"},
        {"input": "30 + 89 =", "target": "119"},
        {"input": "486 + 141 =", "target": "627"},
        {"input": "13 + 476 =", "target": "489"},
        {"input": "773 + 546 =", "target": "1319"},
        {"input": "348 + 227 =", "target": "575"},
    ]

    def __init__(self, config_list):
        self.config_list = config_list

    @staticmethod
    def verif_func(example):
        lhs = eval(re.findall(r"^(.*?)=", example["input"])[0].strip())
        rhs = int(example["target"])

        return lhs == rhs

    def test_func(self, example):
        base_prompt = "{input}"
        config = {
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "model": "text-davinci-003",
        }
        # query['prompt'] = base_prompt.format(example['input'])
        # resp = oai.Completion.create(**query)
        response = oai.Completion.create(example, prompt=base_prompt, config_list=self.config_list, **config)
        return example["target"] == oai.Completion.extract_text(response)[0].strip()


class WikipediaQGen:
    def __init__(self, config_list, search_term='Cornell University'):
        self.config_list = config_list
        r = wikipedia.search(search_term)
        page = wikipedia.page(r[0])
        self.title = page.title
        self.content = page.content
        example_gen_prompt = f"""<|im_start|>system
You are a question generating assistant. Your objective is to take some context and generate questions together with their corresponding answer or possible answers
<|im_end|>
<|im_start|>user
Context
---
# 
{page.title}

{page.content}
<|im_end|>
<|im_start|>user
Generate a series of questions related to {page.title} as follows.

1. Mode = "paragraph"

Write a question for which the answer is a short paragraph.

2. Mode = "few-words"

The answer is at most a few words.

3. Mode = "number"

The answer is a number.

4. Mode = "bool"

Generate a question with a True/False answer.

For each question above, provide the corresponding correct answer. If there is more than one correct answer, provide a list of all possible answers.
<|im_end|>
<|im_start|>assistant
"""
        config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "model": "text-davinci-003",
        }

        response = oai.Completion.create(prompt=example_gen_prompt, config_list=self.config_list, **config)
        answer = oai.Completion.extract_text(response)[0].strip()
        # find qa
        qa_parsed = re.findall(r"(?=Question:)[\s\S]*?(?=[0-9]. Mode|$)", response)
        self.input_examples = []
        for qa in qa_parsed:
            example = {"input":re.findall(r"(?<=Question:)[\s\S]*?(?=Answer:)", qa)[0].strip(), 
                       "target":re.findall(r"(?<=Answer:)", qa)[0].strip()}
            self.input_examples.append(example)


    def add_message(self, content, role="user"):
        self.messages.append({"role": role, "content": content})

    def verif_func(self, example):
        base_prompt = """Respond with Yes or No, does the text below answer the question provided?
        Question: {input}
        Text: {target}
        Answer:
        """
        config = {
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "model": "text-davinci-003",
        }
        response = oai.Completion.create(example, prompt=base_prompt, config_list=self.config_list, **config)
        answer = oai.Completion.extract_text(response)[0].strip()
        return answer == 'Yes'

    def test_func(self, example):
        base_prompt = f"""Answer the following question based on the context provided.
        Question:
        {{input}}
        Context:
        {self.title}
        {self.content}
        Answer:
        """
        config = {
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "model": "text-davinci-003",
        }
        response = oai.Completion.create(example, prompt=base_prompt, config_list=self.config_list, **config)
        answer = oai.Completion.extract_text(response)[0]
        pred_example = {"input": example["input"], "target": answer}
        return self.verif_func(pred_example)



if __name__ == "__main__":
    # import openai
    # import os

    # config_list = oai.config_list_openai_aoai(KEY_LOC)
    # assert len(config_list) >= 3, config_list
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    test_adv_gen()
