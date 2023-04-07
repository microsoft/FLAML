import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
import os
import pytest
import json
import random
from time import sleep
import numpy as np
import re
from flaml import oai
from flaml.autogen.math_utils import last_boxed_only_string, remove_boxed, is_equiv
from flaml.autogen.math_utils import (
    nestmkdir,
    write_json,
    boxed_number,
    strip_math_message,
    voting,
)

from collections import defaultdict
import matplotlib.pyplot as plt
import logging

try:
    import openai

    skip = False
except ImportError:
    skip = True


here = os.path.abspath(os.path.dirname(__file__))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _load_data(
    n_tune_data=20,
    n_test_data=50,
    prob_level="Level 5",
    prob_type="Counting & Probability",
):
    import datasets

    seed = 41
    data = datasets.load_dataset("competition_math")
    train_data = data["train"].shuffle(seed=seed)
    test_data = data["test"].shuffle(seed=seed)
    tune_data = [
        {
            "problem": train_data[x]["problem"],
            "solution": train_data[x]["solution"],
        }
        for x in range(len(train_data))
        if train_data[x]["level"] == prob_level and train_data[x]["type"] == prob_type
    ][:n_tune_data]
    test_data = [
        {
            "problem": test_data[x]["problem"],
            "solution": test_data[x]["solution"],
        }
        for x in range(len(test_data))
        if test_data[x]["level"] == prob_level and test_data[x]["type"] == prob_type
    ][:n_test_data]

    print(len(tune_data), len(test_data))
    print(tune_data[1]["problem"])
    print(tune_data[1]["solution"])
    return tune_data, test_data


def _eval_test(test_data, config):
    import logging

    prompts = [
        "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
    ]
    config = {"model": "gpt-3.5-turbo", "prompt": prompts[0], "max_tokens": 600, "n": 1}
    n1_result = oai.ChatCompletion.test(test_data, config, eval_math_responses)
    print(n1_result)
    return n1_result


def _hint_generator(problem, model, n=3):
    """
    problem: math question
    model: gpt model
    n: how many hints to be generated
    """
    config = {
        "model": model,
        "n": n,
        "prompt": "What is the key trick to this problem? Please state it with an sentence and then stop.\n\nProblem: {problem}",
    }

    context = {"problem": problem["problem"]}

    raw_responses = oai.ChatCompletion.create(context, **config, use_cache=True)

    return [
        r["message"]["content"].rstrip() for r in raw_responses["choices"]
    ]  # strip hints


def _hint_selector(hints, model, n=100):
    """
    hints: list of str,  hints
    model: model used
    """
    config = {"model": model, "n": n, "prompt": "{ask}"}

    ask = "Given a problem, there are several hints to the problem. Choose the best hint that can solve the problem. You should ONLY return the id of that response. For example, if you find the best hint is i, reply '[i]' and then stop. You should have no explanations or any other words.\n\n"
    for i, r in enumerate(hints):
        ask += f"[{i}]: " + r + "\n\n"

    context = {"ask": ask}

    raw_responses = oai.ChatCompletion.create(context, **config, use_cache=True)
    bracket_res = [r["message"]["content"].rstrip() for r in raw_responses["choices"]]
    box_res = []
    for b in bracket_res:
        a = boxed_number(b)
        if a:
            box_res.append(a)

    most_voted_id, _, _ = voting(box_res)

    return hints[int(most_voted_id)]


def _hint_creator(problem, **config):
    """
    problem: math question
    model: gpt model
    n: how many hints to be generated
    """
    model = config["model"]
    n = config["n"]
    hints = _hint_generator(problem, model, n=n)
    voted_hint = _hint_selector(hints, model, n=100)

    print("\n\nHints:", hints)
    print("\n\nVoted Hint:", voted_hint)

    return voted_hint


def solve_with_vote_hint(test_data, model_name="gpt-4"):
    logger.debug("Solving with vote and hints")

    markers = [
        "o",
        "s",
        "D",
        "v",
        "p",
        "h",
        "d",
        "P",
        "X",
        "H",
        "8",
        "4",
        "3",
        "2",
        "1",
        "x",
        "+",
        ">",
        "<",
        "^",
        "v",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "d",
        "D",
        "|",
        "_",
    ]

    # NOTE: about hints
    hint_generator_prompts = [
        "What is the key trick to this problem? Please state it with an sentence and then stop.\n\nProblem: {problem}"
    ]
    hint_number = 3
    hint_config = {
        "model": model_name,
        "prompt": hint_generator_prompts[0],
        "max_tokens": 600,
        "n": hint_number,
    }

    prompts_with_hints = [
        "{problem} Solve the problem carefully with hint. \n\nHint: {hint}. \n\nSimplify your answer as much as possible. Put the final answer in \\boxed{{}}."
    ]
    for j, n in enumerate([10, 30]):
        config_with_hint = {
            "model": model_name,
            "prompt": prompts_with_hints[0],
            "max_tokens": 600,
            "n": n,
        }
        metrics = []
        x, y = [], []
        votes_success = defaultdict(lambda: [0, 0])
        success_num = 0
        for i, data_i in enumerate(test_data):
            hint_i = _hint_creator(problem=data_i, **hint_config)  # generate hint
            # NOTE: adding hint to context
            # data_i.update({"hint": hint_i})

            print(data_i)
            data_with_hint = data_i.copy()
            data_with_hint.update({"hint": hint_i})
            print("data with hint:", data_with_hint)
            response = oai.ChatCompletion.create(
                context=data_with_hint, **config_with_hint
            )
            responses = oai.ChatCompletion.extract_text(response)
            metrics.append(eval_math_responses(responses, **data_i))
            votes = metrics[-1]["votes"]
            success = metrics[-1]["success_vote"]
            votes_success[votes][0] += 1
            votes_success[votes][1] += success
            success_num += success
        logger.debug(f"n: {n}, +hint, agg success rate: {success_num / (i + 1)}")
        for votes in votes_success:
            x.append(votes)
            y.append(votes_success[votes][1] / votes_success[votes][0])
        logger.debug(f"+hint, top vote: {x}")
        logger.debug(f"+hint, success rate: {y}")
        plt.scatter(x, y, marker=markers[j])
        plt.xlabel("top vote")
        plt.ylabel("success rate")
    plt.legend(["n=10", "n=30"])
    plt.savefig("test/openai/math_vote_hint.png")


def solve_with_vote(test_data, model_name="gpt-4"):
    from collections import defaultdict
    import matplotlib.pyplot as plt

    logger.debug("Solving with vote")

    prompts = [
        "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
    ]
    markers = [
        "o",
        "s",
        "D",
        "v",
        "p",
        "h",
        "d",
        "P",
        "X",
        "H",
        "8",
        "4",
        "3",
        "2",
        "1",
        "x",
        "+",
        ">",
        "<",
        "^",
        "v",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "d",
        "D",
        "|",
        "_",
    ]
    for j, n in enumerate([10, 30]):
        config = {"model": model_name, "prompt": prompts[0], "max_tokens": 600, "n": n}

        metrics = []
        x, y = [], []
        votes_success = defaultdict(lambda: [0, 0])
        success_num = 0
        for i, data_i in enumerate(test_data):
            response = oai.ChatCompletion.create(context=data_i, **config)
            responses = oai.ChatCompletion.extract_text(response)
            metrics.append(eval_math_responses(responses, **data_i))
            votes = metrics[-1]["votes"]
            success = metrics[-1]["success_vote"]
            votes_success[votes][0] += 1
            votes_success[votes][1] += success
            success_num += success
        logger.debug(f"n: {n}, agg success rate: {success_num / (i + 1)}")
        for votes in votes_success:
            x.append(votes)
            y.append(votes_success[votes][1] / votes_success[votes][0])
        logger.debug(f"top vote: {x}")
        logger.debug(f"success rate: {y}")
        plt.scatter(x, y, marker=markers[j])
        plt.xlabel("top vote")
        plt.ylabel("success rate")
    plt.legend(["n=10", "n=30"])
    plt.savefig("test/openai/math_vote.png")


if __name__ == "__main__":
    import time

    openai.api_key_path = "test/openai/key.txt"
    from flaml.autogen.math_utils import eval_math_responses

    seed = 41
    # model_name = "gpt-3.5-turbo" # "gpt-4", or "gpt-3.5-turbo"
    model_name = "gpt-4"  # "gpt-4", or "gpt-3.5-turbo"

    oai.ChatCompletion.set_cache(seed)
    prompts = [
        "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
    ]
    start_time = time.time()
    tune_data, test_data = _load_data(
        n_tune_data=20,
        n_test_data=50,
        prob_level="Level 5",
        prob_type="Counting & Probability",
    )
    # print("time used:", time.time() - start_time)
    # config_n1 = {"model": model_name, "prompt": prompts[0], "max_tokens": 600, "n": 1}
    # res_1 = _eval_test(test_data, config_n1)
    # print("result with config 1:", res_1)
    # print("time used:", time.time() - start_time)

    # start_time = time.time()
    # config_n2 = {"model": model_name, "prompt": prompts[0], "max_tokens": 600, "n": 10}
    # res_2 = _eval_test(test_data, config_n2)
    # print("result with config 2:", res_2)
    # print("time used:", time.time() - start_time)

    start_time = time.time()
    solve_with_vote(test_data, model_name)

    solve_with_vote_hint(test_data, model_name)
    print("time used:", time.time() - start_time)
