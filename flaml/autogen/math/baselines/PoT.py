# adapted from https://github.com/wenhuchen/Program-of-Thoughts/blob/main/run_gsm8k_zs.py
import openai
from time import sleep
from tool import synthesize_program
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import os
import json
import argparse
from flaml import oai
import datasets

# Caution: distinguish between the two types imports
from flaml.autogen.math_utils import eval_math_responses, get_answer
from flaml.autogen.math.utils import load_level5_math_each_category, math_type_mapping, write_json, remove_asy_sections, mylogger
from flaml.autogen.code_utils import execute_code


parser = argparse.ArgumentParser()
# parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./PoT", type=str)
parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache/PoT", help="cache folder")
parser.add_argument("--samples_per_category", "-s", help="samples per category", default=20, type=int)
args = parser.parse_args()

# key = os.getenv(args.key)
# print(key)


def PoT_solve(model, problem, max_tokens=None):
    commented_problem = problem["problem"].replace("\n", "\n# ")  # in case the problem is multiline
    commented_problem = remove_asy_sections(commented_problem)
    full_prompt = f"""
import math
import numpy as np
import sympy as sp # added

# Question: {commented_problem}
# Answer this question by implementing a solver() function.
def solver():
    # Let's write a Python program step by step, and then return the answer
    # Firstly, we need define the following variable:
"""
    with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
                f.write(full_prompt)
    if args.dry_run:
        print(full_prompt)
        print("=======================")
        return

    config = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        "n": 1,
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    raw_responses = oai.ChatCompletion.create(None, **config, use_cache=True)
    responses = oai.ChatCompletion.extract_text(raw_responses)

    # TODO: adapt for voting
    program = synthesize_program(responses[0], full_prompt)
    return_code, ans = execute_code(program, timeout=5, use_docker=False)
    ans = ans.decode("ascii").strip() if type(ans) is bytes else ans
    ans = "Error" if return_code != 0 or ans is None else ans
    response_with_ans = "\\boxed{" + str(ans) + "}"

    prompt_price = (
        oai.ChatCompletion.price1K[model][0]
        if type(oai.ChatCompletion.price1K[model]) == tuple
        else oai.ChatCompletion.price1K[model]
    )
    return {
        "cost": oai.ChatCompletion.cost(model, raw_responses),
        "prompt_cost": prompt_price * raw_responses["usage"]["prompt_tokens"] / 1000,
        "response_with_ans": response_with_ans,
        "program": program,
    }


if __name__ == "__main__":
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=41, cache_path=args.cache_folder)
    
    os.makedirs(args.folder, exist_ok=True)
    logger = mylogger(os.path.join(args.folder, "log.txt"))
    logger.log("ans $ correct_ans $ accum_acc", verbose=True)
    
    engine = "gpt-4"
    aggre_correct = 0
    problem_sets = load_level5_math_each_category(samples_per_category=args.samples_per_category)
    for problem_set in problem_sets:  # one problem_set is one category
        for i in range(len(problem_set)):
            problem_set[i]["problem_id"] = str(i)  # assign problem id

        logger.log('Solving' + problem_set[0]["type"], verbose=True)
        saving_folder = os.path.join(args.folder, math_type_mapping[problem_set[0]["type"]])
        os.makedirs(saving_folder, exist_ok=True)
        done_problems = set([int(f.split(".")[0]) for f in os.listdir(saving_folder) if "json" in f])

        correct_counts = 0
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            # 1. if problem already solved, continue
            if int(problem["problem_id"]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                logger.log(
                    f"{problem['voted_answer']} $ {problem['correct_ans']} $ {round(correct_counts / (count + 1), 4)} (This problem is loaded from previous run)",
                    verbose=True,
                )
                aggre_correct += problem["is_correct"]
                correct_counts += problem["is_correct"]
                continue

            results = PoT_solve(engine, problem)
            metrics = eval_math_responses([results["response_with_ans"]], problem["solution"])
            aggre_correct += metrics["success_vote"]
            correct_counts += metrics["success_vote"]

            problem.update(
                {
                    "cost": results["cost"],
                    "is_correct": bool(metrics["success_vote"]),
                    "correct_ans": get_answer(problem["solution"]),
                    "voted_answer": get_answer(metrics["voted_answer"]),
                    "program": results["program"],
                }
            )
            write_json(problem, problem_path)
            logger.log(
                f"{problem['voted_answer']} $ {problem['correct_ans']} $ {round(correct_counts / (count + 1), 4)}",
                verbose=True,
            )
            if args.dry_run:
                break
        print('-----------------------------------')
        break

    logger.log(
        f"Total accuracy: {round(aggre_correct / (len(problem_sets) * len(problem_sets[0])), 4)}",
        verbose=True,
    ) 
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

