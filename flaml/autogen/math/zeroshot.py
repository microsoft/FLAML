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
from utils import (
    load_level5_math_test_each_category,
    math_type_mapping,
    write_json,
    remove_asy_sections,
    mylogger,
    load_all_fixed,
)


parser = argparse.ArgumentParser()
# parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./zeroshot", type=str)
parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache/zeroshot", help="cache folder")
parser.add_argument("--samples_per_category", "-s", help="samples per category", default=20, type=int)
parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")
parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
parser.add_argument("--select", action="store_true")

args = parser.parse_args()
args.folder = args.folder + "_baseline_zeroshot_t" + str(args.temperature) + "_seed" + str(args.seed)

# key = os.getenv(args.key)
# print(key)


def zeroshot_solve(model, problem, max_tokens=None):
    full_prompt = """Solve a math problem carefully. Put the final answer in \\boxed{}.\n\nProblem: """
    full_prompt += remove_asy_sections(problem["problem"])

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
        # 'temperature': args.temperature,
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    if config_list is not None:
        raw_responses = oai.ChatCompletion.create(
            config_list=config_list,
            **config,
        )
    else:
        raw_responses = oai.ChatCompletion.create(None, **config)
    # raw_responses = oai.ChatCompletion.create(config_list=config_list, **config)
    responses = oai.ChatCompletion.extract_text(raw_responses)

    try:
        total_cost = oai.ChatCompletion.cost(raw_responses)
    except TypeError:
        total_cost = oai.ChatCompletion.cost("gpt-4", raw_responses)
    return {
        "cost": total_cost,
        "response_with_ans": responses[0],
    }


if __name__ == "__main__":
    config_list = None

    # from azure.identity import DefaultAzureCredential

    # SCOPE = "https://ml.azure.com"
    # credential = DefaultAzureCredential()
    # token = credential.get_token(SCOPE).token
    # headers = {
    #     "azureml-model-deployment": "gpt4",
    #     "Authorization": f"Bearer {token}",
    #     "Content-Type": "application/json",
    #     **json.load(open("headers.json")),
    # }
    # config_list = [
    #     {
    #         "api_key": open("key.txt").read().strip(),
    #         "api_type": "open_ai",
    #         "api_base": "https://api.openai.com/v1",
    #     },
    #     {
    #         "api_key": open("key_flaml.txt").read().strip(),
    #         "api_type": "azure",
    #         "api_base": open("base_flaml.txt").read().strip(),
    #         "api_version": "2023-03-15-preview",
    #     },
    #     # {
    #     #     "api_key": open("key_gcr.txt").read().strip(),
    #     #     "api_type": "azure",
    #     #     "api_base": open("base_gcr.txt").read().strip(),
    #     #     "api_version": "2023-03-15-preview",
    #     # },
    #     # {
    #     #     "api_key": "nokey",
    #     #     "headers": headers,
    #     #     "api_base": open("base_azure.txt").read().strip(),
    #     # },
    # ]
    problem_sets = load_level5_math_test_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )
    if args.select:
        # problem_sets = load_fixed()
        problem_sets = load_all_fixed()
        # print("hhh")

    selected_samples = {
        "Algebra": [108],  # [8] wrong,  # 8 correct
        # "Counting & Probability": [73, 96], #  0,10,  | 5 correct [2,3,16,18,19], 6 [4,5,13,14,15,17] wrong
        # "Algebra": [14],
        # "Number Theory": [75, 120],
        # "Precalculus": [62],
        # "Number Theory": [19, 26, 75, 120, 140],
        # "Algebra": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # [8] wrong,  # 8 correct
        # "Algebra": [1,2,4,13],
        # "Algebra": [18], # [1, 8] wrong, 9-10 out of 10 correct
        # "Algebra": [2, 5, 13],
        # "Geometry": [],
        # "Algebra": [i for i in range(20)],
        # "Counting & Probability": [i for i in range(20)],
        # "Intermediate Algebra": [0, 8, 15, 17],
        # "Prealgebra": [i for i in range(20)],
        # "Number Theory": [0, 2, 4,6,7,8,10,11,12,13,14,15,17,18],  # [3] always wrong      [1, 5, 9, 16, 19] always right
        # "Prealgebra": [3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17], # [0,7,16] always wrong, [1,2,5,6,10,18,19] always right
    }
    config_list = [
        {
            "api_key": open("key.txt").read().strip(),
            "api_type": "open_ai",
            "api_base": "https://api.openai.com/v1",
        },
        {
            "api_key": open("key_flaml.txt").read().strip(),
            "api_type": "azure",
            "api_base": open("base_flaml.txt").read().strip(),
            "api_version": "2023-03-15-preview",
        },
        {
            "api_key": open("key_aoai.txt").read().strip(),
            "api_type": "azure",
            "api_base": open("base_aoai.txt").read().strip(),
            "api_version": "2023-03-15-preview",
        },
        # {
        #     "api_key": open("key_gcr.txt").read().strip(),
        #     "api_type": "azure",
        #     "api_base": open("base_gcr.txt").read().strip(),
        #     "api_version": "2023-03-15-preview",
        # },
        # {
        #     "api_key": "nokey",
        #     "headers": headers,
        #     "api_base": open("base_azure.txt").read().strip(),
        # },
    ]
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=args.seed, cache_path=args.cache_folder)

    os.makedirs(args.folder, exist_ok=True)
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    engine = "gpt-4"
    aggre_correct = 0

    logger.log("problem id: is_correct $ ans $ correct_ans $ accum_acc", verbose=True)

    for problem_set in problem_sets:  # one problem_set is one category
        for i in range(len(problem_set)):
            problem_set[i]["problem_id"] = str(i)  # assign problem id
        if args.select:
            if problem_set[0]["type"] in selected_samples and len(selected_samples[problem_set[0]["type"]]) > 0:
                problem_set = [problem_set[i] for i in selected_samples[problem_set[0]["type"]]]
                print(problem_set[0]["type"], selected_samples[problem_set[0]["type"]])
            else:
                continue
        logger.log("Solving " + problem_set[0]["type"], verbose=True)
        saving_folder = os.path.join(args.folder, math_type_mapping[problem_set[0]["type"]])
        os.makedirs(saving_folder, exist_ok=True)
        done_problems = set([int(f.split(".")[0]) for f in os.listdir(saving_folder) if "json" in f])

        correct_counts = 0
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            # 1. if problem already solved, continue
            if int(problem["problem_id"]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                aggre_correct += problem["is_correct"]
                correct_counts += problem["is_correct"]
                logger.log(
                    f"{count}: {problem['is_correct']} $ {problem['voted_answer']} $ {problem['correct_ans']} $ {round(correct_counts / (count + 1), 4)} (loaded from previous run)",
                    verbose=True,
                )
                continue

            results = zeroshot_solve(engine, problem)
            if results is None:
                break
            metrics = eval_math_responses([results["response_with_ans"]], problem["solution"])
            aggre_correct += metrics["success_vote"]
            correct_counts += metrics["success_vote"]

            problem.update(
                {
                    "cost": results["cost"],
                    "is_correct": bool(metrics["success_vote"]),
                    "correct_ans": get_answer(problem["solution"]),
                    "voted_answer": get_answer(metrics["voted_answer"]),
                    "response": results["response_with_ans"],
                }
            )
            write_json(problem, problem_path)
            logger.log(
                f"{count}: {problem['is_correct']} $ {problem['voted_answer']} $ {problem['correct_ans']}",
                verbose=True,
            )
        logger.log(
            f"{problem_set[0]['type']} acc: {correct_counts}/{len(problem_set)}= {round(correct_counts / len(problem_set), 4)}",
        )
        logger.log("-----------------------------------")
        if args.dry_run:
            break
        # os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    logger.log(
        f"Total accuracy: {aggre_correct}/{(len(problem_sets) * len(problem_sets[0]))}={round(aggre_correct / (len(problem_sets) * len(problem_sets[0])), 4)}",
    )
    logger.log("****************************\n\n\n\n")
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)
