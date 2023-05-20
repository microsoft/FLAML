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
import random

# Caution: distinguish between the two types imports
from flaml.autogen.math_utils import eval_math_responses, get_answer
from utils import (
    load_level5_math_test_each_category,
    math_type_mapping,
    write_json,
    remove_asy_sections,
    mylogger,
)


parser = argparse.ArgumentParser()
# parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./fewshot", type=str)
parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache/fewshot", help="cache folder")
parser.add_argument("--samples_per_category", "-s", help="samples per category", default=20, type=int)
parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")
parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
parser.add_argument("--k", dest="k", help="k", default=3, type=int)
args = parser.parse_args()
args.folder = args.folder + "_baseline_fewshot_t" + str(args.temperature) + "_seed" + str(args.seed)

# key = os.getenv(args.key)
# print(key)


def random_sample_level5_train_each_category(k=3, category_to_load=None):
    """
    Load level 5 math problems from the train set of  competition dataset.
    Returns:
        A list of list of problems. Each list of problems is of the same category.
    """
    category_to_load = [i for i in range(7)] if not category_to_load or "all" in category_to_load else category_to_load
    category_to_load = [int(x) for x in category_to_load]
    seed = 41
    data = datasets.load_dataset("competition_math")
    train_data = data["train"].shuffle(seed=seed)
    sep_cate = []
    print("******Loading train data******")
    for i, category in enumerate(math_type_mapping.keys()):
        if i not in category_to_load:
            print(i, category, "(skipped)", flush=True)
            continue
        tmp = []
        for x in range(len(train_data)):
            if (
                train_data[x]["level"] == "Level 5"
                and train_data[x]["type"] == category
                and "asy" not in train_data[x]["problem"]
                and "ASY" not in train_data[x]["problem"]
                and "asy" not in train_data[x]["solution"]
                and "ASY" not in train_data[x]["solution"]
            ):
                tmp.append(train_data[x])

        sep_cate.append(tmp[:k])
        print(i, category, f"{len(sep_cate[-1])} problems loaded", flush=True)
    print("******Loading train data done******")

    if len(sep_cate) == 0:
        raise ValueError("No category is loaded.")
    return sep_cate


# def random_select(problem_set, problem, k = 2):
#     examplar_data = [p for p in problem_set if p['problem_id']!=problem['problem_id'] and not 'asy' in p['problem'] and not 'ASY' in p['problem']]
#     assert len(examplar_data) >= k, "Not enough examplars"
#     selected_samples = random.sample(list(examplar_data), k)
#     return selected_samples


def few_shot_template(examplars):
    few_shot_prompt = ""
    for examplar in examplars:
        few_shot_prompt += "\n".join(
            [
                "Problem: " + examplar["problem"],
                "Solution: " + examplar["solution"],
                "\n",
            ]
        )
    return few_shot_prompt


def fewshot_solve(model, problem, prompt, max_tokens=None):
    # few shot examplars
    # examplars = random_select(problem_set, problem, k = k)
    # few_shot_prompt = few_shot_template(examplars)

    prompt += remove_asy_sections(problem["problem"])
    if args.dry_run:
        print(prompt)
        print("=======================")
        return

    config = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
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

    # openai.api_key = open("key_e.txt").read().strip()
    # print(openai.api_key)

    from azure.identity import DefaultAzureCredential

    SCOPE = "https://ml.azure.com"
    credential = DefaultAzureCredential()
    token = credential.get_token(SCOPE).token
    headers = {
        "azureml-model-deployment": "gpt4",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **json.load(open("headers.json")),
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
    random.seed(args.seed)
    os.makedirs(args.folder, exist_ok=True)
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    engine = "gpt-4"
    aggre_correct = 0
    problem_sets = load_level5_math_test_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )

    examplar_data = random_sample_level5_train_each_category(k=args.k, category_to_load=args.categories)
    logger.log("problem id: is_correct $ ans $ correct_ans $ accum_acc", verbose=True)

    for cate_id, problem_set in enumerate(problem_sets):  # one problem_set is one category
        for i in range(len(problem_set)):
            problem_set[i]["problem_id"] = str(i)  # assign problem id

        logger.log("Solving " + problem_set[0]["type"], verbose=True)
        saving_folder = os.path.join(args.folder, math_type_mapping[problem_set[0]["type"]])
        os.makedirs(saving_folder, exist_ok=True)
        done_problems = set([int(f.split(".")[0]) for f in os.listdir(saving_folder) if "json" in f])

        assert examplar_data[cate_id][0]["type"] == problem_set[0]["type"], " examplar and test category mismatch"
        category_prompt = (
            "Solve a math problem carefully. Put the final answer in \\boxed{}.\n\n"
            + few_shot_template(examplar_data[cate_id])
            + """\n\nProblem: """
        )

        with open(os.path.join(args.folder, f"prompt_{math_type_mapping[problem_set[0]['type']]}.txt"), "w") as f:
            f.write(category_prompt)

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
            results = fewshot_solve(engine, problem, category_prompt, max_tokens=None)
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
            if args.dry_run:
                break
        logger.log(
            f"{problem_set[0]['type']} acc: {correct_counts}/{len(problem_set)}= {round(correct_counts / len(problem_set), 4)}",
        )
        logger.log("-----------------------------------")
        if args.dry_run:
            print("------------------------------------")
        # os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    logger.log(
        f"Total accuracy: {aggre_correct}/{(len(problem_sets) * len(problem_sets[0]))}={round(aggre_correct / (len(problem_sets) * len(problem_sets[0])), 4)}",
    )
    logger.log("****************************\n\n\n\n")
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)
