import os
import json
import argparse
from flaml import oai
from flaml.autogen.math_utils import eval_math_responses, get_answer
from utils import (
    write_json,
    math_type_mapping,
    mylogger,
    load_level5_math_test_each_category,
    load_fixed,
    random_sample_MATH,
)

from funccall_python import MathChatFunctionPython
from funccall_wolfram import MathChatFunctionWolfram
from mathchat import MathChat


def solve_one_category(problem_set, saving_folder, solver):
    """
    Solve all problems in a category.
    Assumption 1: all problems are of the same type
    Assumption 2: if resume from a previous run, the sequence of problems are the same as the previous run, using same shuffling seed

    Args:
        problem_set (list): a list of problems
        saving_folder (str): the result folder to save the solved problems, the category folder will be created inside

    Returns:
        None
    """
    logger = mylogger(os.path.join(saving_folder, "log.txt"))

    # assume all problems are of the same type: TODO: ensure this assumption
    saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]["type"]])
    # mkdir if not exist
    os.makedirs(saving_folder, exist_ok=True)

    # from the saving folder load solved problems
    done_problems = set([int(f.split(".")[0]) for f in os.listdir(saving_folder) if "json" in f])

    correct_counts = 0
    logger.log("id : is_correct $ ans $ correct_ans | corrected_ans $ round")
    for _, problem in enumerate(problem_set):
        problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

        # 1. if problem already solved, continue
        if int(problem["problem_id"]) in done_problems:
            problem = json.load(open(problem_path, "r"))
            correct_counts += problem["is_correct"]

            logger.log(
                f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} $ {problem["round"]} $ (from previous run)'
            )
            continue

        # 2. solve the problem
        # file_to_be_saved=os.path.join(saving_folder, problem["problem_id"] + ".txt")
        result = solver.solve_one_problem(problem)
        metrics = eval_math_responses([result["response_with_ans"]], problem["solution"])

        # 3. save the result
        correct_ans = get_answer(problem["solution"])
        problem.update(
            {
                "is_valid_reply": result["is_valid_reply"],
                "is_correct": bool(metrics["success_vote"]),
                "correct_ans": correct_ans,
                "voted_answer": get_answer(metrics["voted_answer"]),
                "round": result["round"],
                "messages": result["messages"],  # the conversation
            }
        )
        write_json(problem, problem_path)

        # 4. continue to next problem
        correct_counts += problem["is_correct"]
        logger.log(
            f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} $ {problem["round"]} $ '
        )

    tp = problem_set[0]["type"]
    logger.log(f"{tp} Accuracy: {correct_counts}/{len(problem_set)} = {correct_counts/len(problem_set)}")
    logger.log("------------------------------------------------------------\n", verbose=True)


def parse_args():
    parser = argparse.ArgumentParser(description="MathChat")
    parser.add_argument("--name", "-n", dest="name", help="the name of current run", default="", type=str)
    parser.add_argument("--solver", dest="solver", help="solver", default="mathchat", type=str)
    parser.add_argument("--max_round", dest="max_round", help="max round", default=10, type=int)
    parser.add_argument(
        "--cache_folder", "-c", dest="cache_folder", default=".cache", help="cache folder"
    )  # cache folder for oai
    parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
    parser.add_argument("--test_run", help="test run", action="store_true")
    parser.add_argument("--samples_per_category", help="samples per category", default=20, type=int)
    parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")

    # not used yet
    parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--sample_all", help="samples per category", default=0, type=int)

    args = parser.parse_args()
    args.folder = "./results/" + args.name + args.solver + "_t" + str(args.temperature)
    if args.seed != 41:
        args.folder = args.folder + "_seed" + str(args.seed)
    if args.refine:
        args.folder = args.folder.replace("_t" + str(args.temperature), "_refine_t" + str(args.temperature))
    if args.sample_all != 0:
        args.folder += "_random_sample"
    os.makedirs(args.folder, exist_ok=True)
    return args


def pseudo_main(config_list):
    # 1. args, settings and logger
    args = parse_args()
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=args.seed, cache_path_root=args.cache_folder)
    logger = mylogger(os.path.join(args.folder, "log.txt"))
    print(f"Running {args.folder} with solver {args.solver}")

    # 2. load solver
    solver_map = {
        "mathchat": MathChat,
        "func_python": MathChatFunctionPython,
        "func_wolfram": MathChatFunctionWolfram,
    }
    if args.solver not in solver_map:
        print(f"Solver {args.solver} not found. Exiting.")
        exit()
    else:
        solver = solver_map[args.solver](
            seed=args.seed,
            config_list=config_list,
            max_consecutive_auto_reply=args.max_round,
        )

    # 3. load math dataset
    # default loading: k level-5 problems per sepcified category
    problem_sets = load_level5_math_test_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )

    # test run: take out 1 problem from each category
    if args.test_run:
        problem_sets = load_level5_math_test_each_category(samples_per_category=1, category_to_load=args.categories)
        logger.log("Take out 1 problem from each category for test run.")

    # select: load fixed problems, need to specify folder, need to check and specify selected_samples
    if args.select:
        problem_sets = load_fixed(folder="22_user_v3select_t1")
    selected_samples = {
        "Intermediate Algebra": [0, 3, 6, 8, 9, 10, 11, 13, 15, 16, 17],
    }

    # random sample: random sample problem from the whole test dataset
    if args.sample_all != 0:
        problem_sets = random_sample_MATH(args.sample_all)

    # 4. solve
    for problem_set in problem_sets:
        if args.select:
            if problem_set[0]["type"] in selected_samples and len(selected_samples[problem_set[0]["type"]]) > 0:
                problem_set = [problem_set[i] for i in selected_samples[problem_set[0]["type"]]]
                print(problem_set[0]["type"], selected_samples[problem_set[0]["type"]])
            else:
                continue
        for i in range(len(problem_set)):
            problem_set[i]["problem_id"] = str(i)  # assign problem id

        solve_one_category(problem_set, saving_folder=args.folder, solver=solver)
        # os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    logger.log("*******************************************************************************\n\n\n", verbose=False)
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)
