import openai
import os
from flaml.autogen.math.math_solver import MathSolver
import datasets
from flaml import oai
from functools import partial
from flaml.autogen.math.math_voting import SelfConsistency
import argparse
from utils import load_level5_math_each_category, math_type_mapping


def vanilla_solving(model, problem, n, max_tokens=None):
    """Solving a problem directly."""
    config = {
        "model": model,
        "n": n,
        "prompt": "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}.",
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    context = {
        "problem": problem["problem"],
    }
    raw_responses = oai.ChatCompletion.create(context, **config, use_cache=True)

    prompt_price = (
        oai.ChatCompletion.price1K[model][0]
        if type(oai.ChatCompletion.price1K[model]) == tuple
        else oai.ChatCompletion.price1K[model]
    )
    return {
        "responses": oai.ChatCompletion.extract_text(raw_responses),
        "cost": oai.ChatCompletion.cost(model, raw_responses),
        "prompt_cost": prompt_price * raw_responses["usage"]["prompt_tokens"] / 1000,
    }


def vanilla_voting_one_category(model, problem_set, saving_folder, n=10, n_per_time=3):
    """Solve one category of problems directly."""
    selfconsistency = SelfConsistency(n=n, n_per_time=n_per_time)
    saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]["type"]])
    os.makedirs(saving_folder, exist_ok=True)
    for problem in problem_set:
        responses = selfconsistency.sequential_reasoning_path_sampling(
            problem=problem,
            saving_folder=saving_folder,
            solving=partial(vanilla_solving, model=model, max_tokens=None),
        )
        results = selfconsistency.vanilla_voting(responses["responses"], problem["solution"])
        print(results["success_vote"], results["votes"])


def tool_voting_one_category(model, problem_set, saving_folder, n=2, n_per_time=1):
    selfconsistency = SelfConsistency(n=n, n_per_time=n_per_time)
    toolsolver = MathSolver(model="gpt-4", tool="both", max_round=10)

    saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]["type"]])
    os.makedirs(saving_folder, exist_ok=True)
    for problem in problem_set:
        responses = selfconsistency.sequential_reasoning_path_sampling(
            problem=problem,
            saving_folder=saving_folder,
            solving=toolsolver.make_conversation,
        )
        results = selfconsistency.vanilla_voting(responses["responses"], problem["solution"])
        print(results["success_vote"], results["votes"])


def parse_args():
    parser = argparse.ArgumentParser(description="Math Solver")
    parser.add_argument("--prompt_type", "-p", dest="prompt_type", help="prompt type", default="select", type=str)
    parser.add_argument("--max_round", dest="max_round", help="max round", default=15, type=int)
    parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./autotools", type=str)
    parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache", help="cache folder")
    parser.add_argument("--samples_per_category", help="samples per category", default=20, type=int)
    parser.add_argument("--test_run", help="test run", action="store_true")

    # not used
    parser.add_argument("--n", dest="n", help="number of samples", default=1, type=int)
    parser.add_argument("--voting", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=41, cache_path=args.cache_folder)
    args.folder = args.folder + "_" + args.prompt_type

    model = "gpt-4"
    problem_sets = load_level5_math_each_category(samples_per_category=args.samples_per_category)
    if args.test_run:
        problem_sets = load_level5_math_each_category(samples_per_category=1)
        print("Take out 1 problem from each category for test run.")

    if not args.voting:
        solver = MathSolver(model=model, prompt_type=args.prompt_type, max_round=args.max_round)

        for problem_set in problem_sets:
            for i in range(len(problem_set)):
                problem_set[i]["problem_id"] = str(i)  # assign problem id

            solver.solve_one_category(problem_set, saving_folder=args.folder)

    else:
        print("Voting is not supported yet.")
        pass

    # problem_sets = load_level5_math_each_category()
    # for problem_set in problem_sets:
    #     for i in range(len(problem_set)):
    #         problem_set[i]['problem_id'] = str(i)

    #     print('Take out 2 problems from each category for testing.')
    #     problem_set = problem_set[:1] # test with only 2 problems first
    #     # vanilla_voting_one_category(model, problem_set, saving_folder='./voting')
    #     break


if __name__ == "__main__":
    main()
