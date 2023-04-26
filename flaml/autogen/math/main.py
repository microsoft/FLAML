import openai
import os
from flaml import oai
from flaml.autogen.math.math_voting import SelfConsistency
from flaml.autogen.math.math_solver import main_solve_with_tools
from utils import load_level5_math_each_category, mylogger, parse_args


def main():
    # 1. keys here

    # 2. args, settings and logger
    args = parse_args()
    args.model = "gpt-4"
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=41, cache_path=args.cache_folder)
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    # 3. load math dataset
    problem_sets = load_level5_math_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )
    if args.test_run:
        problem_sets = load_level5_math_each_category(samples_per_category=1, category_to_load=args.categories)
        logger.log("Take out 1 problem from each category for test run.")

    # 4. solve
    if not args.voting:
        main_solve_with_tools(args=args, problem_sets=problem_sets, logger=logger)
    else:
        logger.log("Voting is not supported yet.")
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
