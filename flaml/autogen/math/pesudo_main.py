import os
from flaml import oai
from math_voting import SelfConsistency
from math_solver import MathSolver
import argparse
from utils import mylogger, load_level5_math_test_each_category, load_fixed, random_sample_MATH


def parse_args():
    parser = argparse.ArgumentParser(description="Math Solver")
    parser.add_argument("--prompt_type", "-ptype", dest="prompt_type", help="prompt type", default="select", type=str)
    parser.add_argument("--prompt_location", dest="prompt_location", help="prompt location", default="user", type=str)
    parser.add_argument("--max_round", dest="max_round", help="max round", default=15, type=int)
    parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./autotools", type=str)
    parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache", help="cache folder")
    parser.add_argument("--samples_per_category", help="samples per category", default=20, type=int)
    parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
    parser.add_argument("--test_run", help="test run", action="store_true")
    parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")
    parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--sample_all", help="samples per category", default=0, type=int)
    parser.add_argument("-systype", dest="systype", help="system type", default="s0", type=str)

    # not used
    parser.add_argument("--n", dest="n", help="number of samples", default=1, type=int)
    parser.add_argument("--voting", action="store_true")
    args = parser.parse_args()
    args.folder = (
        args.folder + args.systype + "_" + args.prompt_location + "_" + args.prompt_type + "_t" + str(args.temperature)
    )
    if args.seed != 41:
        args.folder = args.folder + "_seed" + str(args.seed)
    if args.refine:
        args.folder = args.folder.replace("_t" + str(args.temperature), "_refine_t" + str(args.temperature))
    if args.sample_all != 0:
        args.folder += "_random_sample"
    os.makedirs(args.folder, exist_ok=True)
    return args


def pseudo_main(config_list):
    # 2. args, settings and logger
    args = parse_args()
    args.model = "gpt-4"
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=args.seed, cache_path_root=args.cache_folder)
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    # 3. load math dataset
    problem_sets = load_level5_math_test_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )
    if args.test_run:
        problem_sets = load_level5_math_test_each_category(samples_per_category=1, category_to_load=args.categories)
        logger.log("Take out 1 problem from each category for test run.")

    if args.select:
        problem_sets = load_fixed()

    if args.sample_all != 0:
        problem_sets = random_sample_MATH(args.sample_all)

    print(f"Running {args.folder}")
    # v1
    # selected_samples ={
    #     "Algebra": [0, 1, 2, 4, 10, 11, 13, 14, 17, 18, 19], # number, assume 8 correct 1 wrong (8)
    #     "Counting & Probability": [],
    #     "Geometry": [],
    #     "Intermediate Algebra": [],
    #     "Number Theory": [3, 4, 6, 7, 11, 12, 14, 15, 18], # number, assume 11 correct
    #     "Prealgebra": [],
    #     "Precalculus": [],
    # }

    # v3
    selected_samples = {
        # "Algebra": [9,13,14],  # [8] wrong,  # 8 correct
        # "Algebra": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # [8] wrong,  # 8 correct
        # "Algebra": [1,2,4,13],
        # "Algebra": [18], # [1, 8] wrong, 9-10 out of 10 correct
        # "Algebra": [2, 5, 13],
        # "Counting & Probability": [0,1,8,9], #  0,10,  | 5 correct [2,3,16,18,19], 6 [4,5,13,14,15,17] wrong
        # "Geometry": [],
        # "Algebra": [i for i in range(20)],
        # "Counting & Probability": [i for i in range(20)],
        "Intermediate Algebra": [0, 3, 6, 8, 9, 10, 11, 13, 15, 16, 17],
        # "Number Theory": [i for i in range(20)],
        # "Prealgebra": [i for i in range(20)],
        "Precalculus": [1, 14, 15, 18],
        # "Number Theory": [0, 2, 4,6,7,8,10,11,12,13,14,15,17,18],  # [3] always wrong      [1, 5, 9, 16, 19] always right
        # "Prealgebra": [3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17], # [0,7,16] always wrong, [1,2,5,6,10,18,19] always right
    }

    # 4. solve
    if not args.voting:
        solver = MathSolver(
            config_list=config_list,
            model=args.model,
            prompt_type=args.prompt_type,
            sys_type=args.systype,
            max_round=args.max_round,
            temperature=args.temperature,
            prompt_location=args.prompt_location,
            logger=logger,
            refine=args.refine,
        )
        with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
            f.write(solver.prompt)

        for problem_set in problem_sets:
            for i in range(len(problem_set)):
                problem_set[i]["problem_id"] = str(i)  # assign problem id
            if args.select:
                if problem_set[0]["type"] in selected_samples and len(selected_samples[problem_set[0]["type"]]) > 0:
                    problem_set = [problem_set[i] for i in selected_samples[problem_set[0]["type"]]]
                    print(problem_set[0]["type"], selected_samples[problem_set[0]["type"]])
                else:
                    continue
            solver.solve_one_category(problem_set, saving_folder=args.folder)
            # os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

        logger.log("****************************\n\n\n\n\n", verbose=False)
        os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    else:
        logger.log("Voting is not supported yet.")
        pass

    # problem_sets = load_level5_math_test_each_category()
    # for problem_set in problem_sets:
    #     for i in range(len(problem_set)):
    #         problem_set[i]['problem_id'] = str(i)

    #     print('Take out 2 problems from each category for testing.')
    #     problem_set = problem_set[:1] # test with only 2 problems first
    #     # vanilla_voting_one_category(model, problem_set, saving_folder='./voting')
    #     break
