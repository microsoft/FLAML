import openai
import os
from MathSolver import MathSolver
from flaml.autogen.math_utils  import math_type_mapping
import datasets
from flaml import oai
from functools import partial
from MathVoting import SelfConsistency
import argparse

math_type_mapping = {
    "Algebra" : 'algebra',
    "Counting & Probability" : 'counting_and_probability',
    "Geometry" : 'geometry',
    "Intermediate Algebra" : 'intermediate_algebra',
    "Number Theory" : 'number_theory',
    "Prealgebra" : 'prealgebra',
    "Precalculus" : 'precalculus',
    }

def load_level5_math_each_category(samples_per_category=20):
    """
    Load level 5 math problems from the competition dataset. 
    Returns:
        A list of list of problems. Each list of problems is of the same category.
    """
    seed = 41
    data = datasets.load_dataset("competition_math")
    test_data = data["test"].shuffle(seed=seed)
    sep_cate = []
    for category in math_type_mapping.keys():
        tmp = [test_data[x] for x in range(len(test_data)) if test_data[x]["level"] == "Level 5" and test_data[x]["type"] == category]
        sep_cate.append(tmp[:samples_per_category])
    
    return sep_cate

def vanilla_solving(model, problem, n, max_tokens=None):
    """Solving a problem directly.
    """
    config = {
        'model' : model,
        'n' : n,
        'prompt' : '{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}.',
    }
    if max_tokens is not None:
        config['max_tokens'] = max_tokens
    context = {
        'problem' : problem['problem'],
    }
    raw_responses = oai.ChatCompletion.create(context, **config, use_cache=True)
    
    return {
        'responses': oai.ChatCompletion.extract_text(raw_responses),
        'cost': oai.ChatCompletion.cost(model, raw_responses),
        'prompt_cost': oai.ChatCompletion.price1K[model] * raw_responses["usage"]["prompt_tokens"] / 1000
    }

def vanilla_voting_one_category(model, problem_set, saving_folder, n=10, n_per_time=3):
    """Solve one category of problems directly.

    """
    selfconsistency = SelfConsistency(n=n, n_per_time=n_per_time)
    saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]['type']]) 
    os.makedirs(saving_folder, exist_ok=True)
    for problem in problem_set:
        responses = selfconsistency.sequential_reasoning_path_sampling(
            problem=problem,
            saving_folder=saving_folder,
            solving=partial(vanilla_solving, model=model, max_tokens=None),
        )
        results = selfconsistency.vanilla_voting(responses["responses"], problem['solution'])
        print(results['success_vote'], results['votes'])

def tool_voting_one_category(model, problem_set, saving_folder, n=2, n_per_time=1):
    selfconsistency = SelfConsistency(n=n, n_per_time=n_per_time) 
    toolsolver = MathSolver(model='gpt-4', tool='both', max_round=10)

    saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]['type']])
    os.makedirs(saving_folder, exist_ok=True)
    for problem in problem_set:
        responses = selfconsistency.sequential_reasoning_path_sampling(
            problem=problem,
            saving_folder=saving_folder,
            solving=toolsolver.make_conversation,
        )
        results = selfconsistency.vanilla_voting(responses["responses"], problem['solution'])
        print(results['success_vote'], results['votes'])


def parse_args():
    parser = argparse.ArgumentParser(description='Math Solver')
    parser.add_argument('--prompt_type', '-p',  dest='prompt_type', help='prompt type', default='select', type=str)
    parser.add_argument('--max_round', dest='max_round', help='max round', default=10, type=int)
    parser.add_argument('--folder', '-f', dest='folder', help='saving folder', default='./autotools', type=str)
    parser.add_argument('--cache_folder', '-c', dest='cache_folder', default='.cache', help='cache folder')
    parser.add_argument('--test_run', help='test run', action='store_true')
    
    # not used
    parser.add_argument('--n', dest='n', help='number of samples', default=1, type=int)
    parser.add_argument('--voting', action='store_true')
    return parser.parse_args()



def main():
    args = parse_args()
    oai.ChatCompletion.request_timeout = 60*10 # 10 minutes
    oai.ChatCompletion.set_cache(seed=41, cache_path=args.cache_folder)
    

    model = 'gpt-4'
    problem_sets = load_level5_math_each_category(samples_per_category=20)
    if args.test_run:
        problem_sets = load_level5_math_each_category(samples_per_category=1)
        print('Take out 1 problem from each category for test run.')
    

    if not args.voting:
        solver = MathSolver(model=model, prompt_type=args.prompt_type, max_round=args.max_round)
        
        for problem_set in problem_sets:
            for i in range(len(problem_set)):
                problem_set[i]['problem_id'] = str(i) # assign problem id 
            
            
            solver.solve_one_category(problem_set, saving_folder=args.folder)

    else:
        pass


    # problem_sets = load_level5_math_each_category()
    # for problem_set in problem_sets:
    #     for i in range(len(problem_set)):
    #         problem_set[i]['problem_id'] = str(i)
    
    #     print('Take out 2 problems from each category for testing.')
    #     problem_set = problem_set[:1] # test with only 2 problems first
    #     # vanilla_voting_one_category(model, problem_set, saving_folder='./voting')
    #     break

if __name__ == '__main__':
    main()