import openai
import os
from MathSolver import MathSolver
from flaml.autogen.math_utils  import math_type_mapping
import datasets
from flaml import oai

math_type_mapping = {
    "Algebra" : 'algebra',
    "Counting & Probability" : 'counting_and_probability',
    "Geometry" : 'geometry',
    "Intermediate Algebra" : 'intermediate_algebra',
    "Number Theory" : 'number_theory',
    "Prealgebra" : 'prealgebra',
    "Precalculus" : 'precalculus',
    }

def load_level5_math_each_category():
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
        sep_cate.append([
        test_data[x] for x in range(len(test_data)) if test_data[x]["level"] == "Level 5" and test_data[x]["type"] == category
    ])
    
    return sep_cate





def main():
    openai.api_key = "Your key here"
    
    if "WOLFRAM_ALPHA_APPID" not in os.environ:
        os.environ["WOLFRAM_ALPHA_APPID"] = "W4W68H-R9KA5AU3XX" # yiran's appid
    
    # request timeout = 10 minutes
    oai.ChatCompletion.request_timeout = 60*10 
    
    # make solver
    solver = MathSolver('gpt-4', max_tokens=4096, max_round=10)
    
    problem_sets = load_level5_math_each_category()

    for problem_set in problem_sets:
        print('Take out 2 problems from each category for testing.')
        problem_set = problem_set[:2] # test with only 2 problems first
        
        solver.solve_one_category(problem_set, saving_folder='./autotools')


    # self.saving_path = f'./results/math_{model}_{max_tokens}_{max_round}_{n}'

if __name__ == '__main__':
    main()