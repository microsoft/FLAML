from QueryHandler import QueryHandler
from flaml.autogen.math_utils import eval_math_responses, remove_boxed, last_boxed_only_string, write_json, remove_asy_sections, math_type_mapping
from flaml import oai
import os
import json
import re
import copy


class SelfConsistency:
    def __init__(self, n=10, n_per_time=5, cache_folder='.cache'):
        self.n = n
        self.n_per_time = n_per_time
        self.start_seed = 41
        self.cache_folder = cache_folder

    def vanilla_voting(self, accum_responses, solution):
        if type(accum_responses[0]) == dict:
            accum_responses = [r['response_with_ans'] for r in accum_responses]
        return eval_math_responses(accum_responses, solution)

    def early_stop_voting(self, accum_responses):
        if type(accum_responses[0]) == dict:
            accum_responses = [r['response_with_ans'] for r in accum_responses]
        pass

    def sequential_reasoning_path_sampling(self, problem, saving_folder, solving):
        """

        Args:
            problem (dict): problem dict
            saving_folder (str): saving folder
            solver (function): solver function, either MathSolver.make_conversation or vanilla prompt

        return from vanilla prompt: {
            'responses': responses,
            'cost': oai.ChatCompletion.cost(model, raw_responses),
            'prompt_cost': oai.ChatCompletion.price1K(model, 0) * raw_responses["usage"]["prompt_tokens"] / 1000
        }

        return from math solver: {
            'valid_q_count' : query_handler.valid_q_count, # number of valid queries
            'total_q_count' : query_handler.total_q_count,
            'is_valid_reply': is_valid_reply, # whether the assistant can give a valid reply
            'response_with_ans': response_with_ans,
            'ans': ans,
            'messages': config['messages'],
            'round' : len(config['messages'])//2 + 1,
            'cost' : total_cost,
        }

        """
        accum_responses = [] # can be a list of dicts (for mathsolver) or list of strings
        accum_cost = 0
        file = os.path.join(saving_folder, 'responses_' + problem['problem_id'] + '.json')
        if os.path.exists(file):
            accum_responses = json.load(open(file, 'r'))['responses']
            accum_cost = json.load(open(file, 'r'))['cost']

        query_count = len(accum_responses)
        tmp_n = self.n_per_time
        while query_count < self.n:
            oai.ChatCompletion.set_cache(seed=self.start_seed + query_count, cache_path=self.cache_folder)
            tmp_n = min(tmp_n, self.n - self.n_per_time)

            responses = solving(problem=problem, n=tmp_n)

            if 'responses' in responses.keys(): 
                accum_responses.extend(responses['responses'])
                if query_count != 0:
                    accum_cost -= responses['prompt_cost'] # if not the first round, deduct the prompt cost
            else: # the response comes from math solver, single response
                accum_responses.extend([responses])
            
            accum_cost += responses['cost']
            write_json({'cost': accum_cost,
                        'true_ans': remove_boxed(last_boxed_only_string(problem['solution'])), 
                        'answers': [remove_boxed(last_boxed_only_string(r)) for r in accum_responses],
                        'responses': accum_responses, 
                        },    
                        file) # save the responses each time

            query_count += tmp_n

        # TODO: cost calculation: should prompt for each round being counted?
        return {'responses': accum_responses, 'cost': accum_cost}



    
