from QueryHandler import QueryHandler
from flaml.autogen.math_utils import eval_math_responses, remove_boxed, last_boxed_only_string, nestmkdir, write_json, remove_asy_sections, math_type_mapping
from flaml import oai
import os
import json
import re
import copy


PROMPT = """
Let's use two tools (python code and Wolfram Alpha) to solve this problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram Alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used. When you are querying python, you should: 1.use tab('\\t') for indentation. 2. use 'print' function for the output. 3. always output fractions instead of decimal.
Please format the query in json: 
{ "tool" : "", # "python" or "wolfram"
"query": "", # your query here, either python code or Wolfram query.
} 
4. Wait for me to give the results.
5. Give a new query if the results are invalid or unexpected.
6. When you get the answer, put the answer in \\box{}.

Problem: 
"""


class MathSolver:
    def __init__(self, model, max_tokens, max_round=10, n=1, use_cache=True, cache_folder='.cache'):
        self.max_round = max_round

        self.deafult_config = {
            'model': model,
            "max_tokens": max_tokens,
            'messages' : [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
            'n' : n, # n should be 1 for now
        }
        # set oai cache
        self.cache_folder = cache_folder
        self.use_cache = use_cache
        oai.ChatCompletion.set_cache(seed=41, cache_path=self.cache_folder)
    

    def make_conversation(self, problem, saving_folder):
        query_hanlder = QueryHandler()

        # initialize the conversation
        config = copy.deepcopy(self.deafult_config)
        config['messages'].append({"role": "user", "content": PROMPT + remove_asy_sections(problem['problem'])})
        
        # save a readable conversation in txt file
        convesation_saver = open(os.path.join(saving_folder, problem['problem_id'] + '.txt'), 'a') 
        seperate_line = '\n'+ '-'* 40 + '\n'
        convesation_saver.write(f'Problem: {self.str_splitter(problem["problem"])}\n\n {seperate_line}')
        
        # init parameters
        is_valid_reply = False # only valid when detect \box
        consecutive_fail = False # for query
        token_used, total_cost = 0, 0 
        response_with_ans = "" # save the response with \box to get the answer
        for _ in range(self.max_round):
            # 1. get the response from the assistant
            raw_responses = oai.ChatCompletion.create(None, **config, use_cache=self.use_cache)
            if raw_responses == -1:
                break # catch the error when no valid reply
            responses = [r["message"]["content"].rstrip() for r in raw_responses["choices"]]
            convesation_saver.write(f'assistant: {self.str_splitter(responses[0])}{seperate_line}')
            token_used = raw_responses['usage']['total_tokens']
            total_cost += oai.ChatCompletion.cost(self.deafult_config['model'], raw_responses)
            config['messages'].append({"role": "assistant", "content": responses[0]}) # append the response to the conversation
            if '\\box' in responses[0]:
                # if the assistant gives a valid reply, stop the conversation
                is_valid_reply = True
                response_with_ans = responses[0]
                break
            elif token_used > 8192 - config['max_tokens']:
                # if the assistant uses too many tokens, stop the conversation. max prompt token + max response token allowed = 8192
                break
            assert len(responses) == 1, 'More than one response' # right now we only use one response

            # 2. handle the response and get the query 
            query_response, is_query_sucess = query_hanlder.handle_query(responses[0]) 
            if len(query_response) > 2000:
                # prevent long response by string length, 2000 chars -> around 500-1000 tokens
                query_response = 'Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query.'
                is_query_sucess = False
            config['messages'].append({"role": "user", "content": query_response})
            if not is_query_sucess:
                if consecutive_fail:
                    # if the query is not valid and last query is also failed, replace the last message with a skip query message
                    assert config['messages'][-1]['role'] == 'user', 'The last message should be from user'
                    skip_query_str = 'Please solve this step yourself and do not use the tools. Then start the next step and give new queries if needed.'
                    config['messages'][-1]['content'] = skip_query_str
                    convesation_saver.write(f'****: Replacing {query_response}****\n')
                    consecutive_fail = False
                else:
                    consecutive_fail = True
            convesation_saver.write('user: {a}{s}'.format(a=config['messages'][-1]['content'], s=seperate_line))
            convesation_saver.flush()
        
        convesation_saver.write('Solution: ' + problem['solution'])
        convesation_saver.close()
        return {
            'valid_q_count' : query_hanlder.valid_q_count, # number of valid queries
            'total_q_count' : query_hanlder.total_q_count,
            'is_valid_reply': is_valid_reply, # whether the assistant can give a valid reply
            'response_with_ans': response_with_ans,
            'messages': config['messages'],
            'round' : len(config['messages'])//2 + 1,
            'cost' : total_cost,
        }


    def str_splitter(self, string, length=130):
        """
        Add '\n' every 'length' characters to make the output more readable.
        If at 'length' there is a word, add '\n' before the word.

        Args:
            string (str): The input string to be processed.
            length (int): The maximum number of characters in a line before adding a newline.

        Returns:
            str: The processed string with newlines added.
        """

        words = string.split(' ')
        current_line = []
        current_length = 0
        result = []

        for word in words:
            if current_length + len(word) + len(current_line) > length:
                result.append(' '.join(current_line))
                current_line = []
                current_length = 0

            current_line.append(word)
            current_length += len(word)

        if current_line:
            result.append(' '.join(current_line))

        return '\n'.join(result)


    def solve_one_category(self, problem_set, saving_folder):
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

        # assume all problems are of the same type: TODO: ensure this assumption
        saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]['type']]) 
        # assign temporary problem_id
        for i in range(len(problem_set)):
            problem_set[i]['problem_id'] = str(i)
        # mkdir if not exist
        nestmkdir(saving_folder, verbose=True) 

        # from the saving folder load solved problems
        done_problems = set([int(f.split('.')[0]) for f in os.listdir(saving_folder) if 'json' in f])

        correct_counts = 0
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem['problem_id'] + '.json')

            # 1. if problem already solved, continue
            if int(problem['problem_id']) in done_problems:
                problem = json.load(open(problem_path, 'r'))
                correct_counts += problem['is_correct']
                print(f'{count}: {correct_counts}/{count+1} successes. valid response: {problem["is_valid_reply"]}, Correct: {problem["is_correct"]}, {problem["round"]} rounds. (This problem is loaded from previous run)')
                continue
            
            # 2. solve the problem
            result = self.make_conversation(problem, saving_folder)
            metrics = eval_math_responses([result['response_with_ans']], problem['solution'])

            # 3. save the result
            correct_ans = remove_boxed(last_boxed_only_string(problem['solution']))
            problem.update({
                'is_valid_reply': result['is_valid_reply'],
                'is_correct': bool(metrics['success_vote']),
                'correct_ans': correct_ans,
                'voted_answer': remove_boxed(last_boxed_only_string(metrics['voted_answer'])) ,
                'round': result['round'],
                'valid_q_count': result['valid_q_count'], # total number of valid queries
                'total_q_count': result['total_q_count'], # total number of queries
                'cost': result['cost'], # total cost of the conversation
                'messages': result['messages'], # the conversation
            })
            write_json(problem, problem_path)

            # 4. continue to next problem
            correct_counts += problem['is_correct']
            print(f'{problem["problem_id"]} Is Valid: {problem["is_valid_reply"]}, Is Correct: {bool(problem["is_correct"])}, Conversation Round: {problem["round"]}, Accum Sucesses: {correct_counts}/{count+1}')
            
        tp = problem_set[0]['type']
        print(f'{tp} correct rate: {correct_counts}/{len(problem_set)} = {correct_counts/len(problem_set)}')

    
