from .agent import Agent
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL
from flaml import oai
from flaml.autogen.math_utils import get_answer, remove_asy_sections
import copy
from openai.error import InvalidRequestError, RateLimitError, Timeout
from ._query_handler import QueryHandler


class MathAgent(Agent):
    """Solve a math problem.
    Most of the code is adopted from the math_solver.py file in Yiran's PR:
    https://github.com/microsoft/FLAML/blob/ac11d2a7bb91f0f210ce0c67ec7b628d967e27b5/flaml/autogen/math/math_solver.py
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a math agent.
    You need to solve a math problem carefully.
    when you get the answer, put the answer in \\boxed{}.
    """

    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,  # default model is gpt-4
    }

    PROMPTS = {
        "v0twostage": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Choose the best way from the two cases to solve the problem and be flexible to switch to another way if necessary.
Case 1: If the problem can be solved with python code directly, you can write a program to solve it.
Case 2: Otherwise, please solve it by yourself directly. You can use python code or Wolfram to help you when necessary (for calculations and equations, etc).
Whenenver you have a query, please follow the query requirements below. I will help you run the query and give you results.
Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1. try to use fractions/radical forms instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # v3python only
        "v3python": """Let's use python to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.
Query requirements:
When you write python code, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "python",
"query": "Your code here."
}
First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above, and I will help you run it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
        # v3.3 select
        "v3.3select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.
Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulations (such as simplifying expressions).
First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use tools to check your calculations when necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above, and I will help you run it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
        # v3.2select 1. change case to mode, change mode 1, change mode 3
        "v3.2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three modes to solve the problem, choose the best mode to solve the problem and be flexible to switch between modes if necessary.
Query requirements:
You are provided with python code and Wolfram alpha, please choose the most suitable tool for each query.
You must put the query in json format (otherwise it will not be parsed correctly):
{"tool":"",# select the best tool from "python" or "wolfram".
"query":"" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
First state the key idea to solve the problem. You may choose from three modes to solve the problem:
Mode 1: If the problem is mostly reasoning or only involve simple calculations, you can solve it by yourself directly. After you get the answer, you can use tools to check your answer if necessary.
Mode 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements. and I will help you run it.
Mode 3: If the problem cannot be handled with the above two modes, please follow this process:
1. Output one step (do not over divide the steps). Take out any queries that can be asked with the tools (for example, any calculations or equations that can be calculated) and follow the query requirements above.
2. Wait for me to give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning, or choose a different tool.
After all the queries are completed and you get the answer, put the answer in \\boxed{}.
""",
        # v3.1select
        "v3.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.
Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
First state the key idea to solve the problem and which way you would choose to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem is mostly reasoning and doesn't involve many calculations or symbol manipulations, you can solve it by yourself directly. You can use tools to check your answer if necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above. and I will help you execute it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Output one step. (do not over divide the steps)
2. Take out any queries that can be asked with the tools (for example, any calculations or equations that can be calculated) and format your query following the query requirements above.
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning or choose a different tool.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # v3select
        "v3select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.
Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulations (such as simplifying expressions).
First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem is mostly reasoning and doesn't involve many calculations or symbol manipulations, you can solve it by yourself directly. If you suspect the result might be wrong, or you can use tools to check it.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above. and I will help you execute it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query.
2. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # "v2.1select" :
        "v2.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query. In particular, if you think you can use one query to aggregate all steps to solve the problem, please do so.
Please follow the query requirements below, otherwise it will not be recognized:
    - Select the most suitable tool for the query.
    - Query python: put python code in ```python ... ```. You must 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
    - Query wolfram: put query ``wolfram ... ```. Note: Wolfram might be more suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
3. There should be one or more queries waiting to be executed. I will take the queries and give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # v1.2select
        "v1.2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query. In particular, if you think you can use one query to aggregate all steps to solve the problem, please do so.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
2. There should be one or more queries waiting to be executed. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # v1.1select
        "v1.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Do not overdivide the steps, and try to use python or wolfram to help you with one or more steps. If you think the problem can be solved with one query, please do so.
You must put the python code or wolfram query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the most suitable tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
2. There should be one or more queries waiting to be executed. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
        # v2select  Try to use python or wolfram to help you with as many steps as possible. Choose the best tool for each task.
        "v2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Choose the best tool to be used.
Follow this format:
    - When query python. put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # nostep
        "nostep": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Try to use the tools to help you solve the problem. In particular, you can write a python program or wolfram query to solve the problem in one step if possible. Please use json format:
{ "tool" : "", #  select the best tool from "python" or "wolfram".
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # v1select *** select *** good for user
        "v1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Put the query in json:
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # *** select *** good for both system and user
        "select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.
First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
Please format the query in json:
{ "tool" : "", # "python" or "wolfram"
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use 'print' function for the output.
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # use python
        "python": """Let's use python code to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.
First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated). When you are querying python, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use 'print' function for the output.
Please format the query in json:
{ "tool" : "python",
"query": "" # your code here.
}
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # use wolfram
        "wolfram": """Let's use Wolfram Alpha to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.
First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through Wolfram Alpha (for example, any calculations or equations that can be calculated).
Please format the query in json:
{ "tool" : "wolfram",
"query": "" # your query here. Please use wolfram language.
}
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
        # v1both
        "v1both": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.
First state the key idea to solve the problem. Then follow the process:
1. Output one step. (do not overdivide the steps)
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated).
You can query both tools for each task to cross-check the results. If you don't have query for one tool, just leave it blank.
Please format the query in json:
{ "python": "", # your python code.
"wolfram": "" # your Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\t'). 3. use 'print' function for the output.
4. Wait for me to give the results.
5. Continue to next step if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
Finally, when you get the answer, put the answer in \\boxed{}.
""",
        "v2refine": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Choose the best tool to be used.
Follow this format:
    - When query python, put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get to the answer, please check the problem conditions to validate your answer. Correct yourself if necessary.
7. Finally, when you believe your answer is correct, put the answer in \\boxed{}.
""",
        # v1refine
        "v1refine": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Put the query in json:
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get to the answer, please check the problem conditions to validate your answer. Correct yourself if necessary.
7. Finally, when you believe your answer is correct, put the answer in \\boxed{}.
""",
        # v1nostep
        "v1nostep": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Keep solving the problem and take out any queries that can be asked through python or Wolfram alpha.
Select the best tool and follow this format:
    - When query python. put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
    }

    def __init__(self, name, system_message=DEFAULT_SYSTEM_MESSAGE, work_dir=None, **config):
        super().__init__(name, system_message)
        self._work_dir = work_dir
        self._config = self.DEFAULT_CONFIG.copy()
        self._config.update(config)
        self._sender_dict = {}

        # TODO: add key word args for the convenience of experiments
        # the following code is adopted from Yiran's PR
        self.max_round = 20
        self.prompt_loaction = "user"  # "system" or "user"
        self.max_invalid_q_per_step = 3
        self.use_cache = True
        self.logger = None  # TODO: add logger
        self.prompt_type = "v3select"
        # if the prompt_location is set to system, then the prompt is put in the system message
        messages = (
            [{"role": "system", "content": self.prompt}]
            if self.prompt_loaction == "system"
            else [{"role": "system", "content": "You are a helpful assistant."}]
        )
        self._config.update(
            {
                "messages": messages,
            }
        )
        self.prompt = MathAgent.PROMPTS[self.prompt_type]

    def receive(self, message, sender):
        problem = {"problem": message}

        file_to_be_saved = "test_math.txt"
        query_handler = QueryHandler()
        # initialize the conversation
        config = copy.deepcopy(self._config)
        problem_prompt = {
            "role": "user",
            "content": self.prompt + "\nProblem: " + remove_asy_sections(problem["problem"]),
        }  # put prompt in user message

        # if the prompt_location is set to system, then the prompt is already put in the system message in __init__,
        # then we only need to put the problem in the user message
        if self.prompt_loaction == "system":
            problem_prompt = {"role": "user", "content": remove_asy_sections(problem["problem"])}
        config["messages"].append(problem_prompt)

        # save a readable conversation in txt file
        def save_message_to_file(message):
            if file_to_be_saved is not None:
                with open(file_to_be_saved, "a") as f:
                    f.write(message)
                    f.flush()

        seperate_line = "\n" + "-" * 40 + "\n"
        save_message_to_file(f'Problem: {self._str_splitter(problem["problem"])}\n {seperate_line}')

        # init parameters
        is_valid_reply = False  # only valid when detect \box
        invalid_q = 0  # for query
        total_cost = 0
        response_with_ans = ""  # save the response with \box to get the answer
        rr = 0  # round
        while rr < self.max_round:
            # 1. get the response from the assistant
            try:
                raw_responses = oai.ChatCompletion.create(None, **config, use_cache=self.use_cache)
            except InvalidRequestError as e:
                # TODO: logging
                save_message_to_file(str(e))
                break
            except (RateLimitError, Timeout):
                print("Rate limit or timeout, retrying...")
                continue
            assert raw_responses != -1, "Error in getting response"
            responses = oai.ChatCompletion.extract_text(raw_responses)
            assert len(responses) == 1, "More than one response"  # right now we only use one response
            # TODO: logging
            save_message_to_file(f"assistant: {self._str_splitter(responses[0])}{seperate_line}")
            total_cost += oai.ChatCompletion.cost(raw_responses)
            config["messages"].append({"role": "assistant", "content": responses[0]})
            if get_answer(responses[0]) is not None and get_answer(responses[0]) != "":
                # if the assistant gives a valid reply, stop the conversation
                is_valid_reply = True
                response_with_ans = responses[0]
                break

            # 2. handle the response and get the query
            query_response, is_query_sucess = query_handler.handle_query(responses[0])
            if len(query_response) > 2000:
                # prevent long response by string length, 2000 chars -> around 500-1000 tokens
                save_message_to_file(f"****: Replacing {query_response} ****\n")
                query_response = "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
                is_query_sucess = False
            config["messages"].append({"role": "user", "content": query_response})

            invalid_q = 0 if is_query_sucess else invalid_q + 1
            if invalid_q >= self.max_invalid_q_per_step:
                assert config["messages"][-1]["role"] == "user", "The last message should be from user"
                skip_query_str = "Please revisit the problem statement and your reasoning. If you think this step is correct, solve it yourself and continue the next step. Otherwise, correct this step."
                config["messages"][-1]["content"] = skip_query_str
                save_message_to_file(f"****: Replacing {query_response}****\n")
                invalid_q = 0

            save_message_to_file("user: {a}{s}".format(a=config["messages"][-1]["content"], s=seperate_line))
            if "Continue" in query_response:
                rr -= 0.5
            rr += 1
        # save_message_to_file("Solution: " + problem["solution"])

        result = {
            "valid_q_count": query_handler.valid_q_count,  # number of valid queries
            "total_q_count": query_handler.total_q_count,
            "is_valid_reply": is_valid_reply,  # whether the assistant can give a valid reply
            "response_with_ans": response_with_ans,  # string instead of list
            "messages": config["messages"],
            "round": min(rr + 1, self.max_round),
            "cost": total_cost,
        }
        self._send(message=result, recipient=sender)

    @staticmethod
    def _str_splitter(string, length=130):
        """
        Add '\n' every 'length' characters to make the output more readable.
        If at 'length' there is a word, add '\n' before the word.
        Args:
            string (str): The input string to be processed.
            length (int): The maximum number of characters in a line before adding a newline.
        Returns:
            str: The processed string with newlines added.
        """

        words = string.split(" ")
        current_line = []
        current_length = 0
        result = []

        for word in words:
            if current_length + len(word) + len(current_line) > length:
                result.append(" ".join(current_line))
                current_line = []
                current_length = 0

            current_line.append(word)
            current_length += len(word)

        if current_line:
            result.append(" ".join(current_line))

        return "\n".join(result)
