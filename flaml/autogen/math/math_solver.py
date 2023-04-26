from flaml.autogen.math.query_handler import QueryHandler
from flaml.autogen.math_utils import eval_math_responses, get_answer
from flaml import oai
import os
import json
import re
import copy
from openai.error import InvalidRequestError, RateLimitError, Timeout
from utils import write_json, remove_asy_sections, math_type_mapping, mylogger


def main_solve_with_tools(args, logger, problem_sets):
    solver = MathSolver(
        model=args.model,
        prompt_type=args.prompt_type,
        max_round=args.max_round,
        temperature=args.temperature,
        prompt_location=args.prompt_location,
        logger=logger,
    )
    with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
        f.write(solver.prompt)

    for problem_set in problem_sets:
        for i in range(len(problem_set)):
            problem_set[i]["problem_id"] = str(i)  # assign problem id

        solver.solve_one_category(problem_set, saving_folder=args.folder)
        os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    logger.log("****************************\n\n\n\n\n", verbose=False)
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)


PROMPTS = {
    "select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
Please format the query in json:
{ "tool" : "", # "python" or "wolfram"
"query": "", # your query here, either python code or Wolfram query.
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
"query": "", # your code here.
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
"query": "", # your query here. Please use wolfram language.
}
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
}


class MathSolver:
    def __init__(
        self,
        model,
        prompt_type="select",
        prompt_location="user",
        max_round=10,
        max_invalid_q_per_step=3,
        n=1,
        temperature=1,
        logger=None,
        use_cache=True,
    ):
        self.max_round = max_round
        if prompt_type not in PROMPTS:
            raise ValueError(f"Tool {prompt_type} not supported, choose from {PROMPTS.keys()}")

        self.prompt_type = prompt_type
        self.prompt_loaction = prompt_location
        self.prompt = PROMPTS[prompt_type]
        messages = (
            [{"role": "system", "content": self.prompt}]
            if prompt_location == "system"
            else [{"role": "system", "content": "You are a helpful assistant."}]
        )
        self.deafult_config = {
            "model": model,
            "messages": messages,
            "n": n,  # n should be 1 for now
            # 'temperature' : 1,
        }

        self.max_invalid_q_per_step = max_invalid_q_per_step
        self.use_cache = use_cache
        self.logger = logger

    def make_conversation(self, problem, n=1, file_to_be_saved=None):
        # initialize the query handler
        query_handler = QueryHandler()

        # initialize the conversation
        config = copy.deepcopy(self.deafult_config)
        problem_prompt = {
            "role": "user",
            "content": self.prompt + "\nProblem: " + remove_asy_sections(problem["problem"]),
        }  # put prompt in user message
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
        save_message_to_file(f'Problem: {self.str_splitter(problem["problem"])}\n {seperate_line}')

        # init parameters
        is_valid_reply = False  # only valid when detect \box
        invalid_q = 0  # for query
        total_cost = 0
        response_with_ans = ""  # save the response with \box to get the answer
        for rr in range(self.max_round):
            # 1. get the response from the assistant
            try:
                raw_responses = oai.ChatCompletion.create(None, **config, use_cache=self.use_cache)
            except (InvalidRequestError, RateLimitError, Timeout) as e:
                print(problem["type"], problem["problem_id"], str(e))
                save_message_to_file(str(e))
                break
            assert raw_responses != -1, "Error in getting response"
            responses = oai.ChatCompletion.extract_text(raw_responses)
            assert len(responses) == 1, "More than one response"  # right now we only use one response
            save_message_to_file(f"assistant: {self.str_splitter(responses[0])}{seperate_line}")
            # token_used = raw_responses['usage']['total_tokens']
            total_cost += oai.ChatCompletion.cost(self.deafult_config["model"], raw_responses)
            config["messages"].append({"role": "assistant", "content": responses[0]})
            if get_answer(responses[0]) is not None:
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
        save_message_to_file("Solution: " + problem["solution"])

        return {
            "valid_q_count": query_handler.valid_q_count,  # number of valid queries
            "total_q_count": query_handler.total_q_count,
            "is_valid_reply": is_valid_reply,  # whether the assistant can give a valid reply
            "response_with_ans": response_with_ans,  # string instead of list
            "messages": config["messages"],
            "round": min(rr + 1, self.max_round),
            "cost": total_cost,
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
        if not self.logger:
            self.logger = mylogger(os.path.join(saving_folder, "log.txt"))

        # assume all problems are of the same type: TODO: ensure this assumption
        saving_folder = os.path.join(saving_folder, math_type_mapping[problem_set[0]["type"]])
        # mkdir if not exist
        os.makedirs(saving_folder, exist_ok=True)

        # from the saving folder load solved problems
        done_problems = set([int(f.split(".")[0]) for f in os.listdir(saving_folder) if "json" in f])

        correct_counts = 0
        self.logger.log("id : is_correct $ ans $ correct_ans | is_valid $ round $ accum acc")
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            # 1. if problem already solved, continue
            if int(problem["problem_id"]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                correct_counts += problem["is_correct"]
                self.logger.log(
                    f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} | {problem["is_valid_reply"]} $ {problem["round"]} $ {correct_counts}/{count+1} (from previous run)'
                )
                continue

            # 2. solve the problem
            result = self.make_conversation(
                problem, file_to_be_saved=os.path.join(saving_folder, problem["problem_id"] + ".txt")
            )
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
                    "valid_q_count": result["valid_q_count"],  # total number of valid queries
                    "total_q_count": result["total_q_count"],  # total number of queries
                    "cost": result["cost"],  # total cost of the conversation
                    "messages": result["messages"],  # the conversation
                }
            )
            write_json(problem, problem_path)

            # 4. continue to next problem
            correct_counts += problem["is_correct"]
            self.logger.log(
                f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} | {problem["is_valid_reply"]} $ {problem["round"]} $ {correct_counts}/{count+1}'
            )

        tp = problem_set[0]["type"]
        self.logger.log(f"{tp} Accuracy: {correct_counts}/{len(problem_set)} = {correct_counts/len(problem_set)}")
        self.logger.log("------------------------------------------------------------\n", verbose=True)
