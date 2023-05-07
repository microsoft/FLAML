from query_handler import QueryHandler
from flaml.autogen.math_utils import eval_math_responses, get_answer
from flaml import oai
import os
import json
import re
import copy
from openai.error import InvalidRequestError, RateLimitError, Timeout
from utils import write_json, remove_asy_sections, math_type_mapping, mylogger
from prompts import PROMPTS


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
        refine=False,
        config_list=None,
    ):
        self.max_round = max_round
        if prompt_type not in PROMPTS:
            raise ValueError(f"Tool {prompt_type} not supported, choose from {PROMPTS.keys()}")

        self.prompt_type = prompt_type
        self.prompt_loaction = prompt_location
        self.prompt = PROMPTS[prompt_type]
        self.refine = refine

        # if the prompt_location is set to system, then the prompt is put in the system message
        messages = (
            [{"role": "system", "content": self.prompt}]
            if prompt_location == "system"
            else [
                # {"role": "system", "content": "You are a helpful assistant."}
            ]
        )
        self.deafult_config = {
            "model": model,
            "messages": messages,
            "n": n,  # n should be 1 for now
            "temperature": temperature,
        }

        self.max_invalid_q_per_step = max_invalid_q_per_step
        self.use_cache = use_cache
        self.logger = logger
        self.config_list=config_list

    def make_conversation(self, problem, n=1, file_to_be_saved=None):
        # initialize the query handler
        query_handler = QueryHandler()

        # initialize the conversation
        config = copy.deepcopy(self.deafult_config)
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
        save_message_to_file(f'Problem: {self.str_splitter(problem["problem"])}\n {seperate_line}')

        # for additional refine process
        is_refine_process = False
        response_with_new_ans = ""  # save the corrected answer

        # init parameters
        is_valid_reply = False  # only valid when detect \box
        invalid_q = 0  # for query
        total_cost = 0
        response_with_ans = ""  # save the response with \box to get the answer
        rr = 0  # round
        while rr < self.max_round:
            # 1. get the response from the assistant, handle exceptions
            try:
                raw_responses = oai.ChatCompletion.create(config_list=self.config_list, **config, use_cache=self.use_cache)
            except InvalidRequestError as e:
                print(problem["type"], problem["problem_id"], str(e), flush=True)
                save_message_to_file(str(e))
                break
            except (RateLimitError, Timeout) as e:
                print("Ratelimit or timeout, retrying...", flush=True)
                continue
            assert raw_responses != -1, "Error in getting response"
            responses = oai.ChatCompletion.extract_text(raw_responses)
            assert len(responses) == 1, "More than one response"  # right now we only use one response

            # 2. process response
            save_message_to_file(f"assistant: {self.str_splitter(responses[0])}{seperate_line}")
            # token_used = raw_responses['usage']['total_tokens']
            total_cost += oai.ChatCompletion.cost(raw_responses)
            config["messages"].append({"role": "assistant", "content": responses[0]})
            tmp_msg = ""
            if "[EOF]" in responses[0] or "EOF" in responses[0]:
                _, is_query_exist = query_handler.check_queries(responses[0])
                if not is_query_exist:
                    end_message = "Now that we have solved the problem, please conclude with this sentence: \"Since the problem is asking for ..., the answer is \\boxed{...}.\" Be cautious what the problem is asking and in what format the answer should be, and put that answer in box."
                    config["messages"].append({"role": "user", "content": end_message})
                    save_message_to_file(
                        "user: {a}{s}".format(a=config["messages"][-1]["content"], s=seperate_line)
                    )
                    continue
                tmp_msg = "\nAbove is the returned results. If the problem is solved, conclude with this sentence: \"Since the problem is asking for ..., the answer is \\boxed{...}.\" Be cautious what the problem is asking and in what format the answer should be, and put that answer in box."

            if get_answer(responses[0]) is not None and get_answer(responses[0]) != "":
                tmp_msg, is_query_exist = query_handler.check_queries(responses[0])
                if not is_query_exist:
                    # if the assistant gives a valid reply and no more queries, stop the conversation
                    is_valid_reply = True
                    if not self.refine:  # if not refine, stop the conversation
                        response_with_ans = responses[0]
                        response_with_new_ans = responses[0]
                        break
                    elif not is_refine_process:  # if refine, start the refine process
                        response_with_ans = responses[0]
                        is_refine_process = True
                        refine_message = "Please check your answer to make sure it meets conditions in the problem and you doesn't make any mistakes. If you find any mistake, please correct it and put the corrected answer in box. If you find no mistake, put previous answer in the box."
                        config["messages"].append({"role": "user", "content": refine_message})
                        save_message_to_file(
                            "user: {a}{s}".format(a=config["messages"][-1]["content"], s=seperate_line)
                        )
                        continue
                    else:  # if already in the refine process, then stop the conversation
                        response_with_new_ans = responses[0]
                        break

            # 3. handle the response and get the query
            query_response, is_query_sucess = query_handler.handle_query(responses[0])
            if len(query_response) > 2000:
                # prevent long response by string length, 2000 chars -> around 500-1000 tokens
                save_message_to_file(f"****: Replacing {query_response} ****\n")
                query_response = "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
                is_query_sucess = False
            if "v4." in self.prompt_type and "Continue" in query_response: # to avoid changing queryhandler for python v4, change response here
                query_response = "Continue. (If you think the problem is finished, please reply \"[EOF]\")"
            query_response += tmp_msg  # add the query response from the previous step
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
        save_message_to_file("Solution: " + problem["solution"])

        return {
            "valid_q_count": query_handler.valid_q_count,  # number of valid queries
            "total_q_count": query_handler.total_q_count,
            "is_valid_reply": is_valid_reply,  # whether the assistant can give a valid reply
            "response_with_ans": response_with_ans,  # string instead of list
            "response_with_new_ans": response_with_new_ans,  # string instead of list
            "messages": config["messages"],
            "round": min(rr + 1, self.max_round),
            "cost": total_cost,
        }

    def str_splitter(self, string, length=300):
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
        self.logger.log("id : is_correct $ ans $ correct_ans | corrected_ans $ round")
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            # 1. if problem already solved, continue
            if int(problem["problem_id"]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                correct_counts += problem["is_correct"]
                new_ans = problem["new_ans"] if "new_ans" in problem else ""
                if problem["new_ans"] == problem["voted_answer"]:
                    problem["new_ans"] = "same"
                self.logger.log(
                    f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} | {new_ans} $ {problem["round"]} $ (from previous run)'
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
                    "new_ans": get_answer(result["response_with_new_ans"]),
                    "round": result["round"],
                    "valid_q_count": result["valid_q_count"],  # total number of valid queries
                    "total_q_count": result["total_q_count"],  # total number of queries
                    "cost": result["cost"],  # total cost of the conversation
                    "messages": result["messages"],  # the conversation
                }
            )
            write_json(problem, problem_path)
            if problem["new_ans"] == problem["voted_answer"]:
                problem["new_ans"] = "same"

            # 4. continue to next problem
            correct_counts += problem["is_correct"]
            self.logger.log(
                f'{problem["problem_id"]} : {bool(problem["is_correct"])} $ {problem["voted_answer"]} $ {problem["correct_ans"]} | {problem["new_ans"]} $ {problem["round"]} $'
            )

        tp = problem_set[0]["type"]
        self.logger.log(f"{tp} Accuracy: {correct_counts}/{len(problem_set)} = {correct_counts/len(problem_set)}")
        self.logger.log("------------------------------------------------------------\n", verbose=True)
