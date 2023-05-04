from .agent import Agent
from .execution_agent import ExecutionAgent
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL
from flaml import oai
import copy
from flaml.autogen.code_utils import extract_code


class MathAgent(Agent):
    """Solve a math problem.
    Most of the code is adopted from the math_solver.py file in Yiran's PR:
    https://github.com/microsoft/FLAML/blob/ac11d2a7bb91f0f210ce0c67ec7b628d967e27b5/flaml/autogen/math/math_solver.py
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant.
    """
    AGENT_PREFIX = "math_agent"

    DEFAULT_CONFIG = {
        "model": FAST_MODEL,  # default model is gpt-4
    }
    EXECUTION_AGENT_PREFIX = "execution_agent4"
    SUCCESS_EXIT_CODE = "exitcode: 0\n"

    PROMPTS = {
        "v3.1python": """Let's use python to solve a math problem.
            Query requirements:
            You should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
            You must following the formats below to write your code (otherwise it will not be recognized):
            ```python
            # your code
            ```
            First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
            Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
            Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
            Case 3: If the problem cannot be handled with the two ways above, please follow this process:
            1. Solve the problem step by step (do not overdivide the steps).
            2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
            3. Wait for me to give the results.
            4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
            After all the queries are run and you get the answer, put the answer in \\boxed{}.
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
        self.prompt_type = "v3.1python"
        self.prompt = MathAgent.PROMPTS[self.prompt_type]
        self._system_message = MathAgent.DEFAULT_SYSTEM_MESSAGE
        # self._system_message = self.prompt
        self._file_to_be_saved = "test_math.txt"

        self._seperate_line = "\n" + "-" * 40 + "\n"
        # to save the list of original senders and problems over time
        self._original_senders_and_questions = []

    def _set_prompt(self, prompt):
        """Set the prompt for the agent.
        #TODO: Not using for now. May need to use it in the future.
        """
        self.prompt = prompt

    def _save_message_to_file(self, message):
        if self._file_to_be_saved is not None:
            with open(self._file_to_be_saved, "a") as f:
                f.write(message)
                f.flush()

    @staticmethod
    def _execution_agent_needed(message):
        """Check if the execution agent is needed."""
        _, lang = extract_code(message)
        if lang == "unknown":
            return False
        else:
            return True

    def receive(self, question, sender, clear_conversation=False, archive_conversation=False):
        # TODO: add a clear conversation function
        if archive_conversation:
            self._remember(self._conversations[sender.name])
        if clear_conversation:
            self._conversations = {}
        if sender.name not in self._conversations or len(self._conversations[sender.name]) == 0:
            self._sender_dict[sender.name] = sender
            self._conversations[sender.name] = [{"content": self._system_message, "role": "system"}]
            # TODO: do we need to clear self._sender_dict and self._conversations?
            prompted_question = self.prompt + "\n Problem: " + question  # TODO: pay attention to the executation agent
        else:
            prompted_question = question
        # if the sender is the master agent, then we need to save the original sender and problem
        # there could be multiple turns of conversation between the master agent and the math agent,
        # we only need to save the original sender and problem once
        is_session_starts = len(self._conversations[sender.name]) == 1
        if not sender.name.startswith(self.EXECUTION_AGENT_PREFIX) and is_session_starts:
            # assuming the master agent is the first agent to send the message to the math agent
            self._original_senders_and_questions.append((sender, question))
        super().receive(prompted_question, sender)
        # save a readable conversation in txt file
        # self._save_message_to_file(f"Problem: {self._str_splitter(prompted_question)}\n {self._seperate_line}")
        messages = copy.deepcopy(self._conversations[sender.name])
        raw_responses = oai.ChatCompletion.create(messages=messages, **self._config, use_cache=self.use_cache)
        response = oai.ChatCompletion.extract_text(raw_responses)[0]
        print(f"\n Sender {sender.name}: {question}")
        print(f"\n MATH AGENT: {response}")

        original_sender, _ = self._original_senders_and_questions[-1]
        if self._execution_agent_needed(response):
            if sender.name.startswith(self.EXECUTION_AGENT_PREFIX):
                excution_agent = sender
            else:
                # create an execution agent if an execution agent is needed
                # TODO: should we consider the case where the execution agent is already created in the past?
                excution_agent = ExecutionAgent(f"{self.EXECUTION_AGENT_PREFIX}{sender.name}", work_dir=self._work_dir)
                # initialize the conversation
                self._conversations[excution_agent.name] = self._conversations[sender.name].copy()
                self._sender_dict[excution_agent.name] = excution_agent
            # send the response to the execution agent
            self._send(response, excution_agent)
        else:
            print(f"Execution agent not needed. Sending to original sender {original_sender.name}")
            # look into the conversation history to finalize the answer
            if sender.name.startswith(self.EXECUTION_AGENT_PREFIX):
                execution_agent_name = sender.name
            elif f"{self.EXECUTION_AGENT_PREFIX}{sender.name}" in self._conversations.keys():
                execution_agent_name = f"{self.EXECUTION_AGENT_PREFIX}{sender.name}"
            else:
                execution_agent_name = None
            if execution_agent_name is not None:
                answer = self._get_answer_from_conversation(execution_agent_name, original_sender.name)
            else:
                answer = response
            self._send(answer, original_sender)

    def _get_answer_from_conversation(self, excution_agent_name, user_name):
        """Extract the answer from the conversation history."""
        excution_conv = self._conversations[excution_agent_name]
        # user_conv = self._conversations[user_name]
        # TODO: currently only check the last msg with the execution agent. may need to change later.
        prompt = (
            "check the conversation history and extract the final answer. Put the answer in \\boxed{}. DO NOT include anything else."
            + "\n Conversation history with execution agent:"
            + str(excution_conv[-1:])
        )
        messages = [{"content": prompt, "role": "user"}]
        res = oai.ChatCompletion.create(messages=messages, **self._config, use_cache=self.use_cache)
        answer = oai.ChatCompletion.extract_text(res)[0]
        return answer

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
