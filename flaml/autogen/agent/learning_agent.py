from .assistant_agent import AssistantAgent
from flaml.autogen.code_utils import DEFAULT_MODEL
from flaml import oai
import asyncio


class LearningAgent(AssistantAgent):
    """(Experimental) A learning agent."""

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute. You must indicate the script type in the code block.
    1. When you need to ask the user for some info, use the code to output the info you need, for example, browse or search the web, download/read a file.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly. Solve the task step by step if you need to.
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again. Suggeset the full code instead of partial code or code changes.
    Reply "TERMINATE" in the end when everything is done.
    """

    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
    }

    def __init__(self, name, system_message=DEFAULT_SYSTEM_MESSAGE, **config):
        """
        Args:
            name (str): agent name.
            system_message (str): system message to be sent to the agent.
            **config (dict): other configurations allowed in
              [oai.Completion.create](../oai/Completion#create).
              These configurations will be used when invoking LLM.
        """
        super().__init__(name, system_message, **config)
        self._system_message_learning = """You are a helpful AI assistant."""
        self._learning_objectives = ""
        self._can_handle_data_volume = lambda *args: True

    def _generate_task_prompt(self, learning_results, learning_data):
        """
        Process the message using NLP.
        """
        task_prompt = f"""
        {self._learning_objectives}.
        This is the latest data entry: {learning_data}.
        Renew the current result:
        {learning_results}
        You can try to condense the current result and add a new bullet point to the result.
        """
        return task_prompt

    @staticmethod
    def is_total_token_count_within_threshold(learning_results, learning_data):
        """
        Check if the total token count of learning data and learning results
        is within a specified threshold.
        """

        def _token_counter(input_string):
            from transformers import GPT2Tokenizer

            # Load a pre-trained tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # Tokenize the string
            tokens = tokenizer.tokenize(input_string)
            return len(tokens)

        oai_max_token_size = 4096
        return _token_counter(learning_results) + _token_counter(learning_data) < oai_max_token_size * 0.8

    def _validate_learning_constraints(self, learning_constraints):
        # check if the learning constraints are satisfied
        # do nothing for now
        return True

    async def receive(self, message, sender):
        """Receive a message from another agent."""
        content = message.get("content", None) if isinstance(message, dict) else message
        self._receive(message, sender)
        # NOTE: content and learning settings are mutually exclusive
        if content is not None:
            # if content is provided, perform the default receive function
            super().receive(content, sender)
        else:
            # perform learning based on the learning settings
            learning_func = message.get("learning_func", None)
            learning_objectives = message.get("learning_objectives", None)
            learning_constraints = message.get("learning_constraints", None)
            learning_results = message.get("learning_results", None)
            data4learning = message.get("data4learning", None)
            if learning_objectives:
                self._learning_objectives = learning_objectives
            # when data is available, perform the learning task when learning_constraints are satisfied
            if data4learning and self._validate_learning_constraints(learning_constraints):
                # perform learning
                if learning_func:
                    # assumption: learning_func is a function that takes learning_results and learning_data as input and returns new_learning_results and can_handle_data_volume
                    # when learning_data is None, the learning_func should work as well, outputting the input learning_results as the
                    # new_learning_results and can_handle_data_volume function
                    new_learning_results, self._can_handle_data_volume = learning_func(learning_results, data4learning)
                else:
                    self._can_handle_data_volume = self.is_total_token_count_within_threshold
                    if data4learning:
                        task_prompt = self._generate_task_prompt(learning_results, data4learning)
                        learning_msg = [
                            # {"content": self._system_message_learning, "role": "system"},
                            {"role": "user", "content": task_prompt},
                        ]
                        responses = oai.ChatCompletion.create(messages=learning_msg, **self._config)
                        new_learning_results = oai.ChatCompletion.extract_text(responses)[0]
                    else:
                        new_learning_results = learning_results
                print("*********Current learning results of the learner*********\n", new_learning_results, flush=True)
                print("*" * 50, flush=True)
                await self._send(
                    {"learning_results": new_learning_results, "can_handle_data_volume": self._can_handle_data_volume},
                    sender,
                )
            else:
                await self._send(
                    {"learning_results": learning_results, "can_handle_data_volume": self._can_handle_data_volume},
                    sender,
                )
