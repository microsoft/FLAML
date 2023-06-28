from .assistant_agent import AssistantAgent
from flaml.autogen.code_utils import DEFAULT_MODEL
from flaml import oai


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

    def _generate_task_prompt(self, learning_objectives, learning_results, learning_data):
        """
        Process the message using NLP.
        """
        task_prompt = f"""You should: {learning_objectives} \n
            based on old results {learning_results} and the latest learning data: {learning_data}.
            You should only return the new learning results without any other information.
            """
        return task_prompt

    @staticmethod
    def _is_data_size_feasible_oai(learning_results, learning_data):
        """
        Check if the learning data is feasible.
        """
        max_token_size = 1024
        return len(learning_results) + len(learning_data) < max_token_size

    def _validate_learning_constraints(self, learning_constraints):
        # check if the learning constraints are satisfied
        # do nothing for now
        return True

    def receive(self, message, sender):
        """Receive a message from another agent."""
        content = message.get("content", None) if isinstance(message, dict) else message
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
            # when data is available, perform the learning task when learning_constraints are satisfied
            if self._validate_learning_constraints(learning_constraints):
                # perform learning
                if learning_func:
                    # assumption: learning_func is a function that takes learning_results and learning_data as input and returns new_learning_results and is_data_size_feasible
                    # when learning_data is None, the learning_func should work as well, outputting the input learning_results as the
                    # new_learning_results and is_data_size_feasible function
                    new_learning_results, is_data_size_feasible_func = learning_func(learning_results, data4learning)
                else:
                    is_data_size_feasible_func = self._is_data_size_feasible_oai
                    if data4learning is not None:
                        task_prompt = self._generate_task_prompt(learning_objectives, learning_results, data4learning)
                        learning_msg = [
                            {"content": self._system_message_learning, "role": "system"},
                            {"role": "user", "content": task_prompt},
                        ]
                        responses = oai.ChatCompletion.create(messages=learning_msg, **self._config)
                        new_learning_results = oai.ChatCompletion.extract_text(responses)[0]
                    else:
                        new_learning_results = learning_results
                self._send(
                    {"learning_results": new_learning_results, "is_data_size_feasible": is_data_size_feasible_func},
                    sender,
                )
