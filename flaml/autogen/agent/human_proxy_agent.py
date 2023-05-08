from .agent import Agent
from flaml.autogen.code_utils import extract_code, execute_code


class HumanProxyAgent(Agent):
    """(Experimental) A proxy agent for human, that can execute code and provide feedback to the other agents."""

    DEFAULT_SYSTEM_MESSAGE = """You are human agent. You can execute_code or give feedback to the sender.
    """

    def __init__(
        self, name, system_message="", work_dir=None, human_input_mode="ALWAYS", is_termination_msg=None, **config
    ):
        """
        Args:
            name (str): name of the agent
            system_message (str): system message to be sent to the agent
            work_dir (str): working directory for the agent to execute code
            human_input_mode (bool): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                When "ALWAYS", the agent will ask for human input every time a message is received.
                When "TERMINATE", the agent will ask for human input only when a termination message is received.
                When "NEVER", the agent will never ask for human input.
            is_termination_msg (function): a function that takes a message and returns a boolean value.
                This function is used to determine if a received message is a termination message.
            config (dict): other configurations.
        """
        super().__init__(name, system_message)
        self._work_dir = work_dir
        self._human_input_mode = human_input_mode
        self._is_termination_msg = (
            is_termination_msg if is_termination_msg is not None else (lambda x: x == "TERMINATE")
        )
        self._config = config

    def receive(self, message, sender):
        """Receive a message from the sender agent.
        Every time a message is received, the human agent will give feedback.
        """
        super().receive(message, sender)
        # to determine if the message is a termination message using a function
        terminate = self._is_termination_msg(message)
        feedback = (
            input("Please give feedback to the sender (press enter to skip): ")
            if self._human_input_mode == "ALWAYS" or terminate and self._human_input_mode == "TERMINATE"
            else ""
        )
        if feedback:
            self._send(feedback, sender)
        elif terminate:
            return
        # try to execute the code
        code, lang = extract_code(message)
        if lang == "unknown":
            # no code block is found, lang should be "unknown"
            self._send(feedback, sender)
        else:
            if lang == "bash":
                assert code.startswith("python "), code
                file_name = code[len("python ") :]
                exitcode, logs = execute_code(filename=file_name, work_dir=self._work_dir)
            elif lang == "python":
                exitcode, logs = execute_code(code, work_dir=self._work_dir)
            else:
                # TODO: could this happen?
                raise NotImplementedError
            self._send(f"exitcode: {exitcode}\n{logs.decode('utf-8')}", sender)
