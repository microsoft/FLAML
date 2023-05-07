from .agent import Agent
from flaml.autogen.code_utils import extract_code, execute_code


class HumanAgent(Agent):
    """A proxy agent for human, that can execute code and provide feedback to the other agents."""

    DEFAULT_SYSTEM_MESSAGE = """You are human agent. You can execute_code or give feedback to the sender.
    """

    def __init__(self, name, system_message="", work_dir=None, interactive_mode=True, **context):
        super().__init__(name, system_message)
        self._work_dir = work_dir
        self._interactive_mode = interactive_mode
        self._context = context

    def _is_termination_msg(self, message):
        """Check if the message is a termination message."""
        if "_is_termination_msg" in self._context:
            return self._context["_is_termination_msg"](message)
        return True

    def receive(self, message, sender):
        """Receive a message from the sender agent.
        Every time a message is received, the human agent will give feedback.
        """
        super().receive(message, sender)
        # try to execute the code
        code, lang = extract_code(message)
        # no code block is found, lang should be "unknown"
        if lang == "unknown":
            # to determine if the message is a termination message using a function
            feedback = (
                input("Please give feedback to the sender (press enter to skip): ") if self._interactive_mode else ""
            )
            if feedback:
                self._send(feedback, sender)
            elif self._is_termination_msg(message):
                return
        else:
            if lang == "bash":
                assert code.startswith("python ")
                file_name = code[len("python ") :]
                exitcode, logs = execute_code(filename=file_name, work_dir=self._work_dir)
            elif lang == "python":
                exitcode, logs = execute_code(code, work_dir=self._work_dir)
            else:
                # TODO: could this happen?
                raise NotImplementedError
            self._send(f"exitcode: {exitcode}\n{logs.decode('utf-8')}", sender)
