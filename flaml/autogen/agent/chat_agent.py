from .agent import Agent
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL
from flaml import oai
from collections import defaultdict


class ChatAgent(Agent):
    """(Experimental) Chat."""

    DEFAULT_SYSTEM_MESSAGE = """You are a chat agent.
    """

    DEFAULT_CONFIG = {
        "model": FAST_MODEL,
    }

    def __init__(self, name, system_message=DEFAULT_SYSTEM_MESSAGE, work_dir=None, meta_prompt=None, **config):
        """
        Args:
            name (str): agent name
            system_message (str): system message to be sent to the agent
            work_dir (str): working directory for the agent to execute code
            config (dict): other configurations.
        """
        super().__init__(name, system_message)
        self._work_dir = work_dir
        self._config = self.DEFAULT_CONFIG.copy()
        self._config.update(config)
        self._sender_dict = {}
        self._meta_prompt = [meta_prompt] if meta_prompt else []

    def receive(self, message, sender):
        message = self._meta_prompt.pop() + f"User message is: {message}" if self._meta_prompt else message
        super().receive(message, sender)
        responses = oai.ChatCompletion.create(messages=self._conversations[sender.name], **self._config)
        # cost = oai.ChatCompletion.cost(responses)
        response = oai.ChatCompletion.extract_text(responses)[0]
        self._send(response, sender)
