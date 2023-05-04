from .agent import Agent

# import oai
from flaml import oai
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL


class ReflectionAgent(Agent):
    """Reflect on the conversation.
    Try to criticize the sender's message.
    """

    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
    }
    DEFAULT_SYSTEM_MESSAGE = """You are a reflection agent. You try to criticize the sender's message. If you think the message is correct, add YES! before the original answer and return it. Otherwise give constructive suggestion.
    """
    AGENT_PREFIX = "reflection_agent"

    def __init__(self, name, system_message="", work_dir=None):
        super().__init__(name, system_message)
        self._word_dir = work_dir
        self._conversations = {}
        self._system_message = ReflectionAgent.DEFAULT_SYSTEM_MESSAGE
        self._config = ReflectionAgent.DEFAULT_CONFIG.copy()
        self._sender_dict = {}

    def receive(self, message, sender):
        if sender.name not in self._conversations:
            self._sender_dict[sender.name] = sender
            self._conversations[sender.name] = [{"content": self._system_message, "role": "system"}]
        super().receive(message, sender)
        res = oai.ChatCompletion.create(messages=self._conversations[sender.name], **self._config)
        critique = oai.ChatCompletion.extract_text(res)[0]
        print("The critique is ", critique)
        self._send(critique, sender)

    def receive_conversation(self, conversation, sender):
        if sender.name not in self._conversations:
            self._conversations[sender.name] = conversation
        else:
            self._conversations[sender.name] = self._conversations[sender.name] + conversation
