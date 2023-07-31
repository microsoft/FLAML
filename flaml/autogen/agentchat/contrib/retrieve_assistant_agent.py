from flaml.autogen.agentchat.agent import Agent
from flaml.autogen.agentchat.assistant_agent import AssistantAgent
from typing import Callable, Dict, Optional, Union


class RetrieveAssistantAgent(AssistantAgent):
    """(Experimental) Assistant agent, designed to solve a task with LLM.

    AssistantAgent is a subclass of ResponsiveAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
        """
        super().__init__(
            name,
            **kwargs,
        )

    def _reset(self):
        self._oai_conversations.clear()

    def receive(self, message: Union[Dict, str], sender: "Agent"):
        """Receive a message from another agent.
        If "Update Context" in message, update the context and reset the messages in the conversation.
        """
        message = self._message_to_dict(message)
        if "UPDATE CONTEXT" in message.get("content", "")[-20::].upper():
            self._reset()
            self.send("UPDATE CONTEXT", sender)
        elif "exitcode: 0 (execution succeeded)" in message.get("content", ""):
            self.send("TERMINATE", sender)
        else:
            super().receive(message, sender)
