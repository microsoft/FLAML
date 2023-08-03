from flaml.autogen.agentchat.agent import Agent
from flaml.autogen.agentchat.assistant_agent import AssistantAgent
from typing import Callable, Dict, Optional, Union, List


class RetrieveAssistantAgent(AssistantAgent):
    """(Experimental) Retrieve Assistant agent, designed to solve a task with LLM.

    RetrieveAssistantAgent is a subclass of AssistantAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        assert messages is not None or sender is not None, "Either messages or sender must be provided."
        if messages is None:
            messages = self._oai_messages[sender.name]
        message = self._message_to_dict(messages[-1])
        if "exitcode: 0 (execution succeeded)" in message.get("content", ""):
            # Terminate the conversation when the code execution succeeds. Although sometimes even when the
            # code execution succeeds, the task is not solved, but it's hard to tell. If the human_input_mode
            # of RetrieveUserProxyAgent is "TERMINATE" or "ALWAYS", user can still continue the conversation.
            return "TERMINATE"
        elif "UPDATE CONTEXT" in message.get("content", "")[-20::].upper():
            return "UPDATE CONTEXT"
        else:
            return super().generate_reply(messages, sender)
