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

    def _reset(self):
        # clean only the messages in the conversation, but not _consecutive_auto_reply_counter
        self._oai_messages.clear()

    def send(self, message: Union[Dict, str], recipient: Agent):
        """Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields (either content or function_call must be provided):
                - content (str): the content of the message.
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [autogen.Completion.create](../oai/Completion#create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": "{use_tool_msg}",
            "context": {
                "use_tool_msg": "Use tool X if they are relevant."
            }
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._message_to_dict(message)
        if "UPDATE CONTEXT" in message.get("content", "")[-20::].upper():
            self._reset()
        return super().send(message, recipient)

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        """Reply based on the conversation history.

        First, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.
        Subclasses can override this method to customize the reply.
        Either messages or sender must be provided.

        Args:
            messages: a list of messages in the conversation history.
            default_reply (str or dict): default reply.
            sender: sender of an Agent instance.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        assert messages is not None or sender is not None, "Either messages or sender must be provided."
        if messages is None:
            messages = self._oai_messages[sender.name]
        message = self._message_to_dict(messages[-1])
        if "exitcode: 0 (execution succeeded)" in message.get("content", ""):
            return "TERMINATE"
        elif "UPDATE CONTEXT" in message.get("content", "")[-20::].upper():
            self._reset()
            return "UPDATE CONTEXT"
        else:
            return super().generate_reply(messages, default_reply, sender)
