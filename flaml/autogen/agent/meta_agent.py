from collections import defaultdict
from typing import Optional
from flaml.autogen.agent.agent import Agent
from flaml import oai
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL


class MetaAgent(Agent):
    """(Experimental) A meta agent that can wrap other agents and perform actions based on the messages received."""

    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
    }

    DEFAULT_SYSTEM_MESSAGE = """
    Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely.
    Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would
    quickly and correctly respond in the future.

    ####

    {chat_history}

    ####

    Please reflect on these interactions.

    You should first critique Assistant's performance. What could Assistant have done better?
    What should the Assistant remember about this user? Are there things this user always wants?
    Indicate this with "Critique: ...".

    You should next revise the Instructions so that Assistant would quickly and correctly respond in the future.
    Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
    """

    def __init__(
        self,
        name,
        system_message="",
        agent: Optional[Agent] = None,
        dev_agent: Optional[Agent] = None,
    ):
        """
        Args:
            name (str): name of the meta agent
            system_message (str): system message to be sent to the meta agent
            user_agent (optional): the agent to be wrapped
            dev_agent (optional): the agent to be wrapped for development
        """
        # use super() to call the parent class's __init__ method
        super().__init__(name, system_message=system_message)
        self._system_message = system_message if system_message else self.DEFAULT_SYSTEM_MESSAGE
        # TODO: do we only need to have only user_agent or dev_agent?
        self._agent = agent
        self._dev_agent = dev_agent
        self._meta_prompt_template = """
        Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely.
        Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would
        quickly and correctly respond in the future.

        ####

        {chat_history}

        ####

        Please reflect on these interactions.

        You should first reflect on Assistant's performance. What could Assistant have done better?
        What should the Assistant remember about this user? Are there things this user always wants?
        Indicate this with "Reflection: ...".

        You should next revise the Instructions so that Assistant would quickly and correctly respond in the future.
        Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions.
        Indicate the new Instructions by "Instructions: ...".
        """

    def _receive(self, message, sender):
        """Receive a message from another agent."""
        if self._agent:
            self._agent.receive(message, sender)
        # if self._dev_agent:
        #     self._dev_agent.receive(message, sender)

    def _get_chat_history(self):
        """Get the chat history of the agent."""
        chat_history = ""
        for conversation in self._agent._conversations.values():
            for message in conversation:
                if message["role"] == "user":
                    chat_history += "User: " + message["content"] + "\n"
                else:
                    chat_history += "Assistant: " + message["content"] + "\n"
        return chat_history

    def reflect(self):
        self.receive("reflect", self._dev_agent)
        # """Reflect on the conversations with the agents."""
        # chat_history = self._get_chat_history()
        # meta_prompt = self._meta_prompt_template.format(chat_history=chat_history)
        # responses = oai.ChatCompletion.create(messages=[{"content": meta_prompt, "role": "user"}], **self._config)
        # response = oai.ChatCompletion.extract_text(responses)[0]
        # print(f"Reflecting.....\n{self._name}", response)
        # self._agent = self._agent_class(self._name, meta_prompt=response)
        # TODO: maybe we should also consider adding the instruction as the init prompt
