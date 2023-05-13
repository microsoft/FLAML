from collections import defaultdict
from flaml.autogen.agent.agent import Agent
from flaml.autogen.agent.chat_agent import ChatAgent
from flaml.autogen.agent.coding_agent import PythonAgent
from flaml import oai
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL


class MetaAgent(Agent):
    """(Experimental) A meta agent that can wrap other agents and perform actions based on the messages received."""

    DEFAULT_CONFIG = {
        "model": FAST_MODEL,
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
        self, agent_class, name, system_message="", meta_agent_name="meta-agent", meta_agent_system_message=""
    ):
        """
        Args:
            agent_class (Agent): the class of the agent to be wrapped
            name (str): name of the agent to be wrapped
            system_message (str): system message to be sent to the agent to be wrapped
            meta_agent_name (str): name of the meta agent
            meta_agent_system_message (str): system message to be sent to the meta agent
        """
        self._memory = []
        # a dictionary of conversations, default value is list
        self._conversations = defaultdict(list)
        self._name = meta_agent_name
        self._config = self.DEFAULT_CONFIG
        self._system_message = meta_agent_system_message if meta_agent_system_message else self.DEFAULT_SYSTEM_MESSAGE
        self._agent_class = agent_class
        if system_message:
            self._agent = agent_class(name, system_message=system_message)
        else:
            self._agent = agent_class(name)

        # TODO: Maintain a dictionary of meta prompts for each agent class
        # self._meta_prompt_glossary = {
        #     ChatAgent.__name__: ChatAgent.DEFAULT_SYSTEM_MESSAGE,
        #     PythonAgent.__name__: PythonAgent.DEFAULT_SYSTEM_MESSAGE,
        # }

        # the following meta prompt is based on https://noahgoodman.substack.com/p/meta-prompt-a-simple-self-improving
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
        self._agent.receive(message, sender)

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
        """Reflect on the conversations with the agents."""
        chat_history = self._get_chat_history()
        meta_prompt = self._meta_prompt_template.format(chat_history=chat_history)
        responses = oai.ChatCompletion.create(messages=[{"content": meta_prompt, "role": "user"}], **self._config)
        response = oai.ChatCompletion.extract_text(responses)[0]
        print(f"Reflecting.....\n{self._name}", response)
        self._agent = self._agent_class(self._name, meta_prompt=response)
        # TODO: maybe we should also consider adding the instruction as the init prompt
