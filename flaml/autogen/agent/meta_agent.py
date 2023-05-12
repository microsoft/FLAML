from collections import defaultdict
from flaml.autogen.agent.agent import Agent
from flaml.autogen.agent.chat_agent import ChatAgent
from flaml.autogen.agent.coding_agent import PythonAgent
from langchain import OpenAI, LLMChain, PromptTemplate
from flaml import oai
from flaml.autogen.code_utils import DEFAULT_MODEL, FAST_MODEL


class MetaAgent:
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
        self._system_prompt_glossary = {
            ChatAgent.__name__: ChatAgent.DEFAULT_SYSTEM_MESSAGE,
            PythonAgent.__name__: PythonAgent.DEFAULT_SYSTEM_MESSAGE,
        }
        self._system_message = (
            meta_agent_system_message
            if meta_agent_system_message
            else self._system_prompt_glossary[agent_class.__name__]
        )
        self._agent_class = agent_class

        self._agent = agent_class(name)
        print("Agent class", self._agent_class)
        self._meta_template = """
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

    def _receive(self, message, sender):
        """Receive a message from another agent."""
        self._agent.receive(message, sender)

    def _init_reflect(self):
        PromptTemplate(input_variables=["chat_history"], template=self._system_message)

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

    def _new_task_bookkeeping(self):
        """Bookkeeping for a new task."""
        self._agent._conversations = defaultdict(list)
        self._agent._memory = []
        self._agent._sender_dict = {}

    def reflect(self):
        """Reflect on the conversations with the agents."""
        chat_history = self._get_chat_history()
        # PromptTemplate(
        #     input_variables=["chat_history"],
        #     template=self._meta_template
        # )
        meta_prompt = self._meta_template.format(chat_history=chat_history)

        responses = oai.ChatCompletion.create(messages=[{"content": meta_prompt, "role": "user"}], **self._config)
        response = oai.ChatCompletion.extract_text(responses)[0]
        print(f"{self._name}", response)

        # new_instructions = self._get_new_instructions()
        # self._init_reflect()
        # self._agent._send(chat_history, self)
        # self._agent._send(new_instructions, self)
        # self._system_message = new_instructions
        self._agent = self._agent_class(self._name, meta_prompt=response)
        # TODO: maybe we should also consider adding the instruction as the init prompt

    def receive(self, message, sender, is_new_task=False):
        """Receive a message from another agent.
        This method is called by the sender.
        It needs to be overriden by the subclass to perform followup actions.
        """
        print(f'{self._name} received "{message}" from {sender._name}')
        self._receive(message, sender)
        # perform actions based on the message
