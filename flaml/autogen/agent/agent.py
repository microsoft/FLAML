from collections import defaultdict
from typing import Dict, Union


class Agent:
    """(Experimental) An abstract class for AI agent.
    An agent can communicate with other agents and perform actions.
    Different agents can differ in what actions they perform in the `receive` method.

    """

    def __init__(self, name, system_message=""):
        """
        Args:
            name (str): name of the agent
            system_message (str): system message to be sent to the agent
        """
        # empty memory
        self._memory = []
        # a dictionary of conversations, default value is list
        self._oai_conversations = defaultdict(list)
        self._name = name
        self._system_message = system_message

    @property
    def name(self):
        """Get the name of the agent."""
        return self._name

    def _remember(self, memory):
        """Remember something."""
        self._memory.append(memory)

    def _send(self, message: Union[Dict, str], recipient):
        """Send a message to another agent."""
        if type(message) is str:
            oai_message = {"content": message, "role": "assistant"}
        else:
            # create openai message to be appended to the conversation
            oai_message = {k: message[k] for k in ("content", "function_call", "name") if k in message}
            # When the agent composes and sends the message, the role of the message is "assistant". The role of 'function' will remain unchanged.
            oai_message["role"] = "function" if message.get("role") == "function" else "assistant"
        self._oai_conversations[recipient.name].append(oai_message)

        recipient.receive(message, self)

    def _receive(self, message: Union[Dict, str], sender):
        """Receive a message from another agent.

        Args:
            message (dict or str): message from the sender. If the type is dict, it can contain at most 4 fields:
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments.
                3. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                4. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
            sender: sender of an Agent instance.
        """
        if type(message) is str:
            message = {"content": message, "role": "user"}
        # print the message received
        print(sender.name, "(to", f"{self.name}):\n", flush=True)
        if message.get("role") == "function":
            func_print = f"***** Response from calling function \"{message['name']}\" *****"
            print(func_print, flush=True)
            print(message["content"], flush=True)
            print("*" * len(func_print), flush=True)
        else:
            if message.get("content") is not None:
                print(message["content"], flush=True)
            if "function_call" in message:
                func_print = f"***** Suggested function Call: {message['function_call'].get('name', '(No function name found)')} *****"
                print(func_print, flush=True)
                print(
                    "Arguments: \n",
                    message["function_call"].get("arguments", "(No arguments found)"),
                    flush=True,
                    sep="",
                )
                print("*" * len(func_print), flush=True)
        print("\n", "-" * 80, flush=True, sep="")

        # create openai message to be appended to the conversation
        oai_message = {k: message[k] for k in ("content", "function_call", "name") if k in message}
        oai_message["role"] = "function" if message.get("role") == "function" else "user"
        self._oai_conversations[sender.name].append(oai_message)

    def receive(self, message: Union[Dict, str], sender):
        """Receive a message from another agent.
        This method is called by the sender.
        It needs to be overriden by the subclass to perform followup actions.

        Args:
            message (dict or str): message from the sender. If the type is dict, it can contain at most 4 fields:
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments.
                3. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                4. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
            sender: sender of an Agent instance.
        """
        self._receive(message, sender)
        # perform actions based on the message