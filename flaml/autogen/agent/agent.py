from collections import defaultdict
import copy


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
        self._conversations = defaultdict(list)
        self._name = name
        self._system_message = system_message

    @property
    def name(self):
        """Get the name of the agent."""
        return self._name

    def _remember(self, memory):
        """Remember something."""
        self._memory.append(memory)

    def _send(self, message: dict, recipient: str):
        """Send a message to another agent.

        An agent always assumes itself as the "assistant", and the message it receives is from the "user`".
        """
        # message["role"] can be "assistant" or "function"
        self._conversations[recipient.name].append(copy.deepcopy(message))

        # The agent position iteself as the "user" instead of "assistant" and sends the message to the recipient.
        if message["role"] == "assistant":
            message["role"] = "user"
        recipient.receive(message, self)

    def _receive(self, message: dict, sender: str):
        """Receive a message from another agent."""
        print(sender.name, "(to", f"{self.name}):", flush=True)

        if message["role"] == "function":
            print(f"*****Response from calling function: {message['name']}*****", flush=True)
            print(message["content"], flush=True)
            print("*" * 40, "\n", flush=True)
        else:
            if "content" in message and message["content"] is not None:
                print(message["content"], flush=True)
            if "function_call" in message:
                print(
                    f"*****Calling function: {message['function_call'].get('name', '(No function name found)')}*****",
                    flush=True,
                )
                print("with arguments: ", message["function_call"].get("arguments", "(No arguments found)"), flush=True)
                print("*" * 40, flush=True)
        print("\n", "-" * 80, flush=True, sep="")
        self._conversations[sender.name].append(message)

    def receive(self, message: dict, sender: str):
        """Receive a message from another agent.
        This method is called by the sender.
        It needs to be overriden by the subclass to perform followup actions.
        """
        self._receive(message, sender)
        # perform actions based on the message
