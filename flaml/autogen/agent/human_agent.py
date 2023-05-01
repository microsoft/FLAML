from .agent import Agent


class HumanAgent(Agent):
    """Human Agent."""

    DEFAULT_SYSTEM_MESSAGE = """You are human agent. You can give feedback to the sender.
    """

    def receive(self, message, sender):
        """Receive a message from the sender agent.
        Every time a message is received, the human agent will give feedback.
        """
        super().receive(message, sender)
        print("Human agent received message: ", message)
        # give feedback to the sender via standard input
        print("Please give feedback to the sender (press enter to skip): ")
        feedback = input()
        if feedback:
            self._send(feedback, sender)

    def retrieve_conversation(self, agent_name):
        """retrieve the conversation with the agent"""
        return self._conversations[agent_name][-1]["content"]
