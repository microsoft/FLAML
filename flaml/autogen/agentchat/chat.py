from typing import Dict, List, Tuple
from .responsive_agent import ResponsiveAgent
import random


class Chat:
    """(Experimental) A chat class that can be used to simulate a multi-agent chat using role-play prompts.

    For more details on role-play prompts, please refer to class :class:`flaml.autogen.agentchat.roleplay_agent.RoleplayMixin`.
    """

    chat_history: List[Dict]
    agents: List[ResponsiveAgent]
    user: ResponsiveAgent
    WAIT_FOR_USER_MSG: Dict = {"role": "system", "content": "Waiting for user input."}

    def __init__(self, user: ResponsiveAgent, agents: List[ResponsiveAgent], chat_history: List[Dict] = []):
        """
        Args:
            user: the user agent
            agents: the assistant agents
            chat_history: the chat history
        """
        self.user = user
        self.agents = agents
        self.chat_history = chat_history

    def _get_role_description(self) -> List[Tuple[str, str]]:
        agents = [agent for agent in self.agents] + [self.user]
        description = [(agent.name, agent.describle_role(self.chat_history)) for agent in agents]

        return description

    def _select_next_speaker(self, description: List[Tuple[str, str]], rules: str) -> ResponsiveAgent:
        agent_list = [agent for agent in self.agents] + [self.user]

        # randomly select an agent
        selected_agent = random.choice(self.agents)
        next_agent_index = selected_agent.select_role(self.chat_history, description, rules)
        if next_agent_index < 0:
            return self.user
        else:
            return agent_list[next_agent_index]

    def push_message(self, message: Dict):
        self.chat_history.append(message)

    def _summarize_rule_from_chat_history(self, chat_history: List[Dict], admin: str) -> str:
        # randomly select an agent
        # and summarize the rule
        agent = random.choice(self.agents)
        return agent.summarize_rule_from_chat_history(chat_history, admin)

    def send_single_step(self, message: Dict) -> Dict:
        self.push_message(message)
        description = self._get_role_description()
        rules = self._summarize_rule_from_chat_history(self.chat_history, self.user.name)
        agent = self._select_next_speaker(description, rules)
        print(f"selected agent: {agent.name}")
        if agent.name != self.user.name:
            return agent.role_play(self.chat_history, description, rules)
        else:
            # wait for user input
            message = agent.get_human_input(f"[{agent.name}]:")
            return {"role": agent.name, "content": message}

    def send(self, message: Dict, max_round: int = 10) -> List[Dict]:
        """Send a message to the chat.

        Args:
            message: the initial message. Must be a dictionary with keys "role" and "content". the "role" key must be the name of an agent.
            max_round: the maximum number of rounds
        """

        for _ in range(max_round):
            print(message)
            message = self.send_single_step(message)

        return self.chat_history + [message]
