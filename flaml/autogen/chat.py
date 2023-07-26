from typing import List, Tuple, Union
from .agent import Agent
from .agent.user_proxy_agent import UserProxyAgent
from .agent.roleplay_agent import RoleplayMixin, Message
from .agent.generic_agent import GenericAgent
import random

class Chat(Agent):
    chat_history: List[Message]
    agents: List[GenericAgent]
    user: GenericAgent
    WAIT_FOR_USER_MSG = Message(role="system", content="Waiting for user input.")

    def __init__(self, user: GenericAgent, agents: List[GenericAgent], chat_history: List[Message] = []):
        self.user = user
        self.agents = agents
        self.chat_history = chat_history

    def get_role_description(self) -> List[Tuple[str, str]]:
        agents = [agent for agent in self.agents] + [self.user]
        description = [(agent.name, agent.describle_role(self.chat_history)) for agent in agents]

        return description
    
    def select_role(self, description:List[Tuple[str, str]], rules: str) -> GenericAgent:
        agent_list = [agent for agent in self.agents] + [self.user]

        # randomly select an agent
        selected_agent = random.choice(self.agents)
        next_agent_index = selected_agent.select_role(self.chat_history, description, rules)
        if next_agent_index < 0:
            return self.user
        else:
            return agent_list[next_agent_index]
    
    def push_message(self, message: Message):
        self.chat_history.append(message)

    def summarize_rule_from_chat_history(self, chat_history: List[Message], admin: str) -> str:
        # randomly select an agent
        # and summarize the rule
        agent = random.choice(self.agents)
        return agent.summarize_rule_from_chat_history(chat_history, admin)
    
    def role_play_single_step(self, message: Message) -> Message:
        self.push_message(message)
        description = self.get_role_description()
        rules = self.summarize_rule_from_chat_history(self.chat_history, self.user.name)
        agent = self.select_role(description, rules)
        print(f"selected agent: {agent.name}")
        if agent.name != self.user.name:
            return agent.role_play(self.chat_history, description, rules)
        else:
            # wait for user input
            message = agent.get_human_input(f'[{agent.name}]:')
            return Message(agent.name, message)
    
    def role_play(self, message: Message, max_round: int = 10) -> List[Message]:
        for _ in range(max_round):
            print(message)
            message = self.role_play_single_step(message)
        
        return self.chat_history + [message]
    
