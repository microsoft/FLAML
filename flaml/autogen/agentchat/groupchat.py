import sys
from typing import Dict, List, Optional, Union
import random
from .agent import Agent
from .responsive_agent import ResponsiveAgent


class Room(ResponsiveAgent):
    """(WIP) A room agent that can broadcast messages to multiple agents."""

    agents: List[ResponsiveAgent]

    def __init__(
        self,
        agents: Optional[List[ResponsiveAgent]] = None,
        name: Optional[str] = "room",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        **kwargs,
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            **kwargs,
        )
        self.register_class_specific_reply(Agent, self._generate_reply_for_agent)
        self.agents = agents or []

    def _generate_reply_for_agent(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = None,
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        """Broadcast a message from an agent in the group chat."""
        if messages is None:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        # set the name to sender's name if the role is not function
        if message["role"] != "function":
            message["name"] = sender.name
        # broadcast the message to all agents except the sender
        for agent in self.agents:
            if agent != sender:
                self.send(message, agent)


class ChatManagerAgent(ResponsiveAgent):
    """(WIP) A chat manager agent that can manage a group chat of multiple agents."""

    room: Room
    max_round: int

    def _select_speaker_msg(self):
        return {
            "role": "system",
            "content": f"""You are in a role play game. Read the following conversation.
Then select the next role from {self._agent_names} to play. Only return the role.""",
        }

    def __init__(
        self,
        room,
        max_round: Optional[int] = 10,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        seed: Optional[int] = 4,
        **kwargs,
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            **kwargs,
        )
        self.register_class_specific_reply(Room, self._generate_reply_for_room)
        self.register_class_specific_reply(GroupChatParticipant, self._generate_reply_for_participant)
        self.room = room
        self.max_round = max_round
        self._agent_names = []
        self._next_speaker = None
        self._random = random.Random(seed)

    def _generate_reply_for_room(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        if messages is None:
            messages = self._oai_messages[sender.name]
        # the chat begins
        self._agent_names = [agent.name for agent in self.room.agents]
        self.send(messages[-1], self.room)
        for _ in range(self.max_round):
            self._select_speaker()
            self.send("speak", self._next_speaker)

    def _generate_reply_for_participant(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        if messages is None:
            messages = self._oai_messages[sender.name]
        # speaker selection msg from an agent
        self._next_speaker = self._find_next_speaker(messages[-1])

    def _select_speaker(self):
        """Select the next speaker."""
        i = self._random.randint(0, len(self._agent_names) - 1)  # randomly pick an id
        self.send(self._select_speaker_msg(), self.room.agents[i])

    def _find_next_speaker(self, message: Dict) -> str:
        """Find the next speaker based on the message."""
        return self.room.agents[self._agent_names.index(message["content"])]


class GroupChatParticipant(ResponsiveAgent):
    """(WIP) A group chat participant agent that can participate in a group chat."""

    room: Room
    chat_manager: ChatManagerAgent

    def __init__(
        self,
        name,
        chat_manager=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            **kwargs,
        )
        self.register_class_specific_reply(ChatManagerAgent, self._generate_reply_for_chat_manager)
        self.register_class_specific_reply(Room, self._generate_reply_for_room)
        self.chat_manager = chat_manager

    def _generate_reply_for_room(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        """Overriding this method to participate in a group chat.

        Args:
            messages: a list of messages from other agents in the group chat.
            default_reply: a default reply if no reply is generated.
            sender: the agent who sent the message.

        Returns:
            No reply to the room.
        """
        return

    def _generate_reply_for_chat_manager(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        """Generate reply for the chat manager.

        Returns next speaker's name if the message is for selecting the next speaker. If the message is "speak", then the agent will send msg to the room.
        """
        if messages is None:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        sender = self.chat_manager.room
        if message["content"] == "speak":
            reply = super().generate_reply(
                self.chat_messages[sender.name], default_reply, sender, class_specific_reply=False
            )
            self.send(reply, self.chat_manager.room)
        else:
            return self._speaker_selection(message)

    def _speaker_selection(self, instruction):
        """Select the next speaker."""
        if self.llm_config is False:
            if self.human_input_mode == "NEVER":
                return self.name
            else:
                return self.get_human_input(instruction["content"])
        sender = self.chat_manager.room
        roles_msg = {
            "content": f"""The following roles are available:
{self._participant_roles()}""",
            "role": "system",
        }
        old_system_msg = self.system_message
        self.update_system_message(instruction["content"])
        reply = self._oai_reply([roles_msg] + self.chat_messages[sender.name])
        self.update_system_message(old_system_msg)
        return reply

    def _participant_roles(self):
        return "\n".join([f"{agent.name}: {agent.system_message}" for agent in self.chat_manager.room.agents])
