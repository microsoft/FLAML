from .user_proxy_agent import UserProxyAgent
from typing import Optional, Callable
from transformers import AutoTokenizer
import asyncio


class TeachingAgent(UserProxyAgent):
    """(Experimental) A teaching agent."""

    def __init__(
        self,
        name,
        system_message="",
        work_dir=None,
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=None,
        is_termination_msg=None,
        use_docker=True,
        **config,
    ):
        """
        Args:
            name (str): name of the agent
            system_message (str): system message to be sent to the agent
            work_dir (str): working directory for the agent
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or or when is_termination_msg is True.
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            is_termination_msg (function): a function that takes a message and returns a boolean value.
                This function is used to determine if a received message is a termination message.
            use_docker (bool): whether to use docker to execute the code.
            **config (dict): other configurations.
        """
        super().__init__(
            name,
            system_message,
            work_dir=work_dir,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            is_termination_msg=is_termination_msg,
            use_docker=use_docker,
            **config,
        )
        self._data4learning = []
        self._learning_constraints = None
        self._learning_objectives = None
        self._learning_results = None
        self._learning_func = None
        self._can_handle_data_volume = lambda *args: True
        self._data_available_event = asyncio.Event()

    def setup_learning(
        self,
        learning_func: Optional[Callable] = None,
        learning_objectives: Optional[str] = None,
        learning_constraints: Optional[dict] = None,
        learning_results: Optional[str] = "",
    ):
        """
        Args:
            learning_func (Optional, Callable): the learning function to be executed.
                The learning function should take the following arguments as inputs:
                    (1) data4learning: the data for learning.
                    (2) learning_results: old learning results.
                The learning function should return the new learning results.
            learning_objectives (Optional, str): the learning objectives in natural language.
            learning_constraints (Optional, dict): the learning constraints.
            learning_results (Optional, str): the learning results in natural language.
            #TODO: learning_results could be other types of data, e.g., a list of data.
        Either learning_func or learning_objectives should be provided.
        """
        self._learning_constraints = learning_constraints
        self._learning_objectives = learning_objectives  # already reflected in the learning_func
        self._learning_results = learning_results
        self._learning_func = learning_func
        assert (
            self._learning_func is not None or self._learning_objectives is not None
        ), "learning_func or learning_objectives should be provided"

        self._learning_settings = {
            "learning_func": self._learning_func,
            "learning_objectives": self._learning_objectives,
            "learning_constraints": self._learning_constraints,
            "learning_results": self._learning_results,
            "data4learning": [],
        }

    def generate_init_prompt(self):
        """
        When generating the init prompt, we need to distinguish the two cases where learning_func or learning_objectives is provided.
        """
        self._init_prompt = self._learning_settings.copy()

        return self._init_prompt

    async def add_data(self, data4learning):
        """Add data for learning."""
        self._data4learning += data4learning
        print(f"{len(data4learning)} data entries added for learning!")
        self._data_available_event.set()

    async def auto_reply(self, message, sender, default_reply=""):
        """
        Need to distinguish if the sender is requesting for learning data or not
        """
        learning_results = message.get("learning_results", "")
        can_handle_data_volume = message.get("can_handle_data_volume") or self._can_handle_data_volume
        current_data4learning = []
        # Wait here if no data is available
        while not self._data4learning:
            print("waiting for data...")
            await self._data_available_event.wait()
        # Reset the event as we are going to consume data
        self._data_available_event.clear()
        while self._data4learning:
            combined_data_str = "\n".join(current_data4learning + [self._data4learning[0]])
            if can_handle_data_volume(learning_results, combined_data_str):
                current_data4learning.append(self._data4learning.pop(0))
            else:
                break
        if current_data4learning:
            response = {
                "learning_results": learning_results,
                "data4learning": current_data4learning,
            }
            await self._send(response, sender)
        else:
            print("no data for learning and thus terminate the conversation")
