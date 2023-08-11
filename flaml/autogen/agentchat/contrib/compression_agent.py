
from typing import Callable, Dict, Optional, Union, Tuple, List, Any
from flaml.autogen import oai
from ..agent import Agent
from ..responsive_agent import ResponsiveAgent
from ..agent_utils import count_token, token_left

class CompressionAgent(ResponsiveAgent):
    """(Experimental) Assistant agent, designed to solve a task with LLM.

    AssistantAgent is a subclass of ResponsiveAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant that will summarize and compress previous messages. The user will input a whole chunk of conversation history. Possible titles include "User", "Assistant", "Function Call" and "Function Return".

Please follow the rules:
1. You should summarize each of the message and reserve these titles. You should also reserve important subtitles and structure within each message, for example, "case", "step" or bullet points. 
2. For very short messages, you can choose to not summarize them. For important information like the desription of a problem or task, you should reserve them (if it is not too long).
3. For code snippets, you have two options: 1. reserve the whole exact code snippet. 2. summerize it use this format:
CODE: <code type, python, etc>
GOAL: <purpose of this code snippet in a short sentence>
IMPLEMENTATION: <overall structure of the code>"""

    def __init__(
        self,
        name: str = "compressor" ,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, bool]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            llm_config (dict): llm inference configuration.
                Please refer to [autogen.Completion.create](/docs/reference/autogen/oai/completion#create)
                for available options.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            **kwargs (dict): Please refer to other kwargs in
                [ResponsiveAgent](responsive_agent#__init__).
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs,
        )

        self._reply_func_list.clear()
        self.register_auto_reply(Agent, CompressionAgent.generate_compressed_reply)

    def generate_compressed_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        context: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
            
        # TOTHINK: different models have different max_length, right now we will use context passed in, so the model will be the same with the source model.
        # If original model is gpt-4; we start compressing at 70% of usage, 70% of 8092 = 5664; and we use gpt 3.5 here max_toke = 4096, it will raise error. choosinng model automatically?
        
        # use passed in context and messages
        llm_config = self.llm_config if context is None else context
        if llm_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        if len(messages) <= 1:
            print(f"Warning: the first message contains {count_token(messages)} tokens, which will not be compressed.")
            return False, None

        # 1. put all history into one, except the first one
        user_message = "Chat History:\n"
        for m in messages[1:]:
            if m.get("role") == "function":
                user_message += f"Function Return: \"{m['name']}\"\n {m['content']}\n"
            else:
                user_message += f"{m['role']}: \n {m['content']}\n"
                if "function_call" in m:
                    user_message += f"Function Call: \"{m['function_call']}\"\n"
        
        # 2. ask LLM to compress
        response = oai.ChatCompletion.create(
            context=messages[-1].pop("context", None), messages=self._oai_system_message + [user_message], **llm_config
        )
        compressed_message = oai.ChatCompletion.extract_text_or_function_call(response)[0]
        assert isinstance(compressed_message, str), f"compressed_message should be a string: {compressed_message}"

        # 3. add compressed message to the first message and return
        messages = messages[0] + compressed_message [{"content": "Compressed Content of Chat History:\n" + compressed_message, "role": "user", "name": "Compressed User"}]
        return True, messages





