import chromadb
from flaml.autogen.agentchat.agent import Agent
from flaml.autogen.agentchat import UserProxyAgent
from flaml.autogen.retrieval_utils import create_vector_db_from_dir, query_vector_db, num_tokens_from_text
from flaml.autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang

from typing import Callable, Dict, Optional, Union, List
from IPython import get_ipython


PROMPT = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the
context provided by the user. You should follow the following steps to answer a question:
Step 1, you estimate the user's intent based on the question and context. The intent can be a code generation task or
a QA task.
Step 2, you generate code or answer the question based on the intent.
You should leverage the context provided by the user as much as possible. If you think the context is not enough, you
can reply exactly "UPDATE CONTEXT" to ask the user to provide more contexts.
For code generation, you must obey the following rules:
You MUST NOT install any packages because all the packages needed are already installed.
The code will be executed in IPython, you must follow the formats below to write your code:
```python
# your code
```

User's question is: {input_question}

Context is: {input_context}
"""


def _is_termination_msg_retrievechat(message):
    """Check if a message is a termination message."""
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == "python" or c[0] == "wolfram":
            contain_code = True
            break
    return not contain_code


class RetrieveUserProxyAgent(UserProxyAgent):
    def __init__(
        self,
        name="RetrieveChatAgent",  # default set to RetrieveChatAgent
        is_termination_msg: Optional[Callable[[Dict], bool]] = _is_termination_msg_retrievechat,
        human_input_mode: Optional[str] = "ALWAYS",
        retrieve_config: Optional[Dict] = None,  # config for the retrieve agent
        **kwargs,
    ):
        """
        Args:
            name (str): name of the agent.
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            retrieve_config (dict or None): config for the retrieve agent.
                To use default config, set to None. Otherwise, set to a dictionary with the following keys:
                - client (Optional, chromadb.Client): the chromadb client.
                    If key not provided, a default client `chromadb.Client()` will be used.
                - docs_path (Optional, str): the path to the docs directory.
                    If key not provided, a default path `./docs` will be used.
                - collection_name (Optional, str): the name of the collection.
                    If key not provided, a default name `flaml-docs` will be used.
                - model (Optional, str): the model to use for the retrieve chat.
                    If key not provided, a default model `gpt-3.5` will be used.
                - chunk_token_size (Optional, int): the chunk token size for the retrieve chat.
                    If key not provided, a default size `max_tokens * 0.6` will be used.
            **kwargs (dict): other kwargs in [UserProxyAgent](user_proxy_agent#__init__).
        """
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            **kwargs,
        )

        self._retrieve_config = {} if retrieve_config is None else retrieve_config
        self._client = self._retrieve_config.get("client", chromadb.Client())
        self._docs_path = self._retrieve_config.get("docs_path", "./docs")
        self._collection_name = self._retrieve_config.get("collection_name", "flaml-docs")
        self._model = self._retrieve_config.get("model", "gpt-3.5")
        self._max_tokens = self.get_max_tokens(self._model)
        self._chunk_token_size = int(self._retrieve_config.get("chunk_token_size", self._max_tokens * 0.6))
        self._collection = False  # whether the collection is created
        self._ipython = get_ipython()
        self._doc_idx = -1  # the index of the current used doc
        self._results = []  # the results of the current query

    @staticmethod
    def get_max_tokens(model="gpt-3.5"):
        if "gpt-4-32k" in model:
            return 32000
        elif "gpt-4" in model:
            return 8000
        else:
            return 4000

    def _reset(self):
        self._oai_conversations.clear()

    def receive(self, message: Union[Dict, str], sender: "Agent"):
        """Receive a message from another agent.
        If "Update Context" in message, update the context and reset the messages in the conversation.
        """
        message = self._message_to_dict(message)
        if "UPDATE CONTEXT" in message.get("content", "")[-20::].upper():
            print("Updating context and resetting conversation.")
            self._reset()
            results = self._results
            doc_contents = ""
            _doc_idx = self._doc_idx
            for idx, doc in enumerate(results["documents"][0]):
                if idx <= _doc_idx:
                    continue
                _doc_contents = doc_contents + doc + "\n"
                if num_tokens_from_text(_doc_contents) > self._chunk_token_size:
                    break
                print(f"Adding doc_id {results['ids'][0][idx]} to context.")
                doc_contents = _doc_contents
                self._doc_idx = idx

            if self.customized_prompt:
                message = (
                    self.customized_prompt + "\nUser's question is: " + self.problem + "\nContext is: " + doc_contents
                )
            else:
                message = PROMPT.format(input_question=self.problem, input_context=doc_contents)
            self.send(message, sender)
        else:
            super().receive(message, sender)

    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = ""):
        if not self._collection:
            create_vector_db_from_dir(
                dir_path=self._docs_path,
                max_tokens=self._chunk_token_size,
                client=self._client,
                collection_name=self._collection_name,
            )
            self._collection = True

        results = query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            client=self._client,
            collection_name=self._collection_name,
        )
        self._results = results
        print("doc_ids: ", results["ids"])

    def generate_init_message(
        self, problem: str, customized_prompt: str = "", n_results: int = 20, search_string: str = ""
    ):
        """Generate a prompt for the assitant agent with the given problem and prompt.

        Args:
            problem (str): the problem to be solved.
            customized_prompt (str): a customized prompt to be used. If it is not "", the built-in prompt will be
            ignored.

        Returns:
            str: the generated prompt ready to be sent to the assistant agent.
        """
        self.reset()
        self.retrieve_docs(problem, n_results, search_string)
        results = self._results
        doc_contents = ""
        for idx, doc in enumerate(results["documents"][0]):
            _doc_contents = doc_contents + doc + "\n"
            if num_tokens_from_text(_doc_contents) > self._chunk_token_size:
                break
            print(f"Adding doc_id {results['ids'][0][idx]} to context.")
            doc_contents = _doc_contents
            self._doc_idx = idx

        if customized_prompt:
            self.customized_prompt = customized_prompt
            msg = customized_prompt + "\nUser's question is:" + problem + "\nContext is:" + doc_contents
        else:
            self.customized_prompt = ""
            msg = PROMPT.format(input_question=problem, input_context=doc_contents)
        self.problem = problem
        return msg

    def run_code(self, code, **kwargs):
        lang = kwargs.get("lang", None)
        if code.startswith("!") or code.startswith("pip") or lang in ["bash", "shell", "sh"]:
            return (
                0,
                bytes(
                    "You MUST NOT install any packages because all the packages needed are already installed.", "utf-8"
                ),
                None,
            )
        result = self._ipython.run_cell(code)
        log = str(result.result)
        exitcode = 0 if result.success else 1
        if result.error_before_exec is not None:
            log += f"\n{result.error_before_exec}"
            exitcode = 1
        if result.error_in_exec is not None:
            log += f"\n{result.error_in_exec}"
            exitcode = 1
        return exitcode, bytes(log, "utf-8"), None
