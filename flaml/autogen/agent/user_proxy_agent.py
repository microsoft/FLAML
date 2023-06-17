from .agent import Agent
from flaml.autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang
from collections import defaultdict
import json


class UserProxyAgent(Agent):
    """(Experimental) A proxy agent for the user, that can execute code and provide feedback to the other agents."""

    MAX_CONSECUTIVE_AUTO_REPLY = 100  # maximum number of consecutive auto replies (subject to future change)

    def __init__(
        self,
        name,
        system_message="",
        work_dir=None,
        human_input_mode="ALWAYS",
        functions=defaultdict(callable),
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
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            functions (dict[str, callable]): a dictionary of functions that can be called by the assistant.
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            is_termination_msg (function): a function that takes a message and returns a boolean value.
                This function is used to determine if a received message is a termination message.
            use_docker (bool): whether to use docker to execute the code.
            **config (dict): other configurations.
        """
        super().__init__(name, system_message)
        self._work_dir = work_dir
        self._human_input_mode = human_input_mode
        self._is_termination_msg = (
            is_termination_msg if is_termination_msg is not None else (lambda x: x == "TERMINATE")
        )
        self._config = config
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply if max_consecutive_auto_reply is not None else self.MAX_CONSECUTIVE_AUTO_REPLY
        )
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._use_docker = use_docker
        self._functions = functions

    def _execute_code(self, code_blocks):
        """Execute the code and return the result."""
        logs_all = ""
        for code_block in code_blocks:
            lang, code = code_block
            if not lang:
                lang = infer_lang(code)
            if lang in ["bash", "shell", "sh"]:
                # if code.startswith("python "):
                #     # return 1, f"please do not suggest bash or shell commands like {code}"
                #     file_name = code[len("python ") :]
                #     exitcode, logs = execute_code(filename=file_name, work_dir=self._work_dir, use_docker=self._use_docker)
                # else:
                exitcode, logs, image = execute_code(
                    code, work_dir=self._work_dir, use_docker=self._use_docker, lang=lang
                )
                logs = logs.decode("utf-8")
            elif lang == "python":
                if code.startswith("# filename: "):
                    filename = code[11 : code.find("\n")].strip()
                else:
                    filename = None
                exitcode, logs, image = execute_code(
                    code, work_dir=self._work_dir, filename=filename, use_docker=self._use_docker
                )
                logs = logs.decode("utf-8")
            else:
                # TODO: could this happen?
                exitcode, logs, image = 1, f"unknown language {lang}"
                # raise NotImplementedError
            self._use_docker = image
            logs_all += "\n" + logs
            if exitcode != 0:
                return exitcode, logs_all
        return exitcode, logs_all

    def _extractArgs(self, input_string: str):
        """Extract arguements as a dict from a string.
        Args:
            input_string: string to extract arguements from
        Returns:
            a dictionary or None
        """

        def _remove_newlines_outside_quotes(s):
            """Remove newlines outside of quotes.

            if calling json.loads(s), it will throw an error because of the newline in the query.
            So this function removes the newline in the query outside of quotes.

            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"
            """
            result = []
            inside_quotes = False
            for c in s:
                if c == '"':
                    inside_quotes = not inside_quotes
                if not inside_quotes and c == "\n":
                    continue
                if inside_quotes and c == "\n":
                    c = "\\n"
                if inside_quotes and c == "\t":
                    c = "\\t"
                result.append(c)
            return "".join(result)

        input_string = _remove_newlines_outside_quotes(input_string)
        try:
            args = json.loads(input_string)
            return args
        except json.JSONDecodeError:
            return None

    def _execute_function(self, func_call):
        func_name = func_call.get("name", "")
        func = self._functions.get(func_name, None)

        is_exec_success = False
        if func is not None:
            arguments = self._extractArgs(func_call.get("arguments", ""))
            if arguments is not None:
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
            else:
                content = f"Error: Invalid arguments for function {func_name}."
        else:
            content = f"Error: Function {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content,
        }

    def auto_reply(self, message, sender, default_reply=""):
        """Generate an auto reply."""
        if "function_call" in message:
            is_exec_success, func_return = self._execute_function(message["function_call"])
            self._send(func_return, sender)
            return

        code_blocks = extract_code(message["content"])
        if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
            # no code block is found, lang should be `UNKNOWN``
            self._send({"role": "user", "content": default_reply}, sender)
        else:
            # try to execute the code
            exitcode, logs = self._execute_code(code_blocks)
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            self._send(
                {"role": "user", "content": f"exitcode: {exitcode} ({exitcode2str})\nCode output: {logs}"}, sender
            )

    def receive(self, message, sender):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        super().receive(message, sender)
        # default reply is empty (i.e., no reply, in this case we will try to generate auto reply)
        reply = ""
        if self._human_input_mode == "ALWAYS":
            reply = input(
                "Provide feedback to the sender. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
        elif self._consecutive_auto_reply_counter[
            sender.name
        ] >= self._max_consecutive_auto_reply or self._is_termination_msg(message["content"]):
            if self._human_input_mode == "TERMINATE":
                reply = input(
                    "Please give feedback to the sender. (Press enter or type 'exit' to stop the conversation): "
                )
                reply = reply if reply else "exit"
            else:
                # this corresponds to the case when self._human_input_mode == "NEVER"
                reply = "exit"
        if reply == "exit" or (self._is_termination_msg(message["content"]) and not reply):
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            return
        if reply:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            self._send({"role": "user", "content": reply}, sender)
            return

        self._consecutive_auto_reply_counter[sender.name] += 1
        print(">>>>>>>> NO HUMAN INPUT RECEIVED. USING AUTO REPLY FOR THE USER...", flush=True)
        self.auto_reply(message, sender, default_reply=reply)
