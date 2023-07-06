from .user_proxy_agent import UserProxyAgent
from flaml.autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang
from flaml.autogen.math_utils import get_answer
from collections import defaultdict
import re
import os
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Dict, Optional
from time import sleep


PROMPTS = {
    # default
    "default": """Let's use Python to solve a math problem.

Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.
Case 2: If the problem is mostly reasoning, you can solve it by yourself directly.
Case 3: If the problem cannot be handled in the above two ways, please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem:
""",
    # select python or wolfram
    "two_tools": """Let's use two tools (Python and Wolfram alpha) to solve a math problem.

Query requirements:
You must follow the formats below to write your query:
For Wolfram Alpha:
```wolfram
# one wolfram query
```
For Python:
```python
# your code
```
When using Python, you should always use the 'print' function for the output and use fractions/radical forms instead of decimals. You can use packages like sympy to help you.
When using wolfram, give one query in each code block.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python or Wolfram Alpha, select the most suitable tool to be used (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the final answer in \\boxed{}.

Problem: """,
    # use python step by step
    "python": """Let's use Python to solve a math problem.

Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem: """,
}


def is_termination_msg(x):
    """Check if a message is a termination message."""
    if isinstance(x, dict):
        x = x.get("content")
        if x is None:
            return False
    cb = extract_code(x)
    contain_code = False
    for c in cb:
        if c[0] == "python" or c[0] == "wolfram":
            contain_code = True
            break
    return not contain_code and get_answer(x) is not None and get_answer(x) != ""


def add_print_to_last_line(s):
    """Add print() to the last line of a string."""
    # 1. check if there is already a print statement
    if "print(" in s:
        return s
    # 2. extract the last line, enclose it in print() and return the new string
    lines = s.splitlines()
    last_line = lines[-1]
    if "\t" in last_line or "=" in last_line:
        return s
    if "=" in last_line:
        last_line = "print(" + last_line.split(" = ")[0] + ")"
        lines.append(last_line)
    else:
        lines[-1] = "print(" + last_line + ")"
    # 3. join the lines back together
    return "\n".join(lines)


def remove_print(s):
    """remove all print statements from a string."""
    lines = s.splitlines()
    lines = [line for line in lines if not line.startswith("print(")]
    return "\n".join(lines)


class MathUserProxyAgent(UserProxyAgent):
    """(Experimental) A MathChat agent that can handle math problems."""

    MAX_CONSECUTIVE_AUTO_REPLY = 15  # maximum number of consecutive auto replies (subject to future change)

    def __init__(
        self,
        name="MathChatAgent",  # default set to MathChatAgent
        system_message="",
        work_dir=None,
        function_map=defaultdict(callable),
        human_input_mode="NEVER",  # Fully automated
        max_consecutive_auto_reply=None,
        is_termination_msg=is_termination_msg,
        use_docker=True,
        max_invalid_q_per_step=3,  # a parameter needed in MathChat
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
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            is_termination_msg (function): a function that takes a message in the form of a dictionary and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            use_docker (bool): whether to use docker to execute the code.
            max_invalid_q_per_step (int): (ADDED) the maximum number of invalid queries per step.
            **config (dict): other configurations.
        """
        super().__init__(
            name=name,
            system_message=system_message,
            work_dir=work_dir,
            function_map=function_map,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            is_termination_msg=is_termination_msg,
            use_docker=use_docker,
            **config,
        )

        # fixed var
        self._max_invalid_q_per_step = max_invalid_q_per_step

        # mutable
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ""
        self.last_reply = None

    def generate_init_prompt(self, problem, prompt_type="default", customized_prompt=None):
        """Generate a prompt for the assitant agent with the given problem and prompt.

        Args:
            problem (str): the problem to be solved.
            prompt_type (str): the type of the prompt. Possible values are "default", "python", "wolfram".
                (1) "default": the prompt that allows the agent to choose between 3 ways to solve a problem:
                    1. write a python program to solve it directly.
                    2. solve it directly without python.
                    3. solve it step by step with python.
                (2) "python":
                    a simplified prompt from the third way of the "default" prompt, that asks the assistant
                    to solve the problem step by step with python.
                (3) "two_tools":
                    a simplified prompt similar to the "python" prompt, but allows the model to choose between
                    Python and Wolfram Alpha to solve the problem.
            customized_prompt (str): a customized prompt to be used. If it is not None, the prompt_type will be ignored.

        Returns:
            str: the generated prompt ready to be sent to the assistant agent.
        """
        self._reset()
        if customized_prompt is not None:
            return customized_prompt + problem
        else:
            return PROMPTS[prompt_type] + problem

    def _reset(self):
        self._oai_conversations.clear()
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ""
        self.last_reply = None

    def _execute_one_python_code(self, pycode):
        """Execute python code blocks.

        Previous python code will be saved and executed together with the new code.
        the "print" function will also be added to the last line of the code if needed
        """
        # Need to replace all "; " with "\n" to avoid syntax error when adding `print` to the last line
        pycode = pycode.replace("; ", "\n").replace(";", "\n")
        pycode = self._previous_code + add_print_to_last_line(pycode)

        return_code, output, _ = execute_code(pycode, use_docker=self._use_docker, timeout=5)
        is_success = return_code == 0

        # Decode the output
        if isinstance(output, bytes):
            try:
                output = output.decode("utf-8")
            except Exception:
                is_success = False
                output = "The return cannot be decoded."

        if not is_success:
            # Remove the file information from the error string
            pattern = r'File "/[^"]+\.py", line \d+, in .+\n'
            if type(output) == str:
                output = re.sub(pattern, "", output)
            output = "Error: " + output
        elif output == "":
            # Check if there is any print statement
            if "print" not in pycode:
                output = "No output found. Make sure you print the results."
                is_success = False
            else:
                output = "No output found."
                is_success = True

        if len(output) > 2000:
            output = "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
            is_success = False

        if is_success:
            # remove print and check if it still works
            tmp = self._previous_code + "\n" + remove_print(pycode) + "\n"
            rcode, _, _ = execute_code(tmp, use_docker=self._use_docker)
        else:
            # only add imports and check if it works
            tmp = self._previous_code + "\n"
            for line in pycode.split("\n"):
                if "import" in line:
                    tmp += line + "\n"
            rcode, _, _ = execute_code(tmp, use_docker=self._use_docker)

        if rcode == 0:
            self._previous_code = tmp
        return output, is_success

    def _execute_one_wolfram_query(self, query: str):
        """
        Run one wolfram query and return the output.
        return:
            output: string with the output of the query
            is_success: boolean indicating whether the query was successful
        """
        # wolfram query handler
        wolfram = WolframAlphaAPIWrapper()
        output, is_success = wolfram.run(query)
        if output == "":
            output = "Error: The wolfram query is invalid."
            is_success = False
        return output, is_success

    def auto_reply(self, message, sender, default_reply=""):
        """Generate an auto reply."""
        message = message.get("content", "")
        code_blocks = extract_code(message)

        if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
            # no code block is found, lang should be `UNKNOWN``
            if default_reply == "":
                default_reply = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"
            self._send(default_reply, sender)
        else:
            is_success, all_success = True, True
            reply = ""
            for code_block in code_blocks:
                lang, code = code_block
                if not lang:
                    lang = infer_lang(code)
                if lang == "python":
                    output, is_success = self._execute_one_python_code(code)
                elif lang == "wolfram":
                    output, is_success = self._execute_one_wolfram_query(code)

                reply += output + "\n"
                if not is_success:
                    all_success = False
                    self._valid_q_count -= 1  # count invalid queries

            reply = reply.strip()

            if self.last_reply == reply:
                return (
                    reply + "\nYour query or result is same from the last, please try a new approach.",
                    False,
                )
            self.last_reply = reply

            if not all_success:
                self._accum_invalid_q_per_step += 1
                if self._accum_invalid_q_per_step > self._max_invalid_q_per_step:
                    self._accum_invalid_q_per_step = 0
                    reply = "Please revisit the problem statement and your reasoning. If you think this step is correct, solve it yourself and continue the next step. Otherwise, correct this step."

            self._send(reply, sender)


# Imported from langchain. Langchain is licensed under MIT License:
# The MIT License

# Copyright (c) Harrison Chase

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


def get_from_dict_or_env(data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    elif env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


# Imported from langchain
class WolframAlphaAPIWrapper(BaseModel):
    """Wrapper for Wolfram Alpha.

    Docs for using:

    1. Go to wolfram alpha and sign up for a developer account
    2. Create an app and get your APP ID
    3. Save your APP ID into WOLFRAM_ALPHA_APPID env variable
    4. pip install wolframalpha

    """

    wolfram_client: Any  #: :meta private:
    wolfram_alpha_appid: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        wolfram_alpha_appid = get_from_dict_or_env(values, "wolfram_alpha_appid", "WOLFRAM_ALPHA_APPID")
        values["wolfram_alpha_appid"] = wolfram_alpha_appid

        try:
            import wolframalpha

        except ImportError:
            raise ImportError("wolframalpha is not installed. " "Please install it with `pip install wolframalpha`")
        client = wolframalpha.Client(wolfram_alpha_appid)
        values["wolfram_client"] = client

        return values

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        from urllib.error import HTTPError

        is_success = False  # added
        res = None
        for _ in range(20):
            try:
                res = self.wolfram_client.query(query)
                break
            except HTTPError:
                sleep(1)
            except Exception:
                return (
                    "Wolfram Alpha wasn't able to answer it. Please try a new query for wolfram or use python.",
                    is_success,
                )
        if res is None:
            return (
                "Wolfram Alpha wasn't able to answer it (may due to web error), you can try again or use python.",
                is_success,
            )

        try:
            if not res["@success"]:
                return (
                    "Your Wolfram query is invalid. Please try a new query for wolfram or use python.",
                    is_success,
                )
            assumption = next(res.pods).text
            answer = ""
            for r in res["pod"]:
                if r["@title"] == "Solution":
                    answer = r["subpod"]["plaintext"]
                if r["@title"] == "Results" or r["@title"] == "Solutions":
                    for i, sub in enumerate(r["subpod"]):
                        answer += f"ans {i}: " + sub["plaintext"] + "\n"
                    break
            if answer == "":
                answer = next(res.results).text

        except Exception:
            return (
                "Wolfram Alpha wasn't able to answer it. Please try a new query for wolfram or use python.",
                is_success,
            )

        if answer is None or answer == "":
            # We don't want to return the assumption alone if answer is empty
            return "No good Wolfram Alpha Result was found", is_success
        else:
            is_success = True
            return f"Assumption: {assumption} \nAnswer: {answer}", is_success
