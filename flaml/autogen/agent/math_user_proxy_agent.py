from .user_proxy_agent import UserProxyAgent
from flaml.autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang
from flaml.autogen.math_utils import get_answer
from collections import defaultdict
import re

# PROMPTS = {
#     # default
#     "default": """Let's use Python to solve a math problem.

# Query requirements:
# You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
# You can use packages like sympy to help you.
# You must follow the formats below to write your code:
# ```python
# # your code
# ```

# First state the key idea to solve the problem. You may choose from three ways to solve the problem:
# Case 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.
# Case 2: If the problem is mostly reasoning, you can solve it by yourself directly.
# Case 3: If the problem cannot be handled in the above two ways, please follow this process:
# 1. Solve the problem step by step (do not over-divide the steps).
# 2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).
# 3. Wait for me to give the results.
# 4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

# After all the queries are run and you get the answer, put the answer in \\boxed{}.

# """,
#     # select python or wolfram
#     "two_tools": """Let's use two tools (Python and Wolfram alpha) to solve a math problem.

# Query requirements:
# You must follow the formats below to write your code:
# For Wolfram Alpha:
# ```wolfram
# # your wolfram query
# ```
# For Python:
# ```python
# # your code
# ```
# When using Python, you should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
# You can use packages like sympy to help you.


# Please follow this process:
# 1. Solve the problem step by step (do not over-divide the steps).
# 2. Take out any queries that can be asked through Python or Wolfram Alpha and select the most suitable tool to be used (for example, any calculations or equations that can be calculated).
# 3. Wait for me to give the results.
# 4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

# After all the queries are run and you get the answer, put the final answer in \\boxed{}.

# """,
#     # use python step by step
#     "python": """Let's use Python to solve a math problem.

# Query requirements:
# You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
# You can use packages like sympy to help you.
# You must follow the formats below to write your code:
# ```python
# # your code
# ```

# Please follow this process:
# 1. Solve the problem step by step (do not over-divide the steps).
# 2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).
# 3. Wait for me to give the results.
# 4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

# After all the queries are run and you get the answer, put the answer in \\boxed{}.

# """,
# }
# self.prompt = new_prompt if new_prompt is not None else PROMPTS[prompt_type]


class MathUserProxyAgent(UserProxyAgent):
    """(Experimental) A MathChat agent that can handle math problems."""

    MAX_CONSECUTIVE_AUTO_REPLY = 10  # maximum number of consecutive auto replies (subject to future change)

    def __init__(
        self,
        name="MathChatAgent",
        system_message="",
        work_dir=None,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=None,
        is_termination_msg=None,
        use_docker=True,
        max_invalid_q_per_step=3,  # new parameter
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
        if is_termination_msg is None:

            def is_termination_msg(x):
                return get_answer(x) is not None and get_answer(x) != ""

        super().__init__(
            name=name,
            system_message=system_message,
            work_dir=work_dir,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            is_termination_msg=is_termination_msg,
            use_docker=use_docker,
            **config,
        )

        self._max_invalid_q_per_step = max_invalid_q_per_step
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ""

    def reset(self):
        self._accum_invalid_q_per_step = 0
        self._valid_q_count = 0
        self._total_q_count = 0

    def _add_print_to_last_line(self, s):
        # first check if there is already a print statement
        if "print(" in s:
            return s

        # Input a string, extract the last line, enclose it in print() and return the new string
        lines = s.splitlines()
        last_line = lines[-1]
        if " = " in last_line:
            last_line = "print(" + last_line.split(" = ")[0] + ")"
            lines.append(last_line)
        else:
            lines[-1] = "print(" + last_line + ")"

        # Join the lines back together
        return "\n".join(lines)

    def _remove_print(self, s):
        # remove all print statements from a string
        lines = s.splitlines()
        lines = [line for line in lines if "print(" not in line]
        return "\n".join(lines)

    def _execute_one_python_code(self, pycode):
        pycode = pycode.replace("; ", "\n").replace(";", "\n")
        pycode = self._previous_code + self._add_print_to_last_line(pycode)

        return_code, output = execute_code(pycode, use_docker=False, timeout=5)
        is_success = return_code == 0

        # Decode the output
        if isinstance(output, bytes):
            try:
                output = output.decode("utf-8")
            except Exception:
                try:
                    output = output.decode("ascii")
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
            tmp = self._previous_code + "\n" + self._remove_print(pycode) + "\n"
            rcode, _ = execute_code(tmp, use_docker=False)
        else:
            # only add imports and check if it works
            tmp = self._previous_code + "\n"
            for line in pycode.split("\n"):
                if "import" in line:
                    tmp += line + "\n"
            rcode, _ = execute_code(tmp, use_docker=False)

        if rcode == 0:
            self._previous_code = tmp
        return output, is_success

    def auto_reply(self, message, sender, default_reply=""):
        """Generate an auto reply."""
        code_blocks = extract_code(message)

        reply = default_reply
        is_success = True
        for code_block in code_blocks:
            lang, code = code_block
            if not lang:
                lang = infer_lang(code)
            if lang == "python":
                reply, is_success = self._execute_one_python_code(code)

            elif lang == "wolfram":
                # TODO: Allow wolfram code
                pass

        if reply == "":
            reply = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"

        if not is_success:
            self._accum_invalid_q_per_step += 1
            if self._accum_invalid_q_per_step > self._max_invalid_q_per_step:
                self._accum_invalid_q_per_step = 0
                reply = "Please revisit the problem statement and your reasoning. If you think this step is correct, solve it yourself and continue the next step. Otherwise, correct this step."

        self._send(reply, sender)
