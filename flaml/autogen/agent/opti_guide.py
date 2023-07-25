"""A simplified implementation of OptiGuide framework with FLAML.
For more details, read: https://arxiv.org/abs/2307.03875

The design here is that a user asked a question, then OptiGuide will answer
it.

The OptiGuide agent will interact with the LLMProxy agent to access LLMs.

Notes:
1. We assume there is a Gurobi model `m` in the global scope.
2. Some helper functions can be integrated with other FLAML components, such as
code_utils.py. Please note and update the code accordingly.
3. We do not handle docker here. Please test and add docker support if missing.
4. We simplify the evaluation only to "DATA CODE" and "CONSTRAINT CODE", where
we would insert the newly added code.
"""
import pdb
import re
from typing import Dict, Union

import gurobipy as gp
from eventlet.timeout import Timeout
from flaml import oai
from flaml.autogen.code_utils import DEFAULT_MODEL, extract_code
from gurobipy import GRB
from termcolor import colored

from .agent import Agent
from .assistant_agent import AssistantAgent
from .user_proxy_agent import UserProxyAgent

ASSIST_SYSTEM_MSG = """You are OptiGuide,
an agent to write Python code and to answer users questions for supply chain-related coding project.


You are a chatbot to explain solutions from a Gurobi/Python solver.


--- SOURCE CODE ---
{source_code}

--- DOC STR ---
{doc_str}


Here are some example questions and their answers and codes:
--- EXAMPLES ---
{example_qa}


---


The execution result of the original source code is below.
--- Original Result ---
{execution_result}


Note that your written code will be added to the lines with substring "# OPTIGUIDE *** CODE GOES HERE".
So, you don't need to write other code, such as m.optimize() or m.update().
You just need to write code snippet.
"""

SAFEGUARD_SYSTEM_MSG = """
Given the original source code:
{source_code}

Is the following code safe (not malicious code to break security and privacy) to run?
Answer only one word.
If not safe, answer `DANGER`; else, answer `SAFE`.
"""


DATA_CODE_STR = "# OPTIGUIDE DATA CODE GOES HERE"
CONSTRAINT_CODE_STR = "# OPTIGUIDE CONSTRAINT CODE GOES HERE"


# %%
class OptiGuideAgent(Agent):
    """OptiGuide is an agent to write Python code and to answer users
    questions for supply chain-related coding project.

    Here, the OptiGuide agent manages three agents (coder, safeguard, and interpreter)
    and two assistant agents (pencil and shield).
    """

    def __init__(self, name, source_code, doc_str="", example_qa="", **config):
        """
        Args:
            name (str): agent name.
            source_code (str): The original source code to run.
            system_message (str): system message to be sent to the agent.
            **config (dict): other configurations allowed in
              [oai.Completion.create](../oai/Completion#create).
              These configurations will be used when invoking LLM.
        """
        assert source_code.find(DATA_CODE_STR) >= 0, "DATA_CODE_STR not found."
        assert source_code.find(CONSTRAINT_CODE_STR) >= 0, "CONSTRAINT_CODE_STR not found."

        super().__init__(name, system_message="")
        self._source_code = source_code
        self._sender_dict = {}

        execution_result = _run_with_exec(source_code)
        self.writing_sys_msg = ASSIST_SYSTEM_MSG.format(
            source_code=source_code, doc_str=doc_str, example_qa=example_qa, execution_result=execution_result
        )
        self.shield_sys_msg = SAFEGUARD_SYSTEM_MSG.format(source_code=source_code)

        self._debug_opportunity = 3

    def receive(self, message: Union[Dict, str], sender):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.

        OptiGuide will invoke its components: coder, safeguard, and interpret
        to generate the reply.

        The steps follows the one in Figure 2 of the OptiGuide paper.
        """
        # Step 1: receive the message
        message = self._message_to_dict(message)
        super().receive(message, sender)
        # default reply is empty (i.e., no reply, in this case we will try to generate auto reply)
        reply = ""
        question = message["content"]

        user_chat_history = f"\nHere are the history of discussions:\n{self._oai_conversations[sender.name]}"

        # Spawn coder, interpreter, and safeguard
        # Note: we use the same agent for coding and interpreting, so that they
        # can share the same chat history.
        coder = interpreter = UserProxyAgent(
            "coder interpreter", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
        )
        safeguard = UserProxyAgent(
            "safeguard", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
        )

        # Spawn the assistants for coder, interpreter, and safeguard
        pencil = AssistantAgent("pencil", system_message=self.writing_sys_msg + user_chat_history)
        shield = AssistantAgent("shield", system_message=self.shield_sys_msg + user_chat_history)

        debug_opportunity = self._debug_opportunity

        new_code = None
        execution_rst = None
        success = False
        while execution_rst is None and debug_opportunity > 0:
            # Step 2: Write code
            if new_code is None:
                # The first time we are trying to solve the question in this session
                msg = CODE_PROMPT.format(question=question)
            else:
                # debug code
                msg = DEBUG_PROMPT.format(question, new_code, execution_rst)

            coder.send(message=msg, recipient=pencil)
            new_code = coder.oai_conversations[pencil.name][-1]["content"]
            new_code = extract_code(new_code)[0][1]  # First code block, the code (excluding language)
            print(colored(new_code, "green"))

            # Step 3: safeguard
            safeguard.send(message=SAFEGUARD_PROMPT.format(code=new_code), recipient=shield)
            safe_msg = safeguard.oai_conversations[shield.name][-1]["content"]
            print("Safety:", colored(safe_msg, "blue"))

            if safe_msg.find("DANGER") < 0:
                # Step 4 and 5: Run the code and obtain the results
                src_code = _insert_code(self._source_code, new_code)
                execution_rst = _run_with_exec(src_code)

                print(colored(str(execution_rst), "yellow"))
                if type(execution_rst) in [str, int, float]:
                    success = True
                    break  # we successfully run the code and get the result
            else:
                # DANGER: If not safe, try to debug. Redo coding
                execution_rst = """
                Sorry, this new code is not safe to run. I would not allow you to execute it.
                Please try to find a new way (coding) to answer the question."""

            # Try to debug and write code again
            debug_opportunity -= 1

        if success:
            # Step 6 - 7: interpret results
            interpret_msg = f"""To answer the question: {question}

        I wrote the code: {new_code}

        Here are the execution results: {execution_rst}

        Can you try to organize these information to a human readable answer?
        Remember to compare the new results to the original results you obtained in the beginning.

        --- HUMAN READABLE ANSWER ---
        """
            interpreter.send(message=interpret_msg, recipient=pencil)
            reply = interpreter.oai_conversations[pencil.name][-1]["content"]
        else:
            reply = "Sorry. I cannot answer your question."

        # Finally, step 8: send reply
        self.send(reply, sender)


# %% Helper functions to edit and run code.
# Here, we use a simplified approach to run the code snippet, which would
# replace substrings in the source code to get an updated version of code.
# Then, we use exec to run the code snippet.
# This approach replicate the evaluation section of the OptiGuide paper.


def _run_with_exec(src_code: str) -> object:
    """Run the code snippet with exec.

    Args:
        src_code (str): The source code to run.

    Returns:
        object: The result of the code snippet.
            If the code succeed, returns the objective value (float or string).
            else, return the error (exception)
    """
    locals_dict = {}
    locals_dict.update(locals())
    locals_dict.update(globals())

    timeout = Timeout(60, Exception("This is a timeout exception, in case GPT's code falls into infinite loop."))
    try:
        exec(src_code, locals_dict, locals_dict)
    except Exception as e:
        return e
    finally:
        timeout.cancel()

    try:
        status = locals_dict["m"].Status
        if status != GRB.OPTIMAL:
            if status == GRB.UNBOUNDED:
                ans = "unbounded"
            elif status == GRB.INF_OR_UNBD:
                ans = "inf_or_unbound"
            elif status == GRB.INFEASIBLE:
                ans = "infeasible"
            else:
                ans = "Model Status: " + str(status)
        else:
            ans = "Optimization problem solved. The objective value is: " + str(locals_dict["m"].objVal)
    except Exception as e:
        return e

    return ans


def _replace(src_code: str, old_code: str, new_code: str) -> str:
    """
    Inserts new code into the source code by replacing a specified old code block.

    Args:
        src_code (str): The source code to modify.
        old_code (str): The code block to be replaced.
        new_code (str): The new code block to insert.

    Returns:
        str: The modified source code with the new code inserted.

    Raises:
        None

    Example:
        src_code = 'def hello_world():\n    print("Hello, world!")\n\n# Some other code here'
        old_code = 'print("Hello, world!")'
        new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
        modified_code = insert(src_code, old_code, new_code)
        print(modified_code)
        # Output:
        # def hello_world():
        #     print("Bonjour, monde!")
        #     print("Hola, mundo!")
        # Some other code here
    """
    pattern = r"( *){old_code}".format(old_code=old_code)
    head_spaces = re.search(pattern, src_code, flags=re.DOTALL).group(1)
    new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])
    rst = re.sub(pattern, new_code, src_code)
    return rst


def _insert_code(src_code: str, new_lines: str) -> str:
    """insert a code patch into the source code.


    Args:
        src_code (str): the full source code
        new_lines (str): The new code.

    Returns:
        str: the full source code after insertion (replacement).
    """
    if new_lines.find("addConstr") >= 0:
        return _replace(src_code, CONSTRAINT_CODE_STR, new_lines)
    else:
        return _replace(src_code, DATA_CODE_STR, new_lines)


# %% Prompt for OptiGuide
CODE_PROMPT = """
Question: {question}
Answer Code:
"""

DEBUG_PROMPT = """
To answer a question, the code snippet is added.
--- QUESTION ---
{question}


Pay attention to the added code below
--- Added Code ---
{code_snippet}


While running Python code, you encountered the error:
--- ERROR ---
{error_message}

Please try to resolve this bug, and rewrite the code snippet.
--- NEW CODE ---
"""


SAFEGUARD_PROMPT = """
--- Code ---
{code}

--- One-Word Answer: SAFE or DANGER ---
"""
