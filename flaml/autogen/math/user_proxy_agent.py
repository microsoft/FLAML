import json
import sys
from io import StringIO
import regex
import os
import re
from pydantic import BaseModel, Field, Extra, root_validator
from typing import Any, Dict, Optional
from flaml.autogen.code_utils import execute_code
from time import sleep


class UserProxyAgent:
    def __init__(self):
        self.previous_code = "import sympy\nfrom sympy import symbols, Eq, solve\nfrom fractions import Fraction\n"

        self.valid_q_count = 0
        self.total_q_count = 0

        self.last_query = None
        self.last_return = None
        self.consecutive_continue = 0

    def check_queries(self, response: str):
        """check if there is a query in the response"""
        queries = self.extractJSON(response)  # extract json queries
        if len(queries) == 0:
            queries = self.extractCode(response)  # extract code queries
            if len(queries) == 0:
                if (
                    ("tool" in response and "query" in response)
                    or ("python" in response and "wolfram" in response)
                    or "```" in response
                ):
                    return (
                        "\nYour query is invalid and cannot be parsed. (If you already get the answer, put it in \\boxed{}.)",
                        True,
                    )
                else:
                    return "", False

        return (
            "\nAbove is the result to the queries. If you get to the final answer, put it in \\boxed{}",
            True,
        )

    def handle_query(self, response: str):
        """Handle a list of queries and return the output.
        Args:
            response: string with a list of queries
        returns:
            output: string with the output of the queries
            is_success: boolean indicating whether the queries were successful
        """
        queries = self.extractJSON(response)  # extract json queries
        if len(queries) == 0:
            queries = self.extractCode(response)  # extract code queries
            if len(queries) == 0:
                if (
                    ("tool" in response and "query" in response)
                    or ("python" in response and "wolfram" in response)
                    or "```" in response
                ):
                    return "Your query is invalid and cannot be parsed. Please revise your query format.", False
                else:
                    # self.consecutive_continue += 1
                    # if self.consecutive_continue >= 3:
                    #     self.consecutive_continue = 0
                    #     return "Continue. Please keep solving the problem until you need to query. (If you get to the answer already, put it in \\boxed{}.)", True
                    return (
                        "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)",
                        True,
                    )

        self.consecutive_continue = 0
        self.total_q_count += len(queries)
        self.valid_q_count += len(queries)

        buffer_out = ""
        all_success = True  # all queries are successful
        for i, query in enumerate(queries):
            if "tool" in query:
                # old format of query in json format, ignore
                if query["tool"] == "python":
                    output, is_success = self.execute_python_code(query["query"])
                elif query["tool"] == "wolfram":
                    output, is_success = self.execute_wolfram_query(query["query"])
                else:
                    output = "Error: Unknown tool"
                    is_success = False
            else:
                output = ""
                is_success = False
                if "python" in query and query["python"] != "":
                    pyout, pysucess = self.execute_python_code(query["python"])
                    output += "python: " + pyout + "\n"
                    is_success = is_success or pysucess
                if "wolfram" in query and query["wolfram"] != "":
                    wolframout, wolframsuccess = self.execute_wolfram_query(query["wolfram"])
                    output += "wolfram: " + wolframout + "\n"
                    is_success = is_success or wolframsuccess
                # add new query handling here

            buffer_out += output + "\n"
            if not is_success:
                # TODO: handle situation with several queries and one fails
                all_success = False
                self.valid_q_count -= 1  # invalid query
        buffer_out = buffer_out.strip()
        if self.last_query == tuple(queries) or self.last_return == buffer_out:
            return (
                buffer_out + "\nYour query or result is same from the last, please try a new approach.",
                False,
            )
        self.last_query = tuple(queries)
        self.last_return = buffer_out
        return buffer_out, all_success

    def extractCode(self, input_string: str):
        """Extract code blocks from message."""
        pattern = r"```(.*?)```"
        match = re.findall(pattern, input_string, flags=re.DOTALL)

        queries = []
        for m in match:
            if "python" in m:
                queries.append({"tool": "python", "query": m.replace("python", "").strip()})
            elif "wolfram" in m:
                queries.append({"tool": "wolfram", "query": m.replace("wolfram", "").strip()})
            # add new query handling here
        return queries

    def execute_wolfram_query(self, query: str):
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
        return output, is_success

    # code query handler
    def execute_python_code(self, query: str):
        """Run one code query and return the output.
        params:
            query: string with the code query
        """
        query = query.replace("; ", "\n").replace(";", "\n")
        code = self.previous_code + self.add_print_to_last_line(query)

        # python_repl = PythonREPL()
        # output, is_success = python_repl.run(code)
        return_code, output = execute_code(code, use_docker=False, timeout=5)
        is_success = return_code == 0
        if isinstance(output, bytes):
            try:
                output = output.decode("ascii")
            except Exception:
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

        if not is_success:
            output = "Error: " + output
        elif output == "":
            if "print" not in query:
                output = "No output found. Make sure you print the results."
                is_success = False
            else:
                output = "No output found."
                is_success = True

        if len(output) > 2000:
            output = "You required too much output. Please print only the necessary output."
            is_success = False

        if is_success:
            # remove print and check if it still works
            tmp = self.previous_code + "\n" + self.remove_print(query) + "\n"
            rcode, _ = execute_code(tmp, use_docker=False)
        else:
            tmp = self.previous_code + "\n"
            for line in query.split("\n"):
                if "import" in line:
                    tmp += line + "\n"
            rcode, _ = execute_code(tmp, use_docker=False)
        if rcode == 0:
            self.previous_code = tmp
        return output, is_success

    def add_print_to_last_line(self, s):
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

    def remove_print(self, s):
        # remove all print statements from a string
        lines = s.splitlines()
        lines = [line for line in lines if "print(" not in line]
        return "\n".join(lines)

    def _remove_newlines_outside_quotes(self, s):
        """Remove newlines outside of quotes.

        Return from openai:
            s = "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}"

        if calling json.loads(s), it will throw an error because of the newline in the query.
        So this function removes the newline in the query outside of quotes.

        _remove_newlines_outside_quotes(s) -> "{"tool": "python","query": "print('hello')\nprint('world')"}"


        params:
            s: string to remove newlines from
        returns:
            string with newlines removed

        Example:

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

    def extractJSON(self, input_string: str):
        """
        Extract JSON queries from a string.
        params:
            input_string: string to extract JSON queries from
        returns:
            list of JSON queries
        """
        input_string = input_string.replace(",\n}", "}")
        # bracketed_strings = re.findall(r'\{[\s\S]*?\}', input_string)
        bracketed_strings = regex.findall(r"\{(?:[^{}]|(?R))*\}", input_string)
        # print(bracketed_strings)
        # Extract valid JSON queries
        json_queries = []
        for bracketed_string in bracketed_strings:
            bracketed_string = self._remove_newlines_outside_quotes(bracketed_string)
            try:
                data = json.loads(bracketed_string)
                if ("tool" in data and "query" in data) or "python" in data or "wolfram" in data:
                    json_queries.append(data)
            except json.JSONDecodeError:
                pass

        return json_queries


# Imported from langchain
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

    @root_validator()
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
