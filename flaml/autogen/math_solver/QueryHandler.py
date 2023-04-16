import json
import sys
from io import StringIO
from typing import Dict, Optional
import regex
import os
from pydantic import BaseModel, Field, Extra, root_validator
from typing import Any, Dict, Optional

class QueryHandler():
    def __init__(self):
        self.previous_code = ""

        self.valid_q_count = 0
        self.total_q_count = 0
    
    def handle_query(self, response: list):
        """Handle a list of queries and return the output.
        Args:
            response: list of queries
        returns:
            output: string with the output of the queries
            is_success: boolean indicating whether the queries were successful
        """
        queries = self.extractJSON(response) # extract json queries
        if len(queries) == 0:
            return "No queries found. Please put your queries in JSON format.", False
        self.total_q_count += len(queries)
        self.valid_q_count += len(queries)

        buffer_out = ""
        all_success = True # all queries are successful
        for i, query in enumerate(queries): 
            if query['tool'] == 'python':
                output, is_success = self.run_one_code(query)
            elif query['tool'] == 'wolfram':
                output, is_success = self.wolfram_query(query)
            else:
                output = "Error: Unknown tool"
                is_success = False
            buffer_out += f'Return {i}: ' + output + '\n'
            if not is_success:
                all_success = False
                self.valid_q_count -= 1 # invalid query

        return buffer_out.replace('\n\n', '\n'), all_success

    def wolfram_query(self, query:json):
        """
        Run one wolfram query and return the output.
        return:
            output: string with the output of the query
            is_success: boolean indicating whether the query was successful
        """
        # wolfram query handler
        wolfram = WolframAlphaAPIWrapper()
        output, is_sucess = wolfram.run(query['query'])
        if output == '':
            output = 'Error: The query is invalid.'
        return output, is_sucess

    def _remove_newlines_outside_quotes(self, s):
        """
        Remove newlines outside of quotes.
        params:
            s: string to remove newlines from
        returns:
            string with newlines removed
        """
        result = []
        inside_quotes = False
        for c in s:
            if c == '"':
                inside_quotes = not inside_quotes
            if not inside_quotes and c == '\n':
                continue
            if inside_quotes and c == '\n':
                c = '\\n'
            if inside_quotes and c == '\t':
                c = '\\t'
            result.append(c)
        return ''.join(result)

    def extractJSON(self, input_string: str):
        """
        Extract JSON queries from a string.
        params:
            input_string: string to extract JSON queries from
        returns:
            list of JSON queries
        """

        # bracketed_strings = re.findall(r'\{[\s\S]*?\}', input_string)
        bracketed_strings = regex.findall(r'\{(?:[^{}]|(?R))*\}', input_string)
        # print(bracketed_strings)
        # Extract valid JSON queries
        json_queries = []
        for bracketed_string in bracketed_strings:
            bracketed_string = self._remove_newlines_outside_quotes(bracketed_string)
            try:
                data = json.loads(bracketed_string)
                if 'tool' in data and 'query' in data:
                    json_queries.append(data)
            except json.JSONDecodeError:
                pass

        return json_queries


    # code query handler
    def run_one_code(self, query:json):
        """Run one code query and return the output.
        params:
            query: json object with the following keys: 
                tool: 'python'
                query: string with the code to run
        """
        code = self.previous_code + self.add_print_to_last_line(query['query'])

        python_repl = PythonREPL()
        output, is_success = python_repl.run(code)
        
        if not is_success: 
            output =  "Error: " + output
        elif output == "":
            output = "No output found. Make sure you print the results."

        self.previous_code += '\n' + self.remove_print(code) + '\n'
        return output, is_success

    def add_print_to_last_line(self, s):
        # first check if there is already a print statement
        if "print(" in s:
            return s
        
        # Input a string, extract the last line, enclose it in print() and return the new string
        lines = s.splitlines()
        last_line = lines[-1]
        lines[-1] = "print(" + last_line + ")"

        # Join the lines back together
        return "\n".join(lines)

    def remove_print(self, s):
        # remove all print statements from a string
        lines = s.splitlines()
        lines = [line for line in lines if not line.startswith("print(")]
        return "\n".join(lines)



# Imported from langchain
class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed.

        Add is_success.
        """
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        is_success = True
        try:
            # TODO: need further testing; original exec(command, self.globals, self.locals), 
            # Wrong for '\nfrom math import factorial\nrotations_fixed = [factorial(7 - i) for i in [1, 2, 3, 4, 5]]\nprint(rotations_fixed)'
            combined_globals_and_locals = {**self.globals, **self.locals}
            exec(command, combined_globals_and_locals, combined_globals_and_locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
            is_success = False
        return output, is_success



# Imported from langchain
def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
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
        wolfram_alpha_appid = get_from_dict_or_env(
            values, "wolfram_alpha_appid", "WOLFRAM_ALPHA_APPID"
        )
        values["wolfram_alpha_appid"] = wolfram_alpha_appid

        try:
            import wolframalpha

        except ImportError:
            raise ImportError(
                "wolframalpha is not installed. "
                "Please install it with `pip install wolframalpha`"
            )
        client = wolframalpha.Client(wolfram_alpha_appid)
        values["wolfram_client"] = client

        return values

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        res = self.wolfram_client.query(query)
        is_success = False # added 
        try:
            assumption = next(res.pods).text
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it", is_success

        if answer is None or answer == "":
            # We don't want to return the assumption alone if answer is empty
            return "No good Wolfram Alpha Result was found", is_success
        else:
            is_success = True
            return f"Assumption: {assumption} \nAnswer: {answer}", is_success









# import re
# def parse_std_err(error):
#     if error is None:
#         return error
#     return re.sub("File (.*)\", ", " ", error.decode('utf-8'))

# import signal
# import subprocess
# import sys

# def timeout_handler(signum, frame):
#     raise TimeoutError("Timed out!")

# signal.signal(signal.SIGALRM, timeout_handler)
# max_exec_time = 3  # seconds

# def execute_code(code):
#     code = code.strip()
#     with open("codetest.py", "w") as fout:
#         fout.write(code)

#     try:
#         signal.alarm(max_exec_time)
#         result = subprocess.run(
#             [sys.executable, "codetest.py"],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.PIPE,
#         )
#         signal.alarm(0)
#     except TimeoutError:
#         return -1, "TimeoutError"
#     return int(result.returncode == 0), parse_std_err(result.stderr)
