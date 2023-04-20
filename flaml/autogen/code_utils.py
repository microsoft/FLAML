import signal
import subprocess
import sys
import os
from typing import List, Dict, Tuple, Optional, Union, Callable
import re
from flaml.autogen import oai, DEFAULT_MODEL, FAST_MODEL

# Regular expression for finding a code block
CODE_BLOCK_PATTERN = r"```python\n(.*?)\n```"


def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN) -> str:
    # Use a regular expression to find the code block
    match = re.search(pattern, text, flags=re.DOTALL)
    # If a match is found, return the code
    if match:
        return match.group(1)
    # If no code block is found, return the whole text
    return text


def generate_code(pattern: str = CODE_BLOCK_PATTERN, **args):
    response = oai.Completion.create(**args)
    return extract_code(oai.Completion.extract_text(response)[0], pattern)


_IMPROVE_FUNCTION_CONFIG = {
    "prompt": """Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}""",
    "model": DEFAULT_MODEL,
    "request_timeout": 300,
}


def improve_function(file_name, func_name, objective, **config):
    """(work in progress) Improve the function to achieve the objective."""
    params = {**_IMPROVE_FUNCTION_CONFIG, **config}
    # read the entire file into a str
    with open(file_name, "r") as f:
        file_string = f.read()
    response = oai.Completion.create(
        {"func_name": func_name, "objective": objective, "file_string": file_string}, **params
    )
    cost = oai.Completion.cost(params["model"], response)
    return oai.Completion.extract_text(response)[0], cost


_IMPROVE_CODE_CONFIG = {
    "prompt": """Analyze the code in the following files and return a list of suggestions for improvement{followup}, to achieve the objective of '{objective}'.
{code}
""",
    "model": DEFAULT_MODEL,
    "request_timeout": 600,
}


def improve_code(files, objective, suggest_only=True, **config):
    """Improve the code to achieve a given objective.

    Args:
        files (list): A list of file names containing the source code.
        objective (str): The objective to achieve.
        suggest_only (bool): Whether to return only the suggestions or the improved code.
        config (Optional, dict): The configuration for the API call.

    Returns:
        str: The improved code if suggest_only=False; a list of suggestions if suggest_only=True (default).
        float: The cost of the generation.
    """
    code = ""
    for file_name in files:
        # read the entire file into a string
        with open(file_name, "r") as f:
            file_string = f.read()
        code += f"""{file_name}:
{file_string}

"""
    params = {**_IMPROVE_CODE_CONFIG, **config}
    followup = "" if suggest_only else " followed by the improved code"
    response = oai.Completion.create({"objective": objective, "code": code, "followup": followup}, **params)
    cost = oai.Completion.cost(params["model"], response)
    return oai.Completion.extract_text(response)[0], cost


def timeout_handler(signum, frame):
    raise TimeoutError("Timed out!")


def execute_code(code: str, max_exec_time: Optional[int] = 3) -> Tuple[int, str]:
    """Execute code in a docker container.

    Args:
        code (str): The code to execute.
        max_exec_time (Optional, int): The maximum execution time in seconds.

    Returns:
        int: 1 if the code executes successfully; 0 otherwise.
        str: The error message if the code fails to execute; the stdout otherwise.
    """
    import docker

    # check if already running in a docker container
    in_docker_container = os.path.exists("/.dockerenv")
    if in_docker_container:
        # already running in a docker container
        signal.signal(signal.SIGALRM, timeout_handler)
        code = code.strip()
        # with open("codetest.py", "w") as fout:
        #     fout.write(code)
        try:
            signal.alarm(max_exec_time)
            result = subprocess.run(
                [sys.executable, "-c", code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            signal.alarm(0)
        except TimeoutError:
            return 0, "Timeout"
        return int(result.returncode == 0), result.stderr if result.returncode else result.stdout
    # create a docker client
    client = docker.from_env()
    image = "python:3.9"
    # check if the image exists
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        # pull the image
        print("Pulling image", image)
        client.images.pull(image)
    # create a docker container
    container = client.containers.run(
        image,
        command=["python", "-c", code],
        working_dir="/workspace",
        detach=True,
    )
    try:
        signal.alarm(max_exec_time)
        # wait for the container to finish
        container.wait()
        signal.alarm(0)
    except TimeoutError:
        return 0, "Timeout"
    # get the container logs
    logs = container.logs().decode("utf-8")
    # remove the container
    container.remove()
    # check if the code executed successfully
    exit_code = container.attrs["State"]["ExitCode"]
    success = 0
    # TODO: "Error:" in the logs may not always be reliable
    if exit_code == 0 and "Error:" not in logs:
        success = 1
    # return the logs
    return int(success), logs


_GENERATE_ASSERTIONS_CONFIG = {
    "prompt": """Given the signature and docstring, write the exactly same number of assertion(s) for the provided example(s) in the docstring, without assertion messages.

func signature:
{definition}
assertions:""",
    "model": FAST_MODEL,
    "max_tokens": 256,
    "stop": "\n\n",
}


def generate_assertions(definition: str, **config) -> Tuple[str, float]:
    """Generate assertions for a function.

    Args:
        definition (str): The function definition, including the signature and docstr.
        config (Optional, dict): The configuration for the API call.

    Returns:
        str: The generated assertions.
        float: The cost of the generation.
    """
    params = {**_GENERATE_ASSERTIONS_CONFIG, **config}
    response = oai.Completion.create(
        {"definition": definition},
        **params,
    )
    cost = oai.Completion.cost(params["model"], response)
    assertions = oai.Completion.extract_text(response)[0]
    return assertions, cost


def _remove_check(response):
    """Remove the check function from the response."""
    # find the position of the check function
    pos = response.find("def check(")
    if pos == -1:
        return response
    return response[:pos]


def eval_function_completions(
    responses: List[str],
    definition: str,
    test: Optional[str] = None,
    entry_point: Optional[str] = None,
    assertions: Optional[Union[str, Callable[[str], Tuple[str, float]]]] = None,
) -> Dict:
    """Select a response from a list of responses for the function completion task (using generated assertions), and/or evaluate if the task is successful using a gold test.

    Args:
        responses (list): The list of responses.
        definition (str): The input definition.
        test (Optional, str): The test code.
        entry_point (Optional, str): The name of the function.
        assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.
            When provided, only the responses that pass the assertions will be considered for the actual test (if provided).

    Returns:
        dict: The success metrics.
    """
    n = len(responses)
    if assertions is None:
        # no assertion filter
        success_list = []
        for i in range(n):
            response = _remove_check(responses[i])
            code = (
                f"{response}\n{test}\ncheck({entry_point})"
                if response.startswith("def")
                else f"{definition}{response}\n{test}\ncheck({entry_point})"
            )
            success, _ = execute_code(code)
            success_list.append(success)
        return {
            "expected_success": 1 - pow(1 - sum(success_list) / n, n),
            "success": any(s for s in success_list),
        }
    if callable(assertions) and n > 1:
        # assertion generator
        assertions, gen_cost = assertions(definition)
    else:
        gen_cost = 0
    if n > 1 or test is None:
        for i in range(n):
            response = responses[i] = _remove_check(responses[i])
            code = (
                f"{response}\n{assertions}" if response.startswith("def") else f"{definition}{response}\n{assertions}"
            )
            succeed_assertions, _ = execute_code(code)
            if succeed_assertions:
                break
    else:
        # just test, no need to check assertions
        succeed_assertions = False
        i, response = 0, responses[0]
    if test is None:
        # no test code
        return {
            "index_selected": i,
            "succeed_assertions": succeed_assertions,
            "gen_cost": gen_cost,
            "assertions": assertions,
        }
    code_test = (
        f"{response}\n{test}\ncheck({entry_point})"
        if response.startswith("def")
        else f"{definition}{response}\n{test}\ncheck({entry_point})"
    )
    success, _ = execute_code(code_test)
    return {
        "index_selected": i,
        "succeed_assertions": succeed_assertions,
        "success": success,
        "gen_cost": gen_cost,
        "assertions": assertions,
    }


_FUNC_COMPLETION_PROMPT = "# Python 3{definition}"
_FUNC_COMPLETION_STOP = ["\nclass", "\ndef", "\nif", "\nprint"]
_IMPLEMENT_CONFIGS = [
    {"model": FAST_MODEL, "prompt": _FUNC_COMPLETION_PROMPT, "temperature": 0, "seed": 0},
    {"model": FAST_MODEL, "prompt": _FUNC_COMPLETION_PROMPT, "stop": _FUNC_COMPLETION_STOP, "n": 7, "seed": 0},
    {"model": DEFAULT_MODEL, "prompt": _FUNC_COMPLETION_PROMPT, "temperature": 0, "seed": 1},
    {"model": DEFAULT_MODEL, "prompt": _FUNC_COMPLETION_PROMPT, "stop": _FUNC_COMPLETION_STOP, "n": 2, "seed": 2},
    {"model": DEFAULT_MODEL, "prompt": _FUNC_COMPLETION_PROMPT, "stop": _FUNC_COMPLETION_STOP, "n": 1, "seed": 2},
]


def implement(
    definition: str,
    configs: Optional[List[Dict]] = None,
    assertions: Optional[Union[str, Callable[[str], Tuple[str, float]]]] = generate_assertions,
) -> Tuple[str, float]:
    """Implement a function from a definition.

    Args:
        definition (str): The function definition, including the signature and docstr.
        configs (list): The list of configurations for completion.
        assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.

    Returns:
        str: The implementation.
        float: The cost of the implementation.
        int: The index of the configuration which generates the implementation.
    """
    cost = 0
    configs = configs or _IMPLEMENT_CONFIGS
    if len(configs) > 1 and callable(assertions):
        assertions, cost = assertions(definition)
    for i, config in enumerate(configs):
        response = oai.Completion.create({"definition": definition}, **config)
        cost += oai.Completion.cost(config["model"], response)
        responses = oai.Completion.extract_text(response)
        metrics = eval_function_completions(responses, definition, assertions=assertions)
        assertions = metrics["assertions"]
        cost += metrics["gen_cost"]
        if metrics["succeed_assertions"] or i == len(configs) - 1:
            return responses[metrics["index_selected"]], cost, i
