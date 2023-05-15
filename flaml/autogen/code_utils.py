import signal
import subprocess
import sys
import os
import pathlib
from typing import List, Dict, Tuple, Optional, Union, Callable
import re
import time
from hashlib import md5
from flaml.autogen import oai, DEFAULT_MODEL, FAST_MODEL

# Regular expression for finding a code block
CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extensions")


def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN) -> str:
    # Use a regular expression to find the code block
    match = re.search(pattern, text, flags=re.DOTALL)
    # If a match is found, return the code
    if match:
        return match.group(2), match.group(1)
    # If no code block is found, return the whole text
    return text, "unknown"


def generate_code(pattern: str = CODE_BLOCK_PATTERN, **config) -> Tuple[str, float]:
    """Generate code.

    Args:
        pattern (Optional, str): The regular expression pattern for finding the code block.
            The default pattern is for finding a code block in a markdown file.
        config (Optional, dict): The configuration for the API call.

    Returns:
        str: The generated code.
        float: The cost of the generation.
    """
    response = oai.Completion.create(**config)
    return extract_code(oai.Completion.extract_text(response)[0], pattern), response["cost"]


_IMPROVE_FUNCTION_CONFIG = {
    "prompt": """Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}""",
    "model": DEFAULT_MODEL,
    "request_timeout": 600,
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
    return oai.Completion.extract_text(response)[0], response["cost"]


_IMPROVE_CODE_CONFIG = {
    "prompt": """Analyze the code in the following files and return a list of suggestions for improvement{followup}, to achieve the objective of '{objective}'.
{code}
""",
    "model": DEFAULT_MODEL,
    "request_timeout": 900,
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
    return oai.Completion.extract_text(response)[0], response["cost"]


def timeout_handler(signum, frame):
    raise TimeoutError("Timed out!")


def execute_code(
    code: Optional[str] = None,
    timeout: Optional[int] = 600,
    filename: Optional[str] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[bool] = True,
) -> Tuple[int, bytes]:
    """Execute code in a docker container.
    This function is not tested on MacOS.

    Args:
        code (Optional, str): The code to execute.
            If None, the code from the file specified by filename will be executed.
            Either code or filename must be provided.
        timeout (Optional, int): The maximum execution time in seconds.
        filename (Optional, str): The file name to save the code or where the code is stored when `code` is None.
            If None, a file with a randomly generated name will be created.
            The randomly generated file will be deleted after execution.
            The file name must be a relative path. Relative paths are relative to the working directory.
        work_dir (Optional, str): The working directory for the code execution.
            If None, a default working directory will be used.
            The default working directory is the "extensions" directory under
            "xxx/flaml/autogen", where "xxx" is the path to the flaml package.
        use_docker (Optional, bool): Whether to use a docker container for code execution.
            If True, the code will be executed in a docker container.
            If False, the code will be executed in the current environment.
            Default is True. If the code is executed in the current environment,
            the code must be trusted.

    Returns:
        int: 0 if the code executes successfully.
        bytes: The error message if the code fails to execute; the stdout otherwise.
    """
    assert code is not None or filename is not None, "Either code or filename must be provided."

    original_filename = filename
    if filename is None:
        code_hash = md5(code.encode()).hexdigest()
        # create a file with a automatically generated name
        filename = f"tmp_code_{code_hash}.py"
    if work_dir is None:
        work_dir = WORKING_DIR
    filepath = os.path.join(work_dir, filename)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)

    if code is not None:
        with open(filepath, "w") as fout:
            fout.write(code)
    # check if already running in a docker container
    in_docker_container = os.path.exists("/.dockerenv")
    if not use_docker or in_docker_container:
        # already running in a docker container
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(timeout)
            # run the code in a subprocess in the current docker container in the working directory
            result = subprocess.run(
                [sys.executable, filename],
                cwd=work_dir,
                capture_output=True,
            )
            signal.alarm(0)
        except TimeoutError:
            if original_filename is None:
                os.remove(filepath)
            return 1, "Timeout"
        if original_filename is None:
            os.remove(filepath)
        return result.returncode, result.stderr if result.returncode else result.stdout

    import docker
    from requests.exceptions import ReadTimeout, ConnectionError

    # create a docker client
    client = docker.from_env()
    image_list = ["python:3-alpine", "python:3", "python:3-windowsservercore"]
    for image in image_list:
        # check if the image exists
        try:
            client.images.get(image)
            break
        except docker.errors.ImageNotFound:
            # pull the image
            print("Pulling image", image)
            try:
                client.images.pull(image)
                break
            except docker.errors.DockerException:
                print("Failed to pull image", image)
    # get a randomized str based on current time to wrap the exit code
    exit_code_str = f"exitcode{time.time()}"
    abs_path = pathlib.Path(work_dir).absolute()
    # if sys.platform == "win32":
    #     abs_path = str(abs_path).replace("\\", "/")
    #     abs_path = f"/{abs_path[0].lower()}{abs_path[2:]}"
    # create a docker container
    container = client.containers.run(
        image,
        command=[
            "sh",
            "-c",
            f"python {filename}; exit_code=$?; echo -n {exit_code_str}; echo -n $exit_code; echo {exit_code_str}",
        ],
        working_dir="/workspace",
        detach=True,
        # get absolute path to the working directory
        volumes={abs_path: {"bind": "/workspace", "mode": "rw"}},
    )
    start_time = time.time()
    while container.status != "exited" and time.time() - start_time < timeout:
        # Reload the container object
        container.reload()
    if container.status != "exited":
        container.stop()
        container.remove()
        if original_filename is None:
            os.remove(filepath)
        return 1, "Timeout"
    # try:
    #     container.wait(timeout=timeout)
    # except (ReadTimeout, ConnectionError):
    #     container.stop()
    #     container.remove()
    #     if original_filename is None:
    #         os.remove(filepath)
    #     return 1, "Timeout"
    # get the container logs
    logs = container.logs().decode("utf-8").rstrip()
    # remove the container
    container.remove()
    # check if the code executed successfully
    exit_code = container.attrs["State"]["ExitCode"]
    if exit_code == 0:
        # extract the exit code from the logs
        pattern = re.compile(f"{exit_code_str}(\\d+){exit_code_str}")
        match = pattern.search(logs)
        exit_code = int(match.group(1))
        # remove the exit code from the logs
        logs = pattern.sub("", logs)

    logs = bytes(logs, "utf-8")
    if original_filename is None:
        os.remove(filepath)
    # return the exit code and logs
    return exit_code, logs


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
    assertions = oai.Completion.extract_text(response)[0]
    return assertions, response["cost"]


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
    timeout: Optional[float] = 3,
    use_docker: Optional[bool] = True,
) -> Dict:
    """Select a response from a list of responses for the function completion task (using generated assertions), and/or evaluate if the task is successful using a gold test.

    Args:
        responses (list): The list of responses.
        definition (str): The input definition.
        test (Optional, str): The test code.
        entry_point (Optional, str): The name of the function.
        assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.
            When provided, only the responses that pass the assertions will be considered for the actual test (if provided).
        timeout (Optional, float): The timeout for executing the code.

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
            success = execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
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
            succeed_assertions = execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
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
    success = execute_code(code_test, timeout=timeout, use_docker=use_docker)[0] == 0
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


class PassAssertionFilter:
    def __init__(self, assertions):
        self._assertions = assertions
        self.cost = 0
        self.metrics = self.responses = None

    def pass_assertions(self, context, response, **_):
        """Check if the response passes the assertions."""
        responses = oai.Completion.extract_text(response)
        metrics = eval_function_completions(responses, context["definition"], assertions=self._assertions)
        self._assertions = metrics["assertions"]
        self.cost += metrics["gen_cost"]
        self.metrics = metrics
        self.responses = responses
        return metrics["succeed_assertions"]


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
    assertion_filter = PassAssertionFilter(assertions)
    response = oai.Completion.create(
        {"definition": definition}, config_list=configs, filter_func=assertion_filter.pass_assertions
    )
    cost += assertion_filter.cost + response["cost"]
    return assertion_filter.responses[assertion_filter.metrics["index_selected"]], cost, response["config_id"]

    # for i, config in enumerate(configs):
    #     response = oai.Completion.create({"definition": definition}, **config)
    #     cost += oai.Completion.cost(response)
    #     responses = oai.Completion.extract_text(response)
    #     metrics = eval_function_completions(responses, definition, assertions=assertions)
    #     assertions = metrics["assertions"]
    #     cost += metrics["gen_cost"]
    #     if metrics["succeed_assertions"] or i == len(configs) - 1:
    #         return responses[metrics["index_selected"]], cost, i
