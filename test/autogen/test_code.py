import os
import sys

import pytest

from flaml import autogen
from flaml.autogen.code_utils import (
    UNKNOWN,
    execute_code,
    extract_code,
    improve_code,
    improve_function,
    infer_lang,
)

KEY_LOC = "notebook"
OAI_CONFIG_LIST = "OAI_CONFIG_LIST"
here = os.path.abspath(os.path.dirname(__file__))


# def test_find_code():
#     try:
#         import openai
#     except ImportError:
#         return
#     # need gpt-4 for this task
#     config_list = autogen.config_list_from_json(
#         OAI_CONFIG_LIST,
#         file_location=KEY_LOC,
#         filter_dict={
#             "model": ["gpt-4", "gpt4", "gpt-4-32k", "gpt-4-32k-0314"],
#         },
#     )
#     # config_list = autogen.config_list_from_json(
#     #     OAI_CONFIG_LIST,
#     #     file_location=KEY_LOC,
#     #     filter_dict={
#     #         "model": {
#     #             "gpt-3.5-turbo",
#     #             "gpt-3.5-turbo-16k",
#     #             "gpt-3.5-turbo-16k-0613",
#     #             "gpt-3.5-turbo-0301",
#     #             "chatgpt-35-turbo-0301",
#     #             "gpt-35-turbo-v0301",
#     #         },
#     #     },
#     # )
#     seed = 42
#     messages = [
#         {
#             "role": "user",
#             "content": "Print hello world to a file called hello.txt",
#         },
#         {
#             "role": "user",
#             "content": """
# # filename: write_hello.py
# ```
# with open('hello.txt', 'w') as f:
#     f.write('Hello, World!')
# print('Hello, World! printed to hello.txt')
# ```
# Please execute the above Python code to print "Hello, World!" to a file called hello.txt and print the success message.
# """,
#         },
#     ]
#     codeblocks, _ = find_code(messages, seed=seed, config_list=config_list)
#     assert codeblocks[0][0] == "python", codeblocks
#     messages += [
#         {
#             "role": "user",
#             "content": """
# exitcode: 0 (execution succeeded)
# Code output:
# Hello, World! printed to hello.txt
# """,
#         },
#         {
#             "role": "assistant",
#             "content": "Great! Can I help you with anything else?",
#         },
#     ]
#     codeblocks, content = find_code(messages, seed=seed, config_list=config_list)
#     assert codeblocks[0][0] == "unknown", content
#     messages += [
#         {
#             "role": "user",
#             "content": "Save a pandas df with 3 rows and 3 columns to disk.",
#         },
#         {
#             "role": "assistant",
#             "content": """
# ```
# # filename: save_df.py
# import pandas as pd

# df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
# df.to_csv('df.csv')
# print('df saved to df.csv')
# ```
# Please execute the above Python code to save a pandas df with 3 rows and 3 columns to disk.
# Before you run the code above, run
# ```
# pip install pandas
# ```
# first to install pandas.
# """,
#         },
#     ]
#     codeblocks, content = find_code(messages, seed=seed, config_list=config_list)
#     assert (
#         len(codeblocks) == 2
#         and (codeblocks[0][0] == "sh"
#         and codeblocks[1][0] == "python"
#         or codeblocks[0][0] == "python"
#         and codeblocks[1][0] == "sh")
#     ), content

#     messages += [
#         {
#             "role": "user",
#             "content": "The code is unsafe to execute in my environment.",
#         },
#         {
#             "role": "assistant",
#             "content": "please run python write_hello.py",
#         },
#     ]
#     # codeblocks, content = find_code(messages, config_list=config_list)
#     # assert codeblocks[0][0] != "unknown", content
#     # I'm sorry, but I cannot execute code from earlier messages. Please provide the code again if you would like me to execute it.

#     messages[-1]["content"] = "please skip pip install pandas if you already have pandas installed"
#     codeblocks, content = find_code(messages, seed=seed, config_list=config_list)
#     assert codeblocks[0][0] != "sh", content

#     messages += [
#         {
#             "role": "user",
#             "content": "The code is still unsafe to execute in my environment.",
#         },
#         {
#             "role": "assistant",
#             "content": "Let me try something else. Do you have docker installed?",
#         },
#     ]
#     codeblocks, content = find_code(messages, seed=seed, config_list=config_list)
#     assert codeblocks[0][0] == "unknown", content
#     print(content)


def test_infer_lang():
    assert infer_lang("print('hello world')") == "python"
    assert infer_lang("pip install flaml") == "sh"


def test_extract_code():
    print(extract_code("```bash\npython temp.py\n```"))
    # test extract_code from markdown
    codeblocks = extract_code(
        """
Example:
```
print("hello extract code")
```
"""
    )
    print(codeblocks)

    codeblocks = extract_code(
        """
Example:
```python
def scrape(url):
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("title").text
    text = soup.find("div", {"id": "bodyContent"}).text
    return title, text
```
Test:
```python
url = "https://en.wikipedia.org/wiki/Web_scraping"
title, text = scrape(url)
print(f"Title: {title}")
print(f"Text: {text}")
"""
    )
    print(codeblocks)
    codeblocks = extract_code("no code block")
    assert len(codeblocks) == 1 and codeblocks[0] == (UNKNOWN, "no code block")


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="do not run on MacOS or windows",
)
def test_execute_code():
    try:
        import docker
    except ImportError as exc:
        print(exc)
        docker = None
    exit_code, msg, image = execute_code("print('hello world')", filename="tmp/codetest.py")
    assert exit_code == 0 and msg == "hello world\n", msg
    # read a file
    print(execute_code("with open('tmp/codetest.py', 'r') as f: a=f.read()"))
    # create a file
    exit_code, msg, image = execute_code(
        "with open('tmp/codetest.py', 'w') as f: f.write('b=1')", work_dir=f"{here}/my_tmp", filename="tmp2/codetest.py"
    )
    assert exit_code and 'File "tmp2/codetest.py"' in msg, msg
    print(execute_code("with open('tmp/codetest.py', 'w') as f: f.write('b=1')", work_dir=f"{here}/my_tmp"))
    # execute code in a file
    print(execute_code(filename="tmp/codetest.py"))
    print(execute_code("python tmp/codetest.py", lang="sh"))
    # execute code for assertion error
    exit_code, msg, image = execute_code("assert 1==2")
    assert exit_code, msg
    assert 'File ""' in msg
    # execute code which takes a long time
    exit_code, error, image = execute_code("import time; time.sleep(2)", timeout=1)
    assert exit_code and error == "Timeout"
    assert isinstance(image, str) or docker is None or os.path.exists("/.dockerenv")


def test_execute_code_no_docker():
    exit_code, error, image = execute_code("import time; time.sleep(2)", timeout=1, use_docker=False)
    if sys.platform != "win32":
        assert exit_code and error == "Timeout"
    assert image is None


def test_improve():
    try:
        import openai
    except ImportError:
        return
    config_list = autogen.config_list_openai_aoai(KEY_LOC)
    improved, _ = improve_function(
        "flaml/autogen/math_utils.py",
        "solve_problem",
        "Solve math problems accurately, by avoiding calculation errors and reduce reasoning errors.",
        config_list=config_list,
    )
    with open(f"{here}/math_utils.py.improved", "w") as f:
        f.write(improved)
    suggestion, _ = improve_code(
        ["flaml/autogen/code_utils.py", "flaml/autogen/math_utils.py"],
        "leverage generative AI smartly and cost-effectively",
        config_list=config_list,
    )
    print(suggestion)
    improvement, cost = improve_code(
        ["flaml/autogen/code_utils.py", "flaml/autogen/math_utils.py"],
        "leverage generative AI smartly and cost-effectively",
        suggest_only=False,
        config_list=config_list,
    )
    print(cost)
    with open(f"{here}/suggested_improvement.txt", "w") as f:
        f.write(improvement)


if __name__ == "__main__":
    # test_infer_lang()
    # test_extract_code()
    test_execute_code()
    # test_find_code()
