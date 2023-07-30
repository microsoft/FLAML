import sys
import os
import pytest
from flaml import autogen
from flaml.autogen.code_utils import (
    UNKNOWN,
    extract_code,
    execute_code,
    infer_lang,
    find_code,
    improve_code,
    improve_function,
)

KEY_LOC = "notebook"
OAI_CONFIG_LIST = "OAI_CONFIG_LIST"
here = os.path.abspath(os.path.dirname(__file__))


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


def test_find_code():
    try:
        import openai
    except ImportError:
        return
    config_list = autogen.config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "model": {
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "chatgpt-35-turbo-0301",
                "gpt-35-turbo-v0301",
            },
        },
    )

    messages = [
        {
            "role": "assistant",
            "content": "Print hello world to a file called hello.txt",
        },
        {
            "role": "user",
            "content": """
# filename: write_hello.py
```
with open('hello.txt', 'w') as f:
    f.write('Hello, World!')
print('Hello, World! printed to hello.txt')
```
Please execute the above Python code to print "Hello, World!" to a file called hello.txt and print the success message.
""",
        },
    ]
    codeblocks = find_code(messages, config_list=config_list)
    assert codeblocks[0][0] == "python", codeblocks
    messages += [
        {
            "role": "assistant",
            "content": """
exitcode: 0 (execution succeeded)
Code output:
Hello, World! printed to hello.txt
""",
        },
        {
            "role": "user",
            "content": "Great! Can I help you with anything else?",
        },
    ]
    codeblocks = find_code(messages, config_list=config_list)
    assert codeblocks[0][0] == "unknown", codeblocks
    messages += [
        {
            "role": "assistant",
            "content": "Save a pandas df with 3 rows and 3 columns to disk.",
        },
        {
            "role": "user",
            "content": """
```
# filename: save_df.py
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df.to_csv('df.csv')
print('df saved to df.csv')
```
Please execute the above Python code to save a pandas df with 3 rows and 3 columns to disk.
Before you run the code above, run
```
pip install pandas
```
first to install pandas.
""",
        },
    ]
    # need gpt-4 for this task
    config_list = autogen.config_list_from_json(OAI_CONFIG_LIST, file_location=KEY_LOC)
    codeblocks = find_code(messages, config_list=config_list)
    assert codeblocks[0][0] == "sh" and codeblocks[1][0] == "python", codeblocks


if __name__ == "__main__":
    # test_infer_lang()
    # test_extract_code()
    # test_execute_code()
    test_find_code()
