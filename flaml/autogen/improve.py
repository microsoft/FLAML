from flaml import oai

IMPROVE_FUNCTION_CONFIG = {
    "prompt": """Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}""",
    "model": "gpt-4",
    "request_timeout": 300,
}


def improve_function(file_name, func_name, objective, test_cases=None):
    """(work in progress) Improve the function to achieve the objective."""

    # read the entire file into a string
    with open(file_name, "r") as f:
        file_string = f.read()
    response = oai.Completion.create(locals(), **IMPROVE_FUNCTION_CONFIG)
    return oai.Completion.extract_text(response)[0]


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    # print(improve_function("flaml/autogen/improve.py", "improve_function", "test"))
