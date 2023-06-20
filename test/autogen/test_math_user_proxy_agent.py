from flaml import oai
from flaml.autogen.agent.math_user_proxy_agent import MathUserProxyAgent, remove_print, add_print_to_last_line
import pytest
import sys

KEY_LOC = "test/autogen"


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="do not run on MacOS or windows",
)
def test_math_user_proxy_agent():
    try:
        import openai
    except ImportError:
        return

    from flaml.autogen.agent.assistant_agent import AssistantAgent

    conversations = {}
    oai.ChatCompletion.start_logging(conversations)

    config_list = oai.config_list_openai_aoai(key_file_path=KEY_LOC)
    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        request_timeout=600,
        seed=42,
        config_list=config_list,
    )

    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")
    assistant.reset()

    math_problem = "$x^3=125$. What is x?"
    assistant.receive(
        message=mathproxyagent.generate_prompt(math_problem),
        sender=mathproxyagent,
    )
    print(conversations)


def test_add_remove_print():
    # test add print
    code = "a = 4\nb = 5\na,b"
    assert add_print_to_last_line(code) == "a = 4\nb = 5\nprint(a,b)"

    # test remove print
    code = """print("hello")\na = 4*5\nprint("wolrld")"""
    assert remove_print(code) == "a = 4*5"

    # test remove print. Only remove prints without indentation
    code = "if 4 > 5:\n\tprint('True')"
    assert remove_print(code) == code


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="do not run on MacOS or windows",
)
def test_execute_one_python_code():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")

    # no output found 1
    code = "x=3"
    assert mathproxyagent._execute_one_python_code(code)[0] == "No output found. Make sure you print the results."

    # no output found 2
    code = "if 4 > 5:\n\tprint('True')"

    assert mathproxyagent._execute_one_python_code(code)[0] == "No output found."

    # return error
    code = "2+'2'"
    assert "Error:" in mathproxyagent._execute_one_python_code(code)[0]

    # save previous status
    mathproxyagent._execute_one_python_code("x=3\ny=x*2")
    assert mathproxyagent._execute_one_python_code("print(y)")[0].strip() == "6"

    code = "print('*'*2001)"
    assert (
        mathproxyagent._execute_one_python_code(code)[0]
        == "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
    )


def test_generate_prompt():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")

    mathproxyagent._execute_one_python_code("x=3\nx")

    assert "customized" in mathproxyagent.generate_prompt(
        problem="2x=4", prompt_type="python", customized_prompt="customized"
    )

    # previous code cleared
    assert mathproxyagent._previous_code == ""


if __name__ == "__main__":
    test_add_remove_print()
    test_execute_one_python_code()
    test_generate_prompt()
    test_math_user_proxy_agent()
