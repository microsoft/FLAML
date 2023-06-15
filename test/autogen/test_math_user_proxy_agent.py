from flaml import oai
from flaml.autogen.agent.math_user_proxy_agent import MathUserProxyAgent

KEY_LOC = "test/autogen"


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

    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER", use_docker=False)
    assistant.reset()

    math_problem = "$x^3=125$. What is x?"
    assistant.receive(
        message=mathproxyagent.generate_prompt(math_problem),
        sender=mathproxyagent,
    )
    print(conversations)


def test_add_remove_print():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER", use_docker=False)

    # test add print
    code = "a = 4\nb = 5\na,b"
    assert mathproxyagent._add_print_to_last_line(code) == "a = 4\nb = 5\nprint(a,b)"

    # test remove print
    code = """print("hello")\na = 4*5\nprint("wolrld")"""
    assert mathproxyagent._remove_print(code) == "a = 4*5"

    # test remove print. Only remove prints without indentation
    code = "if 4 > 5:\n\tprint('True')"
    assert mathproxyagent._remove_print(code) == code


def test_execution_code():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER", use_docker=False)

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
    print(mathproxyagent._execute_one_python_code("print(y)")[0])
    assert mathproxyagent._execute_one_python_code("print(y)")[0].strip() == "6"


if __name__ == "__main__":
    test_add_remove_print()
    test_execution_code()
    test_math_user_proxy_agent()
