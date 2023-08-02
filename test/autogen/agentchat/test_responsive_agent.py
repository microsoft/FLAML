import sys
from io import StringIO
import pytest
from flaml.autogen.agentchat import ResponsiveAgent


def test_responsive_agent(monkeypatch):
    dummy_agent_1 = ResponsiveAgent(name="dummy_agent_1", human_input_mode="ALWAYS")
    dummy_agent_2 = ResponsiveAgent(name="dummy_agent_2", human_input_mode="TERMINATE")

    monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.receive("hello", dummy_agent_2)  # receive a str
    monkeypatch.setattr(sys, "stdin", StringIO("TERMINATE\n\n"))
    dummy_agent_1.receive(
        {
            "content": "hello {name}",
            "context": {
                "name": "dummy_agent_2",
            },
        },
        dummy_agent_2,
    )  # receive a dict
    assert "context" in dummy_agent_1.chat_messages["dummy_agent_2"][-2]
    # receive dict without openai fields to be printed, such as "content", 'function_call'. There should be no error raised.
    pre_len = len(dummy_agent_1.chat_messages["dummy_agent_2"])
    with pytest.raises(ValueError):
        dummy_agent_1.receive({"message": "hello"}, dummy_agent_2)
    assert pre_len == len(
        dummy_agent_1.chat_messages["dummy_agent_2"]
    ), "When the message is not an valid openai message, it should not be appended to the oai conversation."

    monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.send("TERMINATE", dummy_agent_2)  # send a str
    monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.send(
        {
            "content": "TERMINATE",
        },
        dummy_agent_2,
    )  # send a dict

    # send dict with no openai fields
    pre_len = len(dummy_agent_1.chat_messages["dummy_agent_2"])
    with pytest.raises(ValueError):
        dummy_agent_1.send({"message": "hello"}, dummy_agent_2)

    assert pre_len == len(
        dummy_agent_1.chat_messages["dummy_agent_2"]
    ), "When the message is not a valid openai message, it should not be appended to the oai conversation."

    # update system message
    dummy_agent_1.update_system_message("new system message")
    assert dummy_agent_1._oai_system_message[0]["content"] == "new system message"


def test_long_auto_reply():
    # prepare code and function call
    longcodeblock = [("python", "print('hello world')"), ("python", "print('*' * 5000)")]

    def return_long_out():
        return "*" * 5000

    func_call = {"name": "return_long_out", "arguments": "{}"}

    # create agent with pre-defined token limit
    dummy_agent = ResponsiveAgent(
        name="dummy_agent",
        human_input_mode="ALWAYS",
        function_map={"return_long_out": return_long_out},
        auto_reply_token_limit=50,
    )
    long_error = dummy_agent.execute_code_blocks(longcodeblock)
    assert (
        "hello world" in long_error[1]
    ), f"Output from previous code block should be shown in the message. Reuturn: {long_error}"
    assert (
        "Error: The output exceeds the length limit and is truncated." in long_error[1]
    ), "The error message for long reply should be shown in the message."
    assert (
        dummy_agent.execute_function(func_call)[1]["content"]
        == "Error: The return from this call exceeds the token limit."
    ), "The error message for long reply should be shown in the message."

    # create agent with no token limit
    dummy_agent = ResponsiveAgent(
        name="dummy_agent", human_input_mode="ALWAYS", function_map={"return_long_out": return_long_out}
    )
    long_return = dummy_agent.execute_code_blocks(longcodeblock)
    assert "****" in long_return[1], f"The output should be valid. Return: {long_return}"
    assert "****" in dummy_agent.execute_function(func_call)[1]["content"], "The return from this call should be valid."


if __name__ == "__main__":
    test_responsive_agent(pytest.monkeypatch)
    test_long_auto_reply()
