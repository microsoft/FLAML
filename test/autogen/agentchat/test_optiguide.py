import pytest
import sys
import requests  # for loading the example source code
from flaml import autogen
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="do not run on MacOS or windows",
)
def test_optiguide():
    try:
        import openai
    except ImportError:
        return

    from flaml.autogen.agentchat import UserProxyAgent
    from flaml.autogen.agentchat.contrib.opti_guide import OptiGuideAgent

    conversations = {}
    autogen.ChatCompletion.start_logging(conversations)

    config_list = autogen.config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "model": ["gpt-4", "gpt4", "gpt-4-32k", "gpt-4-32k-0314"],
        },
    )
    code_url = "https://www.beibinli.com/docs/optiguide/coffee.py"
    code = requests.get(code_url).text

    assistant = OptiGuideAgent(
        "optiguide",
        source_code=code,
        example_qa="",
        llm_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
        },
    )

    user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)
    user_proxy.send("What if we prohibit shipping from supplier 1 to roastery 2?", assistant)
    user_proxy.send(
        "What is the impact of supplier1 being able to supply only half the quantity at present?", assistant
    )
    user_proxy.send("What if Roastery 1 is exclusively for Cafe 3", assistant)

    # test danger case
    assistant.reset()
    assistant.debug_times = 1
    assistant._shield.generate_reply = lambda **_: "DANGER"
    user_proxy.send("What's the weather today", assistant)
    assert (
        assistant._debug_times_left == 0
        and assistant.last_message(user_proxy)["content"] == "Sorry. I cannot answer your question."
    )
    # print(conversations)


if __name__ == "__main__":
    test_optiguide()
