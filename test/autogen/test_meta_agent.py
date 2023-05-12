from flaml.autogen.agent.chat_agent import ChatAgent
from flaml.autogen.agent.coding_agent import PythonAgent

from flaml.autogen.agent.human_proxy_agent import HumanProxyAgent
from flaml.autogen.math_utils import eval_math_responses, get_answer
from flaml import oai
from flaml.autogen.agent.meta_agent import MetaAgent


def test_meta_prompt():
    """
    Based on:
    https://python.langchain.com/en/latest/use_cases/autonomous_agents/meta_prompt.html
    """

    tasks = [
        "Provide a systematic argument for why we should always eat pasta with olives.",
        "Provide a systematic argument for why we should always eat noodle with garlic.",
        "What should I eat for dinner",
        "What should I eat for lunch",
    ]
    oai.ChatCompletion.start_logging()
    agent = MetaAgent(ChatAgent, name="chat_agent", meta_agent_name="meta_agent")
    user = HumanProxyAgent(
        "human user", work_dir="test/autogen", human_input_mode="ALWAYS", max_consecutive_auto_reply=10
    )
    for i, task in enumerate(tasks):
        agent.receive(task, user)
        print(f"====={i+1}-th task finished!=====")
        agent.reflect()


if __name__ == "__main__":
    import openai

    # openai.api_key_path = "test/openai/key.txt"
    openai.api_key_path = "test/openai/key_gpt3.txt"
    test_meta_prompt()
