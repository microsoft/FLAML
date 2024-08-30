from flaml.autogen.agent.coding_agent import PythonAgent
from flaml.autogen.agent.user_proxy_agent import UserProxyAgent
from flaml.autogen.agent.meta_agent import MetaAgent


def test_meta_prompt():
    tasks = [
        "Create and execute a script to plot a rocket without using matplotlib.",
        "Create and execute a script to plot a helicopter without using matplotlib",
    ]

    ## User mode:
    python_agent = PythonAgent("python agent")
    assistant = MetaAgent(name="meta agent", agent=python_agent)
    user = UserProxyAgent("human user", work_dir="test/autogen", human_input_mode="ALWAYS")
    for i, task in enumerate(tasks):
        print(f".........Starting the {i+1}-th task!.........")
        assistant.receive(task, user)
        print(f".........{i+1}-th task finished!.........")

    ## Dev mode:
    dev = UserProxyAgent("expert", work_dir="test/autogen", human_input_mode="ALWAYS")
    assistant = MetaAgent(name="meta agent", agent=python_agent, dev_agent=dev)
    user = UserProxyAgent("human user", work_dir="test/autogen", human_input_mode="ALWAYS")
    for i, task in enumerate(tasks[0:2]):
        assistant.receive(task, user)
        assistant.reflect()

    ### Can also be used in the following way:
    for i, task in enumerate(tasks):
        print(f".........Starting the {i+1}-th task!.........")
        assistant.receive(task, dev)
        assistant.reflect()
        print(f".........{i+1}-th task finished!.........")


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    # openai.api_key_path = "test/openai/key_gpt3.txt"
    test_meta_prompt()
