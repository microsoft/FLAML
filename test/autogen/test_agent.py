from flaml.autogen.code_utils import extract_code
from flaml import oai


def test_extract_code():
    print(extract_code("```bash\npython temp.py\n```"))


def test_coding_agent():
    try:
        import openai
    except ImportError:
        return
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.agent import Agent

    conversations = {}
    oai.Completion.book_keeping_start(conversations)
    agent = PythonAgent("coding_agent")
    user = Agent("user")
    agent.receive(
        """Create a temp.py file with the following content:
```
print('Hello world!')
```""",
        user,
    )
    agent.receive("""Execute temp.py""", user)
    oai.Completion.book_keeping_end()
    print(conversations)


def test_tsp():
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.agent import Agent

    hard_questions = [
        "What if we must go from node 1 to node 2?",
        "Can we double all distances?",
        "Can we add a new point to the graph? It's distance should be randomly between 0 - 5 to each of the existing points.",
    ]

    oai.Completion.book_keeping_start()
    agent = PythonAgent("coding_agent", work_dir="test/autogen", temperature=0)
    user = Agent("user")
    with open("test/autogen/tsp_prompt.txt", "r") as f:
        prompt = f.read()
    # agent.receive(prompt.format(question=hard_questions[0]), user)
    # agent.receive(prompt.format(question=hard_questions[1]), user)
    agent.receive(prompt.format(question=hard_questions[2]), user)
    print(oai.Completion.logged_messages)
    oai.Completion.book_keeping_end()


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    # if you use Azure OpenAI, comment the above line and uncomment the following lines
    # openai.api_type = "azure"
    # openai.api_base = "https://<your_endpoint>.openai.azure.com/"
    # openai.api_version = "2023-03-15-preview"  # change if necessary
    # openai.api_key = "<your_api_key>"
    # test_extract_code()
    test_coding_agent()
    test_tsp()
