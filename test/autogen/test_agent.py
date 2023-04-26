from flaml.autogen.code_utils import extract_code


def test_extract_code():
    print(extract_code("```bash\npython temp.py\n```"))


def test_coding_agent():
    try:
        import openai
    except ImportError:
        return
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.agent import Agent

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


def _test_tsp():
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.agent import Agent

    hard_questions = [
        "What if we must go from node 1 to node 2?",
        "Can we double all distances?",
        "Can we add a new point to the graph? It's distance should be randomly between 0 - 5 to each of the existing points.",
    ]

    agent = PythonAgent("coding_agent", work_dir="test/autogen", temperature=0)
    user = Agent("user")
    with open("test/autogen/tsp_prompt.txt", "r") as f:
        prompt = f.read()
    # agent.receive(prompt.format(question=hard_questions[0]), user)
    # agent.receive(prompt.format(question=hard_questions[1]), user)
    agent.receive(prompt.format(question=hard_questions[2]), user)


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    _test_tsp()
    # test_extract_code()
    # test_coding_agent()
