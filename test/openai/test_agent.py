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


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    test_extract_code()
    test_coding_agent()
