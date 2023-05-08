from flaml.autogen.code_utils import extract_code
from flaml import oai


def test_extract_code():
    print(extract_code("```bash\npython temp.py\n```"))


def test_coding_agent(interactive_mode=False):
    try:
        import openai
    except ImportError:
        return
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.human_agent import HumanProxyAgent

    conversations = {}
    oai.ChatCompletion.start_logging(conversations)
    agent = PythonAgent("coding_agent", request_timeout=600, seed=42)
    user = HumanProxyAgent(
        "user", interactive_mode=interactive_mode, is_termination_msg=lambda x: x.rstrip().endswith("TERMINATE")
    )
    #     agent.receive("""Find $a+b+c$, given that $x+y\\neq -1$ and  \\begin{align*}
    # ax+by+c&=x+7,\\\\
    # a+bx+cy&=2x+6y,\\\\
    # ay+b+cx&=4x+y.
    # \end{align*}
    # Solve the problem smartly.""", user)
    #     agent.reset()
    #     agent.receive("""Let $a_1,a_2,a_3,\\dots$ be an arithmetic sequence. If $\\frac{a_4}{a_2} = 3$, what is $\\frac{a_5}{a_3}$? Solve the problem smartly.""", user)
    #     agent.reset()
    #     agent.receive("""The product of the first and the third terms of an arithmetic sequence is $5$. If all terms of the sequence are positive integers, what is the fourth term? Solve the problem smartly.""", user)
    agent.reset()
    agent.receive(
        """Create a temp.py file with the following content:
```
print('Hello world!')
```""",
        user,
    )
    print(conversations)
    oai.ChatCompletion.start_logging(compact=False)
    agent.receive("""Execute temp.py""", user)
    print(oai.ChatCompletion.logged_history)
    oai.ChatCompletion.stop_logging()


def test_tsp(interactive_mode=False):
    try:
        import openai
    except ImportError:
        return
    from flaml.autogen.agent.coding_agent import PythonAgent
    from flaml.autogen.agent.human_agent import HumanProxyAgent

    hard_questions = [
        "What if we must go from node 1 to node 2?",
        "Can we double all distances?",
        "Can we add a new point to the graph? It's distance should be randomly between 0 - 5 to each of the existing points.",
    ]

    oai.ChatCompletion.start_logging()
    agent = PythonAgent("coding_agent", temperature=0)
    user = HumanProxyAgent("user", work_dir="test/autogen", interactive_mode=interactive_mode)
    with open("test/autogen/tsp_prompt.txt", "r") as f:
        prompt = f.read()
    # agent.receive(prompt.format(question=hard_questions[0]), user)
    # agent.receive(prompt.format(question=hard_questions[1]), user)
    agent.receive(prompt.format(question=hard_questions[2]), user)
    print(oai.ChatCompletion.logged_history)
    oai.ChatCompletion.stop_logging()


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    # if you use Azure OpenAI, comment the above line and uncomment the following lines
    # openai.api_type = "azure"
    # openai.api_base = "https://<your_endpoint>.openai.azure.com/"
    # openai.api_version = "2023-03-15-preview"  # change if necessary
    # openai.api_key = "<your_api_key>"
    # test_extract_code()
    test_coding_agent(interactive_mode="TERMINATE")
    # test_tsp(interactive_mode=True)
