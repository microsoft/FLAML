from flaml.autogen.agent.math_agent import MathAgent
from flaml.autogen.agent.human_agent import HumanAgent
from flaml.autogen.math_utils import eval_math_responses, get_answer


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    user_agent = HumanAgent("human agent")
    math_agent = MathAgent("math_agent")
    problems = [
        {
            "problem": "solve the equation x^3=125",
            "solution": "\\boxed{x=5}",
        },
        {
            "problem": "solve the equation x^2=4",
            "solution": "\\boxed{x=2,-2}",
        },
    ]

    for i, problem in enumerate(problems):
        print("hello")
        math_agent.receive(message=problem["problem"], sender=user_agent)
        # get the answer from the math agent
        result = user_agent.retrieve_conversation("math_agent")
        # evaluate how good the answer is
        metrics = eval_math_responses([result["response_with_ans"]], problem["solution"])
        # get the result
        correct_ans = get_answer(problem["solution"])
        print("answer:", result["response_with_ans"])
        print("\n correct answer is:", correct_ans)
        print("metrics:", metrics)
