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
            "problem": "For what negative value of $k$ is there exactly one solution to the system of equations \\begin{align*}\ny &= 2x^2 + kx + 6 \\\\\ny &= -x + 4?\n\\end{align*}",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "Setting the two expressions for $y$ equal to each other, it follows that $2x^2 + kx + 6 = -x + 4$. Re-arranging, $2x^2 + (k+1)x + 2 = 0$. For there to be exactly one solution for $x$, then the discriminant of the given quadratic must be equal to zero. Thus, $(k+1)^2 - 4 \\cdot 2 \\cdot 2 = (k+1)^2 - 16 = 0$, so $k+1 = \\pm 4$. Taking the negative value, $k = \\boxed{-5}$.",
            "problem_id": "3",
            "correct_ans": "-5",
            "voted_answer": "-5",
        },
        {
            "problem": "Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. Express your answer in interval notation.",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "If these two graphs intersect then the points of intersection occur when  \\[x^2+a=ax,\\] or  \\[x^2-ax+a=0.\\] This quadratic has solutions exactly when the discriminant is nonnegative: \\[(-a)^2-4\\cdot1\\cdot a\\geq0.\\] This simplifies to  \\[a(a-4)\\geq0.\\] This quadratic (in $a$) is nonnegative when $a$ and $a-4$ are either both $\\ge 0$ or both $\\le 0$. This is true for $a$ in $$(-\\infty,0]\\cup[4,\\infty).$$ Therefore the line and quadratic intersect exactly when $a$ is in $\\boxed{(-\\infty,0]\\cup[4,\\infty)}$.",
            "problem_id": "6",
            "correct_ans": "(-\\infty,0]\\cup[4,\\infty)",
            "voted_answer": "a=0",
            "round": 2.5,
            "valid_q_count": 1,
            "total_q_count": 1,
        },
    ]

    for i, problem in enumerate(problems):
        print("hello")
        # send the problem to the math agent add the option to clear past conversation
        math_agent.clear_conversation()
        math_agent.receive(message=problem["problem"], sender=user_agent)
        # get the answer from the math agent
        result = user_agent.retrieve_conversation("math_agent")
        # evaluate how good the answer is
        result_with_ans = result if isinstance(result, str) else result["response_with_ans"]
        metrics = eval_math_responses([result_with_ans], problem["solution"])
        # get the result
        correct_ans = get_answer(problem["solution"])
        print("answer:", result_with_ans)
        print("\n correct answer is:", correct_ans)
        print("metrics:", metrics)
