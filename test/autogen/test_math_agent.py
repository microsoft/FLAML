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
            "problem": "What is the sum of all positive integers $r$ that satisfy $$\\mathop{\\text{lcm}}[r,700] = 7000~?$$",
            "level": "Level 5",
            "type": "Number Theory",
            "solution": "Note the prime factorizations $700=2^2\\cdot 5^2\\cdot 7$ and $7000=2^3\\cdot 5^3\\cdot 7$.\n\nIf $\\mathop{\\text{lcm}}[r,700]=7000$, then in particular, $r$ is a divisor of $7000$, so we can write $r=2^\\alpha\\cdot 5^\\beta\\cdot 7^\\gamma$, where $0\\le\\alpha\\le 3$, $0\\le\\beta\\le 3$, and $0\\le\\gamma\\le 1$.\n\nMoreover, we know that $\\mathop{\\text{lcm}}[r,700]=2^{\\max\\{\\alpha,2\\}}\\cdot 5^{\\max\\{\\beta,2\\}}\\cdot 7^{\\max\\{\\gamma,1\\}}$, and we know that this is equal to $7000=2^3\\cdot 5^3\\cdot 7$. This is possible only if $\\alpha=3$ and $\\beta=3$, but $\\gamma$ can be $0$ or $1$, giving us two choices for $r$: $$r = 2^3\\cdot 5^3\\cdot 7^0 = 1000 \\text{~~or~~} r=2^3\\cdot 5^3\\cdot 7^1 = 7000.$$So the sum of all solutions is $1000+7000=\\boxed{8000}$.",
            "problem_id": "6",
            "is_valid_reply": True,
            "is_correct": False,
            "correct_ans": "8000",
            "voted_answer": "1440",
        },
        {
            "problem": "Find the value of $a_2+a_4+a_6+a_8+\\dots+a_{98}$ if $a_1, a_2, a_3, \\ldots$ is an arithmetic progression with common difference $1$ and \\[a_1+a_2+a_3+\\dots+a_{98}=137.\\]",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "Let $S = a_1 + a_3 + \\dots + a_{97}$ and $T = a_2 + a_4 + \\dots + a_{98}$. Then the given equation states that $S + T = 137$, and we want to find $T$.\n\nWe can build another equation relating $S$ and $T$: note that \\[\\begin{aligned} T-S &= (a_2-a_1) + (a_4-a_3) + \\dots + (a_{98}-a_{97}) \\\\ &= \\underbrace{1 + 1 + \\dots + 1}_{49 \\text{ times }} \\\\ &= 49 \\end{aligned}\\]since $(a_n)$ has common difference $1$. Then, adding the two equations $S+T=137$ and $T-S=49$, we get $2T=137+49=186$, so $T = \\tfrac{186}{2} = \\boxed{93}$.",
            "problem_id": "1",
            "is_valid_reply": True,
            "is_correct": False,
            "correct_ans": "93",
            "voted_answer": "1269",
            "round": 6.5,
        },
        {
            "problem": "Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. Express your answer in interval notation.",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "If these two graphs intersect then the points of intersection occur when  \\[x^2+a=ax,\\] or  \\[x^2-ax+a=0.\\] This quadratic has solutions exactly when the discriminant is nonnegative: \\[(-a)^2-4\\cdot1\\cdot a\\geq0.\\] This simplifies to  \\[a(a-4)\\geq0.\\] This quadratic (in $a$) is nonnegative when $a$ and $a-4$ are either both $\\ge 0$ or both $\\le 0$. This is true for $a$ in $$(-\\infty,0]\\cup[4,\\infty).$$ Therefore the line and quadratic intersect exactly when $a$ is in $\\boxed{(-\\infty,0]\\cup[4,\\infty)}$.",
            "problem_id": "6",
            "is_valid_reply": True,
            "is_correct": False,
            "correct_ans": "(-\\infty,0]\\cup[4,\\infty)",
            "voted_answer": "a=0",
            "round": 2.5,
            "valid_q_count": 1,
            "total_q_count": 1,
        },
    ]

    for i, problem in enumerate(problems):
        print("\n Hello! math_agent is solving this problem: {problem}".format(problem=problem))
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
