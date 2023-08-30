
from flaml.autogen import AssistantAgent, UserProxyAgent

import sys
import pytest

try:
    import openai
    skip = False
except ImportError:
    skip = True


from flaml import autogen
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4"],
    },
)


@pytest.mark.skipif(
    skip or not sys.version.startswith("3.10"),
    reason="do not run if openai is not installed or py!=3.10",
)
def test_prompt_with_lever5_math(save_dir=None):
    # level 5 math problems from MATH dataset
    # 17 problems
    problems = [
    "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.",
    "Find the value of $a_2+a_4+a_6+a_8+\\dots+a_{98}$ if $a_1, a_2, a_3, \\ldots$ is an arithmetic progression with common difference $1$ and \\[a_1+a_2+a_3+\\dots+a_{98}=137.\\]",
    "Tina the tourist goes on a trip. She starts at the origin and drives north (in the positive $y$ direction) for $10$ units. Then she turns east (the positive $x$ direction) and as she's turning her camera flies out the window and lands exactly at $(0,10)$. She then drives $9$ units east, turns and drives $8$ units north.  She continues this pattern of turning and driving one unit less than after the previous turn, until stopping after driving $1$ unit east. She reaches for her camera only to find it missing! She activates the GPS homing device on her camera and drives back to it in a straight line. What is the equation of this line? Express your answer as $ax+by=c$, where $a$, $b$, and $c$ are integers, $a>0$, and $a$ is as small as possible.",
    "For what negative value of $k$ is there exactly one solution to the system of equations \\begin{align*}\ny &= 2x^2 + kx + 6 \\\\\ny &= -x + 4?\n\\end{align*}",
    "If $\\frac{3x^2-4x+1}{x-1}=m$, and $x$ can be any real number except $1$, what real values can $m$ NOT have?",
    "Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. Express your answer in interval notation.",
    
    'If $\\displaystyle{f(x)=x^{(x+1)}(x+2)^{(x+3)}}$, then find the value of $f(0)+f(-1)+f(-2)+f(-3)$.',
    'An envelope contains eight bills: 2 ones, 2 fives, 2 tens, and 2 twenties. Two bills are drawn at random without replacement. What is the probability that their sum is $\\$20$ or more?',
    'Find the coefficient of $x^2$ in the expansion of the product $$(1-x)(1+2x)(1-3x)\\dotsm(1+14x)(1-15x).$$',
    'All 50 states as well as the District of Columbia and Puerto Rico, have distinct two-letter postal abbreviations. If a two-letter sequence of letters (such as CO or EE) is chosen at random, what is the probability that it is a postal abbreviation for one of the 50 states, the District of Columbia, or Puerto Rico? Express your answer as a common fraction.',
    'Let $x$ and $y$ be real numbers.  Find the set of possible values of\n\\[\\frac{(x + y)(1 - xy)}{(1 + x^2)(1 + y^2)}.\\]',
    'On a number line, the coordinates of $P$ and $Q$ are 8 and 48, respectively. The midpoint of $\\overline{PQ}$ is $B$, the midpoint of $\\overline{BQ}$ is $C$, and the midpoint of $\\overline{PC}$ is $D$. What is the coordinate of $D$?',
    'Find $24^{-1} \\pmod{11^2}$. That is, find the residue $b$ for which $24b \\equiv 1\\pmod{11^2}$.\n\nExpress your answer as an integer from $0$ to $11^2-1$, inclusive.',
    'There are two cameras that take pictures of a traffic intersection. Camera A starts taking pictures at $6$ AM and takes a picture every $11$ minutes. Camera B starts taking pictures at $7$ AM and takes pictures every $7$ minutes. Camera A and Camera B take a picture at the same time at four different times before noon. When Camera A and Camera B take their last picture together, how many minutes before noon is it?',
    'Let $z$ be a complex number such that $z^{13} = 1.$  Let $w_1,$ $w_2,$ $\\dots,$ $w_k$ be all the possible values of\n\\[z + z^3 + z^4 + z^9 + z^{10} + z^{12}.\\]Find $w_1^2 + w_2^2 + \\dots + w_k^2.$',
    'There are 190 people on the beach. 110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?',
    'Completely simplify and rationalize the denominator: $$\\frac{\\sqrt{160}}{\\sqrt{252}}\\times\\frac{\\sqrt{245}}{\\sqrt{108}}$$']

    answers = [
        # 6 algebra problems
        "(-\\infty, -14)\\cup(-3,\\infty)",
        "93",
        "4x-5y=-50",
        "-5",
        "2",
        "(-\\infty,0]\\cup[4,\\infty)",

        # 11 problems, 2 from each category, (1 algebra is deleted)
        '\\frac{10}{9}',
        '\\frac{1}{2}',
        '-588',
        ' \\frac{1}{13}',
        '\\left[ -\\frac{1}{2}, \\frac{1}{2} \\right]',
        '23',
        '116',
        '41',
        '43',
        '10',
        '\\frac{5\\sqrt{42}}{27}'
    ]

    # define solver chat
    agentchat_assistant = AssistantAgent(
        name="assistant",
        llm_config={
            "seed": 42,
            "config_list": config_list,
            "request_timeout": 600,
        },
    )
    user = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            },
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "").rstrip().endswith("TERMINATE") or x.get("content", "").rstrip().endswith("TERMINATE.")),
    )
    print(agentchat_assistant.system_message, flush=True)
    stars = "*"*80
    print(stars + "\n" + stars, flush=True)

    # define answer checker chat
    answer_checker = AssistantAgent(
        name="checker",
        llm_config={
            "seed": 42,
            "config_list": config_list,
            "request_timeout": 600,
        },
        system_message="""You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "The answer is <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer. 
    - When comparison is not obvious (for example, 3*\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code. 
    - You should also note what the problem is asking for. For example, if the problem is asking to simplify a fraction to rational form, but the answer is in decimal form, you should mark the answer as incorrect even if they are the same number.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """,
)
    checker_proxy = UserProxyAgent(
            name="checker_proxy",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            },
            max_consecutive_auto_reply=5,
            is_termination_msg= lambda x: x.get("content", "").lower() and ("the answer is correct" in x.get("content", "").lower() or "the answer is incorrect" in x.get("content", "").lower() or "the reply doesn't contain an answer" in x.get("content", "").lower() or "the answer is approximated but should be correct" in x.get("content", "").lower()),
    )
    

    correct_count = 0
    accum_prints = ""
    for i in range(len(problems)):
        # ---------between "user" and "assistant"---------
        # solve a problem
        agentchat_assistant.reset()
        user.reset()
        user.initiate_chat(agentchat_assistant, message=problems[i])

        # extract reply
        response_with_ans = ""
        messages = agentchat_assistant._oai_messages[user]
        for j in range(len(messages)-1, -1, -1):
            if messages[j]['role'] == 'assistant' and messages[j]['content'].strip() != 'TERMINATE' and messages[j]['content'].strip() != 'TERMINATE.':
                response_with_ans = messages[j]['content']
                break
        

        # ---------between "answer_checker" and "checker_proxy"---------
        # check answer
        message_to_check = f"Problem: {problems[i]}\n\nReply: {response_with_ans}\n\nGround truth answer: {answers[i]}"
        checker_proxy.reset()
        answer_checker.reset()
        checker_proxy.initiate_chat(answer_checker, message=message_to_check)

        # record result
        check_result = answer_checker._oai_messages[checker_proxy][-1]["content"].lower()
        if "the answer is correct" in check_result or "the answer is approximated but should be correct" in check_result:
            correct_count += 1
            
            accum_prints += f"{stars}\nProblem {i} | Is_correct: True | Correct Answer: {answers[i]}\n\nReply: {response_with_ans}\n{stars}\n"
        else:
            check_reply = check_result
            accum_prints += f"Problem {i} | Is_correct: False | Correct Answer: {answers[i]}\nReply: {response_with_ans}\nCheck: {check_reply}\n\n"
        print("*"*10 + f"Problem {i}:", check_result + "*"*10, flush=True)
        
    print(f"Correct count: {correct_count}/{len(problems)}", flush=True)
    print(accum_prints, flush=True)
    assert correct_count > 7, f"The correctness rate is too low. Solve {correct_count}/{len(problems)} problems. Required at least 8/17."



if __name__ == "__main__":
    test_prompt_with_lever5_math()