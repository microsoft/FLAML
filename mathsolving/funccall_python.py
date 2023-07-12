from flaml.autogen.agent import AssistantAgent, UserProxyAgent, MathUserProxyAgent
from flaml.autogen.agent.math_user_proxy_agent import is_termination_msg


class MathChatFunctionPython:
    def __init__(self, seed, config_list, max_consecutive_auto_reply):
        system_message = """You are an advanced AI with the capability to solve complex math problems. You can write Python code to help you by calling the function execute_python.
There are two ways to utilize Python:
1. You can write code that solves the problem directly.
2. You can solve the problem by yourself, and try to use Python as a tool whenever possible during the solving process, such as simplifications, calculations, equation solving, enumerations, etc.

If the result is invalid or unexpected, please correct your code or reasoning.
Put the final answer in \\boxed{} when everything is done."""

        oai_config = {
            "model": "gpt-4-0613",
            "functions": [
                {
                    "name": "execute_python",
                    "description": "Return execution result of a python code. the return is a tuple of (output, is_success).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pycode": {
                                "type": "string",
                                "description": "The python code to be executed. Be careful with the indentation and necessary imports of the code.",
                            }
                        },
                        "required": ["pycode"],
                    },
                },
            ],
            "function_call": "auto",
        }

        # 1. create an AssistantAgent instance named "assistant"
        self.assistant = AssistantAgent(
            name="assistant",
            system_message=system_message,
            request_timeout=600,
            seed=seed,
            config_list=config_list,
            **oai_config
        )

        # 2. create a MathUserProxyAgent for built-in functions only
        self.mathfunc = MathUserProxyAgent(use_docker=False)

        # 3. create the UserProxyAgent instance named "user"
        self.user_agent = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            function_map={"execute_python": self.mathfunc._execute_one_python_code},
        )

    def solve_one_problem(self, problem):
        """

        Args:
            problem (dict): a problem dict. Use problem["problem"] to extract the problem text.
        """
        # reset
        self.assistant.reset()
        self.mathfunc._reset()
        self.user_agent._oai_conversations.clear()

        # solve
        self.assistant.receive(problem["problem"], self.user_agent)
        response_with_ans = self.assistant._oai_conversations["user"][-1]["content"]
        return {
            "response_with_ans": response_with_ans,
            "is_valid_reply": True if is_termination_msg(response_with_ans) else False,
            "round": (len(self.assistant._oai_conversations["user"]) - 1) // 2,
            "messages": self.assistant._oai_conversations,
        }
