from flaml.autogen.agent import AssistantAgent, UserProxyAgent, MathUserProxyAgent
from flaml.autogen.agent.math_user_proxy_agent import is_termination_msg


class MathChatFunctionWolfram:
    def __init__(self, seed, config_list, max_consecutive_auto_reply):
        system_message = """You are an advanced AI with the capability to solve math problems with Wolfram Alpha.
Wolfram Alpha is provided as an external service through the function "query_wolfram", and you are encouraged to use Wolfram Alpha whenever necessary. For example, you can use it to help you with simplifications, calculations, equation solving, enumerations, etc.
If you keep getting errors from using Wolfram Alpha, you may try to decompose the queries and try again.
Put the final answer in \\boxed{} when everything is done."""

        oai_config = {
            "model": "gpt-4-0613",
            "functions": [
                {
                    "name": "query_wolfram",
                    "description": "Return the API result from the Wolfram Alpha. the return is a tuple of (result, is_success).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Wolfram Alpha code to be executed. You should write the code in Wolfram Alpha format",
                            }
                        },
                        "required": ["query"],
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
            function_map={"query_wolfram": self.mathfunc._execute_one_wolfram_query},
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
