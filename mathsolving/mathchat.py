from flaml.autogen.agent import AssistantAgent, UserProxyAgent, MathUserProxyAgent
from flaml.autogen.agent.math_user_proxy_agent import is_termination_msg


class MathChat:
    def __init__(self, seed, config_list, max_consecutive_auto_reply):
        """Initialize the MathChat instance.

        Args:
            seed (int): random seed.
            config_list (list): list of config dicts.
            max_consecutive_auto_reply (int): maximum number of consecutive auto replies.
        """

        # create an AssistantAgent instance named "assistant"
        self.assistant = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            request_timeout=600,
            seed=seed,
            config_list=config_list,
        )

        # create the UserProxyAgent instance named "user"
        self.math_user_agent = MathUserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
        )

    def solve_one_problem(self, problem):
        """Solve one problem.

        Args:
            problem (dict): a problem dict. Use problem["problem"] to extract the problem text.
        """
        # reset
        self.assistant.reset()

        # solve
        self.assistant.receive(
            message=self.math_user_agent.generate_init_prompt(problem["problem"]),
            sender=self.math_user_agent,
        )

        response_with_ans = self.assistant._oai_conversations["user"][-1]["content"]
        return {
            "response_with_ans": response_with_ans,
            "is_valid_reply": True if is_termination_msg(response_with_ans) else False,
            "round": (len(self.assistant._oai_conversations["user"]) - 1) // 2,
            "messages": self.assistant._oai_conversations,
        }
