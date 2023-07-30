from typing import Dict, List, Tuple
import re


class Message:
    role: str
    content: str

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __str__(self):
        return f"[{self.role}]: {self.content}"


class RoleplayMixin:
    def __init__(self) -> None:
        raise Exception("RoleplayMixin is a mixin class, it should not be instantiated.")

    def describle_role(self, chat_history: List[Dict]) -> str:
        return self._system_message

    def _render_role_information(self, roles: List[Tuple[str, str]]) -> str:
        return "\n".join([f"{name}: {description}" for name, description in roles])

    def _render_message(self, message: Dict) -> str:
        # remove newline characters
        content = message["content"].replace("\n", " ")
        return f"[{message['role']}]: {content}"

    def _render_chat_history(self, chat_history: List[Dict]) -> str:
        return "\n".join([self._render_message(message) for message in chat_history])

    def _call_chat(self, content: str) -> str:
        message = self._message_to_dict(content)
        message["role"] = "user"
        return self._oai_reply([message])

    def _render_role_play(self, chat_history: List[Dict], roles: List[Tuple[str, str]], rule: str) -> str:
        prompt = f"""### role information ###
{self._render_role_information(roles)}
### end of role information ###

### role-play rule ###
{rule}
### end of role-play rule ###

You are in a multi-role play game. Your role is {self.name}, {self.describle_role(chat_history)}.
please follow the role-play rule and continue the conversation!"""

        return prompt

    def _render_rule_prompt(self, chat_history: List[Message], admin: str) -> str:
        prompt = f"""### chat history ###
{self._render_chat_history(chat_history)}
### end of chat history ###

list all the rules announced by {admin} from the chat history above.
rules:
"""

        return prompt

    def summarize_rule_from_chat_history(self, chat_history: List[Dict], admin: str) -> str:
        if len(chat_history) == 0:
            rule = f"""always listen to {admin}"""
        else:
            prompt = self._render_rule_prompt(chat_history, admin)
            rule = self._call_chat(prompt)

        return rule

    def select_role(self, chat_history: List[Dict], roles: List[Tuple[str, str]], rules: str) -> int:
        task_prompt = f"""### rule ###
{rules}

### role information ###
{self._render_role_information(roles)}

You are in a multi-role play game and your task is to continue writing conversation. Follow the rule and pick a role to continue conversation. Put role's name in square brackets.
(e.g: [role]: // short and precise message)"""

        chat_history = [{"role": "user", "content": str(message)} for message in chat_history]
        task_message = {"role": "user", "content": task_prompt}

        reply = self._oai_reply([task_message] + chat_history)

        try:
            candidates = [name.lower() for name, _ in roles]

            # reply: [role]: // message
            # retrieve role using regex
            selected_role = re.search(r"\[(.*?)\]", reply).group(1).lower()

            # find the index of the selected role
            return candidates.index(selected_role)
        except Exception:
            # invalid reply, return -1
            return -1

    def role_play(self, chat_history: List[Dict], role_description: List[Tuple[str, str]], rule: str) -> Dict:
        prompt = self._render_role_play(chat_history, role_description, rule)
        task_message = {"role": "user", "content": prompt}
        chat_history = [{"role": "user", "content": str(message)} for message in chat_history]
        # only get the last 5 messages
        chat_history = chat_history[-5:]
        new_message = {"role": "user", "content": f"[{self.name}]:"}
        reply = self.generate_reply([task_message] + chat_history + [new_message])
        return {"role": self.name, "content": reply}
