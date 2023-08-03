from flaml import autogen


def test_chat_manager():
    chat_manager = autogen.ChatManagerAgent(max_round=2, llm_config=False)
    agent1 = autogen.GroupChatParticipant(
        "alice",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice sepaking.",
        chat_manager=chat_manager,
    )
    agent2 = autogen.GroupChatParticipant(
        "bob",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
        chat_manager=chat_manager,
    )
    chat_manager.agents = [agent1, agent2]
    agent1.send("start", chat_manager)

    assert len(agent1.chat_messages[chat_manager.name]) == 2

    chat_manager.reset()
    agent1.reset()
    agent2.reset()
    agent2.send("start", chat_manager)


if __name__ == "__main__":
    # test_broadcast()
    test_chat_manager()
