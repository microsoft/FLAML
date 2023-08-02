from flaml import autogen


def test_broadcast():
    agent1 = autogen.ResponsiveAgent("alice", max_consecutive_auto_reply=0, human_input_mode="NEVER")
    agent2 = autogen.ResponsiveAgent("bob", max_consecutive_auto_reply=0, human_input_mode="NEVER")
    broadcaster = autogen.BroadcastAgent(agents=[agent1, agent2])
    agent1.send("start", broadcaster)
    # no auto reply
    assert len(agent1.chat_messages[broadcaster.name]) == 1
    assert len(agent2.chat_messages[broadcaster.name]) == 1
    assert agent2.last_message(broadcaster)["name"] == agent1.name


def test_chat_manager():
    broadcaster = autogen.BroadcastAgent()
    chat_manager = autogen.ChatManagerAgent(broadcaster=broadcaster, max_round=2)
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
    broadcaster.agents = [agent1, agent2]
    broadcaster.send("start", chat_manager)

    # no auto reply
    assert len(broadcaster.chat_messages[chat_manager.name]) == 2


if __name__ == "__main__":
    # test_broadcast()
    test_chat_manager()
