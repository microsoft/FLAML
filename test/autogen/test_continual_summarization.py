try:
    import openai
except ImportError:
    openai = None
import pytest
from flaml import oai
from flaml.autogen.agent import LearningAgent, TeachingAgent

KEY_LOC = "test/autogen"


@pytest.mark.skipif(openai is None, reason="openai not installed")
def test_continual_summarization():
    import feedparser

    research_teacher = TeachingAgent(name="research_teacher", human_input_mode="NEVER")
    research_teacher.setup_learning(
        learning_constraints={"learning_trigger": True, "cpu": 1},
        learning_objectives="Briefly summarize research topics into bullet points.",
        learning_results=" ",
    )
    # get data and add to research_teacher
    ml_feed = feedparser.parse("http://export.arxiv.org/rss/cs.LG")
    ai_feed = feedparser.parse("http://export.arxiv.org/rss/cs.AI")
    ml_data, ai_data = [], []

    # for demo purpose, only use 3 entries from ml_feed and ai_feed
    for entry in ml_feed.entries[0:3]:
        ml_data.append(entry.summary)
    research_teacher.add_data(ml_data)
    for entry in ai_feed.entries[0:3]:
        ai_data.append(entry.summary)
    research_teacher.add_data(ai_data)

    config_list = oai.config_list_from_models(key_file_path=KEY_LOC, model_list=["gpt-4"])
    research_learner = LearningAgent(name="research_learner", config_list=config_list)
    research_learner.receive(research_teacher.generate_init_prompt(), research_teacher)


if __name__ == "__main__":
    test_continual_summarization()
