try:
    import openai
except ImportError:
    openai = None
import pytest
from flaml import oai
from flaml.autogen.agent import LearningAgent, TeachingAgent


@pytest.mark.skipif(openai is None, reason="openai not installed")
def test_continual_summarization():
    import feedparser

    research_teacher = TeachingAgent(name="research_teacher")
    research_teacher.setup_learning(
        learning_constraints={"learning_trigger": True, "cpu": 1},
        learning_objectives="Summarize the latest research trends.",
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
    print(len(ml_data), len(ai_data))
    print("ML data", ml_data[0])
    print("AI data", ai_data[0])
    research_teacher.add_data(ai_data)

    research_learner = LearningAgent(name="research_learner", model="gpt-3.5-turbo")  # model="gpt-3.5-turbo"
    research_learner.receive(research_teacher.generate_init_prompt(), research_teacher)


if __name__ == "__main__":
    test_continual_summarization()
