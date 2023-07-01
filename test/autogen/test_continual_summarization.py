try:
    import openai
except ImportError:
    openai = None
import pytest
from flaml import oai
from flaml.autogen.agent import LearningAgent, TeachingAgent

KEY_LOC = "test/autogen/"


@pytest.mark.skipif(openai is None, reason="openai not installed")
def test_continual_summarization():
    def LLM_related(input_string):
        if "Large Language Models" in input_string or "LLM" in input_string or "GPT" in input_string:
            return True
        else:
            return False

    import feedparser

    research_teacher = TeachingAgent(name="research_teacher", human_input_mode="NEVER")
    research_teacher.setup_learning(
        learning_constraints={"learning_trigger": True, "cpu": 1},
        learning_objectives="""Condense the provided data, which consists of titles and abstracts of research papers from arXiv, into a research digest.
        Create a single bullet point for each entry, ensuring clarity and coherence.
        """,
        learning_results=" ",
        # learning_func=oai.summarize,
    )
    # get data and add to research_teacher
    ml_feed = feedparser.parse("http://export.arxiv.org/rss/cs.LG")
    ai_feed = feedparser.parse("http://export.arxiv.org/rss/cs.AI")
    ml_data, ai_data = [], []

    # for demo purpose, only use 3 entries from ml_feed and ai_feed
    for entry in ml_feed.entries:
        title_and_abstract = f"Title: {entry.title}. \n Abstract: {entry.summary}"
        if LLM_related(title_and_abstract):
            ml_data.append(title_and_abstract)
    research_teacher.add_data(ml_data)
    for entry in ai_feed.entries:
        title_and_abstract = f"Title: {entry.title}. \n Abstract: {entry.summary}"
        if LLM_related(title_and_abstract):
            ai_data.append(entry.summary)
    research_teacher.add_data(ai_data)

    # config_list = oai.config_list_from_models(key_file_path=KEY_LOC, model_list=["gpt-3.5-turbo-0613"])
    config_list = oai.config_list_from_models(key_file_path=KEY_LOC, model_list=["gpt-4"])
    research_learner = LearningAgent(name="research_learner", config_list=config_list)
    research_learner.receive(research_teacher.generate_init_prompt(), research_teacher)


if __name__ == "__main__":
    test_continual_summarization()
