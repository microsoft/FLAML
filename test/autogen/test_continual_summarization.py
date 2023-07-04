try:
    import openai
except ImportError:
    openai = None
import pytest
from flaml import oai
from flaml.autogen.agent import LearningAgent, TeachingAgent
import asyncio

KEY_LOC = "test/autogen/"


@pytest.mark.skipif(openai is None, reason="openai not installed")
async def test_continual_summarization():
    def LLM_related(input_string):
        if "Large Language Models" in input_string or "LLM" in input_string or "GPT" in input_string:
            return True
        else:
            return False

    import feedparser

    research_teacher = TeachingAgent(name="research_teacher", human_input_mode="NEVER")
    research_teacher.setup_learning(
        learning_constraints={"learning_trigger": True, "cpu": 1},
        learning_objectives="""Condense the provided data, which consists of titles and abstracts of research papers, into a research digest.
        Create a single bullet point for each entry, ensuring clarity and coherence.
        """,
        learning_results=" ",
        # learning_func=oai.summarize,
    )

    # get data from ml arxiv feed
    ml_feed = feedparser.parse("http://export.arxiv.org/rss/cs.LG")
    ml_data = []
    for entry in ml_feed.entries:
        title_and_abstract = f"Title: {entry.title}. \n Abstract: {entry.summary}"
        if LLM_related(title_and_abstract):
            ml_data.append(title_and_abstract)
    await research_teacher.add_data(ml_data)

    config_list = oai.config_list_from_models(key_file_path=KEY_LOC, model_list=["gpt-3.5-turbo-0613"], exclude="aoai")
    research_learner = LearningAgent(name="research_learner", config_list=config_list)
    asyncio.create_task(research_learner.receive(research_teacher.generate_init_prompt(), research_teacher))

    # get data from ai arxiv feed
    await asyncio.sleep(5)
    ai_feed = feedparser.parse("http://export.arxiv.org/rss/cs.AI")
    ai_data = []
    for entry in ai_feed.entries:
        title_and_abstract = f"Title: {entry.title}. \n Abstract: {entry.summary}"
        if LLM_related(title_and_abstract):
            ai_data.append(title_and_abstract)
    print("adding AI data...")
    await research_teacher.add_data(ai_data)
    ai_data.append(entry.summary)
    research_teacher.add_data(ai_data)


if __name__ == "__main__":
    asyncio.run(test_continual_summarization())
