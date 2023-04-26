# Self-Improving API Agent
This code describes a scenario where an API agent is designed to self-improve
and interact with a user. The API agent tries to answer a given question using the provided API or suggests a new API if the existing one is not suitable.

The code is divided into two modes: User mode and Dev mode. The user mode provides a basic interaction between the API agent and the user agent, while the Dev mode involves an expert agent for verifying the correctness of the answers provided by the API agent.

## User mode
In User mode, the API agent receives a question and an API to use for answering the question. It attempts to answer the question or create new APIs if the provided API cannot answer the question.

```python
# An API agent that performs self-improvement by default
api_agent = APIAgent("api_agent", self_improve=True)

# Question and API to use for answering the question
question = "What if we must go from node 1 to node 2?"
api = "change_dist"
user_prompt = "answer question: {question}, with api: {api}".format(question=question, api=api)

# User mode
# The API agent tries to create new APIs if the provided API cannot answer the provided question.
user = Agent("user")
api_agent.receive(user_prompt, user)
```

## Dev mode
In Dev mode, the expert agent verifies the correctness of the answer provided by the API agent. The process continues in a loop until the expert agent is satisfied with the answer (the verification returns "correct"). Once the expert agent is satisfied, the API agent has successfully answered the question.

```python
# Dev mode with an expert agent (could be an LLM or human)
expert = ExpertAgent("expert")
# If the expert agent is satisfied with the answer (the verification returns "correct"), the loop ends,
# and the API agent has successfully answered the question.
expert_feedback = None
while expert_feedback != "correct":
    prompt = user_prompt + "Expert verification: " + expert_feedback
    answer, new_api_suggested = api_agent.receive(prompt, expert)

    result = {
        "question": question,
        "api": api if not new_api_suggested else new_api_suggested,
        "answer": answer,
        "expert_feedback": expert_feedback
    }
    dev_prompt = "Verify the correctness of the answer. question: {question}, answer: {answer}.\
    The api used to get the answer is {api}".format(**result)

    # The expert agent provides feedback
    expert_feedback = expert.receive(dev_prompt, api_agent)
```
