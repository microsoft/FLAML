# Self-Improving API Agent
This code describes a scenario where an API agent is designed to self-improve
and interact with a user. The API agent tries to answer a given question using the provided API or suggests a new API if the existing one is not suitable.

The code is divided into two modes: User mode and Dev mode. The user mode provides a basic interaction between the API agent and the user agent, while the Dev mode involves an expert agent for verifying the correctness of the answers provided by the API agent.

## User mode
In User mode, the API agent use `receive` the function call to receive questions from users and answer the questions. It attempts to answer the question with existing apis and will try to add new apis if the question cannot be answered by existing ones properly.

```python
# An API agent that performs self-improvement by default
api_agent = APIAgent("api_agent", self_improve=True, api="api.json")

# User mode
user = HumanAgent("human user")
# Question from the user
question = "What if we must go from node 1 to node 2?"
# the api_agent will send the answer to user
api_agent.receive(question, user)
```

## Dev mode
In Dev mode, the expert agent verifies the correctness of the answer provided by the API agent. The process continues until the expert agent is satisfied with the answer (the verification returns "correct") or an interaction number upper limit is reached. Once the expert agent is satisfied, the API agent has successfully answered the question.

```python
# Dev mode with an expert agent (could be an LLM or human)
expert = ExpertAgent("expert")
# The api agent shall communicate with the expert
api_agent.receive(question, expert)
```


The code below is the old code for the self-improving API agent. It is not used anymore.


```python
expert_feedback = None
while expert_feedback != "correct":
    prompt = user_prompt + "Expert verification: " + expert_feedback
    answer, new_api_suggested = api_agent.receive(prompt, expert)

    result = {
        "question": question,
        # "api": api if not new_api_suggested else new_api_suggested,
        "answer": answer,
        "expert_feedback": expert_feedback
    }
    dev_prompt = "Verify the correctness of the answer. question: {question}, answer: {answer}.\
    The api used to get the answer is {api}".format(**result)

    # The expert agent provides feedback
    expert_feedback = expert.receive(dev_prompt, api_agent)
```
