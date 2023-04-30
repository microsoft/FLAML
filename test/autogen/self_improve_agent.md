# Self-Improving Agent
This code describes a scenario where an agent (SIAgent) is designed to be able to interact with a user and self-improve.

The code is divided into two modes: User mode and Dev mode. The user mode provides a basic interaction between the SIAgent and the user agent, while the Dev mode involves an expert agent for verifying the correctness of the answers provided by a SIAgent.

## User mode
In User mode, the SIAgent use `receive` the function call to receive questions from users and answer the questions. It attempts to answer the question with existing apis and will try to add new apis if the question cannot be answered by existing ones properly.

```python
# An SIAgent that performs self-improvement by default
api_agent =SIAgent("api_agent", self_improve=True, context="api.json")

# User mode
user = HumanAgent("human user")
# Question from the user
question = "What if we must go from node 1 to node 2?"
# the api_agent will communicate with the user
api_agent.receive(question, user)
# the api_agent will either answer the question (with the existing api or newly created api), or ask for more information from the user (e.g., to clarify the question)
```

## Dev mode
In Dev mode, the expert agent verifies the correctness of the answer provided by the SIAgent. The process continues until the expert agent is satisfied with the answer (the verification returns "correct") or an interaction number upper limit is reached. Once the expert agent is satisfied, the SIAgent has successfully answered the question.

```python
# Dev mode with an expert agent (could be an LLM or human)
expert = ExpertAgent("expert")
# The SIAgent shall communicate with the expert
api_agent.receive(question, expert)
```

## User experience after the receive function call
To answer the question
"Describe what happens after the receive call
To be more specific, do we want to assume what experience happens after receive()? For example, like ChatGPT, like AutoGPT, or neither, or both."

Option 1: ChatGPT
Conversation between the user and an SiAgent similar to the conversation between the user and ChatGPT.

Option 2: autoGPT
An SIAgent show the progress of the task execution and have pre-defined messages to ask for more information (or permission to continue) from the user

Option 3: branching conversation model

The SIAgent initiates multiple conversation sessions or threads, allowing for diverse paths to be considered simultaneously. Agent B then has the option to selectively engage with one or more of these threads, leading to a potentially rich and varied conversation. As the conversation progresses, it can continue to branch out as new responses or subthreads are introduced, creating a dynamic and non-linear dialogue structure.
