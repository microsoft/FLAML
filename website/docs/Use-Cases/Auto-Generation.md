# Auto Generation

`flaml.autogen` is a package for automating generation tasks (in preview), featuring:
* Leveraging [`flaml.tune`](../reference/tune/tune) to adapt LLMs to applications, such that:
  - Maximize the utility out of using expensive foundation models.
  - Reduce the inference cost by using cheaper models or configurations which achieve equal or better performance.
* An enhanced inference API as a drop-in replacement of `openai.Completion.create` or `openai.ChatCompletion.create` with utilities like API unification, caching, error handling, multi-config inference, context programming etc.
* Higher-level components like LLM-based intelligent agents which can perform tasks autonomously or with human feedback, including tasks that require using tools via code.

The package is under active development with more features upcoming.

## Tune Inference Parameters

### Choices to optimize

The cost of using foundation models for text generation is typically measured in terms of the number of tokens in the input and output combined. From the perspective of an application builder using foundation models, the use case is to maximize the utility of the generated text under an inference budget constraint (e.g., measured by the average dollar cost needed to solve a coding problem). This can be achieved by optimizing the hyperparameters of the inference,
which can significantly affect both the utility and the cost of the generated text.

The tunable hyperparameters include:
1. model - this is a required input, specifying the model ID to use.
1. prompt/messages - the input prompt/messages to the model, which provides the context for the text generation task.
1. max_tokens - the maximum number of tokens (words or word pieces) to generate in the output.
1. temperature - a value between 0 and 1 that controls the randomness of the generated text. A higher temperature will result in more random and diverse text, while a lower temperature will result in more predictable text.
1. top_p - a value between 0 and 1 that controls the sampling probability mass for each token generation. A lower top_p value will make it more likely to generate text based on the most likely tokens, while a higher value will allow the model to explore a wider range of possible tokens.
1. n - the number of responses to generate for a given prompt. Generating multiple responses can provide more diverse and potentially more useful output, but it also increases the cost of the request.
1. stop - a list of strings that, when encountered in the generated text, will cause the generation to stop. This can be used to control the length or the validity of the output.
1. presence_penalty, frequency_penalty - values that control the relative importance of the presence and frequency of certain words or phrases in the generated text.
1. best_of - the number of responses to generate server-side when selecting the "best" (the one with the highest log probability per token) response for a given prompt.

The cost and utility of text generation are intertwined with the joint effect of these hyperparameters.
There are also complex interactions among subsets of the hyperparameters. For example,
the temperature and top_p are not recommended to be altered from their default values together because they both control the randomness of the generated text, and changing both at the same time can result in conflicting effects; n and best_of are rarely tuned together because if the application can process multiple outputs, filtering on the server side causes unnecessary information loss; both n and max_tokens will affect the total number of tokens generated, which in turn will affect the cost of the request.
These interactions and trade-offs make it difficult to manually determine the optimal hyperparameter settings for a given text generation task.

*Do the choices matter? Check this [blogpost](/blog/2023/04/21/LLM-tuning-math) to find example tuning results about gpt-3.5-turbo and gpt-4.*


With `flaml.autogen`, the tuning can be performed with the following information:
1. Validation data.
1. Evaluation function.
1. Metric to optimize.
1. Search space.
1. Budgets: inference and optimization respectively.

### Validation data

Collect a diverse set of instances. They can be stored in an iterable of dicts. For example, each instance dict can contain "problem" as a key and the description str of a math problem as the value; and "solution" as a key and the solution str as the value.

### Evaluation function

The evaluation function should take a list of responses, and other keyword arguments corresponding to the keys in each validation data instance as input, and output a dict of metrics. For example,

```python
def eval_math_responses(responses: List[str], solution: str, **args) -> Dict:
    # select a response from the list of responses
    answer = voted_answer(responses)
    # check whether the answer is correct
    return {"success": is_equivalent(answer, solution)}
```

[`flaml.autogen.code_utils`](../reference/autogen/code_utils) and [`flaml.autogen.math_utils`](../reference/autogen/math_utils) offer some example evaluation functions for code generation and math problem solving.

### Metric to optimize

The metric to optimize is usually an aggregated metric over all the tuning data instances. For example, users can specify "success" as the metric and "max" as the optimization mode. By default, the aggregation function is taking the average. Users can provide a customized aggregation function if needed.

### Search space

Users can specify the (optional) search range for each hyperparameter.

1. model. Either a constant str, or multiple choices specified by `flaml.tune.choice`.
1. prompt/messages. Prompt is either a str or a list of strs, of the prompt templates. messages is a list of dicts or a list of lists, of the message templates.
Each prompt/message template will be formatted with each data instance. For example, the prompt template can be:
"{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
And `{problem}` will be replaced by the "problem" field of each data instance.
1. max_tokens, n, best_of. They can be constants, or specified by `flaml.tune.randint`, `flaml.tune.qrandint`, `flaml.tune.lograndint` or `flaml.qlograndint`. By default, max_tokens is searched in [50, 1000); n is searched in [1, 100); and best_of is fixed to 1.
1. stop. It can be a str or a list of strs, or a list of lists of strs or None. Default is None.
1. temperature or top_p. One of them can be specified as a constant or by `flaml.tune.uniform` or `flaml.tune.loguniform` etc.
Please don't provide both. By default, each configuration will choose either a temperature or a top_p in [0, 1] uniformly.
1. presence_penalty, frequency_penalty. They can be constants or specified by `flaml.tune.uniform` etc. Not tuned by default.

### Budgets

One can specify an inference budget and an optimization budget.
The inference budget refers to the average inference cost per data instance.
The optimization budget refers to the total budget allowed in the tuning process. Both are measured by dollars and follow the price per 1000 tokens.

### Perform tuning

Now, you can use [`flaml.oai.Completion.tune`](../reference/autogen/oai/completion#tune) for tuning. For example,

```python
from flaml import oai

config, analysis = oai.Completion.tune(
    data=tune_data,
    metric="success",
    mode="max",
    eval_func=eval_func,
    inference_budget=0.05,
    optimization_budget=3,
    num_samples=-1,
)
```

`num_samples` is the number of configurations to sample. -1 means unlimited (until optimization budget is exhausted).
The returned `config` contains the optimized configuration and `analysis` contains an [ExperimentAnalysis](../reference/tune/analysis#experimentanalysis-objects) object for all the tried configurations and results.

The tuend config can be used to perform inference.

*Refer to this [page](../Examples/AutoGen-OpenAI) for a full example.*

## Perform Inference

One can use [`flaml.oai.Completion.create`](../reference/autogen/oai/completion#create) to perform inference.
There are a number of benefits of using `flaml.oai.Completion.create` to perform inference.

### API unification

`flaml.oai.Completion.create` is compatible with both `openai.Completion.create` and `openai.ChatCompletion.create`, and both OpenAI API and Azure OpenAI API. So models such as "text-davinci-003", "gpt-3.5-turbo" and "gpt-4" can share a common API.
When chat models are used and `prompt` is given as the input to `flaml.oai.Completion.create`, the prompt will be automatically converted into `messages` to fit the chat completion API requirement. One advantage is that one can experiment with both chat and non-chat models for the same prompt in a unified API.

For local LLMs, one can spin up an endpoint using a package like [simple_ai_server](https://github.com/lhenault/simpleAI), and then use the same API to send a request.

When only working with the chat-based models, `flaml.oai.ChatCompletion` can be used. It also does automatic conversion from prompt to messages, if prompt is provided instead of messages.

### Caching

API call results are cached locally and reused when the same request is issued. This is useful when repeating or continuing experiments for reproducibility and cost saving. It still allows controlled randomness by setting the "seed", using [`set_cache`](../reference/autogen/oai/completion#set_cache) or specifying in `create()`.

### Error handling

#### Runtime error

It is easy to hit error when calling OpenAI APIs, due to connection, rate limit, or timeout. Some of the errors are transient. `flaml.oai.Completion.create` deals with the transient errors and retries automatically. Initial request timeout, retry timeout and retry time interval can be configured via `flaml.oai.request_timeout`, `flaml.oai.retry_timeout` and `flaml.oai.retry_time`.

Moreover, one can pass a list of configurations of different models/endpoints to mitigate the rate limits. For example,

```python
response = oai.Completion.create(
    config_list=[
        {
            "model": "gpt-4",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "api_base": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2023-03-15-preview",
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "open_ai",
            "api_base": "https://api.openai.com/v1",
            "api_version": None,
        },
        {
            "model": "llama-7B",
            "api_base": "http://127.0.0.1:8080",
            "api_type": "open_ai",
            "api_version": None,
        }
    ],
    prompt="Hi",
)
```

It will try querying Azure OpenAI gpt-4, OpenAI gpt-3.5-turbo, and a locally hosted llama-7B one by one, ignoring AuthenticationError, RateLimitError and Timeout,
until a valid result is returned. This can speed up the development process where the rate limit is a bottleneck. An error will be raised if the last choice fails. So make sure the last choice in the list has the best availability.

#### Logic error

Another type of error is that the returned response does not satisfy a requirement. For example, if the response is required to be a valid json string, one would like to filter the responses that are not. This can be achieved by providing a list of configurations and a filter function. For example,

```python
def valid_json_filter(context, config, response):
    for text in oai.Completion.extract_text(response):
        try:
            json.loads(text)
            return True
        except ValueError:
            pass
    return False

response = oai.Completion.create(
    config_list=[{"model": "text-ada-001"}, {"model": "gpt-3.5-turbo"}, {"model": "text-davinci-003"}],
    prompt="How to construct a json request to Bing API to search for 'latest AI news'? Return the JSON request.",
    filter_func=valid_json_filter,
)
```

The example above will try to use text-ada-001, gpt-3.5-turbo, and text-davinci-003 iteratively, until a valid json string is returned or the last config is used. One can also repeat the same model in the list for multiple times to try one model multiple times for increasing the robustness of the final response.

*Advanced use case: Check this [blogpost](/blog/2023/05/18/GPT-adaptive-humaneval) to find how to improve GPT-4's coding performance from 68% to 90% while reducing the inference cost.*

### Templating

If the provided prompt or message is a template, it will be automatically materialized with a given context. For example,

```python
response = oai.Completion.create(
    context={"problem": "How many positive integers, not exceeding 100, are multiples of 2 or 3 but not 4?"},
    prompt="{problem} Solve the problem carefully.",
    **config
)
```

A template is either a format str, like the example above, or a function which produces a str from several input fields, like the example below.

```python
def content(turn, **context):
    return "\n".join(
        [
            context[f"user_message_{turn}"],
            context[f"external_info_{turn}"]
        ]
    )

messages = [
    {
        "role": "system",
        "content": "You are a teaching assistant of math.",
    },
    {
        "role": "user",
        "content": partial(content, turn=0),
    },
]
context = {
    "user_message_0": "Could you explain the solution to Problem 1?",
    "external_info_0": "Problem 1: ...",
}

response = oai.ChatCompletion.create(context, messages=messages, **config)
messages.append(
    {
        "role": "assistant",
        "content": oai.ChatCompletion.extract_text(response)[0]
    }
)
messages.append(
    {
        "role": "user",
        "content": partial(content, turn=1),
    },
)
context.append(
    {
        "user_message_1": "Why can't we apply Theorem 1 to Equation (2)?",
        "external_info_1": "Theorem 1: ...",
    }
)
response = oai.ChatCompletion.create(context, messages=messages, **config)
```

### Logging (Experimental)

When debugging or diagnosing an LLM-based system, it is often convenient to log the API calls and analyze them. `flaml.oai.Completion` and `flaml.oai.ChatCompletion` offer an easy way to collect the API call histories. For example, to log the chat histories, simply run:
```python
flaml.oai.ChatCompletion.start_logging()
```
The API calls made after this will be automatically logged. They can be retrieved at any time by:
```python
flaml.oai.ChatCompletion.logged_history
```
To stop logging, use
```python
flaml.oai.ChatCompletion.stop_logging()
```
If one would like to append the history to an existing dict, pass the dict like:
```python
flaml.oai.ChatCompletion.start_logging(history_dict=existing_history_dict)
```
By default, the counter of API calls will be reset at `start_logging()`. If no reset is desired, set `reset_counter=False`.

There are two types of logging formats: compact logging and individual API call logging. The default format is compact.
Set `compact=False` in `start_logging()` to switch.

* Example of a history dict with compact logging.
```python
{
    """
    [
        {
            'role': 'system',
            'content': system_message,
        },
        {
            'role': 'user',
            'content': user_message_1,
        },
        {
            'role': 'assistant',
            'content': assistant_message_1,
        },
        {
            'role': 'user',
            'content': user_message_2,
        },
        {
            'role': 'assistant',
            'content': assistant_message_2,
        },
    ]""": {
        "created_at": [0, 1],
        "cost": [0.1, 0.2],
    }
}
```

* Example of a history dict with individual API call logging.
```python
{
    0: {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message_1,
                }
            ],
            ... # other parameters in the request
        },
        "response": {
            "choices": [
                "messages": {
                    "role": "assistant",
                    "content": assistant_message_1,
                },
            ],
            ... # other fields in the response
        }
    },
    1: {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message_1,
                },
                {
                    "role": "assistant",
                    "content": assistant_message_1,
                },
                {
                    "role": "user",
                    "content": user_message_2,
                },
            ],
            ... # other parameters in the request
        },
        "response": {
            "choices": [
                "messages": {
                    "role": "assistant",
                    "content": assistant_message_2,
                },
            ],
            ... # other fields in the response
        }
    },
}
```
It can be seen that the individual API call history contain redundant information of the conversation. For a long conversation the degree of redundancy is high.
The compact history is more efficient and the individual API call history contains more details.

### Other Utilities

- a [`cost`](../reference/autogen/oai/completion#cost) function to calculate the cost of an API call.
- a [`test`](../reference/autogen/oai/completion#test) function to conveniently evaluate the configuration over test data.
- a [`extract_text`](../reference/autogen/oai/completion#extract_text) function to extract the text from a completion or chat response.


## Agents (Experimental)

[`flaml.autogen.agents`](../reference/autogen/agent/agent) contains an experimental implementation of interactive agents which can adapt to human or simulated feedback. This subpackage is under active development.

*Interested in trying it yourself? Please check the following notebook example:*
* [Use agents in FLAML to perform tasks with code](https://github.com/microsoft/FLAML/blob/main/notebook/autogen_agent_auto_feedback_from_code_execution.ipynb)

## Utilities for Applications

### Code

[`flaml.autogen.code_utils`](../reference/autogen/code_utils) offers code-related utilities, such as:
- a [`improve_code`](../reference/autogen/code_utils#improve_code) function to improve code for a given objective.
- a [`generate_assertions`](../reference/autogen/code_utils#generate_assertions) function to generate assertion statements from function signature and docstr.
- a [`implement`](../reference/autogen/code_utils#implement) function to implement a function from a definition.
- a [`eval_function_completions`](../reference/autogen/code_utils#eval_function_completions) function to evaluate the success of a function completion task, or select a response from a list of responses using generated assertions.

### Math

[`flaml.autogen.math_utils`](../reference/autogen/math_utils) offers utilities for math problems, such as:
- a [eval_math_responses](../reference/autogen/math_utils#eval_math_responses) function to select a response using voting, and check if the final answer is correct if the canonical solution is provided.


*Interested in trying it yourself? Please check the following notebook examples:*
* [Optimize for Code Gen](https://github.com/microsoft/FLAML/blob/main/notebook/autogen_openai_completion.ipynb)
* [Optimize for Math](https://github.com/microsoft/FLAML/blob/main/notebook/autogen_chatgpt_gpt4.ipynb)
