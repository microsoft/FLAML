import datasets
import sys
import numpy as np
import pytest
from functools import partial
from flaml import oai
from flaml.autogen.code_utils import (
    eval_function_completions,
    generate_assertions,
    implement,
)
from flaml.autogen.math_utils import eval_math_responses


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="do not run on windows",
)
def test_humaneval(num_samples=1):
    eval_with_generated_assertions = partial(eval_function_completions, assertions=generate_assertions)

    seed = 41
    data = datasets.load_dataset("openai_humaneval")["test"].shuffle(seed=seed)
    n_tune_data = 20
    tune_data = [
        {
            "definition": data[x]["prompt"],
            "test": data[x]["test"],
            "entry_point": data[x]["entry_point"],
        }
        for x in range(n_tune_data)
    ]
    test_data = [
        {
            "definition": data[x]["prompt"],
            "test": data[x]["test"],
            "entry_point": data[x]["entry_point"],
        }
        for x in range(n_tune_data, len(data))
    ]
    oai.Completion.set_cache(seed)
    try:
        import openai
        import diskcache
    except ImportError as exc:
        print(exc)
        return
    # a minimal tuning example
    config, _ = oai.Completion.tune(
        data=tune_data,
        metric="success",
        mode="max",
        eval_func=eval_function_completions,
        n=1,
        prompt="{definition}",
    )
    responses = oai.Completion.create(context=test_data[0], **config)
    # a minimal tuning example for tuning chat completion models using the Completion class
    config, _ = oai.Completion.tune(
        data=tune_data,
        metric="succeed_assertions",
        mode="max",
        eval_func=eval_with_generated_assertions,
        n=1,
        model="gpt-3.5-turbo",
        prompt="{definition}",
    )
    responses = oai.Completion.create(context=test_data[0], **config)
    # a minimal tuning example for tuning chat completion models using the Completion class
    config, _ = oai.ChatCompletion.tune(
        data=tune_data,
        metric="expected_success",
        mode="max",
        eval_func=eval_function_completions,
        n=1,
        messages=[{"role": "user", "content": "{definition}"}],
    )
    responses = oai.ChatCompletion.create(context=test_data[0], **config)
    print(responses)
    code, cost, _ = implement(tune_data[1], [config])
    print(code)
    print(cost)
    print(eval_function_completions([code], **tune_data[1]))
    # a more comprehensive tuning example
    config2, analysis = oai.Completion.tune(
        data=tune_data,
        metric="success",
        mode="max",
        eval_func=eval_with_generated_assertions,
        log_file_name="logs/humaneval.log",
        inference_budget=0.002,
        optimization_budget=2,
        num_samples=num_samples,
        prompt=[
            "{definition}",
            "# Python 3{definition}",
            "Complete the following Python function:{definition}",
        ],
        stop=[["\nclass", "\ndef", "\nif", "\nprint"], None],  # the stop sequences
    )
    print(config2)
    print(analysis.best_result)
    print(test_data[0])
    responses = oai.Completion.create(context=test_data[0], **config2)
    print(responses)
    oai.Completion.data = test_data[:num_samples]
    result = oai.Completion._eval(analysis.best_config, prune=False, eval_only=True)
    print("result without pruning", result)
    result = oai.Completion.test(test_data[:num_samples], config=config2)
    print(result)
    code, cost, selected = implement(tune_data[1], [config2, config])
    print(selected)
    print(eval_function_completions([code], **tune_data[1]))


def test_math(num_samples=-1):
    seed = 41
    data = datasets.load_dataset("competition_math")
    train_data = data["train"].shuffle(seed=seed)
    test_data = data["test"].shuffle(seed=seed)
    n_tune_data = 20
    tune_data = [
        {
            "problem": train_data[x]["problem"],
            "solution": train_data[x]["solution"],
        }
        for x in range(len(train_data))
        if train_data[x]["level"] == "Level 1"
    ][:n_tune_data]
    test_data = [
        {
            "problem": test_data[x]["problem"],
            "solution": test_data[x]["solution"],
        }
        for x in range(len(test_data))
        if test_data[x]["level"] == "Level 1"
    ]
    print(
        "max tokens in tuning data's canonical solutions",
        max([len(x["solution"].split()) for x in tune_data]),
    )
    print(len(tune_data), len(test_data))
    # prompt template
    prompts = [
        lambda data: "%s Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{}."
        % data["problem"]
    ]

    try:
        import openai
        import diskcache
    except ImportError as exc:
        print(exc)
        return

    oai.ChatCompletion.set_cache(seed)
    vanilla_config = {
        "model": "gpt-3.5-turbo",
        "temperature": 1,
        "max_tokens": 2048,
        "n": 1,
        "prompt": prompts[0],
        "stop": "###",
    }
    test_data_sample = test_data[0:3]
    result = oai.ChatCompletion.test(test_data_sample, vanilla_config, eval_math_responses)
    test_data_sample = test_data[3:6]
    result = oai.ChatCompletion.test(
        test_data_sample,
        vanilla_config,
        eval_math_responses,
        use_cache=False,
        agg_method="median",
    )

    def my_median(results):
        return np.median(results)

    def my_average(results):
        return np.mean(results)

    result = oai.ChatCompletion.test(
        test_data_sample,
        vanilla_config,
        eval_math_responses,
        use_cache=False,
        agg_method=my_median,
    )
    result = oai.ChatCompletion.test(
        test_data_sample,
        vanilla_config,
        eval_math_responses,
        use_cache=False,
        agg_method={
            "expected_success": my_median,
            "success": my_average,
            "success_vote": my_average,
            "votes": np.mean,
        },
    )

    print(result)

    config, _ = oai.ChatCompletion.tune(
        data=tune_data,  # the data for tuning
        metric="expected_success",  # the metric to optimize
        mode="max",  # the optimization mode
        eval_func=eval_math_responses,  # the evaluation function to return the success metrics
        # log_file_name="logs/math.log",  # the log file name
        inference_budget=0.002,  # the inference budget (dollar)
        optimization_budget=0.01,  # the optimization budget (dollar)
        num_samples=num_samples,
        prompt=prompts,  # the prompt templates to choose from
        stop="###",  # the stop sequence
    )
    print("tuned config", config)
    result = oai.ChatCompletion.test(test_data_sample, config)
    print("result from tuned config:", result)
    print("empty responses", eval_math_responses([], None))


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    test_humaneval(1)
    # test_math(1)
