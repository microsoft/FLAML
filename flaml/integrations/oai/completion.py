from time import sleep
import logging
import openai
from openai.error import (
    ServiceUnavailableError,
    RateLimitError,
    APIError,
    InvalidRequestError,
    APIConnectionError,
)
import diskcache
import numpy as np
from flaml import tune, BlendSearch

logger = logging.getLogger(__name__)


def get_key(config):
    """Get a unique identifier of a configuration.

    Args:
        config (dict or list): A configuration.

    Returns:
        tuple: A unique identifier which can be used as a key for a dict.
    """
    if isinstance(config, dict):
        return tuple(get_key(x) for x in sorted(config.items()))
    if isinstance(config, list):
        return tuple(get_key(x) for x in config)
    return config


class Completion:

    price1K = {
        "text-ada-001": 0.0004,
        "text-babbage-001": 0.0005,
        "text-curie-001": 0.002,
        "code-cushman-001": 0.002,  # TODO: update when available
        "code-davinci-002": 0.02,  # TODO: update when available
        "text-davinci-002": 0.02,
        "text-davinci-003": 0.02,
    }

    default_search_space = {
        "model": tune.choice(list(price1K.keys())),
        "temperature_or_top_p": tune.choice(
            [
                {"temperature": tune.uniform(0, 1)},
                {"top_p": tune.uniform(0, 1)},
            ]
        ),
        "max_tokens": tune.lograndint(100, 1000),
        "n": tune.randint(1, 100),
        "prompt": "{prompt}",
    }

    seed = 41
    # retry after this many seconds
    retry_time = 10
    # fail a request after hitting RateLimitError for this many seconds
    retry_timeout = 60

    @classmethod
    def set_cache(cls, seed=41, cache_path=".cache"):
        """Set cache path.

        Args:
            seed (int, Optional): The integer identifier for the pseudo seed.
                Results corresponding to different seeds will be cached in different places.
            cache_path (str, Optional): The root path for the cache.
                The complete cache path will be {cache_path}/{seed}.
        """
        cls.seed = seed
        cls.cache_path = f"{cache_path}/{seed}"

    @classmethod
    def _get_response(cls, config: dict, eval_only=False):
        """Get the response from the openai api call.

        Try cache first. If not found, call the openai api. If the api call fails, retry after retry_time.
        """
        key = get_key(config)
        response = cls._cache.get(key, None)
        if response is not None and (response != -1 or not eval_only):
            # print("using cached response")
            return response
        retry = 0
        while eval_only or retry * cls.retry_time < cls.retry_timeout:
            try:
                response = openai.Completion.create(**config)
                cls._cache.set(key, response)
                return response
            except (
                ServiceUnavailableError,
                APIError,
                InvalidRequestError,
                APIConnectionError,
            ):
                logger.info(f"retrying in {cls.retry_time} seconds...", exc_info=1)
                sleep(cls.retry_time)
            except RateLimitError:
                logger.info(f"retrying in {cls.retry_time} seconds...", exc_info=1)
                retry += 1
        logger.warning(
            f"Failed to get response from openai api due to getting RateLimitError for {cls.retry_timeout} seconds."
        )
        response = -1
        cls._cache.set(key, response)
        return response

    @classmethod
    def _get_max_safe_n(cls, key, max_tokens):
        # find the max value in max_safe_n_per_max_tokens whose key is equal or larger than max_tokens
        return max(
            (
                value
                for key, value in cls._max_safe_n_per_max_tokens.get(key, {}).items()
                if key >= max_tokens
            ),
            default=1,
        )

    @classmethod
    def _get_min_unsafe_n(cls, key, max_tokens):
        # find the min value in min_unsafe_n_per_max_tokens whose key is equal or smaller than max_tokens
        return min(
            (
                value
                for key, value in cls._min_unsafe_n_per_max_tokens.get(key, {}).items()
                if key <= max_tokens
            ),
            default=None,
        )

    @classmethod
    def _get_region_key(cls, config):
        # get a key for the safe/unsafe region corresponding to the given config
        return (config["model"], config["prompt"], config.get("stop"))

    @classmethod
    def eval(cls, config: dict, prune=True, eval_only=False):
        """Evaluate the given config as the hyperparameter setting for the openai api call.

        Args:
            config (dict): Hyperparameter setting for the openai api call.
            prune (bool, optional): Whether to enable pruning. Defaults to True.
            eval_only (bool, optional): Whether to evaluate only. Defaults to False.

        Returns:
            dict: Evaluation results.
        """
        cost = 0
        data = cls.data
        target_n_tokens = 1000 * cls.inference_budget / cls.price1K[config["model"]]
        prune_hp = cls._prune_hp
        metric = cls._metric
        config_n = config[prune_hp]
        max_tokens = config["max_tokens"]
        region_key = cls._get_region_key(config)
        prompt = cls._prompts[config["prompt"]]
        stop = cls._stops and cls._stops[config["stop"]]
        if prune:
            max_safe_n = cls._get_max_safe_n(region_key, max_tokens)
            min_unsafe_n = cls._get_min_unsafe_n(region_key, max_tokens)
            if min_unsafe_n is not None and config_n >= min_unsafe_n:
                if config_n > max_safe_n:
                    # prune this config
                    return {
                        "inference_cost": np.inf,
                        metric: np.inf if cls._mode == "min" else -np.inf,
                        "cost": cost,
                    }
                # since config_n<=max_safe_n, there is a chance config_n is safe
                start_n = config_n
            else:
                # start from a safe n
                start_n = min(max_safe_n, config_n)
        else:
            start_n = config_n
        params = config.copy()
        params["stop"] = stop
        temperature_or_top_p = params.pop("temperature_or_top_p", None)
        if temperature_or_top_p:
            params.update(temperature_or_top_p)
        data_length = len(data)
        n, previous_n = start_n, 0
        n_tokens_list, result, responses_list = [], {}, []
        while True:  # n <= config_n
            params[prune_hp] = n - previous_n
            data_limit = 1 if prune else data_length
            prev_data_limit = 0
            data_early_stop = False  # whether data early stop happens for this n
            while True:  # data_limit <= data_length
                # limit the number of data points to avoid rate limit
                for i in range(prev_data_limit, data_limit):
                    x = data[i]
                    params["prompt"] = prompt.format(**x)
                    response = cls._get_response(params, eval_only)
                    if response == -1:  # rate limit error, treat as invalid
                        if prune:
                            # update unsafe n and prune this config
                            cls._min_unsafe_n_per_max_tokens[
                                region_key
                            ] = unsafe_n = cls._min_unsafe_n_per_max_tokens.get(
                                region_key, {}
                            )
                            unsafe_n[max_tokens] = min(
                                n, unsafe_n.get(max_tokens, np.inf)
                            )
                        result[metric] = 0
                        result["cost"] = cost
                        return result
                    # evaluate the quality of the responses
                    responses = [r["text"].rstrip() for r in response["choices"]]
                    n_tokens = (
                        response["usage"]["completion_tokens"]
                        if previous_n
                        else response["usage"]["total_tokens"]
                    )
                    query_cost = (
                        response["usage"]["total_tokens"]
                        * cls.price1K[config["model"]]
                        / 1000
                    )
                    cls._total_cost += query_cost
                    cost += query_cost
                    if cls._total_cost >= cls.optimization_budget and not eval_only:
                        # limit the total tuning cost
                        return {
                            metric: 0,
                            "total_cost": cls._total_cost,
                            "cost": cost,
                        }
                    if previous_n:
                        n_tokens_list[i] += n_tokens
                        responses_list[i].extend(responses)
                    else:
                        n_tokens_list.append(n_tokens)
                        responses_list.append(responses)
                n_tokens = np.mean(n_tokens_list[:data_limit])
                rho = (
                    (1 - data_limit / data_length) * (1 + 1 / data_limit)
                    if data_limit << 1 > data_length
                    else (1 - (data_limit - 1) / data_length)
                )
                # Hoeffding-Serfling bound
                ratio = 0.1 * np.sqrt(rho / data_limit)
                if n_tokens > target_n_tokens * (1 + ratio) and not eval_only:
                    if prune:
                        # update unsafe n and prune this config
                        cls._min_unsafe_n_per_max_tokens[
                            region_key
                        ] = unsafe_n = cls._min_unsafe_n_per_max_tokens.get(
                            region_key, {}
                        )
                        unsafe_n[max_tokens] = min(n, unsafe_n.get(max_tokens, np.inf))
                    result[metric] = 0
                    result["total_cost"] = cls._total_cost
                    result["cost"] = cost
                    return result
                if (
                    prune
                    and n_tokens <= target_n_tokens * (1 - ratio)
                    and (n < config_n or n == config_n and data_limit == data_length)
                ):
                    # update safe n
                    cls._max_safe_n_per_max_tokens[
                        region_key
                    ] = safe_n = cls._max_safe_n_per_max_tokens.get(region_key, {})
                    safe_n[max_tokens] = max(n, safe_n.get(max_tokens, 0))
                    if n < config_n:
                        # safe already, skip the rest of the data
                        data_limit = data_length
                        data_early_stop = True
                        break
                prev_data_limit = data_limit
                if data_limit < data_length:
                    data_limit = min(data_limit << 1, data_length)
                else:
                    break
            # use exponential search to increase n
            if n == config_n:
                for i in range(data_limit):
                    x = data[i]
                    responses = responses_list[i]
                    metrics = cls._eval_func(responses, **x)
                    if result:
                        for key, value in metrics.items():
                            result[key] += value
                    else:
                        result = metrics
                for key in result.keys():
                    result[key] /= data_limit
                result["total_cost"] = cls._total_cost
                result["cost"] = cost
                result["inference_cost"] = (
                    n_tokens * cls.price1K[config["model"]] / 1000
                )
                break
            else:
                if data_early_stop:
                    previous_n = 0
                    n_tokens_list.clear()
                    responses_list.clear()
                else:
                    previous_n = n
                n = min(n << 1, config_n)
        return result

    @classmethod
    def tune(
        cls,
        data,
        metric,
        mode,
        eval_func,
        log_file_name=None,
        inference_budget=None,
        optimization_budget=None,
        num_samples=1,
        **config,
    ):
        """Tune the parameters for the OpenAI API call.

        Args:
            data (list): The list of data points.
            metric (str): The metric to optimize.
            mode (str): The optimization mode, "min" or "max.
            eval_func (Callable): The evaluation function for responses.
            log_file_name (str): The log file.
            inference_budget (float): The inference budget.
            optimization_budget (float): The optimization budget.
            num_samples (int): The number of samples to evaluate.
            search_space (dict): The search space to update over the default search.

        Returns:
            dict: The optimized hyperparameter setting.
            tune.ExperimentAnalysis: The tuning results.
        """
        space = Completion.default_search_space.copy()
        if config is not None:
            space.update(config)
            temperature = space.pop("temperature", None)
            top_p = space.pop("top_p", None)
            if temperature is not None and top_p is None:
                space["temperature_or_top_p"] = {"temperature": temperature}
            elif temperature is None and top_p is not None:
                space["temperature_or_top_p"] = {"top_p": top_p}
            elif temperature is not None and top_p is not None:
                space.pop("temperature_or_top_p")
                space["temperature"] = temperature
                space["top_p"] = top_p
        with diskcache.Cache(cls.cache_path) as cls._cache:
            cls._max_safe_n_per_max_tokens, cls._min_unsafe_n_per_max_tokens = {}, {}
            cls.optimization_budget = optimization_budget
            cls.inference_budget = inference_budget
            cls._prune_hp = "best_of" if space.get("best_of", 1) != 1 else "n"
            cls._prompts = space["prompt"]
            space["prompt"] = tune.choice(list(range(len(cls._prompts))))
            cls._stops = space.get("stop")
            if cls._stops:
                assert isinstance(
                    cls._stops, (str, list)
                ), "stop must be None, str or list."
                if isinstance(cls._stops, str) or not isinstance(cls._stops[0], list):
                    cls._stops = [cls._stops]
                space["stop"] = tune.choice(list(range(len(cls._stops))))
            cls._metric, cls._mode = metric, mode
            cls._total_cost = 0  # total optimization cost
            cls._eval_func = eval_func
            cls.data = data

            search_alg = BlendSearch(
                cost_attr="cost",
                cost_budget=optimization_budget,
                metric=metric,
                mode=mode,
                space=space,
            )
            analysis = tune.run(
                cls.eval,
                search_alg=search_alg,
                num_samples=num_samples,
                log_file_name=log_file_name,
                verbose=3,
            )
            config = analysis.best_config
            params = config.copy()
            params["prompt"] = cls._prompts[config["prompt"]]
            stop = cls._stops and cls._stops[config["stop"]]
            params["stop"] = stop
            temperature_or_top_p = params.pop("temperature_or_top_p", None)
            if temperature_or_top_p:
                params.update(temperature_or_top_p)
        return params, analysis

    @classmethod
    def create(cls, context, use_cache=True, **config):
        """Make a completion for a given context.

        Args:
            context (dict): The context to instantiate the prompt.
                It needs to contain keys that are used by the prompt template.
                E.g., `prompt="Complete the following sentence: {prefix}"`.
                `context={"prefix": "Today I feel"}`.
                The actual prompt sent to OpenAI will be:
                "Complete the following sentence: Today I feel".
            use_cache (bool, Optional): Whether to use cached responses.

        Returns:
            Responses from OpenAI API.
        """
        params = config.copy()
        params["prompt"] = config["prompt"].format(**context)
        if use_cache:
            return cls._get_response(params)
        return openai.Completion.create(**params)
