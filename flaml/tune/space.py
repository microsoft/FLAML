
try:
    from ray.tune import sample
except ImportError:
    from . import sample
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def define_by_run_func(
    trial, space: Dict, path: str = ""
) -> Optional[Dict[str, Any]]:
    """Define-by-run function to create the search space.

    Returns:
        None or a dict with constant values.
    """
    config = {}
    for key, domain in space.items():
        if path:
            key = path + '/' + key
        if not isinstance(domain, sample.Domain):
            config[key] = domain
            continue
        sampler = domain.get_sampler()
        quantize = None
        if isinstance(sampler, sample.Quantized):
            quantize = sampler.q
            sampler = sampler.sampler
            if isinstance(sampler, sample.LogUniform):
                logger.warning(
                    "Optuna does not handle quantization in loguniform "
                    "sampling. The parameter will be passed but it will "
                    "probably be ignored.")
        if isinstance(domain, sample.Float):
            if isinstance(sampler, sample.LogUniform):
                if quantize:
                    logger.warning(
                        "Optuna does not support both quantization and "
                        "sampling from LogUniform. Dropped quantization.")
                trial.suggest_float(
                    key, domain.lower, domain.upper, log=True)
            elif isinstance(sampler, sample.Uniform):
                if quantize:
                    trial.suggest_float(
                        key, domain.lower, domain.upper, step=quantize)
                trial.suggest_float(key, domain.lower, domain.upper)
        elif isinstance(domain, sample.Integer):
            if isinstance(sampler, sample.LogUniform):
                trial.suggest_int(
                    key, domain.lower,
                    domain.upper - int(bool(not quantize)),
                    step=quantize or 1, log=True)
            elif isinstance(sampler, sample.Uniform):
                # Upper bound should be inclusive for quantization and
                # exclusive otherwise
                trial.suggest_int(
                    key, domain.lower,
                    domain.upper - int(bool(not quantize)),
                    step=quantize or 1)
        elif isinstance(domain, sample.Categorical):
            if isinstance(sampler, sample.Uniform):
                if not hasattr(domain, 'choices'):
                    domain.choices = list(range(len(domain.categories)))
                choices = domain.choices
                # This choice needs to be removed from the final config
                index = trial.suggest_categorical(key + '_choice_', choices)
                choice = domain.categories[index]
                if isinstance(choice, dict):
                    key += f":{index}"
                    # the suffix needs to be removed from the final config
                    config[key] = define_by_run_func(trial, choice, key)
        else:
            raise ValueError(
                "Optuna search does not support parameters of type "
                "`{}` with samplers of type `{}`".format(
                    type(domain).__name__,
                    type(domain.sampler).__name__))
    # Return all constants in a dictionary.
    return config


def exclusive_to_inclusive(space: Dict) -> Dict:
    """Change the upper bound from exclusive to inclusive.

    Returns:
        A copied dict with modified upper bound.
    """
    space = space.copy()
    for key in space:
        domain = space[key]
        if not isinstance(domain, sample.Domain):
            continue
        sampler = domain.get_sampler()
        if isinstance(sampler, sample.Quantized):
            continue
        if isinstance(domain, sample.Integer):
            if isinstance(sampler, sample.LogUniform):
                space[key] = sample.lograndint(
                    domain.lower, domain.upper - 1, sampler.base)
            elif isinstance(sampler, sample.Uniform):
                space[key] = sample.randint(
                    domain.lower, domain.upper - 1)
    return space


def add_cost_to_space(space: Dict, low_cost_point: Dict, choice_cost: Dict):
    """Update the space in place by adding low_cost_point and choice_cost
    """
    for key in space:
        domain = space[key]
        if not isinstance(domain, sample.Domain):
            continue
        low_cost = low_cost_point.get(key)
        choice_cost_list = choice_cost.get(key)
        if isinstance(domain, sample.Categorical):
            for i, cat in enumerate(domain.categories):
                if isinstance(cat, dict):
                    if isinstance(low_cost, list):
                        low_cost_dict = low_cost[i]
                    else:
                        low_cost_dict = {}
                    if choice_cost_list:
                        choice_cost_dict = choice_cost_list[i]
                    else:
                        choice_cost_dict = {}
                    add_cost_to_space(cat, low_cost_dict, choice_cost_dict)
            if choice_cost_list:
                if len(choice_cost_list) == len(domain.categories):
                    domain.choice_cost = choice_cost_list
                else:
                    domain.choice_cost = choice_cost_list[-1]
                # sort the choices by cost
                cost = np.array(domain.choice_cost)
                ind = np.argsort(cost)
                domain.categories = np.array(domain.categories)[ind].tolist()
                domain.choice_cost = cost[ind]
                domain.ordered = True
            elif all(isinstance(x, int) or isinstance(x, float)
                     for x in domain.categories):
                # sort the choices by value
                domain.categories.sort()
                domain.ordered = True
            else:
                domain.ordered = False
            if isinstance(low_cost, list) and low_cost not in domain.categories:
                domain.low_cost_point = low_cost[-1]
                return
        if low_cost:
            domain.low_cost_point = low_cost

