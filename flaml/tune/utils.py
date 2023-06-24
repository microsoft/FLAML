from typing import Sequence

try:
    from ray import __version__ as ray_version

    assert ray_version >= "1.10.0"
    if ray_version.startswith("1."):
        from ray.tune import sample
    else:
        from ray.tune.search import sample
except (ImportError, AssertionError):
    from . import sample


def choice(categories: Sequence, order=None):
    """Sample a categorical value.
    Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
    ``np.random.choice([1, 2])``

    Args:
        categories (Sequence): Sequence of categories to sample from.
        order (bool): Whether the categories have an order. If None, will be decided autoamtically:
            Numerical categories have an order, while string categories do not.
    """
    domain = sample.Categorical(categories).uniform()
    domain.ordered = order if order is not None else all(isinstance(x, (int, float)) for x in categories)
    return domain


def get_lexico_bound(metric, mode, lexico_objectives, f_best):
    """Get targeted vector according to the historical points.
    LexiFlow uses targeted vector to justify the order of different configurations.
    """
    k_target = lexico_objectives["targets"][metric] if mode == "min" else -1 * lexico_objectives["targets"][metric]
    if not isinstance(lexico_objectives["tolerances"][metric], str):
        tolerance_bound = f_best[metric] + lexico_objectives["tolerances"][metric]
    else:
        assert (
            lexico_objectives["tolerances"][metric][-1] == "%"
        ), "String tolerance of {} should use %% as the suffix".format(metric)
        tolerance_bound = f_best[metric] * (1 + 0.01 * float(lexico_objectives["tolerances"][metric].replace("%", "")))
    bound = max(tolerance_bound, k_target)
    return bound
