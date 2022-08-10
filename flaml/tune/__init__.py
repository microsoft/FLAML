try:
    from ray import __version__ as ray_version

    assert ray_version >= "1.10.0"
    from ray.tune import (
        uniform,
        quniform,
        randint,
        qrandint,
        randn,
        qrandn,
        loguniform,
        qloguniform,
        lograndint,
        qlograndint,
        sample,
    )
except (ImportError, AssertionError):
    from .sample import (
        uniform,
        quniform,
        randint,
        qrandint,
        randn,
        qrandn,
        loguniform,
        qloguniform,
        lograndint,
        qlograndint,
    )
    from . import sample
from .tune import run, report, INCUMBENT_RESULT
from .sample import polynomial_expansion_set
from .sample import PolynomialExpansionSet, Categorical, Float
from .trial import Trial
from typing import Sequence


def choice(categories: Sequence, ordered=None):
    """Sample a categorical value.
    Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
    ``np.random.choice([1, 2])``

    Args:
        categories (Sequence): Sequence of categories to sample from.
        ordered (bool): Whether the categories have an order. If None, will be decided autoamtically:
            Numerical categories are ordered, while string categories are not.
    """
    domain = sample.Categorical(categories).uniform()
    domain.ordered = (
        ordered
        if ordered is not None
        else all(isinstance(x, int) or isinstance(x, float) for x in categories)
    )
    return domain
