try:
    from ray import __version__ as ray_version

    assert ray_version >= "1.10.0"
    from ray.tune import (
        lograndint,
        loguniform,
        qlograndint,
        qloguniform,
        qrandint,
        qrandn,
        quniform,
        randint,
        randn,
        uniform,
    )

    if ray_version.startswith("1."):
        from ray.tune import sample
    else:
        from ray.tune.search import sample
except (ImportError, AssertionError):
    from . import sample
    from .sample import (
        lograndint,
        loguniform,
        qlograndint,
        qloguniform,
        qrandint,
        qrandn,
        quniform,
        randint,
        randn,
        uniform,
    )
from .sample import Categorical, Float, PolynomialExpansionSet, polynomial_expansion_set
from .trial import Trial
from .tune import INCUMBENT_RESULT, report, run
from .utils import choice
