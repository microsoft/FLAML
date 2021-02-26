try:
    from ray.tune import (uniform, quniform, choice, randint, qrandint, randn,
 qrandn, loguniform, qloguniform, lograndint)
except:
    from .sample import (uniform, quniform, choice, randint, qrandint, randn,
 qrandn, loguniform, qloguniform, lograndint)
from .tune import run, report