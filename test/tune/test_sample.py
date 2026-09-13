import numpy as np

from flaml.tune import choice
from flaml.tune.sample import (
    BaseSampler,
    Domain,
    PolynomialExpansionSet,
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


def test_sampler():
    print(randn().sample(size=2))
    print(PolynomialExpansionSet(), BaseSampler())
    print(qrandn(2, 10, 2).sample(size=2))
    c = choice([1, 2])
    print(c.domain_str, len(c), c.is_valid(3))
    c = choice([1, 2], order=False)
    print(c.domain_str, len(c), c.ordered)
    i = randint(1, 10)
    print(i.domain_str, i.is_valid(10))
    d = Domain()
    print(d.domain_str, d.is_function())
    d.default_sampler_cls = BaseSampler
    print(d.get_sampler())


def _sampled_values(domain, n=5000, seed=0):
    rs = np.random.RandomState(seed)
    return {domain.sample(spec=None, random_state=rs) for _ in range(n)}


def test_qrandint_reaches_inclusive_upper_bound():
    # docstring: "lower is inclusive, upper is also inclusive (!)"
    values = _sampled_values(qrandint(1, 10))
    assert 10 in values
    assert values <= set(range(1, 11))

    # the exact q=1 shape used by flaml.automl.time_series.ts_model's own ARIMA/SARIMAX
    # search spaces (e.g. tune.qrandint(lower=0, upper=6, q=1))
    values = _sampled_values(qrandint(0, 6, q=1))
    assert 6 in values
    assert values <= set(range(0, 7))


def test_qrandint_reaches_inclusive_upper_bound_q_gt_1():
    # 0..10 in steps of 2 is exactly 10 = 5*2, so 10 is a valid grid point and must be
    # reachable. Also verifies test_space.py's own documented -21..12 step-3 example,
    # which the pre-fix rounding happens to satisfy by coincidence for that shape.
    values = _sampled_values(qrandint(0, 10, 2))
    assert 10 in values
    assert all(v % 2 == 0 for v in values)
    assert values <= set(range(0, 11))

    values = _sampled_values(qrandint(-21, 12, 3))
    assert 12 in values
    assert all(v % 3 == 0 for v in values)
    assert values <= set(range(-21, 13))


def test_qlograndint_reaches_inclusive_upper_bound():
    values = _sampled_values(qlograndint(1, 100, q=1))
    assert 100 in values
    assert values <= set(range(1, 101))


def test_randint_stays_exclusive_upper_bound():
    # plain randint is documented "lower is inclusive, upper is exclusive" and must be
    # untouched by the qrandint/qlograndint inclusive-upper fix.
    values = _sampled_values(randint(1, 10))
    assert 10 not in values
    assert values <= set(range(1, 10))
