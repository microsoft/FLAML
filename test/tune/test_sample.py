from collections import Counter

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


def test_qrandint_returns_plain_int_scalar_and_batched():
    # the q=1 default path used to route through float division, turning an integer
    # domain's samples into np.float64 and losing precision above 2**53. A batched draw
    # historically came back as an integer ndarray (not a Python list of ints), so check
    # the container and dtype, not just each element's Python-level type.
    domain = qrandint(0, 6, q=1)
    scalar = domain.sample(spec=None, random_state=np.random.RandomState(0))
    assert type(scalar) is int

    batched = domain.sample(spec=None, size=5, random_state=np.random.RandomState(0))
    assert isinstance(batched, np.ndarray)
    assert np.issubdtype(batched.dtype, np.integer)

    # q > 1 historically returned a plain Python list, not an ndarray; only the q == 1
    # default path returns the integer ndarray checked above.
    q_domain = qrandint(0, 20, 5)
    batched_q = q_domain.sample(spec=None, size=5, random_state=np.random.RandomState(0))
    assert isinstance(batched_q, list)
    assert all(isinstance(v, (int, np.integer)) for v in batched_q)
    assert all(v % 5 == 0 for v in batched_q)


def test_qrandint_large_bound_keeps_integer_precision():
    # values above 2**53 cannot round-trip through float64; a grid point must come back
    # exactly, not off by a rounding error introduced by the sampler.
    lower, upper, q = 0, 2**60, 2**50
    rs = np.random.RandomState(0)
    for _ in range(20):
        v = qrandint(lower, upper, q).sample(spec=None, random_state=rs)
        assert lower <= v <= upper
        assert v % q == 0


def test_qrandint_large_nonzero_lower_bound_batched():
    # a nonzero lower bound above 2**53 used to reach `domain.lower / self.q` as a float
    # division, which can drop precision and land a scalar sample below the requested
    # lower bound, or overflow a narrower int dtype (e.g. int32 on Windows) once the batch
    # path multiplies the sampled index back up by q.
    lower, upper, q = 2**55, 2**55 + 2**50 * 8, 2**50
    rs = np.random.RandomState(0)
    batched = qrandint(lower, upper, q).sample(spec=None, size=50, random_state=rs)
    # q > 1 (as here) returns a plain Python list, not an ndarray; see
    # test_qrandint_returns_plain_int_scalar_and_batched for the q == 1 ndarray contract.
    assert isinstance(batched, list)
    assert all(isinstance(v, (int, np.integer)) for v in batched)
    for v in batched:
        v = int(v)
        assert lower <= v <= upper, v
        assert (v - lower) % q == 0, v


def test_qrandint_grid_bins_are_uniform_including_top_bin():
    # the pre-fix implementation extended the raw draw's range by a full q and clamped
    # the overshoot into the top bin, giving it roughly 1.4x the width of an interior
    # bin. With 5 equal-width bins (0, 5, 10, 15, 20) over 200000 draws, every bin's
    # count should sit close to the uniform expectation of 40000.
    rs = np.random.RandomState(0)
    n = 200000
    counts = Counter(qrandint(0, 20, 5).sample(spec=None, random_state=rs) for _ in range(n))
    assert set(counts) == {0, 5, 10, 15, 20}
    expected = n / 5
    for bin_value, count in counts.items():
        assert abs(count - expected) / expected < 0.05, (bin_value, count, expected)


def test_qlograndint_large_bound_keeps_integer_precision():
    lower, upper, q = 1, 2**60, 2**50
    rs = np.random.RandomState(0)
    for _ in range(20):
        v = qlograndint(lower, upper, q).sample(spec=None, random_state=rs)
        assert lower <= v <= upper
        assert v % q == 0


def test_qlograndint_stays_log_uniform_when_lower_much_greater_than_q():
    # qlograndint(1000, 1010, 1) must sample log-uniformly over the actual grid
    # {1000, 1001, ..., 1010}, not over an index rebased to start at 1: LogUniform is
    # scale-invariant under multiplication (X ~ LogUniform(a, b) => qX ~ LogUniform(qa,
    # qb)) but not under an arbitrary additive shift. The pre-fix rebase-to-1
    # implementation instead sampled log-uniformly over the unrelated index range
    # {1..11}, which for this domain has an expected value around 1003.4 instead of the
    # analytic LogUniform(1000, 1010) mean below; the gap (~1.6) is over 200 standard
    # errors of the mean at this sample size, so this is not a flaky comparison.
    lower, upper, q = 1000, 1010, 1
    rs = np.random.RandomState(0)
    n = 200000
    samples = [qlograndint(lower, upper, q).sample(spec=None, random_state=rs) for _ in range(n)]
    empirical_mean = sum(samples) / n
    analytic_mean = (upper - lower) / np.log(upper / lower)
    assert abs(empirical_mean - analytic_mean) < 0.1, (empirical_mean, analytic_mean)
