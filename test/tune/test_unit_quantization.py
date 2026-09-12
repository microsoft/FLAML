import numpy as np
import pytest

from flaml.tune.sample import qloguniform, qrandn, quniform


@pytest.mark.parametrize(
    "factory", [lambda q: quniform(2, 10, q), lambda q: qloguniform(2, 10, q), lambda q: qrandn(0, 2, q)]
)
@pytest.mark.parametrize("size", [1, 100])
@pytest.mark.parametrize("q", [1, 2, 0.5])
def test_unit_quantization_rounds_float_samples(factory, size, q):
    values = np.asarray(factory(q).sample(size=size, random_state=42))
    np.testing.assert_array_equal(values / q, np.round(values / q))
    np.testing.assert_array_equal(values, factory(q).sample(size=size, random_state=42))
