---
sidebar_label: sample
title: tune.sample
---

## Domain Objects

```python
class Domain()
```

Base class to specify a type and valid range to sample parameters from.
This base class is implemented by parameter spaces, like float ranges
(``Float``), integer ranges (``Integer``), or categorical variables
(``Categorical``). The ``Domain`` object contains information about
valid values (e.g. minimum and maximum values), and exposes methods that
allow specification of specific samplers (e.g. ``uniform()`` or
``loguniform()``).

#### cast

```python
def cast(value)
```

Cast value to domain type

#### is\_valid

```python
def is_valid(value: Any)
```

Returns True if `value` is a valid value in this domain.

## Grid Objects

```python
class Grid(Sampler)
```

Dummy sampler used for grid search

#### uniform

```python
def uniform(lower: float, upper: float)
```

Sample a float value uniformly between ``lower`` and ``upper``.
Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
``np.random.uniform(1, 10))``

#### quniform

```python
def quniform(lower: float, upper: float, q: float)
```

Sample a quantized float value uniformly between ``lower`` and ``upper``.
Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
``np.random.uniform(1, 10))``
The value will be quantized, i.e. rounded to an integer increment of ``q``.
Quantization makes the upper bound inclusive.

#### loguniform

```python
def loguniform(lower: float, upper: float, base: float = 10)
```

Sugar for sampling in different orders of magnitude.

**Arguments**:

- `lower` _float_ - Lower boundary of the output interval (e.g. 1e-4)
- `upper` _float_ - Upper boundary of the output interval (e.g. 1e-2)
- `base` _int_ - Base of the log. Defaults to 10.

#### qloguniform

```python
def qloguniform(lower: float, upper: float, q: float, base: float = 10)
```

Sugar for sampling in different orders of magnitude.
The value will be quantized, i.e. rounded to an integer increment of ``q``.
Quantization makes the upper bound inclusive.

**Arguments**:

- `lower` _float_ - Lower boundary of the output interval (e.g. 1e-4)
- `upper` _float_ - Upper boundary of the output interval (e.g. 1e-2)
- `q` _float_ - Quantization number. The result will be rounded to an
  integer increment of this value.
- `base` _int_ - Base of the log. Defaults to 10.

#### choice

```python
def choice(categories: Sequence)
```

Sample a categorical value.
Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
``np.random.choice([1, 2])``

#### randint

```python
def randint(lower: int, upper: int)
```

Sample an integer value uniformly between ``lower`` and ``upper``.
``lower`` is inclusive, ``upper`` is exclusive.
Sampling from ``tune.randint(10)`` is equivalent to sampling from
``np.random.randint(10)``

#### lograndint

```python
def lograndint(lower: int, upper: int, base: float = 10)
```

Sample an integer value log-uniformly between ``lower`` and ``upper``,
with ``base`` being the base of logarithm.
``lower`` is inclusive, ``upper`` is exclusive.

#### qrandint

```python
def qrandint(lower: int, upper: int, q: int = 1)
```

Sample an integer value uniformly between ``lower`` and ``upper``.

``lower`` is inclusive, ``upper`` is also inclusive (!).

The value will be quantized, i.e. rounded to an integer increment of ``q``.
Quantization makes the upper bound inclusive.

#### qlograndint

```python
def qlograndint(lower: int, upper: int, q: int, base: float = 10)
```

Sample an integer value log-uniformly between ``lower`` and ``upper``,
with ``base`` being the base of logarithm.
``lower`` is inclusive, ``upper`` is also inclusive (!).
The value will be quantized, i.e. rounded to an integer increment of ``q``.
Quantization makes the upper bound inclusive.

#### randn

```python
def randn(mean: float = 0.0, sd: float = 1.0)
```

Sample a float value normally with ``mean`` and ``sd``.

**Arguments**:

- `mean` _float_ - Mean of the normal distribution. Defaults to 0.
- `sd` _float_ - SD of the normal distribution. Defaults to 1.

#### qrandn

```python
def qrandn(mean: float, sd: float, q: float)
```

Sample a float value normally with ``mean`` and ``sd``.

The value will be quantized, i.e. rounded to an integer increment of ``q``.

**Arguments**:

- `mean` - Mean of the normal distribution.
- `sd` - SD of the normal distribution.
- `q` - Quantization number. The result will be rounded to an
  integer increment of this value.

