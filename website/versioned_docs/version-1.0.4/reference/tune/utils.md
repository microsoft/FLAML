---
sidebar_label: utils
title: tune.utils
---

#### choice

```python
def choice(categories: Sequence, order=None)
```

Sample a categorical value.
Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
``np.random.choice([1, 2])``

**Arguments**:

- `categories` _Sequence_ - Sequence of categories to sample from.
- `order` _bool_ - Whether the categories have an order. If None, will be decided autoamtically:
  Numerical categories have an order, while string categories do not.

