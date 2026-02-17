---
sidebar_label: greedy
title: default.greedy
---

#### construct\_portfolio

```python
def construct_portfolio(regret_matrix, meta_features, regret_bound)
```

The portfolio construction algorithm.

(Reference)[https://arxiv.org/abs/2202.09927].

**Arguments**:

- `regret_matrix` - A dataframe of regret matrix.
- `meta_features` - None or a dataframe of metafeatures matrix.
  When set to None, the algorithm uses greedy strategy.
  Otherwise, the algorithm uses greedy strategy with feedback
  from the nearest neighbor predictor.
- `regret_bound` - A float of the regret bound.
  

**Returns**:

  A list of configuration names.

