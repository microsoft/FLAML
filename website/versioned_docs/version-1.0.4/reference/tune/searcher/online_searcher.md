---
sidebar_label: online_searcher
title: tune.searcher.online_searcher
---

## BaseSearcher Objects

```python
class BaseSearcher()
```

Abstract class for an online searcher.

## ChampionFrontierSearcher Objects

```python
class ChampionFrontierSearcher(BaseSearcher)
```

The ChampionFrontierSearcher class.

NOTE about the correspondence about this code and the research paper:
[ChaCha for Online AutoML](https://arxiv.org/pdf/2106.04815.pdf).
This class serves the role of ConfigOralce as described in the paper.

#### \_\_init\_\_

```python
def __init__(init_config: Dict, space: Optional[Dict] = None, metric: Optional[str] = None, mode: Optional[str] = None, random_seed: Optional[int] = 2345, online_trial_args: Optional[Dict] = {}, nonpoly_searcher_name: Optional[str] = "CFO")
```

Constructor.

**Arguments**:

- `init_config` - A dictionary of initial configuration.
- `space` - A dictionary to specify the search space.
- `metric` - A string of the metric name to optimize for.
- `mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `random_seed` - An integer of the random seed.
- `online_trial_args` - A dictionary to specify the online trial
  arguments for experimental purpose.
- `nonpoly_searcher_name` - A string to specify the search algorithm
  for nonpoly hyperparameters.

#### set\_search\_properties

```python
def set_search_properties(metric: Optional[str] = None, mode: Optional[str] = None, config: Optional[Dict] = {}, setting: Optional[Dict] = {}, init_call: Optional[bool] = False)
```

Construct search space with the given config, and setup the search.

#### next\_trial

```python
def next_trial()
```

Return a trial from the _challenger_list.

