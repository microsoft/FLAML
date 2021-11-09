---
sidebar_label: flow2
title: searcher.flow2
---

## FLOW2 Objects

```python
class FLOW2(Searcher)
```

Local search algorithm FLOW2, with adaptive step size.

#### \_\_init\_\_

```python
def __init__(init_config: dict, metric: Optional[str] = None, mode: Optional[str] = None, space: Optional[dict] = None, prune_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, resource_multiple_factor: Optional[float] = 4, cost_attr: Optional[str] = "time_total_s", seed: Optional[int] = 20)
```

Constructor.

**Arguments**:

- `init_config` - a dictionary of a partial or full initial config,
  e.g., from a subset of controlled dimensions
  to the initial low-cost values.
  E.g., {&#x27;epochs&#x27;: 1}.
- `metric` - A string of the metric name to optimize for.
- `mode` - A string in [&#x27;min&#x27;, &#x27;max&#x27;] to specify the objective as
  minimization or maximization.
- `cat_hp_cost` - A dictionary from a subset of categorical dimensions
  to the relative cost of each choice.
  e.g.,
  
  .. code-block:: python
  
- `{&#x27;tree_method&#x27;` - [1, 1, 2]}
  
  i.e., the relative cost of the
  three choices of &#x27;tree_method&#x27; is 1, 1 and 2 respectively.
- `space` - A dictionary to specify the search space.
- `prune_attr` - A string of the attribute used for pruning.
  Not necessarily in space.
  When prune_attr is in space, it is a hyperparameter, e.g.,
  &#x27;n_iters&#x27;, and the best value is unknown.
  When prune_attr is not in space, it is a resource dimension,
  e.g., &#x27;sample_size&#x27;, and the peak performance is assumed
  to be at the max_resource.
- `min_resource` - A float of the minimal resource to use for the
  prune_attr; only valid if prune_attr is not in space.
- `max_resource` - A float of the maximal resource to use for the
  prune_attr; only valid if prune_attr is not in space.
- `resource_multiple_factor` - A float of the multiplicative factor
  used for increasing resource.
- `cost_attr` - A string of the attribute used for cost.
- `seed` - An integer of the random seed.

#### complete\_config

```python
def complete_config(partial_config: Dict, lower: Optional[Dict] = None, upper: Optional[Dict] = None) -> Tuple[Dict, Dict]
```

Generate a complete config from the partial config input.

Add minimal resource to config if available.

#### normalize

```python
def normalize(config, recursive=False) -> Dict
```

normalize each dimension in config to [0,1].

#### denormalize

```python
def denormalize(config)
```

denormalize each dimension in config from [0,1].

#### on\_trial\_complete

```python
def on_trial_complete(trial_id: str, result: Optional[Dict] = None, error: bool = False)
```

Compare with incumbent.
If better, move, reset num_complete and num_proposed.
If not better and num_complete &gt;= 2*dim, num_allowed += 2.

#### on\_trial\_result

```python
def on_trial_result(trial_id: str, result: Dict)
```

Early update of incumbent.

#### suggest

```python
def suggest(trial_id: str) -> Optional[Dict]
```

Suggest a new config, one of the following cases:
1. same incumbent, increase resource.
2. same resource, move from the incumbent to a random direction.
3. same resource, move from the incumbent to the opposite direction.

#### can\_suggest

```python
@property
def can_suggest() -> bool
```

Can&#x27;t suggest if 2*dim configs have been proposed for the incumbent
while fewer are completed.

#### config\_signature

```python
def config_signature(config, space: Dict = None) -> tuple
```

Return the signature tuple of a config.

#### converged

```python
@property
def converged() -> bool
```

Whether the local search has converged.

#### reach

```python
def reach(other: Searcher) -> bool
```

whether the incumbent can reach the incumbent of other.

