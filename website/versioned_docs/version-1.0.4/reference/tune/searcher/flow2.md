---
sidebar_label: flow2
title: tune.searcher.flow2
---

## FLOW2 Objects

```python
class FLOW2(Searcher)
```

Local search algorithm FLOW2, with adaptive step size.

#### \_\_init\_\_

```python
def __init__(init_config: dict, metric: Optional[str] = None, mode: Optional[str] = None, space: Optional[dict] = None, resource_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, resource_multiple_factor: Optional[float] = None, cost_attr: Optional[str] = "time_total_s", seed: Optional[int] = 20, lexico_objectives=None)
```

Constructor.

**Arguments**:

- `init_config` - a dictionary of a partial or full initial config,
  e.g., from a subset of controlled dimensions
  to the initial low-cost values.
  E.g., {'epochs': 1}.
- `metric` - A string of the metric name to optimize for.
- `mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `space` - A dictionary to specify the search space.
- `resource_attr` - A string to specify the resource dimension and the best
  performance is assumed to be at the max_resource.
- `min_resource` - A float of the minimal resource to use for the resource_attr.
- `max_resource` - A float of the maximal resource to use for the resource_attr.
- `resource_multiple_factor` - A float of the multiplicative factor
  used for increasing resource.
- `cost_attr` - A string of the attribute used for cost.
- `seed` - An integer of the random seed.
- `lexico_objectives` - dict, default=None | It specifics information needed to perform multi-objective
  optimization with lexicographic preferences. When lexico_objectives is not None, the arguments metric,
  mode will be invalid. This dictionary shall contain the following fields of key-value pairs:
  - "metrics":  a list of optimization objectives with the orders reflecting the priorities/preferences of the
  objectives.
  - "modes" (optional): a list of optimization modes (each mode either "min" or "max") corresponding to the
  objectives in the metric list. If not provided, we use "min" as the default mode for all the objectives
  - "targets" (optional): a dictionary to specify the optimization targets on the objectives. The keys are the
  metric names (provided in "metric"), and the values are the numerical target values.
  - "tolerances"(optional): a dictionary to specify the optimality tolerances on objectives. The keys are the
  metric names (provided in "metrics"), and the values are the numerical tolerances values.
  E.g.,
  ```python
  lexico_objectives = {
- `"metrics"` - ["error_rate", "pred_time"],
- `"modes"` - ["min", "min"],
- `"tolerances"` - {"error_rate": 0.01, "pred_time": 0.0},
- `"targets"` - {"error_rate": 0.0},
  }
  ```

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
If not better and num_complete >= 2*dim, num_allowed += 2.

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

Can't suggest if 2*dim configs have been proposed for the incumbent
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

