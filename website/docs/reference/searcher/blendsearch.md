---
sidebar_label: blendsearch
title: searcher.blendsearch
---

## BlendSearch Objects

```python
class BlendSearch(Searcher)
```

class for BlendSearch algorithm.

#### \_\_init\_\_

```python
def __init__(metric: Optional[str] = None, mode: Optional[str] = None, space: Optional[dict] = None, low_cost_partial_config: Optional[dict] = None, cat_hp_cost: Optional[dict] = None, points_to_evaluate: Optional[List[dict]] = None, evaluated_rewards: Optional[List] = None, time_budget_s: Union[int, float] = None, num_samples: Optional[int] = None, prune_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, reduction_factor: Optional[float] = None, global_search_alg: Optional[Searcher] = None, config_constraints: Optional[
            List[Tuple[Callable[[dict], float], str, float]]
        ] = None, metric_constraints: Optional[List[Tuple[str, str, float]]] = None, seed: Optional[int] = 20, experimental: Optional[bool] = False)
```

Constructor.

**Arguments**:

- `metric` - A string of the metric name to optimize for.
- `mode` - A string in [&#x27;min&#x27;, &#x27;max&#x27;] to specify the objective as
  minimization or maximization.
- `space` - A dictionary to specify the search space.
- `low_cost_partial_config` - A dictionary from a subset of
  controlled dimensions to the initial low-cost values.
  e.g.,
  
  .. code-block:: python
  
- `{&#x27;n_estimators&#x27;` - 4, &#x27;max_leaves&#x27;: 4}
  
- `cat_hp_cost` - A dictionary from a subset of categorical dimensions
  to the relative cost of each choice.
  e.g.,
  
  .. code-block:: python
  
- `{&#x27;tree_method&#x27;` - [1, 1, 2]}
  
  i.e., the relative cost of the
  three choices of &#x27;tree_method&#x27; is 1, 1 and 2 respectively.
- `points_to_evaluate` - Initial parameter suggestions to be run first.
- `evaluated_rewards` _list_ - If you have previously evaluated the
  parameters passed in as points_to_evaluate you can avoid
  re-running those trials by passing in the reward attributes
  as a list so the optimiser can be told the results without
  needing to re-compute the trial. Must be the same length as
  points_to_evaluate.
- `time_budget_s` - int or float | Time budget in seconds.
- `num_samples` - int | The number of configs to try.
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
- `reduction_factor` - A float of the reduction factor used for
  incremental pruning.
- `global_search_alg` - A Searcher instance as the global search
  instance. If omitted, Optuna is used. The following algos have
  known issues when used as global_search_alg:
  - HyperOptSearch raises exception sometimes
  - TuneBOHB has its own scheduler
- `config_constraints` - A list of config constraints to be satisfied.
  e.g.,
  
  .. code-block: python
  
  config_constraints = [(mem_size, &#x27;&lt;=&#x27;, 1024**3)]
  
  mem_size is a function which produces a float number for the bytes
  needed for a config.
  It is used to skip configs which do not fit in memory.
- `metric_constraints` - A list of metric constraints to be satisfied.
  e.g., `[&#x27;precision&#x27;, &#x27;&gt;=&#x27;, 0.9]`
- `seed` - An integer of the random seed.
- `experimental` - A bool of whether to use experimental features.

#### save

```python
def save(checkpoint_path: str)
```

save states to a checkpoint path.

#### restore

```python
def restore(checkpoint_path: str)
```

restore states from checkpoint.

#### on\_trial\_complete

```python
def on_trial_complete(trial_id: str, result: Optional[Dict] = None, error: bool = False)
```

search thread updater and cleaner.

#### on\_trial\_result

```python
def on_trial_result(trial_id: str, result: Dict)
```

receive intermediate result.

#### suggest

```python
def suggest(trial_id: str) -> Optional[Dict]
```

choose thread, suggest a valid config.

## BlendSearchTuner Objects

```python
class BlendSearchTuner(BlendSearch,  NNITuner)
```

Tuner class for NNI.

#### receive\_trial\_result

```python
def receive_trial_result(parameter_id, parameters, value, **kwargs)
```

Receive trial&#x27;s final result.

**Arguments**:

- `parameter_id` - int.
- `parameters` - object created by `generate_parameters()`.
- `value` - final metrics of the trial, including default metric.

#### generate\_parameters

```python
def generate_parameters(parameter_id, **kwargs) -> Dict
```

Returns a set of trial (hyper-)parameters, as a serializable object.

**Arguments**:

- `parameter_id` - int.

#### update\_search\_space

```python
def update_search_space(search_space)
```

Required by NNI.

Tuners are advised to support updating search space at run-time.
If a tuner can only set search space once before generating first hyper-parameters,
it should explicitly document this behaviour.

**Arguments**:

- `search_space` - JSON object created by experiment owner.

## CFO Objects

```python
class CFO(BlendSearchTuner)
```

class for CFO algorithm.

## RandomSearch Objects

```python
class RandomSearch(CFO)
```

Class for random search.

