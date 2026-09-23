---
sidebar_label: blendsearch
title: tune.searcher.blendsearch
---

## BlendSearch Objects

```python
class BlendSearch(Searcher)
```

class for BlendSearch algorithm.

#### \_\_init\_\_

```python
def __init__(metric: Optional[str] = None, mode: Optional[str] = None, space: Optional[dict] = None, low_cost_partial_config: Optional[dict] = None, cat_hp_cost: Optional[dict] = None, points_to_evaluate: Optional[List[dict]] = None, evaluated_rewards: Optional[List] = None, time_budget_s: Union[int, float] = None, num_samples: Optional[int] = None, resource_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, reduction_factor: Optional[float] = None, global_search_alg: Optional[Searcher] = None, config_constraints: Optional[
            List[Tuple[Callable[[dict], float], str, float]]
        ] = None, metric_constraints: Optional[List[Tuple[str, str, float]]] = None, seed: Optional[int] = 20, cost_attr: Optional[str] = "auto", experimental: Optional[bool] = False, lexico_objectives: Optional[dict] = None, use_incumbent_result_in_evaluation=False, allow_empty_config=False)
```

Constructor.

**Arguments**:

- `metric` - A string of the metric name to optimize for.
- `mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `space` - A dictionary to specify the search space.
- `low_cost_partial_config` - A dictionary from a subset of
  controlled dimensions to the initial low-cost values.
  E.g., ```{'n_estimators': 4, 'max_leaves': 4}```.
- `cat_hp_cost` - A dictionary from a subset of categorical dimensions
  to the relative cost of each choice.
  E.g., ```{'tree_method': [1, 1, 2]}```.
  I.e., the relative cost of the three choices of 'tree_method'
  is 1, 1 and 2 respectively.
- `points_to_evaluate` - Initial parameter suggestions to be run first.
- `evaluated_rewards` _list_ - If you have previously evaluated the
  parameters passed in as points_to_evaluate you can avoid
  re-running those trials by passing in the reward attributes
  as a list so the optimiser can be told the results without
  needing to re-compute the trial. Must be the same or shorter length than
  points_to_evaluate. When provided, `mode` must be specified.
- `time_budget_s` - int or float | Time budget in seconds.
- `num_samples` - int | The number of configs to try.
- `resource_attr` - A string to specify the resource dimension and the best
  performance is assumed to be at the max_resource.
- `min_resource` - A float of the minimal resource to use for the resource_attr.
- `max_resource` - A float of the maximal resource to use for the resource_attr.
- `reduction_factor` - A float of the reduction factor used for
  incremental pruning.
- `global_search_alg` - A Searcher instance as the global search
  instance. If omitted, Optuna is used. The following algos have
  known issues when used as global_search_alg:
  - HyperOptSearch raises exception sometimes
  - TuneBOHB has its own scheduler
- `config_constraints` - A list of config constraints to be satisfied.
  E.g., ```config_constraints = [(mem_size, '<=', 1024**3)]```.
  `mem_size` is a function which produces a float number for the bytes
  needed for a config.
  It is used to skip configs which do not fit in memory.
- `metric_constraints` - A list of metric constraints to be satisfied.
  E.g., `['precision', '>=', 0.9]`. The sign can be ">=" or "<=".
- `seed` - An integer of the random seed.
- `cost_attr` - Choose from ["auto", None] to specify the attribute to evaluate the cost of different trials.
  Default is "auto", which means that we will automatically chose the cost attribute to use (depending
  on the nature of the resource budget). When cost_attr is set to None, cost differences between different trials will be omitted
  in our search algorithm.
- `lexico_objectives` - dict, default=None | It specifics information needed to perform multi-objective
  optimization with lexicographic preferences. This is only supported in CFO currently.
  When lexico_objectives is not None, the arguments metric, mode will be invalid.
  This dictionary shall contain the  following fields of key-value pairs:
  - "metrics":  a list of optimization objectives with the orders reflecting the priorities/preferences of the
  objectives.
  - "modes" (optional): a list of optimization modes (each mode either "min" or "max") corresponding to the
  objectives in the metric list. If not provided, we use "min" as the default mode for all the objectives.
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

#### results

```python
@property
def results() -> List[Dict]
```

A list of dicts of results for each evaluated configuration.

Each dict has "config" and metric names as keys.
The returned dict includes the initial results provided via `evaluated_reward`.

## BlendSearchTuner Objects

```python
class BlendSearchTuner(BlendSearch,  NNITuner)
```

Tuner class for NNI.

#### receive\_trial\_result

```python
def receive_trial_result(parameter_id, parameters, value, **kwargs)
```

Receive trial's final result.

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

