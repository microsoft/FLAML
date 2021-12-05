---
sidebar_label: tune
title: tune.tune
---

## ExperimentAnalysis Objects

```python
class ExperimentAnalysis(EA)
```

Class for storing the experiment results.

#### report

```python
def report(_metric=None, **kwargs)
```

A function called by the HPO application to report final or intermediate
results.

**Example**:

  
```python
import time
from flaml import tune

def compute_with_config(config):
    current_time = time.time()
    metric2minimize = (round(config['x'])-95000)**2
    time2eval = time.time() - current_time
    tune.report(metric2minimize=metric2minimize, time2eval=time2eval)

analysis = tune.run(
    compute_with_config,
    config={
        'x': tune.lograndint(lower=1, upper=1000000),
        'y': tune.randint(lower=1, upper=1000000)
    },
    metric='metric2minimize', mode='min',
    num_samples=1000000, time_budget_s=60, use_ray=False)

print(analysis.trials[-1].last_result)
```
  

**Arguments**:

- `_metric` - Optional default anonymous metric for ``tune.report(value)``.
  (For compatibility with ray.tune.report)
- `**kwargs` - Any key value pair to be reported.

#### run

```python
def run(training_function, config: Optional[dict] = None, low_cost_partial_config: Optional[dict] = None, cat_hp_cost: Optional[dict] = None, metric: Optional[str] = None, mode: Optional[str] = None, time_budget_s: Union[int, float] = None, points_to_evaluate: Optional[List[dict]] = None, evaluated_rewards: Optional[List] = None, prune_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, reduction_factor: Optional[float] = None, report_intermediate_result: Optional[bool] = False, search_alg=None, verbose: Optional[int] = 2, local_dir: Optional[str] = None, num_samples: Optional[int] = 1, resources_per_trial: Optional[dict] = None, config_constraints: Optional[
        List[Tuple[Callable[[dict], float], str, float]]
    ] = None, metric_constraints: Optional[List[Tuple[str, str, float]]] = None, max_failure: Optional[int] = 100, use_ray: Optional[bool] = False)
```

The trigger for HPO.

**Example**:

  
```python
import time
from flaml import tune

def compute_with_config(config):
    current_time = time.time()
    metric2minimize = (round(config['x'])-95000)**2
    time2eval = time.time() - current_time
    tune.report(metric2minimize=metric2minimize, time2eval=time2eval)

analysis = tune.run(
    compute_with_config,
    config={
        'x': tune.lograndint(lower=1, upper=1000000),
        'y': tune.randint(lower=1, upper=1000000)
    },
    metric='metric2minimize', mode='min',
    num_samples=-1, time_budget_s=60, use_ray=False)

print(analysis.trials[-1].last_result)
```
  

**Arguments**:

- `training_function` - A user-defined training function.
- `config` - A dictionary to specify the search space.
- `low_cost_partial_config` - A dictionary from a subset of
  controlled dimensions to the initial low-cost values.
  e.g., ```{'n_estimators': 4, 'max_leaves': 4}```
  
- `cat_hp_cost` - A dictionary from a subset of categorical dimensions
  to the relative cost of each choice.
  e.g., ```{'tree_method': [1, 1, 2]}```
  i.e., the relative cost of the
  three choices of 'tree_method' is 1, 1 and 2 respectively
- `metric` - A string of the metric name to optimize for.
- `mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `time_budget_s` - int or float | The time budget in seconds.
- `points_to_evaluate` - A list of initial hyperparameter
  configurations to run first.
- `evaluated_rewards` _list_ - If you have previously evaluated the
  parameters passed in as points_to_evaluate you can avoid
  re-running those trials by passing in the reward attributes
  as a list so the optimiser can be told the results without
  needing to re-compute the trial. Must be the same length as
  points_to_evaluate.
  e.g.,
  
```python
points_to_evaluate = [
    {"b": .99, "cost_related": {"a": 3}},
    {"b": .99, "cost_related": {"a": 2}},
]
evaluated_rewards=[3.0, 1.0]
```
  
  means that you know the reward for the two configs in
  points_to_evaluate are 3.0 and 1.0 respectively and want to
  inform run()
  
- `prune_attr` - A string of the attribute used for pruning.
  Not necessarily in space.
  When prune_attr is in space, it is a hyperparameter, e.g.,
  'n_iters', and the best value is unknown.
  When prune_attr is not in space, it is a resource dimension,
  e.g., 'sample_size', and the peak performance is assumed
  to be at the max_resource.
- `min_resource` - A float of the minimal resource to use for the
  prune_attr; only valid if prune_attr is not in space.
- `max_resource` - A float of the maximal resource to use for the
  prune_attr; only valid if prune_attr is not in space.
- `reduction_factor` - A float of the reduction factor used for incremental
  pruning.
- `report_intermediate_result` - A boolean of whether intermediate results
  are reported. If so, early stopping and pruning can be used.
- `search_alg` - An instance of BlendSearch as the search algorithm
  to be used. The same instance can be used for iterative tuning.
  e.g.,
  
```python
from flaml import BlendSearch
algo = BlendSearch(metric='val_loss', mode='min',
        space=search_space,
        low_cost_partial_config=low_cost_partial_config)
for i in range(10):
    analysis = tune.run(compute_with_config,
        search_alg=algo, use_ray=False)
    print(analysis.trials[-1].last_result)
```
  
- `verbose` - 0, 1, 2, or 3. Verbosity mode for ray if ray backend is used.
  0 = silent, 1 = only status updates, 2 = status and brief trial
  results, 3 = status and detailed trial results. Defaults to 2.
- `local_dir` - A string of the local dir to save ray logs if ray backend is
  used; or a local dir to save the tuning log.
- `num_samples` - An integer of the number of configs to try. Defaults to 1.
- `resources_per_trial` - A dictionary of the hardware resources to allocate
  per trial, e.g., `{'cpu': 1}`. Only valid when using ray backend.
- `config_constraints` - A list of config constraints to be satisfied.
  e.g., ```config_constraints = [(mem_size, '<=', 1024**3)]```
  
  mem_size is a function which produces a float number for the bytes
  needed for a config.
  It is used to skip configs which do not fit in memory.
- `metric_constraints` - A list of metric constraints to be satisfied.
  e.g., `['precision', '>=', 0.9]`.
- `max_failure` - int | the maximal consecutive number of failures to sample
  a trial before the tuning is terminated.
- `use_ray` - A boolean of whether to use ray as the backend.

