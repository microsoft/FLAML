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
  

**Raises**:

  StopIteration (when not using ray, i.e., _use_ray=False):
  A StopIteration exception is raised if the trial has been signaled to stop.
  SystemExit (when using ray):
  A SystemExit exception is raised if the trial has been signaled to stop by ray.

#### run

```python
def run(evaluation_function, config: Optional[dict] = None, low_cost_partial_config: Optional[dict] = None, cat_hp_cost: Optional[dict] = None, metric: Optional[str] = None, mode: Optional[str] = None, time_budget_s: Union[int, float] = None, points_to_evaluate: Optional[List[dict]] = None, evaluated_rewards: Optional[List] = None, resource_attr: Optional[str] = None, min_resource: Optional[float] = None, max_resource: Optional[float] = None, reduction_factor: Optional[float] = None, scheduler=None, search_alg=None, verbose: Optional[int] = 2, local_dir: Optional[str] = None, num_samples: Optional[int] = 1, resources_per_trial: Optional[dict] = None, config_constraints: Optional[
        List[Tuple[Callable[[dict], float], str, float]]
    ] = None, metric_constraints: Optional[List[Tuple[str, str, float]]] = None, max_failure: Optional[int] = 100, use_ray: Optional[bool] = False, use_incumbent_result_in_evaluation: Optional[bool] = None, log_file_name: Optional[str] = None, lexico_objectives: Optional[dict] = None, **ray_args, ,)
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
    # if the evaluation fails unexpectedly and the exception is caught,
    # and it doesn't inform the goodness of the config,
    # return {}
    # if the failure indicates a config is bad,
    # report a bad metric value like np.inf or -np.inf
    # depending on metric mode being min or max

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

- `evaluation_function` - A user-defined evaluation function.
  It takes a configuration as input, outputs a evaluation
  result (can be a numerical value or a dictionary of string
  and numerical value pairs) for the input configuration.
  For machine learning tasks, it usually involves training and
  scoring a machine learning model, e.g., through validation loss.
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
  needing to re-compute the trial. Must be the same or shorter length than
  points_to_evaluate.
  e.g.,
  
```python
points_to_evaluate = [
    {"b": .99, "cost_related": {"a": 3}},
    {"b": .99, "cost_related": {"a": 2}},
]
evaluated_rewards = [3.0]
```
  
  means that you know the reward for the first config in
  points_to_evaluate is 3.0 and want to inform run().
  
- `resource_attr` - A string to specify the resource dimension used by
  the scheduler via "scheduler".
- `min_resource` - A float of the minimal resource to use for the resource_attr.
- `max_resource` - A float of the maximal resource to use for the resource_attr.
- `reduction_factor` - A float of the reduction factor used for incremental
  pruning.
- `scheduler` - A scheduler for executing the experiment. Can be None, 'flaml',
  'asha' (or  'async_hyperband', 'asynchyperband') or a custom instance of the TrialScheduler class. Default is None:
  in this case when resource_attr is provided, the 'flaml' scheduler will be
  used, otherwise no scheduler will be used. When set 'flaml', an
  authentic scheduler implemented in FLAML will be used. It does not
  require users to report intermediate results in evaluation_function.
  Find more details about this scheduler in this paper
  https://arxiv.org/pdf/1911.04706.pdf).
  When set 'asha', the input for arguments "resource_attr",
  "min_resource", "max_resource" and "reduction_factor" will be passed
  to ASHA's "time_attr",  "max_t", "grace_period" and "reduction_factor"
  respectively. You can also provide a self-defined scheduler instance
  of the TrialScheduler class. When 'asha' or self-defined scheduler is
  used, you usually need to report intermediate results in the evaluation
  function via 'tune.report()'.
  If you would like to do some cleanup opearation when the trial is stopped
  by the scheduler, you can catch the `StopIteration` (when not using ray)
  or `SystemExit` (when using ray) exception explicitly,
  as shown in the following example.
  Please find more examples using different types of schedulers
  and how to set up the corresponding evaluation functions in
  test/tune/test_scheduler.py, and test/tune/example_scheduler.py.
```python
def easy_objective(config):
    width, height = config["width"], config["height"]
    for step in range(config["steps"]):
        intermediate_score = evaluation_fn(step, width, height)
        try:
            tune.report(iterations=step, mean_loss=intermediate_score)
        except (StopIteration, SystemExit):
            # do cleanup operation here
            return
```
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
  per trial, e.g., `{'cpu': 1}`. It is only valid when using ray backend
  (by setting 'use_ray = True'). It shall be used when you need to do
  [parallel tuning](../../Use-Cases/Tune-User-Defined-Function#parallel-tuning).
- `config_constraints` - A list of config constraints to be satisfied.
  e.g., ```config_constraints = [(mem_size, '<=', 1024**3)]```
  
  mem_size is a function which produces a float number for the bytes
  needed for a config.
  It is used to skip configs which do not fit in memory.
- `metric_constraints` - A list of metric constraints to be satisfied.
  e.g., `['precision', '>=', 0.9]`. The sign can be ">=" or "<=".
- `max_failure` - int | the maximal consecutive number of failures to sample
  a trial before the tuning is terminated.
- `use_ray` - A boolean of whether to use ray as the backend.
- `log_file_name` - A string of the log file name. Default to None.
  When set to None:
  if local_dir is not given, no log file is created;
  if local_dir is given, the log file name will be autogenerated under local_dir.
  Only valid when verbose > 0 or use_ray is True.
- `lexico_objectives` - dict, default=None | It specifics information needed to perform multi-objective
  optimization with lexicographic preferences. When lexico_objectives is not None, the arguments metric,
  mode, will be invalid, and flaml's tune uses CFO
  as the `search_alg`, which makes the input (if provided) `search_alg' invalid.
  This dictionary shall contain the following fields of key-value pairs:
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
- `**ray_args` - keyword arguments to pass to ray.tune.run().
  Only valid when use_ray=True.

