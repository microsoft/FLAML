---
sidebar_label: suggestion
title: tune.searcher.suggestion
---

## Searcher Objects

```python
class Searcher()
```

Abstract class for wrapping suggesting algorithms.
Custom algorithms can extend this class easily by overriding the
`suggest` method provide generated parameters for the trials.
Any subclass that implements ``__init__`` must also call the
constructor of this class: ``super(Subclass, self).__init__(...)``.
To track suggestions and their corresponding evaluations, the method
`suggest` will be passed a trial_id, which will be used in
subsequent notifications.
Not all implementations support multi objectives.

**Arguments**:

- `metric` _str or list_ - The training result objective value attribute. If
  list then list of training result objective value attributes
- `mode` _str or list_ - If string One of {min, max}. If list then
  list of max and min, determines whether objective is minimizing
  or maximizing the metric attribute. Must match type of metric.
  
```python
class ExampleSearch(Searcher):
    def __init__(self, metric="mean_loss", mode="min", **kwargs):
        super(ExampleSearch, self).__init__(
            metric=metric, mode=mode, **kwargs)
        self.optimizer = Optimizer()
        self.configurations = {}
    def suggest(self, trial_id):
        configuration = self.optimizer.query()
        self.configurations[trial_id] = configuration
    def on_trial_complete(self, trial_id, result, **kwargs):
        configuration = self.configurations[trial_id]
        if result and self.metric in result:
            self.optimizer.update(configuration, result[self.metric])
tune.run(trainable_function, search_alg=ExampleSearch())
```

#### set\_search\_properties

```python
def set_search_properties(metric: Optional[str], mode: Optional[str], config: Dict) -> bool
```

Pass search properties to searcher.
This method acts as an alternative to instantiating search algorithms
with their own specific search spaces. Instead they can accept a
Tune config through this method. A searcher should return ``True``
if setting the config was successful, or ``False`` if it was
unsuccessful, e.g. when the search space has already been set.

**Arguments**:

- `metric` _str_ - Metric to optimize
- `mode` _str_ - One of ["min", "max"]. Direction to optimize.
- `config` _dict_ - Tune config dict.

#### on\_trial\_result

```python
def on_trial_result(trial_id: str, result: Dict)
```

Optional notification for result during training.
Note that by default, the result dict may include NaNs or
may not include the optimization metric. It is up to the
subclass implementation to preprocess the result to
avoid breaking the optimization process.

**Arguments**:

- `trial_id` _str_ - A unique string ID for the trial.
- `result` _dict_ - Dictionary of metrics for current training progress.
  Note that the result dict may include NaNs or
  may not include the optimization metric. It is up to the
  subclass implementation to preprocess the result to
  avoid breaking the optimization process.

#### metric

```python
@property
def metric() -> str
```

The training result objective value attribute.

#### mode

```python
@property
def mode() -> str
```

Specifies if minimizing or maximizing the metric.

## ConcurrencyLimiter Objects

```python
class ConcurrencyLimiter(Searcher)
```

A wrapper algorithm for limiting the number of concurrent trials.

**Arguments**:

- `searcher` _Searcher_ - Searcher object that the
  ConcurrencyLimiter will manage.
- `max_concurrent` _int_ - Maximum concurrent samples from the underlying
  searcher.
- `batch` _bool_ - Whether to wait for all concurrent samples
  to finish before updating the underlying searcher.

**Example**:

```python
from ray.tune.suggest import ConcurrencyLimiter  # ray version < 2
search_alg = HyperOptSearch(metric="accuracy")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
tune.run(trainable, search_alg=search_alg)
```

#### validate\_warmstart

```python
def validate_warmstart(parameter_names: List[str], points_to_evaluate: List[Union[List, Dict]], evaluated_rewards: List, validate_point_name_lengths: bool = True)
```

Generic validation of a Searcher's warm start functionality.
Raises exceptions in case of type and length mismatches between
parameters.
If ``validate_point_name_lengths`` is False, the equality of lengths
between ``points_to_evaluate`` and ``parameter_names`` will not be
validated.

## OptunaSearch Objects

```python
class OptunaSearch(Searcher)
```

A wrapper around Optuna to provide trial suggestions.
[Optuna](https://optuna.org/)
is a hyperparameter optimization library.
In contrast to other libraries, it employs define-by-run style
hyperparameter definitions.
This Searcher is a thin wrapper around Optuna's search algorithms.
You can pass any Optuna sampler, which will be used to generate
hyperparameter suggestions.

**Arguments**:

- `space` _dict|Callable_ - Hyperparameter search space definition for
  Optuna's sampler. This can be either a class `dict` with
  parameter names as keys and ``optuna.distributions`` as values,
  or a Callable - in which case, it should be a define-by-run
  function using ``optuna.trial`` to obtain the hyperparameter
  values. The function should return either a class `dict` of
  constant values with names as keys, or None.
  For more information, see
  [tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html).
  Warning - No actual computation should take place in the define-by-run
  function. Instead, put the training logic inside the function
  or class trainable passed to tune.run.
- `metric` _str_ - The training result objective value attribute. If None
  but a mode was passed, the anonymous metric `_metric` will be used
  per default.
- `mode` _str_ - One of {min, max}. Determines whether objective is
  minimizing or maximizing the metric attribute.
- `points_to_evaluate` _list_ - Initial parameter suggestions to be run
  first. This is for when you already have some good parameters
  you want to run first to help the algorithm make better suggestions
  for future parameters. Needs to be a list of dicts containing the
  configurations.
- `sampler` _optuna.samplers.BaseSampler_ - Optuna sampler used to
  draw hyperparameter configurations. Defaults to ``TPESampler``.
- `seed` _int_ - Seed to initialize sampler with. This parameter is only
  used when ``sampler=None``. In all other cases, the sampler
  you pass should be initialized with the seed already.
- `evaluated_rewards` _list_ - If you have previously evaluated the
  parameters passed in as points_to_evaluate you can avoid
  re-running those trials by passing in the reward attributes
  as a list so the optimiser can be told the results without
  needing to re-compute the trial. Must be the same length as
  points_to_evaluate.
  
  Tune automatically converts search spaces to Optuna's format:
  
````python
from ray.tune.suggest.optuna import OptunaSearch  # ray version < 2
config = { "a": tune.uniform(6, 8),
           "b": tune.loguniform(1e-4, 1e-2)}
optuna_search = OptunaSearch(metric="loss", mode="min")
tune.run(trainable, config=config, search_alg=optuna_search)
````
  
  If you would like to pass the search space manually, the code would
  look like this:
  
```python
from ray.tune.suggest.optuna import OptunaSearch  # ray version < 2
import optuna
config = { "a": optuna.distributions.UniformDistribution(6, 8),
           "b": optuna.distributions.LogUniformDistribution(1e-4, 1e-2)}
optuna_search = OptunaSearch(space,metric="loss",mode="min")
tune.run(trainable, search_alg=optuna_search)
# Equivalent Optuna define-by-run function approach:
def define_search_space(trial: optuna.Trial):
    trial.suggest_float("a", 6, 8)
    trial.suggest_float("b", 1e-4, 1e-2, log=True)
    # training logic goes into trainable, this is just
    # for search space definition
optuna_search = OptunaSearch(
    define_search_space,
    metric="loss",
    mode="min")
tune.run(trainable, search_alg=optuna_search)
.. versionadded:: 0.8.8
```

