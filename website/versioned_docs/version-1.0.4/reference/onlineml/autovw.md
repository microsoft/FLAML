---
sidebar_label: autovw
title: onlineml.autovw
---

## AutoVW Objects

```python
class AutoVW()
```

Class for the AutoVW algorithm.

#### \_\_init\_\_

```python
def __init__(max_live_model_num: int, search_space: dict, init_config: Optional[dict] = {}, min_resource_lease: Optional[Union[str, float]] = "auto", automl_runner_args: Optional[dict] = {}, scheduler_args: Optional[dict] = {}, model_select_policy: Optional[str] = "threshold_loss_ucb", metric: Optional[str] = "mae_clipped", random_seed: Optional[int] = None, model_selection_mode: Optional[str] = "min", cb_coef: Optional[float] = None)
```

Constructor.

**Arguments**:

- `max_live_model_num` - An int to specify the maximum number of
  'live' models, which, in other words, is the maximum number
  of models allowed to update in each learning iteraction.
- `search_space` - A dictionary of the search space. This search space
  includes both hyperparameters we want to tune and fixed
  hyperparameters. In the latter case, the value is a fixed value.
- `init_config` - A dictionary of a partial or full initial config,
  e.g. {'interactions': set(), 'learning_rate': 0.5}
- `min_resource_lease` - string or float | The minimum resource lease
  assigned to a particular model/trial. If set as 'auto', it will
  be calculated automatically.
- `automl_runner_args` - A dictionary of configuration for the OnlineTrialRunner.
  If set {}, default values will be used, which is equivalent to using
  the following configs.

**Example**:

  
```python
automl_runner_args = {
    "champion_test_policy": 'loss_ucb', # the statistic test for a better champion
    "remove_worse": False,              # whether to do worse than test
}
```
  
- `scheduler_args` - A dictionary of configuration for the scheduler.
  If set {}, default values will be used, which is equivalent to using the
  following config.

**Example**:

  
```python
scheduler_args = {
    "keep_challenger_metric": 'ucb',  # what metric to use when deciding the top performing challengers
    "keep_challenger_ratio": 0.5,     # denotes the ratio of top performing challengers to keep live
    "keep_champion": True,            # specifcies whether to keep the champion always running
}
```
  
- `model_select_policy` - A string in ['threshold_loss_ucb',
  'threshold_loss_lcb', 'threshold_loss_avg', 'loss_ucb', 'loss_lcb',
  'loss_avg'] to specify how to select one model to do prediction from
  the live model pool. Default value is 'threshold_loss_ucb'.
- `metric` - A string in ['mae_clipped', 'mae', 'mse', 'absolute_clipped',
  'absolute', 'squared'] to specify the name of the loss function used
  for calculating the progressive validation loss in ChaCha.
- `random_seed` - An integer of the random seed used in the searcher
  (more specifically this the random seed for ConfigOracle).
- `model_selection_mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `cb_coef` - A float coefficient (optional) used in the sample complexity bound.

#### predict

```python
def predict(data_sample)
```

Predict on the input data sample.

**Arguments**:

- `data_sample` - one data example in vw format.

#### learn

```python
def learn(data_sample)
```

Perform one online learning step with the given data sample.

**Arguments**:

- `data_sample` - one data example in vw format. It will be used to
  update the vw model.

#### get\_ns\_feature\_dim\_from\_vw\_example

```python
@staticmethod
def get_ns_feature_dim_from_vw_example(vw_example) -> dict
```

Get a dictionary of feature dimensionality for each namespace singleton.

