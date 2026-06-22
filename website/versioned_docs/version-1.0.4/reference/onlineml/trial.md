---
sidebar_label: trial
title: onlineml.trial
---

#### get\_ns\_feature\_dim\_from\_vw\_example

```python
def get_ns_feature_dim_from_vw_example(vw_example) -> dict
```

Get a dictionary of feature dimensionality for each namespace singleton.

## OnlineResult Objects

```python
class OnlineResult()
```

Class for managing the result statistics of a trial.

#### \_\_init\_\_

```python
def __init__(result_type_name: str, cb_coef: Optional[float] = None, init_loss: Optional[float] = 0.0, init_cb: Optional[float] = 100.0, mode: Optional[str] = "min", sliding_window_size: Optional[int] = 100)
```

Constructor.

**Arguments**:

- `result_type_name` - A String to specify the name of the result type.
- `cb_coef` - a string to specify the coefficient on the confidence bound.
- `init_loss` - a float to specify the inital loss.
- `init_cb` - a float to specify the intial confidence bound.
- `mode` - A string in ['min', 'max'] to specify the objective as
  minimization or maximization.
- `sliding_window_size` - An int to specify the size of the sliding windown
  (for experimental purpose).

#### update\_result

```python
def update_result(new_loss, new_resource_used, data_dimension, bound_of_range=1.0, new_observation_count=1.0)
```

Update result statistics.

## BaseOnlineTrial Objects

```python
class BaseOnlineTrial(Trial)
```

Class for the online trial.

#### \_\_init\_\_

```python
def __init__(config: dict, min_resource_lease: float, is_champion: Optional[bool] = False, is_checked_under_current_champion: Optional[bool] = True, custom_trial_name: Optional[str] = "mae", trial_id: Optional[str] = None)
```

Constructor.

**Arguments**:

- `config` - The configuration dictionary.
- `min_resource_lease` - A float specifying the minimum resource lease.
- `is_champion` - A bool variable indicating whether the trial is champion.
- `is_checked_under_current_champion` - A bool indicating whether the trial
  has been used under the current champion.
- `custom_trial_name` - A string of a custom trial name.
- `trial_id` - A string for the trial id.

#### set\_resource\_lease

```python
def set_resource_lease(resource: float)
```

Sets the resource lease accordingly.

#### set\_status

```python
def set_status(status)
```

Sets the status of the trial and record the start time.

## VowpalWabbitTrial Objects

```python
class VowpalWabbitTrial(BaseOnlineTrial)
```

The class for Vowpal Wabbit online trials.

#### \_\_init\_\_

```python
def __init__(config: dict, min_resource_lease: float, metric: str = "mae", is_champion: Optional[bool] = False, is_checked_under_current_champion: Optional[bool] = True, custom_trial_name: Optional[str] = "vw_mae_clipped", trial_id: Optional[str] = None, cb_coef: Optional[float] = None)
```

Constructor.

**Arguments**:

- `config` _dict_ - the config of the trial (note that the config is a set
  because the hyperparameters are).
- `min_resource_lease` _float_ - the minimum resource lease.
- `metric` _str_ - the loss metric.
- `is_champion` _bool_ - indicates whether the trial is the current champion or not.
- `is_checked_under_current_champion` _bool_ - indicates whether this trials has
  been paused under the current champion.
- `trial_id` _str_ - id of the trial (if None, it will be generated in the constructor).

#### train\_eval\_model\_online

```python
def train_eval_model_online(data_sample, y_pred)
```

Train and evaluate model online.

#### predict

```python
def predict(x)
```

Predict using the model.

