---
sidebar_label: trial_runner
title: onlineml.trial_runner
---

## OnlineTrialRunner Objects

```python
class OnlineTrialRunner()
```

Class for the OnlineTrialRunner.

#### \_\_init\_\_

```python
def __init__(max_live_model_num: int, searcher=None, scheduler=None, champion_test_policy="loss_ucb", **kwargs)
```

Constructor.

**Arguments**:

- `max_live_model_num` - The maximum number of 'live'/running models allowed.
- `searcher` - A class for generating Trial objects progressively.
  The ConfigOracle is implemented in the searcher.
- `scheduler` - A class for managing the 'live' trials and allocating the
  resources for the trials.
- `champion_test_policy` - A string to specify what test policy to test for
  champion. Currently can choose from ['loss_ucb', 'loss_avg', 'loss_lcb', None].

#### champion\_trial

```python
@property
def champion_trial() -> Trial
```

The champion trial.

#### running\_trials

```python
@property
def running_trials()
```

The running/'live' trials.

#### step

```python
def step(data_sample=None, prediction_trial_tuple=None)
```

Schedule one trial to run each time it is called.

**Arguments**:

- `data_sample` - One data example.
- `prediction_trial_tuple` - A list of information containing
  (prediction_made, prediction_trial).

#### get\_top\_running\_trials

```python
def get_top_running_trials(top_ratio=None, top_metric="ucb") -> list
```

Get a list of trial ids, whose performance is among the top running trials.

#### get\_trials

```python
def get_trials() -> list
```

Return the list of trials managed by this TrialRunner.

#### add\_trial

```python
def add_trial(new_trial)
```

Add a new trial to this TrialRunner.
Trials may be added at any time.

**Arguments**:

- `new_trial` _Trial_ - Trial to queue.

#### stop\_trial

```python
def stop_trial(trial)
```

Stop a trial: set the status of a trial to be
Trial.TERMINATED and perform other subsequent operations.

#### pause\_trial

```python
def pause_trial(trial)
```

Pause a trial: set the status of a trial to be Trial.PAUSED
and perform other subsequent operations.

#### run\_trial

```python
def run_trial(trial)
```

Run a trial: set the status of a trial to be Trial.RUNNING
and perform other subsequent operations.

