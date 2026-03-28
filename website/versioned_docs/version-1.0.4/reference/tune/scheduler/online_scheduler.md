---
sidebar_label: online_scheduler
title: tune.scheduler.online_scheduler
---

## OnlineScheduler Objects

```python
class OnlineScheduler(TrialScheduler)
```

Class for the most basic OnlineScheduler.

#### on\_trial\_result

```python
def on_trial_result(trial_runner, trial: Trial, result: Dict)
```

Report result and return a decision on the trial's status.

#### choose\_trial\_to\_run

```python
def choose_trial_to_run(trial_runner) -> Trial
```

Decide which trial to run next.

## OnlineSuccessiveDoublingScheduler Objects

```python
class OnlineSuccessiveDoublingScheduler(OnlineScheduler)
```

class for the OnlineSuccessiveDoublingScheduler algorithm.

#### \_\_init\_\_

```python
def __init__(increase_factor: float = 2.0)
```

Constructor.

**Arguments**:

- `increase_factor` - A float of multiplicative factor
  used to increase resource lease. Default is 2.0.

#### on\_trial\_result

```python
def on_trial_result(trial_runner, trial: Trial, result: Dict)
```

Report result and return a decision on the trial's status.

## ChaChaScheduler Objects

```python
class ChaChaScheduler(OnlineSuccessiveDoublingScheduler)
```

class for the ChaChaScheduler algorithm.

#### \_\_init\_\_

```python
def __init__(increase_factor: float = 2.0, **kwargs)
```

Constructor.

**Arguments**:

- `increase_factor` - A float of multiplicative factor
  used to increase resource lease. Default is 2.0.

#### on\_trial\_result

```python
def on_trial_result(trial_runner, trial: Trial, result: Dict)
```

Report result and return a decision on the trial's status.

