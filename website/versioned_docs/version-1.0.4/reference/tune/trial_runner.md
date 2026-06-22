---
sidebar_label: trial_runner
title: tune.trial_runner
---

## Nologger Objects

```python
class Nologger()
```

Logger without logging.

## SimpleTrial Objects

```python
class SimpleTrial(Trial)
```

A simple trial class.

## BaseTrialRunner Objects

```python
class BaseTrialRunner()
```

Implementation of a simple trial runner.

Note that the caller usually should not mutate trial state directly.

#### get\_trials

```python
def get_trials()
```

Returns the list of trials managed by this TrialRunner.

Note that the caller usually should not mutate trial state directly.

#### add\_trial

```python
def add_trial(trial)
```

Adds a new trial to this TrialRunner.

Trials may be added at any time.

**Arguments**:

- `trial` _Trial_ - Trial to queue.

#### stop\_trial

```python
def stop_trial(trial)
```

Stops trial.

## SequentialTrialRunner Objects

```python
class SequentialTrialRunner(BaseTrialRunner)
```

Implementation of the sequential trial runner.

#### step

```python
def step() -> Trial
```

Runs one step of the trial event loop.

Callers should typically run this method repeatedly in a loop. They
may inspect or modify the runner's state in between calls to step().

**Returns**:

  a trial to run.

