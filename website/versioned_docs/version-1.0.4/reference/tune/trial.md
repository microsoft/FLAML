---
sidebar_label: trial
title: tune.trial
---

#### unflatten\_dict

```python
def unflatten_dict(dt, delimiter="/")
```

Unflatten dict. Does not support unflattening lists.

## Trial Objects

```python
class Trial()
```

A trial object holds the state for one model training run.
Trials are themselves managed by the TrialRunner class, which implements
the event loop for submitting trial runs to a Ray cluster.
Trials start in the PENDING state, and transition to RUNNING once started.
On error it transitions to ERROR, otherwise TERMINATED on success.

**Attributes**:

- `trainable_name` _str_ - Name of the trainable object to be executed.
- `config` _dict_ - Provided configuration dictionary with evaluated params.
- `trial_id` _str_ - Unique identifier for the trial.
- `local_dir` _str_ - Local_dir as passed to tune.run.
- `logdir` _str_ - Directory where the trial logs are saved.
- `evaluated_params` _dict_ - Evaluated parameters by search algorithm,
- `experiment_tag` _str_ - Identifying trial name to show in the console.
- `resources` _Resources_ - Amount of resources that this trial will use.
- `status` _str_ - One of PENDING, RUNNING, PAUSED, TERMINATED, ERROR/
- `error_file` _str_ - Path to the errors that this trial has raised.

#### set\_status

```python
def set_status(status)
```

Sets the status of the trial.

