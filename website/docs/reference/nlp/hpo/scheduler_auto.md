---
sidebar_label: scheduler_auto
title: nlp.hpo.scheduler_auto
---

## AutoScheduler Objects

```python
class AutoScheduler()
```

This is a class for getting the scheduler based on the scheduler name
(a string variable) instantiated as one of the schedulers of the library when
created with the `~flaml.nlp.hpo.AutoScheduler.from_scheduler_name` method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_scheduler\_name

```python
@classmethod
def from_scheduler_name(cls, scheduler_name, **kwargs)
```

Instantiate one of the schedulers using the scheduler names

**Arguments**:

  scheduler_name:
  A string variable for the scheduler name
  

**Example**:

  &gt;&gt;&gt; AutoScheduler.from_scheduler_name(&quot;asha&quot;)

