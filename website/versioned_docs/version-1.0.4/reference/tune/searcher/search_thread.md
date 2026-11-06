---
sidebar_label: search_thread
title: tune.searcher.search_thread
---

## SearchThread Objects

```python
class SearchThread()
```

Class of global or local search thread.

#### \_\_init\_\_

```python
def __init__(mode: str = "min", search_alg: Optional[Searcher] = None, cost_attr: Optional[str] = "time_total_s", eps: Optional[float] = 1.0)
```

When search_alg is omitted, use local search FLOW2.

#### suggest

```python
def suggest(trial_id: str) -> Optional[Dict]
```

Use the suggest() of the underlying search algorithm.

#### on\_trial\_complete

```python
def on_trial_complete(trial_id: str, result: Optional[Dict] = None, error: bool = False)
```

Update the statistics of the thread.

#### reach

```python
def reach(thread) -> bool
```

Whether the incumbent can reach the incumbent of thread.

#### can\_suggest

```python
@property
def can_suggest() -> bool
```

Whether the thread can suggest new configs.

