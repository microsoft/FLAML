---
sidebar_label: space
title: tune.space
---

#### define\_by\_run\_func

```python
def define_by_run_func(trial, space: Dict, path: str = "") -> Optional[Dict[str, Any]]
```

Define-by-run function to create the search space.

**Returns**:

  A dict with constant values.

#### unflatten\_hierarchical

```python
def unflatten_hierarchical(config: Dict, space: Dict) -> Tuple[Dict, Dict]
```

Unflatten hierarchical config.

#### add\_cost\_to\_space

```python
def add_cost_to_space(space: Dict, low_cost_point: Dict, choice_cost: Dict)
```

Update the space in place by adding low_cost_point and choice_cost.

**Returns**:

  A dict with constant values.

#### normalize

```python
def normalize(config: Dict, space: Dict, reference_config: Dict, normalized_reference_config: Dict, recursive: bool = False)
```

Normalize config in space according to reference_config.

Normalize each dimension in config to [0,1].

#### indexof

```python
def indexof(domain: Dict, config: Dict) -> int
```

Find the index of config in domain.categories.

#### complete\_config

```python
def complete_config(partial_config: Dict, space: Dict, flow2, disturb: bool = False, lower: Optional[Dict] = None, upper: Optional[Dict] = None) -> Tuple[Dict, Dict]
```

Complete partial config in space.

**Returns**:

  config, space.

