---
sidebar_label: variant_generator
title: tune.searcher.variant_generator
---

## TuneError Objects

```python
class TuneError(Exception)
```

General error class raised by ray.tune.

#### generate\_variants

```python
def generate_variants(unresolved_spec: Dict, constant_grid_search: bool = False, random_state: "RandomState" = None) -> Generator[Tuple[Dict, Dict], None, None]
```

Generates variants from a spec (dict) with unresolved values.
There are two types of unresolved values:
Grid search: These define a grid search over values. For example, the
following grid search values in a spec will produce six distinct
variants in combination:
"activation": grid_search(["relu", "tanh"])
"learning_rate": grid_search([1e-3, 1e-4, 1e-5])
Lambda functions: These are evaluated to produce a concrete value, and
can express dependencies or conditional distributions between values.
They can also be used to express random search (e.g., by calling
into the `random` or `np` module).
"cpu": lambda spec: spec.config.num_workers
"batch_size": lambda spec: random.uniform(1, 1000)
Finally, to support defining specs in plain JSON / YAML, grid search
and lambda functions can also be defined alternatively as follows:
"activation": {"grid_search": ["relu", "tanh"]}
"cpu": {"eval": "spec.config.num_workers"}
Use `format_vars` to format the returned dict of hyperparameters.

**Yields**:

  (Dict of resolved variables, Spec object)

#### grid\_search

```python
def grid_search(values: List) -> Dict[str, List]
```

Convenience method for specifying grid search over a value.

**Arguments**:

- `values` - An iterable whose parameters will be gridded.

