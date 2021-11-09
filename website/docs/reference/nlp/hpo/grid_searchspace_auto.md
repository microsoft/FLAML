---
sidebar_label: grid_searchspace_auto
title: nlp.hpo.grid_searchspace_auto
---

## AutoGridSearchSpace Objects

```python
class AutoGridSearchSpace()
```

This is a class for getting the recommended grid search space of a pre-trained LM that will be
instantiated as one of the search spaces of the library when created with the
`~flaml.nlp.hpo.AutoGridSearchSpace.from_model_and_dataset_name` method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_model\_and\_dataset\_name

```python
@classmethod
def from_model_and_dataset_name(cls, model_type, model_size_type, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

Instantiate one of the classes for getting the recommended grid search space of a pre-trained LM from
the model type, model size type, dataset name, sub dataset name and algorithm mode

**Arguments**:

  model_type:
  A string variable which is the model type, e.g. &quot;electra&quot;
  
  model_size_type:
  A string variable which is the size of the model, e.g., &quot;small&quot;
  
  dataset_name_list:
  A string variable which is the dataset name, e.g., &quot;glue&quot;
  
  subdataset_name:
  A string variable which is the sub dataset name,e.g., &quot;rte&quot;
  
  algo_mode:
  A string variable which is the algorithm mode for grid search, e.g., &quot;gridbert&quot;
  

**Example**:

  &gt;&gt;&gt; AutoGridSearchSpace.from_model_and_dataset_name(&quot;electra&quot;, &quot;small&quot;, [&quot;glue&quot;], &quot;rte&quot;, &quot;grid&quot;)

