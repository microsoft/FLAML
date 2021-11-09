---
sidebar_label: hpo_searchspace
title: nlp.hpo.hpo_searchspace
---

#### hpo\_space\_custom

```python
def hpo_space_custom(model_type=None, model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None, **custom_hpo_args)
```

The 5 arguments here cannot be deleted, they need to be kept consistent with
other functions in HPO_SEARCH_SPACE_MAPPING

## AutoHPOSearchSpace Objects

```python
class AutoHPOSearchSpace()
```

This is a class for getting the hpo search space based on the search space mode
(a string variable) instantiated as one of the HPO search spaces of the library when
created with the `~flaml.nlp.hpo.AutoHPOSearchSpace.from_model_and_dataset_name` method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_model\_and\_dataset\_name

```python
@classmethod
def from_model_and_dataset_name(cls, hpo_searchspace_mode, model_type, model_size_type, dataset_name_list: list = None, subdataset_name=None, algo_mode=None, **custom_hpo_args)
```

Instantiate one of the classes for getting the hpo search space from the search space name, model type,
model size type, dataset name and sub dataset name

**Arguments**:

  
  hpo_searchspace_mode:
  A string variable which is the mode of the hpo search space, it must be chosen from the following options:
  - uni: the union of BERT, RoBERTa and Electra&#x27;s grid configs
  - grid: the recommended grid config of the LM specified in jobconfig.pre
  - gnr: the generic continuous search space
  - uni_test: the search space for smoke test
  - cus: user customized search space, specified in the &quot;hpo_space&quot; argument in AutoTransformers.fit
  - buni: bounded grid union search space
  
  model_type:
  A string variable which is the type of the model, e.g., &quot;electra&quot;
  
  model_size_type:
  A string variable which is the type of the model size, e.g., &quot;small&quot;
  
  dataset_name:
  A string variable which is the dataset name, e.g., &quot;glue&quot;
  
  subdataset_name:
  A string variable which is the sub dataset name,e.g., &quot;rte&quot;
  
  custom_hpo_args:
  Any additional keyword argument to be used for the function for the HPO search space
  

**Example**:

  &gt;&gt;&gt; AutoHPOSearchSpace.from_model_and_dataset_name(&quot;uni&quot;, &quot;electra&quot;, &quot;small&quot;, [&quot;glue&quot;], &quot;rte&quot;, &quot;hpo&quot;)

