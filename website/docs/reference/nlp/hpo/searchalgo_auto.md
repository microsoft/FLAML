---
sidebar_label: searchalgo_auto
title: nlp.hpo.searchalgo_auto
---

## AutoSearchAlgorithm Objects

```python
class AutoSearchAlgorithm()
```

This is a class for getting the search algorithm based on the search algorithm name
(a string variable) instantiated as one of the algorithms of the library when
created with the `~flaml.nlp.hpo.AutoSearchAlgorithm.from_method_name` method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_method\_name

```python
@classmethod
def from_method_name(cls, search_algo_name, search_algo_args_mode, hpo_search_space, time_budget, metric_name, metric_mode_name, **custom_hpo_args)
```

Instantiating one of the search algorithm classes based on the search algorithm name, search algorithm
argument mode, hpo search space and other keyword args

**Arguments**:

  search_algo_name:
  A string variable that specifies the search algorithm name, e.g., &quot;bs&quot;
  
  search_algo_args_mode:
  A string variable that specifies the mode for the search algorithm args, e.g., &quot;dft&quot; means
  initializing using the default mode
  
  hpo_search_space:
  The hpo search space
  
  custom_hpo_args:
  The customized arguments for the search algorithm (specified by user)
  

**Example**:

  &gt;&gt;&gt; from flaml.nlp.hpo.hpo_searchspace import AutoHPOSearchSpace
  &gt;&gt;&gt; search_space_hpo=AutoHPOSearchSpace.from_model_and_dataset_name(&quot;uni&quot;, &quot;electra&quot;, &quot;small&quot;, [&quot;glue&quot;], &quot;rte&quot;)
  &gt;&gt;&gt; search_algo = AutoSearchAlgorithm.from_method_name(&quot;bs&quot;, &quot;cus&quot;, search_space_hpo,
- `{&quot;points_to_evaluate&quot;` - [{&quot;learning_rate&quot;: 1e-5, &quot;num_train_epochs&quot;: 10}])

