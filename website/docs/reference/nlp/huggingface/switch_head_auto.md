---
sidebar_label: switch_head_auto
title: nlp.huggingface.switch_head_auto
---

## AutoSeqClassificationHead Objects

```python
class AutoSeqClassificationHead()
```

This is a class for getting classification head class based on the name of the LM
instantiated as one of the ClassificationHead classes of the library when
created with the `~flaml.nlp.huggingface.AutoSeqClassificationHead.from_model_type_and_config` method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_model\_type\_and\_config

```python
@classmethod
def from_model_type_and_config(cls, model_type, config)
```

Instantiate one of the classification head classes from the mode_type and model configuration.

**Arguments**:

  model_type:
  A string, which desribes the model type, e.g., &quot;electra&quot;
  config (:class:`~transformers.PretrainedConfig`):
  The huggingface class of the model&#x27;s configuration:
  
  Examples::
  &gt;&gt;&gt; from transformers import AutoConfig
  &gt;&gt;&gt; model_config = AutoConfig.from_pretrained(&quot;google/electra-base-discriminator&quot;)
  &gt;&gt;&gt; AutoSeqClassificationHead.from_model_type_and_config(&quot;electra&quot;, model_config)

