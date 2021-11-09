---
sidebar_label: dataprocess_auto
title: nlp.dataset.dataprocess_auto
---

#### tokenize\_superglue\_wic

```python
def tokenize_superglue_wic(this_example, this_tokenizer, dataset_name, subdataset_name=None, **kwargs)
```

tokenize the data from the wic task (word-in-context dataset),
e.g., sentence 1: &quot;There&#x27;s a lot of trash on the bed of the river&quot;
sentence 2: &quot;I keep a glass of water next to my bed when I sleep&quot;,
label = False (different word senses)
In the superglue data, the position of the word in sentence 1 and 2 are provided
What this function does is to update the span position after tokenization, based on each LM&#x27;s own tokenizer,
The key is to insert an [SEP] before and after the original sentence, then feed it into the LM&#x27;s tokenizer.
There are two challenges:
   (1) Each LM&#x27;s tokenizations are different, e.g., in XLNet&#x27;s tokenizer, the paddings are on the left&#x27;
   (2) Some LM&#x27;s tokenization would add an underline symbol before the word, e.g., &quot;There&#x27;s a lot&quot;
   -&gt; [_There, _&#x27;, _s, _a, _lot]
   When underline meets special char such as &#x27;&quot;&#x27;, &quot;&#x27;&quot;, the tokenized sequence after adding [SEP] needs to be
   aligned with the sequence tokenized without [SEP]. We use a two pointer algorithm for the alignment

## AutoEncodeText Objects

```python
class AutoEncodeText()
```

This is a generic input text tokenization class that will be instantiated as one of the
tokenization classes of the library when created with the
`~flaml.nlp.dataset.AutoEncodeText.from_model_and_dataset_name` class method.

This class cannot be instantiated directly using ``__init__()`` (throws an error).

#### from\_model\_and\_dataset\_name

```python
@classmethod
def from_model_and_dataset_name(cls, data_raw, model_checkpoint_path, dataset_name_list: list = None, subdataset_name=None, **kwargs)
```

Instantiate one of the input text tokenization classes from the raw data, model checkpoint path, dataset name
and sub dataset name. The raw data is used for creating a mapping function from the raw tokens to the
tokenized token ids.

**Arguments**:

  data_raw:
  The raw data (a datasets.Dataset object)
  
  model_checkpoint_path:
  A string variable which specifies the model path, e.g., &quot;google/electra-base-discriminator&quot;
  
  dataset_name_list:
  A list which is the dataset name, e.g., [&quot;glue&quot;]
  
  subdataset_name:
  A string variable which is the sub dataset name,e.g., &quot;rte&quot;
  
  kwargs:
  The values in kwargs of any keys will be used for the mapping function
  

**Examples**:

  &gt;&gt;&gt; from datasets import load_dataset
  &gt;&gt;&gt; data_raw = load_dataset(&quot;glue&quot;, &quot;rte&quot;)
  &gt;&gt;&gt; AutoEncodeText.from_model_and_dataset_name(data_raw, &quot;google/electra-base-discriminator&quot;, [&quot;glue&quot;], &quot;rte&quot;)

