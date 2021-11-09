---
sidebar_label: get_grid_search_space
title: nlp.hpo.get_grid_search_space
---

#### get\_space\_union\_and\_unique

```python
def get_space_union_and_unique(search_space_common, search_space_unique, this_case_tags: list)
```

get the recommended search configs for each pre-trained language models

**Arguments**:

  search_space_common:
  the union of configs recommended by the LM for all cases;
  search_space_unique:
  the recommended config by the LM for a specific condition, e.g., small model
  this_case_tags:
  a list, which contains the tags describing the specific condition, e.g., [&quot;small&quot;]

#### get\_deberta\_space

```python
def get_deberta_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION: Table 9
https://arxiv.org/abs/2006.03654

#### get\_longformer\_space

```python
def get_longformer_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

TODO: Longformer: The Long-Document Transformer

#### get\_funnel\_space

```python
def get_funnel_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
https://arxiv.org/abs/2006.03236

#### get\_bert\_space

```python
def get_bert_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/pdf/1810.04805.pdf

#### get\_electra\_space

```python
def get_electra_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
https://arxiv.org/pdf/2003.10555.pdf

#### get\_mobilebert\_space

```python
def get_mobilebert_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
https://arxiv.org/pdf/2004.02984.pdf

#### get\_albert\_space

```python
def get_albert_space(model_size_type=None, dataset_name_list: list = None, subdataset_name=None, algo_mode=None)
```

Hyperparameters for downstream tasks are shown in Table 14. We adapt these hyperparameters
from Liu et al. (2019), Devlin et al. (2019), and Yang et al. (2019).

LR BSZ ALBERT DR Classifier DR TS WS MSL
CoLA 1.00E-05 16 0 0.1 5336 320 512
STS 2.00E-05 16 0 0.1 3598 214 512
SST-2 1.00E-05 32 0 0.1 20935 1256 512
MNLI 3.00E-05 128 0 0.1 10000 1000 512
QNLI 1.00E-05 32 0 0.1 33112 1986 512
QQP 5.00E-05 128 0.1 0.1 14000 1000 512
RTE 3.00E-05 32 0.1 0.1 800 200 512
MRPC 2.00E-05 32 0 0.1 800 200 512
WNLI 2.00E-05 16 0.1 0.1 2000 250 512
SQuAD v1.1 5.00E-05 48 0 0.1 3649 365 384
SQuAD v2.0 3.00E-05 48 0 0.1 8144 814 512
RACE 2.00E-05 32 0.1 0.1 12000 1000 512

