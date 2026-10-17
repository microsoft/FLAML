---
sidebar_label: trainer
title: automl.nlp.huggingface.trainer
---

## TrainerForAuto Objects

```python
class TrainerForAuto(Seq2SeqTrainer)
```

#### evaluate

```python
def evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")
```

Overriding transformers.Trainer.evaluate by saving metrics and checkpoint path.

