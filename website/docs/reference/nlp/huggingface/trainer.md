---
sidebar_label: trainer
title: nlp.huggingface.trainer
---

## TrainerForAutoTransformers Objects

```python
class TrainerForAutoTransformers(TFTrainer)
```

#### evaluate

```python
def evaluate(eval_dataset=None)
```

Overriding transformers.Trainer.evaluate by saving state with save_state

**Arguments**:

  eval_dataset:
  the dataset to be evaluated

#### save\_state

```python
def save_state()
```

Overriding transformers.Trainer.save_state. It is only through saving
the states can best_trial.get_best_checkpoint return a non-empty value.

