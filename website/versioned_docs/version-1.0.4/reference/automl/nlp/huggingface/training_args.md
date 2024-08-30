---
sidebar_label: training_args
title: automl.nlp.huggingface.training_args
---

## TrainingArgumentsForAuto Objects

```python
@dataclass
class TrainingArgumentsForAuto(TrainingArguments)
```

FLAML custom TrainingArguments.

**Arguments**:

- `task` _str_ - the task name for NLP tasks, e.g., seq-classification, token-classification
- `output_dir` _str_ - data root directory for outputing the log, etc.
- `model_path` _str, optional, defaults to "facebook/muppet-roberta-base"_ - A string,
  the path of the language model file, either a path from huggingface
  model card huggingface.co/models, or a local path for the model.
- `fp16` _bool, optional, defaults to "False"_ - A bool, whether to use FP16.
- `max_seq_length` _int, optional, defaults to 128_ - An integer, the max length of the sequence.
  For token classification task, this argument will be ineffective.
  pad_to_max_length (bool, optional, defaults to "False"):
  whether to pad all samples to model maximum sentence length.
  If False, will pad the samples dynamically when batching to the maximum length in the batch.
- `per_device_eval_batch_size` _int, optional, defaults to 1_ - An integer, the per gpu evaluation batch size.
- `label_list` _List[str], optional, defaults to None_ - A list of string, the string list of the label names.
  When the task is sequence labeling/token classification, there are two formats of the labels:
  (1) The token labels, i.e., [B-PER, I-PER, B-LOC]; (2) Id labels. For (2), need to pass the label_list (e.g., [B-PER, I-PER, B-LOC])
  to convert the Id to token labels when computing the metric with metric_loss_score.
  See the example in [a simple token classification example](../../../../Examples/AutoML-NLP#a-simple-token-classification-example).

