# AutoML - NLP

### Requirements

This example requires GPU. Install the [nlp] option:
```python
pip install "flaml[nlp]"
```

### A simple sequence classification example

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas()
dev_dataset = load_dataset("glue", "mrpc", split="validation").to_pandas()
test_dataset = load_dataset("glue", "mrpc", split="test").to_pandas()
custom_sent_keys = ["sentence1", "sentence2"]
label_key = "label"
X_train, y_train = train_dataset[custom_sent_keys], train_dataset[label_key]
X_val, y_val = dev_dataset[custom_sent_keys], dev_dataset[label_key]
X_test = test_dataset[custom_sent_keys]

automl = AutoML()

import ray
if not ray.is_initialized():
    ray.init()

automl_settings = {
    "time_budget": 1200,                  # setting the time budget
    "task": "seq-classification",       # setting the task as seq-classification
    "fit_kwargs_by_estimator": {
        "transformer": {
            "output_dir": "data/output/",   # setting the output directory
            "model_path": "bert-base-uncased",  # if model_path is not set, the default model is facebook/muppet-roberta-base: https://huggingface.co/facebook/muppet-roberta-base
        }
    },
    "gpu_per_trial": 1,                 # set to 0 if no GPU is available
    "log_file_name": "seqclass.log",    # set the file to save the log for HPO
    "log_type": "all",                  # the log type for trials: "all" if logging all the trials, "better" if only keeping the better trials
    "use_ray": {"local_dir": "data/output/"},                    # set whether to use Ray
    "n_concurrent_trials": 4,
    "keep_search_state": True,          # keeping the search state
}
automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
automl.predict(X_test)
```

#### Sample output

```
== Status ==
Current time: 2022-07-21 13:27:54 (running for 00:20:14.52)
Memory usage on this node: 22.9/376.6 GiB
Using FIFO scheduling algorithm.
Resources requested: 0/96 CPUs, 0/4 GPUs, 0.0/252.58 GiB heap, 0.0/112.24 GiB objects (0.0/1.0 accelerator_type:V100)
Current best trial: a478b276 with val_loss=0.12009803921568629 and parameters={'learning_rate': 2.1872511767624938e-05, 'num_train_epochs': 3, 'per_device_train_batch_size': 4, 'seed': 7, 'global_max_steps': 9223372036854775807, 'learner': 'transformer'}
Result logdir: /data/xliu127/projects/hyperopt/FLAML/notebook/data/output/train_2022-07-21_13-07-38
Number of trials: 33/1000000 (33 TERMINATED)
```

### A simple sequence regression example

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("glue", "stsb", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("glue", "stsb", split="train").to_pandas()
)
custom_sent_keys = ["sentence1", "sentence2"]
label_key = "label"
X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]
X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()
automl_settings = {
    "gpu_per_trial": 0,
    "time_budget": 20,
    "task": "seq-regression",
    "metric": "rmse",
}
automl_settings["fit_kwargs_by_estimator"] = {  # setting the huggingface arguments
    "transformer": {
        "model_path": "google/electra-small-discriminator", # if model_path is not set, the default model is facebook/muppet-roberta-base: https://huggingface.co/facebook/muppet-roberta-base
        "output_dir": "data/output/",                       # setting the output directory
        "fp16": False,
    }   # setting whether to use FP16
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```

#### Sample output

```
[flaml.automl: 12-20 11:47:28] {1965} INFO - task = seq-regression
[flaml.automl: 12-20 11:47:28] {1967} INFO - Data split method: uniform
[flaml.automl: 12-20 11:47:28] {1971} INFO - Evaluation method: holdout
[flaml.automl: 12-20 11:47:28] {2063} INFO - Minimizing error metric: rmse
[flaml.automl: 12-20 11:47:28] {2115} INFO - List of ML learners in AutoML Run: ['transformer']
[flaml.automl: 12-20 11:47:28] {2355} INFO - iteration 0, current learner transformer
```

### A simple summarization example

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("xsum", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("xsum", split="validation").to_pandas()
)
custom_sent_keys = ["document"]
label_key = "summary"

X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]

X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()

import ray
if not ray.is_initialized():
    ray.init()

automl_settings = {
    "time_budget": 500,         # setting the time budget
    "task": "summarization",    # setting the task as summarization
    "fit_kwargs_by_estimator": {  # if model_path is not set, the default model is t5-small: https://huggingface.co/t5-small
        "transformer": {
            "output_dir": "data/output/",  # setting the output directory
            "model_path": "t5-small",
            "per_device_eval_batch_size": 16,  # the batch size for validation (inference)
        }
    },
    "gpu_per_trial": 1,  # set to 0 if no GPU is available
    "log_file_name": "seqclass.log",  # set the file to save the log for HPO
    "log_type": "all",   # the log type for trials: "all" if logging all the trials, "better" if only keeping the better trials
    "use_ray": {"local_dir": "data/output/"},  # set whether to use Ray
    "metric": "rouge1",
    "n_concurrent_trials": 4,  # sample: False # if the time is sufficient (e.g., longer than one trial's running time), you can set
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```
#### Sample Output

```
== Status ==
Current time: 2022-03-19 14:55:00 (running for 00:08:31.38)
Memory usage on this node: 23.1/376.6 GiB
Using FIFO scheduling algorithm.
Resources requested: 0/96 CPUs, 0/4 GPUs, 0.0/250.17 GiB heap, 0.0/111.21 GiB objects (0.0/1.0 accelerator_type:V100)
Current best trial: 08b6571c with val_loss=0.8569452656271894 and parameters={'learning_rate': 1.0000000000000003e-05, 'num_train_epochs': 1.0, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.0, 'weight_decay': 0.0, 'adam_epsilon': 1e-06, 'seed': 42, 'global_max_steps': 9223372036854775807, 'learner': 'transformer', 'FLAML_sample_size': 10000}
Result logdir: /data/xliu127/projects/hyperopt/FLAML/notebook/data/output/train_2022-03-19_14-46-29
Number of trials: 8/1000000 (8 TERMINATED)
```

### A simple token classification example

There are two ways to define the label for a token classification task. The first is to define the token labels:

```python
from flaml import AutoML
import pandas as pd

train_dataset = {
    "id": ["0", "1"],
    "ner_tags": [
        ["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"],
        ["B-PER", "I-PER"],
    ],
    "tokens": [
        [
            "EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".",
        ],
        ["Peter", "Blackburn"],
    ],
}
dev_dataset = {
    "id": ["0"],
    "ner_tags": [
        ["O"],
    ],
    "tokens": [
        ["1996-08-22"]
    ],
}
test_dataset = {
    "id": ["0"],
    "ner_tags": [
        ["O"],
    ],
    "tokens": [
        ['.']
    ],
}
custom_sent_keys = ["tokens"]
label_key = "ner_tags"

train_dataset = pd.DataFrame(train_dataset)
dev_dataset = pd.DataFrame(dev_dataset)
test_dataset = pd.DataFrame(test_dataset)

X_train, y_train = train_dataset[custom_sent_keys], train_dataset[label_key]
X_val, y_val = dev_dataset[custom_sent_keys], dev_dataset[label_key]
X_test = test_dataset[custom_sent_keys]

automl = AutoML()
automl_settings = {
    "time_budget": 10,
    "task": "token-classification",
    "fit_kwargs_by_estimator": {
        "transformer":
            {
                "output_dir": "data/output/"
                # if model_path is not set, the default model is facebook/muppet-roberta-base: https://huggingface.co/facebook/muppet-roberta-base
            }
    },  # setting the huggingface arguments: output directory
    "gpu_per_trial": 1,  # set to 0 if no GPU is available
    "metric": "seqeval:overall_f1"
}

automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
automl.predict(X_test)
```

The second is to define the id labels + a token [label list](https://microsoft.github.io/FLAML/docs/reference/nlp/huggingface/training_args):

```python
from flaml import AutoML
import pandas as pd

train_dataset = {
        "id": ["0", "1"],
        "ner_tags": [
            [3, 0, 7, 0, 0, 0, 7, 0, 0],
            [1, 2],
        ],
        "tokens": [
            [
                "EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".",
            ],
            ["Peter", "Blackburn"],
        ],
    }
dev_dataset = {
    "id": ["0"],
    "ner_tags": [
        [0],
    ],
    "tokens": [
        ["1996-08-22"]
    ],
}
test_dataset = {
    "id": ["0"],
    "ner_tags": [
        [0],
    ],
    "tokens": [
        ['.']
    ],
}
custom_sent_keys = ["tokens"]
label_key = "ner_tags"

train_dataset = pd.DataFrame(train_dataset)
dev_dataset = pd.DataFrame(dev_dataset)
test_dataset = pd.DataFrame(test_dataset)

X_train, y_train = train_dataset[custom_sent_keys], train_dataset[label_key]
X_val, y_val = dev_dataset[custom_sent_keys], dev_dataset[label_key]
X_test = test_dataset[custom_sent_keys]

automl = AutoML()
automl_settings = {
    "time_budget": 10,
    "task": "token-classification",
    "fit_kwargs_by_estimator": {
        "transformer":
            {
                "output_dir": "data/output/",
                # if model_path is not set, the default model is facebook/muppet-roberta-base: https://huggingface.co/facebook/muppet-roberta-base
                "label_list": [ "O","B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC" ]
            }
    },  # setting the huggingface arguments: output directory
    "gpu_per_trial": 1,  # set to 0 if no GPU is available
    "metric": "seqeval:overall_f1"
}

automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
automl.predict(X_test)
```

#### Sample Output

```
[flaml.automl: 06-30 03:10:02] {2423} INFO - task = token-classification
[flaml.automl: 06-30 03:10:02] {2425} INFO - Data split method: stratified
[flaml.automl: 06-30 03:10:02] {2428} INFO - Evaluation method: holdout
[flaml.automl: 06-30 03:10:02] {2497} INFO - Minimizing error metric: seqeval:overall_f1
[flaml.automl: 06-30 03:10:02] {2637} INFO - List of ML learners in AutoML Run: ['transformer']
[flaml.automl: 06-30 03:10:02] {2929} INFO - iteration 0, current learner transformer
```

For tasks that are not currently supported, use `flaml.tune` for [customized tuning](Tune-HuggingFace).

### Link to Jupyter notebook

To run these examples in our Jupyter notebook, please go to:

[Link to notebook](https://github.com/microsoft/FLAML/blob/main/notebook/automl_nlp.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/FLAML/blob/main/notebook/automl_nlp.ipynb)