# AutoML - NLP

### Requirements

This example requires GPU. Install the [nlp] option:
```python
pip install "flaml[nlp]"
```

### A simple sequence classification example

To perform the sequence classification task in FLAML, simply load your data, convert it to the pandas format, and use `AutoML.fit()`:

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = load_dataset("glue", "mrpc", split="train").to_pandas()
dev_dataset = load_dataset("glue", "mrpc", split="validation").to_pandas()
test_dataset = load_dataset("glue", "mrpc", split="test").to_pandas()

custom_sent_keys = ["sentence1", "sentence2"]          # specify the column names of the input sentences
label_key = "label"                                    # specify the column name of the label

X_train, y_train = train_dataset[custom_sent_keys], train_dataset[label_key]
X_val, y_val = dev_dataset[custom_sent_keys], dev_dataset[label_key]
X_test = test_dataset[custom_sent_keys]

automl = AutoML()
automl_settings = {
    "time_budget": 100,                                 # setting the time budget
    "task": "seq-classification",                       # specifying your task is seq-classification 
    "custom_hpo_args": {"output_dir": "data/output/"},  # specifying your output directory
    "gpu_per_trial": 1,                                 # set to 0 if no GPU is available
}
automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
automl.predict(X_test)
```

#### Sample output

```
[flaml.automl: 12-06 08:21:39] {1943} INFO - task = seq-classification
[flaml.automl: 12-06 08:21:39] {1945} INFO - Data split method: stratified
[flaml.automl: 12-06 08:21:39] {1949} INFO - Evaluation method: holdout
[flaml.automl: 12-06 08:21:39] {2019} INFO - Minimizing error metric: 1-accuracy
[flaml.automl: 12-06 08:21:39] {2071} INFO - List of ML learners in AutoML Run: ['transformer']
[flaml.automl: 12-06 08:21:39] {2311} INFO - iteration 0, current learner transformer
{'data/output/train_2021-12-06_08-21-53/train_8947b1b2_1_n=1e-06,s=9223372036854775807,e=1e-05,s=-1,s=0.45765,e=32,d=42,o=0.0,y=0.0_2021-12-06_08-21-53/checkpoint-53': 53}
[flaml.automl: 12-06 08:22:56] {2424} INFO - Estimated sufficient time budget=766860s. Estimated necessary time budget=767s.
[flaml.automl: 12-06 08:22:56] {2499} INFO -  at 76.7s, estimator transformer's best error=0.1740,      best estimator transformer's best error=0.1740
[flaml.automl: 12-06 08:22:56] {2606} INFO - selected model: <flaml.nlp.huggingface.trainer.TrainerForAuto object at 0x7f49ea8414f0>
[flaml.automl: 12-06 08:22:56] {2100} INFO - fit succeeded
[flaml.automl: 12-06 08:22:56] {2101} INFO - Time taken to find the best model: 76.69802761077881
[flaml.automl: 12-06 08:22:56] {2112} WARNING - Time taken to find the best model is 77% of the provided time budget and not all estimators' hyperparameter search converged. Consider increasing the time budget.
```

### A simple sequence regression example

Sequence regression is for tasks such as predicting the likert scales from reviews. The usage is similar to sequence classification:  

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("glue", "stsb", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("glue", "stsb", split="train").to_pandas()
)

custom_sent_keys = ["sentence1", "sentence2"]   # specify the column names of the input sentences
label_key = "label"                             # specify the column name of the label

X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]
X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()
automl_settings = {
    "gpu_per_trial": 0,
    "time_budget": 20,                          # specifying the time budget
    "task": "seq-regression",                   # setting the task to sequence regression
    "metric": "rmse",                           # setting the evaluation metric 
}
automl_settings["custom_hpo_args"] = {
    "model_path": "google/electra-small-discriminator",
    "output_dir": "data/output/",
    "ckpt_per_epoch": 5,
    "fp16": False,
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```


### A simple summarization example

Similarly, you can use FLAML for summarizing a long document into a short document: 

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("xsum", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("xsum", split="validation").to_pandas()
)

custom_sent_keys = ["document"]                  # specify the column names of the input document, i.e., the original document
label_key = "summary"                            # specify the column name of the output summarization

X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]

X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()
automl_settings = {
    "gpu_per_trial": 1,
    "time_budget": 20,                           # specifying the time budget
    "task": "summarization",                     # setting the task to summarization
    "metric": "rouge1",                          # setting the metric 
}
automl_settings["custom_hpo_args"] = {
    "model_path": "t5-small",
    "output_dir": "data/output/",
    "ckpt_per_epoch": 5,
    "fp16": False,
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```

### A simple token classification example

The token classification can be used for tasks such as named entity recognition and part-of-speech tagging. An example of using FLAML for NER is:

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("conll2003", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("conll2003", split="validation").to_pandas()
)

custom_sent_keys = ["tokens"]                   # specify the column names of the input data
label_key = "ner_tags"                          # specify the column name of the output token tags

X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]

X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()
automl_settings = {
    "gpu_per_trial": 0,
    "time_budget": 50,                       # specify the time budget 
    "task": "token-classification",         # specify the task to token classification
    "metric": "seqeval",                    # specify the metric
}

automl_settings["custom_hpo_args"] = {
    "model_path": "bert-base-uncased",
    "output_dir": "test/data/output/",
    "ckpt_per_epoch": 1,
    "fp16": False,
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```

### A simple multiple choice example

The multiple choice task is for selecting from options to finish a sentence based on common sense reasoning. For example:

On stage, a woman takes a seat at the piano. She
a) sits on a bench as her sister plays with the doll.
b) smiles with someone as the music plays.
c) is in the crowd, watching the dancers.
d) nervously sets her fingers on the keys

An example of running FLAML for the multiple choice task:

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("swag", split="train").to_pandas()
)
dev_dataset = (
    load_dataset("swag", split="validation").to_pandas()
)

custom_sent_keys = [                            # specify the column names of the first sentence and choices
    "sent1",
    "sent2",
    "ending0",
    "ending1",
    "ending2",
    "ending3",
    "gold-source",
    "video-id",
    "startphrase",
    "fold-ind",
]
label_key = "label"                         # specify the column name of the output token tags

X_train = train_dataset[custom_sent_keys]
y_train = train_dataset[label_key]

X_val = dev_dataset[custom_sent_keys]
y_val = dev_dataset[label_key]

automl = AutoML()
automl_settings = {
    "gpu_per_trial": 0,
    "time_budget": 50,                      # specify the time budget 
    "task": "multichoice-classification",  # specify the task to multiple choice
    "metric": "accuracy",                  # specify the evaluation metric
    "log_file_name": "seqclass.log",
}

automl_settings["custom_hpo_args"] = {
    "model_path": "google/electra-small-discriminator",
    "output_dir": "test/data/output/",
    "ckpt_per_epoch": 1,
    "fp16": False,
}
automl.fit(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
)
```

### A simple example of running your own dataset

If you want to run your own dataset in csv or json, simply load it into a pandas dataframe and the rest are the same:

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = pd.read_csv("train.tsv", delimiter="\t", quoting=3)
dev_dataset = pd.read_csv("dev.tsv", delimiter="\t", quoting=3)
test_dataset = pd.read_csv("test.tsv", delimiter="\t", quoting=3)
```

For tasks that are not currently supported, use `flaml.tune` for [customized tuning](Tune-HuggingFace).
