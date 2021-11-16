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
automl_settings = {
    "time_budget": 100,
    "task": "seq-classification",
    "custom_hpo_args": {"output_dir": "data/output/"},
    "gpu_per_trial": 1,  # set to 0 if no GPU is available
}
automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
automl.predict(X_test)
```

#### Sample output

```
```

### A simple sequence regression example

```python
from flaml import AutoML
from datasets import load_dataset

train_dataset = (
    load_dataset("glue", "stsb", split="train[:1%]").to_pandas().iloc[0:4]
)
dev_dataset = (
    load_dataset("glue", "stsb", split="train[1%:2%]").to_pandas().iloc[0:4]
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