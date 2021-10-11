# Hyperparameter Optimization for Huggingface Transformers

AutoTransformers is an AutoML class for fine-tuning pre-trained language models based on the transformers library.

An example of using AutoTransformers:

```python
from flaml.nlp.autotransformers import AutoTransformers
autohf = AutoTransformers()

autohf_settings = {
    "dataset_config": ["glue", "mrpc"],
    "model_path": "google/electra-small-discriminator",
    "output_dir": "data/",
    "resources_per_trial": {"cpu": 1, "gpu": 1},
    "resplit_mode": "rspt",
    "sample_num": -1,
    "time_budget": 300,
}

validation_metric, analysis = autohf.fit(**autohf_settings)
if validation_metric is not None:
    predictions, test_metric = autohf.predict()


```

The current use cases that are supported:

1. A simplified version of fine-tuning the GLUE dataset using HuggingFace;
2. For selecting better search space for fine-tuning the GLUE dataset;
3. Use the search algorithms in flaml for more efficient fine-tuning of HuggingFace.

The use cases that can be supported in future:

1. HPO fine-tuning for text generation;
2. HPO fine-tuning for question answering.

## Troubleshooting fine-tuning HPO for pre-trained language models

To reproduce the results for our ACL2021 paper:

* [An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models](https://arxiv.org/abs/2106.09204). Xueqing Liu, Chi Wang. ACL-IJCNLP 2021.

```bibtex
@inproceedings{liu2021hpo,
    title={An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models},
    author={Xueqing Liu and Chi Wang},
    year={2021},
    booktitle={ACL-IJCNLP},
}
```

Please refer to the following jupyter notebook: [Troubleshooting HPO for fine-tuning pre-trained language models](https://github.com/microsoft/FLAML/blob/main/notebook/research/acl2021.ipynb)