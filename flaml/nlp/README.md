How to use AutoTransformers:

```python
from flaml.nlp.autotransformers import AutoTransformers

autohf = AutoTransformers()
autohf_settings = {"resources_per_trial": {"cpu": 1},
                    "num_samples": 1,
                    "time_budget": 100000,
                    "ckpt_per_epoch": 1,
                    "fp16": False,
                   }
validation_metric, analysis = \
    autohf.fit(**autohf_settings,)

```

The current use cases that are supported:
1. A simplified version of fine-tuning the GLUE dataset using HuggingFace;
2. For selecting better search space for fine-tuning the GLUE dataset;
3. Use the search algorithms in flaml for more efficient fine-tuning of HuggingFace;

The use cases that can be supported in future:
1. HPO fine-tuning for text generation;
2. HPO fine-tuning for question answering;