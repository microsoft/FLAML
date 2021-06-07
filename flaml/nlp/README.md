How to use AutoTransformers:

```python
from flaml.nlp.autotransformers import AutoTransformers

autohf = AutoTransformers()
autohf.fit(train_dataset,
           eval_dataset)

```