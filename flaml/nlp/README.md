# AutoML for NLP

This directory contains utility functions used by AutoNLP. Currently we support four NLP tasks: sequence classification, sequence regression, multiple choice and summarization. 

Please refer to this [link](https://microsoft.github.io/FLAML/docs/Examples/AutoML-NLP) for examples.


# Troubleshooting fine-tuning HPO for pre-trained language models

When using AutoNLP for hypperparameter tuning, there can be multiple reasons for a suboptimal results. To help users troubleshooting such tuning failures, we have investigated the following research question: how to systematically trouble shoot AutoNLP's failure and improve the performance? Our findings can be seen in the following paper published in ACL 2021:

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
