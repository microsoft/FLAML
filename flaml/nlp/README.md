# AutoML for NLP

This directory contains utility functions for the nlp module, currently we support four tasks: sequence classification, sequence regression, multiple choice and summarization. 

For how to install NLP and usage cases please refer to:

https://microsoft.github.io/FLAML/docs/Examples/AutoML-NLP

# Troubleshooting fine-tuning HPO for pre-trained language models

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
