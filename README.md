[![PyPI version](https://badge.fury.io/py/FLAML.svg)](https://badge.fury.io/py/FLAML)
![Conda version](https://img.shields.io/conda/vn/conda-forge/flaml)
[![Build](https://github.com/microsoft/FLAML/actions/workflows/python-package.yml/badge.svg)](https://github.com/microsoft/FLAML/actions/workflows/python-package.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/FLAML)](https://pypi.org/project/FLAML/)
[![Downloads](https://pepy.tech/badge/flaml)](https://pepy.tech/project/flaml)
[![](https://img.shields.io/discord/1025786666260111483?logo=discord&style=flat)](https://discord.gg/Cppx2vSPVP)

<!-- [![Join the chat at https://gitter.im/FLAMLer/community](https://badges.gitter.im/FLAMLer/community.svg)](https://gitter.im/FLAMLer/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) -->

# A Fast Library for Automated Machine Learning & Tuning

<p align="center">
    <img src="https://github.com/microsoft/FLAML/blob/main/website/static/img/flaml.svg"  width=200>
    <br>
</p>

:fire: FLAML supports AutoML and Hyperparameter Tuning in [Microsoft Fabric Data Science](https://learn.microsoft.com/en-us/fabric/data-science/automated-machine-learning-fabric). In addition, we've introduced Python 3.11 and 3.12 support, along with a range of new estimators, and comprehensive integration with MLflow—thanks to contributions from the Microsoft Fabric product team.

:fire: Heads-up: [AutoGen](https://microsoft.github.io/autogen/) has moved to a dedicated [GitHub repository](https://github.com/microsoft/autogen). FLAML no longer includes the `autogen` module—please use AutoGen directly.

## What is FLAML

FLAML is a lightweight Python library for efficient automation of machine
learning and AI operations. It automates workflow based on large language models, machine learning models, etc.
and optimizes their performance.

- FLAML enables economical automation and tuning for ML/AI workflows, including model selection and hyperparameter optimization under resource constraints.
- For common machine learning tasks like classification and regression, it quickly finds quality models for user-provided data with low computational resources. It is easy to customize or extend. Users can find their desired customizability from a smooth range.
- It supports fast and economical automatic tuning (e.g., inference hyperparameters for foundation models, configurations in MLOps/LMOps workflows, pipelines, mathematical/statistical models, algorithms, computing experiments, software configurations), capable of handling large search space with heterogeneous evaluation cost and complex constraints/guidance/early stopping.

FLAML is powered by a series of [research studies](https://microsoft.github.io/FLAML/docs/Research/) from Microsoft Research and collaborators such as Penn State University, Stevens Institute of Technology, University of Washington, and University of Waterloo.

FLAML has a .NET implementation in [ML.NET](http://dot.net/ml), an open-source, cross-platform machine learning framework for .NET.

## Installation

The latest version of FLAML requires **Python >= 3.10 and \< 3.13**. While other Python versions may work for core components, full model support is not guaranteed. FLAML can be installed via `pip`:

```bash
pip install flaml
```

Minimal dependencies are installed without extra options. You can install extra options based on the feature you need. For example, use the following to install the dependencies needed by the [`automl`](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML) module.

```bash
pip install "flaml[automl]"
```

Find more options in [Installation](https://microsoft.github.io/FLAML/docs/Installation).
Each of the [`notebook examples`](https://github.com/microsoft/FLAML/tree/main/notebook) may require a specific option to be installed.

## Quickstart

- With three lines of code, you can start using this economical and fast
  AutoML engine as a [scikit-learn style estimator](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML).

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification")
```

- You can restrict the learners and use FLAML as a fast hyperparameter tuning
  tool for XGBoost, LightGBM, Random Forest etc. or a [customized learner](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#estimator-and-search-space).

```python
automl.fit(X_train, y_train, task="classification", estimator_list=["lgbm"])
```

- You can also run generic hyperparameter tuning for a [custom function](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function).

```python
from flaml import tune

tune.run(
    evaluation_function, config={…}, low_cost_partial_config={…}, time_budget_s=3600
)
```

- [Zero-shot AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Zero-Shot-AutoML) allows using the existing training API from lightgbm, xgboost etc. while getting the benefit of AutoML in choosing high-performance hyperparameter configurations per task.

```python
from flaml.default import LGBMRegressor

# Use LGBMRegressor in the same way as you use lightgbm.LGBMRegressor.
estimator = LGBMRegressor()
# The hyperparameters are automatically set according to the training data.
estimator.fit(X_train, y_train)
```

## Documentation

You can find a detailed documentation about FLAML [here](https://microsoft.github.io/FLAML/).

In addition, you can find:

- [Research](https://microsoft.github.io/FLAML/docs/Research) and [blogposts](https://microsoft.github.io/FLAML/blog) around FLAML.

- [Discord](https://discord.gg/Cppx2vSPVP).

- [Contributing guide](https://microsoft.github.io/FLAML/docs/Contribute).

- ML.NET documentation and tutorials for [Model Builder](https://learn.microsoft.com/dotnet/machine-learning/tutorials/predict-prices-with-model-builder), [ML.NET CLI](https://learn.microsoft.com/dotnet/machine-learning/tutorials/sentiment-analysis-cli), and [AutoML API](https://learn.microsoft.com/dotnet/machine-learning/how-to-guides/how-to-use-the-automl-api).

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Contributors Wall

<a href="https://github.com/microsoft/flaml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/flaml&max=204" />
</a>
