# FLAML - Fast and Lightweight AutoML

FLAML is a Python library designed to automatically produce accurate machine
learning models with low computational cost. It frees users from selecting
learners and hyperparameters for each learner. It is fast, cheap and scalable.
It scales to large datasets and has robust performance across different tasks.
The simple and lightweight design makes it easy to use and extend, such as
adding new learners. FLAML is powered by a new, cost-effective hyperparameter
optimization and learner selection method invented by Microsoft Research.
FLAML can:

1. serve as an economical and fast AutoML engine and a drop-in replacement of
a scikit-learn style estimator,

2. be used as a fast hyperparameter tuning tool for XGBoost, LightGBM,
Random Forest etc. or a customized learner, or

3. be embedded in self-tuning software for _just-in-time_ tuning with low
latency & resource consumption.

Example:

```python
from flaml import AutoML

# Initialize the FLAML learner.
automl = AutoML()

# Provide configurations for the learner.
settings = {
    "time_budget": 60,
    "metric": 'accuracy',
    "objective_name": 'classification',
    "log_file_name": 'mylog.log',
}

# Train with labeled input data.
automl.fit(X_train = X_train, y_train = y_train, **settings)

# Export the learned model.
model = automl.model
```

## Installation

FLAML requires **Python version >= 3.6**. It can be installed from pip:

```bash
pip install flaml
```

To run the [`notebook example`](/notebook),
install flaml with the [notebook] option:

```bash
pip install flaml[notebook]
```

## Examples

A basic classification example.

```python
from flaml import AutoML
from sklearn.datasets import load_iris
automl_experiment = AutoML()
automl_settings = {
    "time_budget": 10,
    "metric": 'accuracy',
    "objective_name": 'classification',
    "log_file_name": "test/iris.log",
    "model_history": True
}
X_train, y_train = load_iris(return_X_y=True)
automl_experiment.fit(X_train=X_train, y_train=y_train,
                        **automl_settings)
print(automl_experiment.predict_proba(X_train))
print(automl_experiment.model)
print(automl_experiment.config_history)
print(automl_experiment.model_history)
```

A basic regression example.

```python
from flaml import AutoML
from sklearn.datasets import load_boston
automl_experiment = AutoML()
automl_settings = {
    "time_budget": 10,
    "metric": 'r2',
    "objective_name": 'regression',
    "log_file_name": "test/boston.log",
    "model_history": True
}
X_train, y_train = load_boston(return_X_y=True)
automl_experiment.fit(X_train=X_train, y_train=y_train,
                        **automl_settings)
print(automl_experiment.predict(X_train))
print(automl_experiment.model)
print(automl_experiment.config_history)
print(automl_experiment.model_history)
```

More examples: see the [notebook](notebook/flaml_demo.ipynb)

## Documentation

The API documentation is [`here`]().

<!-- You can also read about FLAML in our blog post [`here`](). -->

For more technical details, please check our papers:

* Qingyun Wu, Chi Wang, Silu Huang, 
[Cost Effective Optimization for Cost-related Hyperparameters](https://arxiv.org/abs/2005.01571), arXiv pre-print.

* Chi Wang, Qingyun Wu, 
[FLO: Fast and Lightweight Hyperparameter Optimization for AutoML](https://arxiv.org/abs/1911.04706), arXiv pre-print.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

<!-- # Community

Join our community! 

For more formal enquiries, you can [`contact us`](). -->

## Authors

* Chi Wang
* Qingyun Wu
* Erkang Zhu

## License

[MIT License](LICENSE)
