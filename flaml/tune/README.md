# Economical Hyperparameter Optimization

`flaml.tune` is a module for economical hyperparameter tuning. It frees users from manually tuning many hyperparameters for a software, such as machine learning training procedures. 
The API is compatible with ray tune.

Example:

```python
from flaml import tune
import time

def evaluate_config(config):
    '''evaluate a hyperparameter configuration'''
    # we uss a toy example with 2 hyperparameters
    metric = (round(config['x'])-85000)**2 - config['x']/config['y']
    # usually the evaluation takes an non-neglible cost
    # and the cost could be related to certain hyperparameters
    # in this example, we assume it's proportional to x
    time.sleep(config['x']/100000)
    # use tune.report to report the metric to optimize    
    tune.report(metric=metric) 

analysis = tune.run(
    evaluate_config,    # the function to evaluate a config
    config={
        'x': tune.qloguniform(lower=1, upper=100000, q=1),
        'y': tune.randint(lower=1, upper=100000)
    }, # the search space
    init_config={'x':1},    # a initial (partial) config with low cost
    metric='metric',    # the name of the metric used for optimization
    mode='min',         # the optimization mode, 'min' or 'max'
    num_samples=-1,    # the maximal number of configs to try, -1 means infinite
    time_budget_s=60,   # the time budget in seconds
    local_dir='logs/',  # the local directory to store logs
    # verbose=0,          # verbosity    
    # use_ray=True, # uncomment when performing parallel tuning using ray
    )

print(analysis.best_trial.last_result)  # the best trial's result
print(analysis.best_config) # the best config
```

Or, using ray tune's API:
```python
from ray import tune as raytune
from flaml import CFO, BlendSearch
import time

def evaluate_config(config):
    '''evaluate a hyperparameter configuration'''
    # we uss a toy example with 2 hyperparameters
    metric = (round(config['x'])-85000)**2 - config['x']/config['y']
    # usually the evaluation takes an non-neglible cost
    # and the cost could be related to certain hyperparameters
    # in this example, we assume it's proportional to x
    time.sleep(config['x']/100000)
    # use tune.report to report the metric to optimize    
    tune.report(metric=metric) 

analysis = raytune.run(
    evaluate_config,    # the function to evaluate a config
    config={
        'x': tune.qloguniform(lower=1, upper=100000, q=1),
        'y': tune.randint(lower=1, upper=100000)
    }, # the search space
    metric='metric',    # the name of the metric used for optimization
    mode='min',         # the optimization mode, 'min' or 'max'
    num_samples=-1,    # the maximal number of configs to try, -1 means infinite
    time_budget_s=60,   # the time budget in seconds
    local_dir='logs/',  # the local directory to store logs
    search_alg=CFO(points_to_evaluate=[{'x':1}]) # or BlendSearch
    # other algo example: raytune.create_searcher('optuna'),
    )

print(analysis.best_trial.last_result)  # the best trial's result
print(analysis.best_config) # the best config
```

For more examples, please check out 
[notebooks](https://github.com/microsoft/FLAML/tree/main/notebook/).


`flaml` offers two HPO methods: CFO and BlendSearch. 
`flaml.tune` uses BlendSearch by default.

## CFO: Frugal Optimization for Cost-related Hyperparameters

<p>
    <img src="https://github.com/microsoft/FLAML/raw/v0.2.2/docs/images/CFO.png"  width=200>
    <br>
</p>

CFO uses a local search method with adaptive stepsize and random restart. 
It requires a low-cost initial point as input if such point exists.
The search begins with the low-cost initial point and gradually move to
high cost region if needed. The local search method has a provable convergence
rate and bounded cost. 

Example:

```python
from flaml import CFO
tune.run(...
    search_alg = CFO(points_to_evaluate=[init_config]),
)
```

Recommended scenario: there exist cost-related hyperparameters and a low-cost
initial point is known before optimization. 
If the search space is complex and CFO gets trapped into local optima, consider
using BlendSearch. 

## BlendSearch: Economical Hyperparameter Optimization With Blended Search Strategy

<p>
    <img src="https://github.com/microsoft/FLAML/raw/v0.2.2/docs/images/BlendSearch.png"  width=200>
    <br>
</p>

BlendSearch combines local search with global search. It leverages the frugality
of CFO and the space exploration ability of global search methods such as
Bayesian optimization. Like CFO, BlendSearch requires a low-cost initial point
as input if such point exists, and starts the search from there. Different from
CFO, BlendSearch will not wait for the local search to fully converge before 
trying new start points. The new start points are suggested by the global search
method and filtered based on their distance to the existing points in the
cost-related dimensions. BlendSearch still gradually increases the trial cost.
It prioritizes among the global search thread and multiple local search threads
based on optimism in face of uncertainty.

Example:

```python
from flaml import BlendSearch
tune.run(...
    search_alg = BlendSearch(points_to_evaluate=[init_config]),
)
```

Recommended scenario: cost-related hyperparameters exist, a low-cost
initial point is known, and the search space is complex such that local search
is prone to be stuck at local optima.

For more technical details, please check our papers.

* [Frugal Optimization for Cost-related Hyperparameters](https://arxiv.org/abs/2005.01571). Qingyun Wu, Chi Wang, Silu Huang. To appear in AAAI 2021.

```
@inproceedings{wu2021cfo,
    title={Frugal Optimization for Cost-related Hyperparameters},
    author={Qingyun Wu and Chi Wang and Silu Huang},
    year={2021},
    booktitle={AAAI'21},
}
```

* Economical Hyperparameter Optimization With Blended Search Strategy. Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. To appear in ICLR 2021.

```
@inproceedings{wang2021blendsearch,
    title={Economical Hyperparameter Optimization With Blended Search Strategy},
    author={Chi Wang and Qingyun Wu and Silu Huang and Amin Saied},
    year={2021},
    booktitle={ICLR'21},
}
```