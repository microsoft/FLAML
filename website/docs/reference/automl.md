---
sidebar_label: automl
title: automl
---

#### size

```python
def size(state: AutoMLState, config: dict) -> float
```

Size function

**Returns**:

  The mem size in bytes for a config

## AutoML Objects

```python
class AutoML()
```

The AutoML class.

**Example**:

  
```python  
automl = AutoML()
automl_settings = {
    "time_budget": 60,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": 'mylog.log',
}
automl.fit(X_train = X_train, y_train = y_train,
    **automl_settings)
```

#### \_\_init\_\_

```python
def __init__(**settings)
```

Constructor.

Many settings in fit() can be passed to the constructor too.
If an argument in fit() is provided, it will override the setting passed to the constructor.
If an argument in fit() is not provided but provided in the constructor, the value passed to the constructor will be used.

**Arguments**:

- `metric` - A string of the metric name or a function,
  e.g., 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
  'f1', 'micro_f1', 'macro_f1', 'log_loss', 'mae', 'mse', 'r2',
  'mape'. Default is 'auto'.
  If passing a customized metric function, the function needs to
  have the follwing signature:
```python
def custom_metric(
    X_test, y_test, estimator, labels,
    X_train, y_train, weight_test=None, weight_train=None,
    config=None, groups_test=None, groups_train=None,
):
    return metric_to_minimize, metrics_to_log
```
  
  which returns a float number as the minimization objective,
  and a dictionary as the metrics to log.
- `task` - A string of the task type, e.g.,
  'classification', 'regression', 'ts_forecast', 'rank',
  'seq-classification', 'seq-regression'.
- `n_jobs` - An integer of the number of threads for training.
- `gpu_per_trial` - A float of the number of gpus per trial, only used by TransformersEstimator.
- `log_file_name` - A string of the log file name. To disable logging,
  set it to be an empty string "".
- `estimator_list` - A list of strings for estimator names, or 'auto'
  e.g., ```['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']```
  
- `time_budget` - A float number of the time budget in seconds.
  Use -1 if no time limit.
- `max_iter` - An integer of the maximal number of iterations.
- `sample` - A boolean of whether to sample the training data during
  search.
- `ensemble` - boolean or dict | default=False. Whether to perform
  ensemble after search. Can be a dict with keys 'passthrough'
  and 'final_estimator' to specify the passthrough and
  final_estimator in the stacker.
- `eval_method` - A string of resampling strategy, one of
  ['auto', 'cv', 'holdout'].
- `split_ratio` - A float of the valiation data percentage for holdout.
- `n_splits` - An integer of the number of folds for cross - validation.
- `log_type` - A string of the log type, one of
  ['better', 'all'].
  'better' only logs configs with better loss than previos iters
  'all' logs all the tried configs.
- `model_history` - A boolean of whether to keep the best
  model per estimator. Make sure memory is large enough if setting to True.
- `log_training_metric` - A boolean of whether to log the training
  metric for each model.
- `mem_thres` - A float of the memory size constraint in bytes.
- `pred_time_limit` - A float of the prediction latency constraint in seconds.
- `train_time_limit` - A float of the training time constraint in seconds.
- `verbose` - int, default=3 | Controls the verbosity, higher means more
  messages.
- `retrain_full` - bool or str, default=True | whether to retrain the
  selected model on the full training data when using holdout.
  True - retrain only after search finishes; False - no retraining;
  'budget' - do best effort to retrain without violating the time
  budget.
- `split_type` - str, default="auto" | the data split type.
  For classification tasks, valid choices are [
  "auto", 'stratified', 'uniform', 'time']. "auto" -> stratified.
  For regression tasks, valid choices are ["auto", 'uniform', 'time'].
  "auto" -> uniform.
  For ts_forecast tasks, must be "auto" or 'time'.
  For ranking task, must be "auto" or 'group'.
- `hpo_method` - str, default="auto" | The hyperparameter
  optimization method. By default, CFO is used for sequential
  search and BlendSearch is used for parallel search.
  No need to set when using flaml's default search space or using
  a simple customized search space. When set to 'bs', BlendSearch
  is used. BlendSearch can be tried when the search space is
  complex, for example, containing multiple disjoint, discontinuous
  subspaces. When set to 'random', random search is used.
- `starting_points` - A dictionary to specify the starting hyperparameter
  config for the estimators.
  Keys are the name of the estimators, and values are the starting
  hyperparamter configurations for the corresponding estimators.
  The value can be a single hyperparamter configuration dict or a list
  of hyperparamter configuration dicts.
  In the following code example, we get starting_points from the
  automl_experiment and use them in the new_automl_experiment.
  e.g.,
  
  
```python
from flaml import AutoML
automl_experiment = AutoML()
X_train, y_train = load_iris(return_X_y=True)
automl_experiment.fit(X_train, y_train)
starting_points = automl_experiment.best_config_per_estimator

new_automl_experiment = AutoML()
new_automl_experiment.fit(X_train, y_train,
    starting_points=starting_points)
```
  
- `seed` - int or None, default=None | The random seed for np.random.
- `n_concurrent_trials` - [Experimental] int, default=1 | The number of
  concurrent trials. For n_concurrent_trials > 1, installation of
  ray is required: `pip install flaml[ray]`.
- `keep_search_state` - boolean, default=False | Whether to keep search
  state after fit(). By default the state is deleted for space
  saving.
- `early_stop` - boolean, default=False | Whether to stop early if the
  search is considered to converge.
- `append_log` - boolean, default=False | Whetehr to directly append the log
  records to the input log file if it exists.
- `auto_augment` - boolean, default=True | Whether to automatically
  augment rare classes.
- `min_sample_size` - int, default=MIN_SAMPLE_TRAIN | the minimal sample
  size when sample=True.
- `use_ray` - boolean, default=False | Whether to use ray to run the training
  in separate processes. This can be used to prevent OOM for large
  datasets, but will incur more overhead in time. Only use it if
  you run into OOM failures.

#### config\_history

```python
@property
def config_history()
```

A dictionary of iter->(estimator, config, time),
storing the best estimator, config, and the time when the best
model is updated each time.

#### model

```python
@property
def model()
```

An object with `predict()` and `predict_proba()` method (for
classification), storing the best trained model.

#### best\_model\_for\_estimator

```python
def best_model_for_estimator(estimator_name)
```

Return the best model found for a particular estimator.

**Arguments**:

- `estimator_name` - a str of the estimator's name.
  

**Returns**:

  An object with `predict()` and `predict_proba()` method (for
  classification), storing the best trained model for estimator_name.

#### best\_estimator

```python
@property
def best_estimator()
```

A string indicating the best estimator found.

#### best\_iteration

```python
@property
def best_iteration()
```

An integer of the iteration number where the best
config is found.

#### best\_config

```python
@property
def best_config()
```

A dictionary of the best configuration.

#### best\_config\_per\_estimator

```python
@property
def best_config_per_estimator()
```

A dictionary of all estimators' best configuration.

#### best\_loss\_per\_estimator

```python
@property
def best_loss_per_estimator()
```

A dictionary of all estimators' best loss.

#### best\_loss

```python
@property
def best_loss()
```

A float of the best loss found.

#### best\_config\_train\_time

```python
@property
def best_config_train_time()
```

A float of the seconds taken by training the best config.

#### classes\_

```python
@property
def classes_()
```

A list of n_classes elements for class labels.

#### time\_to\_find\_best\_model

```python
@property
def time_to_find_best_model() -> float
```

Time taken to find best model in seconds.

#### predict

```python
def predict(X_test: Union[np.array, DataFrame, List[str], List[List[str]]])
```

Predict label from features.

**Arguments**:

- `X_test` - A numpy array of featurized instances, shape n * m,
  or for 'ts_forecast' task:
  a pandas dataframe with the first column containing
  timestamp values (datetime type) or an integer n for
  the predict steps (only valid when the estimator is
  arima or sarimax). Other columns in the dataframe
  are assumed to be exogenous variables (categorical
  or numeric).
  
```python                    
multivariate_X_test = pd.DataFrame({
    'timeStamp': pd.date_range(start='1/1/2022', end='1/07/2022'),
    'categorical_col': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes'],
    'continuous_col': [105, 107, 120, 118, 110, 112, 115]
})
model.predict(multivariate_X_test)
```
  

**Returns**:

  A array-like of shape n * 1 - - each element is a predicted
  label for an instance.

#### predict\_proba

```python
def predict_proba(X_test)
```

Predict the probability of each class from features, only works for
classification problems.

**Arguments**:

- `X_test` - A numpy array of featurized instances, shape n * m.
  

**Returns**:

  A numpy array of shape n * c. c is the  # classes. Each element at
  (i, j) is the probability for instance i to be in class j.

#### add\_learner

```python
def add_learner(learner_name, learner_class)
```

Add a customized learner.

**Arguments**:

- `learner_name` - A string of the learner's name.
- `learner_class` - A subclass of flaml.model.BaseEstimator.

#### get\_estimator\_from\_log

```python
def get_estimator_from_log(log_file_name, record_id, task)
```

Get the estimator from log file.

**Arguments**:

- `log_file_name` - A string of the log file name.
- `record_id` - An integer of the record ID in the file,
  0 corresponds to the first trial.
- `task` - A string of the task type,
  'binary', 'multi', 'regression', 'ts_forecast', 'rank'.
  

**Returns**:

  An estimator object for the given configuration.

#### retrain\_from\_log

```python
def retrain_from_log(log_file_name, X_train=None, y_train=None, dataframe=None, label=None, time_budget=np.inf, task=None, eval_method=None, split_ratio=None, n_splits=None, split_type=None, groups=None, n_jobs=-1, gpu_per_trial=0, train_best=True, train_full=False, record_id=-1, auto_augment=None, **fit_kwargs, ,)
```

Retrain from log file.

**Arguments**:

- `log_file_name` - A string of the log file name.
- `X_train` - A numpy array or dataframe of training data in shape n*m.
  For 'ts_forecast' task, the first column of X_train
  must be the timestamp column (datetime type). Other
  columns in the dataframe are assumed to be exogenous
  variables (categorical or numeric).
- `y_train` - A numpy array or series of labels in shape n*1.
- `dataframe` - A dataframe of training data including label column.
  For 'ts_forecast' task, dataframe must be specified and should
  have at least two columns: timestamp and label, where the first
  column is the timestamp column (datetime type). Other columns
  in the dataframe are assumed to be exogenous variables
  (categorical or numeric).
- `label` - A str of the label column name, e.g., 'label';
- `Note` - If X_train and y_train are provided,
  dataframe and label are ignored;
  If not, dataframe and label must be provided.
- `time_budget` - A float number of the time budget in seconds.
- `task` - A string of the task type, e.g.,
  'classification', 'regression', 'ts_forecast', 'rank',
  'seq-classification', 'seq-regression'.
- `eval_method` - A string of resampling strategy, one of
  ['auto', 'cv', 'holdout'].
- `split_ratio` - A float of the validation data percentage for holdout.
- `n_splits` - An integer of the number of folds for cross-validation.
- `split_type` - str, default="auto" | the data split type.
  For classification tasks, valid choices are [
  "auto", 'stratified', 'uniform', 'time', 'group']. "auto" -> stratified.
  For regression tasks, valid choices are ["auto", 'uniform', 'time'].
  "auto" -> uniform.
  For ts_forecast tasks, must be "auto" or 'time'.
  For ranking task, must be "auto" or 'group'.
- `groups` - None or array-like | Group labels (with matching length to
  y_train) or groups counts (with sum equal to length of y_train)
  for training data.
- `n_jobs` - An integer of the number of threads for training. Use all
  available resources when n_jobs == -1.
- `gpu_per_trial` - A float of the number of gpus per trial. Only used by TransformersEstimator.
- `train_best` - A boolean of whether to train the best config in the
  time budget; if false, train the last config in the budget.
- `train_full` - A boolean of whether to train on the full data. If true,
  eval_method and sample_size in the log file will be ignored.
- `record_id` - the ID of the training log record from which the model will
  be retrained. By default `record_id = -1` which means this will be
  ignored. `record_id = 0` corresponds to the first trial, and
  when `record_id >= 0`, `time_budget` will be ignored.
- `auto_augment` - boolean, default=True | Whether to automatically
  augment rare classes.
- `**fit_kwargs` - Other key word arguments to pass to fit() function of
  the searched learners, such as sample_weight.

#### search\_space

```python
@property
def search_space() -> dict
```

Search space.

Must be called after fit(...)
(use max_iter=0 and retrain_final=False to prevent actual fitting).

**Returns**:

  A dict of the search space.

#### low\_cost\_partial\_config

```python
@property
def low_cost_partial_config() -> dict
```

Low cost partial config.

**Returns**:

  A dict.
  (a) if there is only one estimator in estimator_list, each key is a
  hyperparameter name.
  (b) otherwise, it is a nested dict with 'ml' as the key, and
  a list of the low_cost_partial_configs as the value, corresponding
  to each learner's low_cost_partial_config; the estimator index as
  an integer corresponding to the cheapest learner is appended to the
  list at the end.

#### cat\_hp\_cost

```python
@property
def cat_hp_cost() -> dict
```

Categorical hyperparameter cost

**Returns**:

  A dict.
  (a) if there is only one estimator in estimator_list, each key is a
  hyperparameter name.
  (b) otherwise, it is a nested dict with 'ml' as the key, and
  a list of the cat_hp_cost's as the value, corresponding
  to each learner's cat_hp_cost; the cost relative to lgbm for each
  learner (as a list itself) is appended to the list at the end.

#### points\_to\_evaluate

```python
@property
def points_to_evaluate() -> dict
```

Initial points to evaluate

**Returns**:

  A list of dicts. Each dict is the initial point for each learner

#### prune\_attr

```python
@property
def prune_attr() -> Optional[str]
```

Attribute for pruning

**Returns**:

  A string for the sample size attribute or None

#### min\_resource

```python
@property
def min_resource() -> Optional[float]
```

Attribute for pruning.

**Returns**:

  A float for the minimal sample size or None.

#### max\_resource

```python
@property
def max_resource() -> Optional[float]
```

Attribute for pruning.

**Returns**:

  A float for the maximal sample size or None.

#### trainable

```python
@property
def trainable() -> Callable[[dict], Optional[float]]
```

Training function.

**Returns**:

  A function that evaluates each config and returns the loss.

#### metric\_constraints

```python
@property
def metric_constraints() -> list
```

Metric constraints.

**Returns**:

  A list of the metric constraints.

#### fit

```python
def fit(X_train=None, y_train=None, dataframe=None, label=None, metric=None, task=None, n_jobs=None, gpu_per_trial=0, log_file_name=None, estimator_list=None, time_budget=None, max_iter=None, sample=None, ensemble=None, eval_method=None, log_type=None, model_history=None, split_ratio=None, n_splits=None, log_training_metric=None, mem_thres=None, pred_time_limit=None, train_time_limit=None, X_val=None, y_val=None, sample_weight_val=None, groups_val=None, groups=None, verbose=None, retrain_full=None, split_type=None, learner_selector=None, hpo_method=None, starting_points=None, seed=None, n_concurrent_trials=None, keep_search_state=None, early_stop=None, append_log=None, auto_augment=None, min_sample_size=None, use_ray=None, **fit_kwargs, ,)
```

Find a model for a given task.

**Arguments**:

- `X_train` - A numpy array or a pandas dataframe of training data in
  shape (n, m). For 'ts_forecast' task, the first column of X_train
  must be the timestamp column (datetime type). Other columns in
  the dataframe are assumed to be exogenous variables (categorical or numeric).
- `y_train` - A numpy array or a pandas series of labels in shape (n, ).
- `dataframe` - A dataframe of training data including label column.
  For 'ts_forecast' task, dataframe must be specified and must have
  at least two columns, timestamp and label, where the first
  column is the timestamp column (datetime type). Other columns in
  the dataframe are assumed to be exogenous variables (categorical or numeric).
- `label` - A str of the label column name for, e.g., 'label';
- `Note` - If X_train and y_train are provided,
  dataframe and label are ignored;
  If not, dataframe and label must be provided.
- `metric` - A string of the metric name or a function,
  e.g., 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
  'f1', 'micro_f1', 'macro_f1', 'log_loss', 'mae', 'mse', 'r2',
  'mape'. Default is 'auto'.
  If passing a customized metric function, the function needs to
  have the follwing signature:
  
```python
def custom_metric(
    X_test, y_test, estimator, labels,
    X_train, y_train, weight_test=None, weight_train=None,
    config=None, groups_test=None, groups_train=None,
):
    return metric_to_minimize, metrics_to_log
```
  
  which returns a float number as the minimization objective,
  and a dictionary as the metrics to log.
- `task` - A string of the task type, e.g.,
  'classification', 'regression', 'ts_forecast', 'rank',
  'seq-classification', 'seq-regression'.
- `n_jobs` - An integer of the number of threads for training.
- `gpu_per_trial` - A float of the number of gpus per trial, only used by TransformersEstimator.
- `log_file_name` - A string of the log file name. To disable logging,
  set it to be an empty string "".
- `estimator_list` - A list of strings for estimator names, or 'auto'
  e.g., ```['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']```
  
- `time_budget` - A float number of the time budget in seconds.
  Use -1 if no time limit.
- `max_iter` - An integer of the maximal number of iterations.
- `sample` - A boolean of whether to sample the training data during
  search.
- `ensemble` - boolean or dict | default=False. Whether to perform
  ensemble after search. Can be a dict with keys 'passthrough'
  and 'final_estimator' to specify the passthrough and
  final_estimator in the stacker.
- `eval_method` - A string of resampling strategy, one of
  ['auto', 'cv', 'holdout'].
- `split_ratio` - A float of the valiation data percentage for holdout.
- `n_splits` - An integer of the number of folds for cross - validation.
- `log_type` - A string of the log type, one of
  ['better', 'all'].
  'better' only logs configs with better loss than previos iters
  'all' logs all the tried configs.
- `model_history` - A boolean of whether to keep the best
  model per estimator. Make sure memory is large enough if setting to True.
- `log_training_metric` - A boolean of whether to log the training
  metric for each model.
- `mem_thres` - A float of the memory size constraint in bytes.
- `pred_time_limit` - A float of the prediction latency constraint in seconds.
- `train_time_limit` - A float of the training time constraint in seconds.
- `X_val` - None or a numpy array or a pandas dataframe of validation data.
- `y_val` - None or a numpy array or a pandas series of validation labels.
- `sample_weight_val` - None or a numpy array of the sample weight of
  validation data of the same shape as y_val.
- `groups_val` - None or array-like | group labels (with matching length
  to y_val) or group counts (with sum equal to length of y_val)
  for validation data. Need to be consistent with groups.
- `groups` - None or array-like | Group labels (with matching length to
  y_train) or groups counts (with sum equal to length of y_train)
  for training data.
- `verbose` - int, default=3 | Controls the verbosity, higher means more
  messages.
- `retrain_full` - bool or str, default=True | whether to retrain the
  selected model on the full training data when using holdout.
  True - retrain only after search finishes; False - no retraining;
  'budget' - do best effort to retrain without violating the time
  budget.
- `split_type` - str, default="auto" | the data split type.
  For classification tasks, valid choices are [
  "auto", 'stratified', 'uniform', 'time']. "auto" -> stratified.
  For regression tasks, valid choices are ["auto", 'uniform', 'time'].
  "auto" -> uniform.
  For ts_forecast tasks, must be "auto" or 'time'.
  For ranking task, must be "auto" or 'group'.
- `hpo_method` - str, default="auto" | The hyperparameter
  optimization method. By default, CFO is used for sequential
  search and BlendSearch is used for parallel search.
  No need to set when using flaml's default search space or using
  a simple customized search space. When set to 'bs', BlendSearch
  is used. BlendSearch can be tried when the search space is
  complex, for example, containing multiple disjoint, discontinuous
  subspaces. When set to 'random', random search is used.
- `starting_points` - A dictionary to specify the starting hyperparameter
  config for the estimators.
  Keys are the name of the estimators, and values are the starting
  hyperparamter configurations for the corresponding estimators.
  The value can be a single hyperparamter configuration dict or a list
  of hyperparamter configuration dicts.
  In the following code example, we get starting_points from the
  automl_experiment and use them in the new_automl_experiment.
  e.g.,
  
```python
from flaml import AutoML
automl_experiment = AutoML()
X_train, y_train = load_iris(return_X_y=True)
automl_experiment.fit(X_train, y_train)
starting_points = automl_experiment.best_config_per_estimator

new_automl_experiment = AutoML()
new_automl_experiment.fit(X_train, y_train,
    starting_points=starting_points)
```
  
- `seed` - int or None, default=None | The random seed for np.random.
- `n_concurrent_trials` - [Experimental] int, default=1 | The number of
  concurrent trials. For n_concurrent_trials > 1, installation of
  ray is required: `pip install flaml[ray]`.
- `keep_search_state` - boolean, default=False | Whether to keep search
  state after fit(). By default the state is deleted for space
  saving.
- `early_stop` - boolean, default=False | Whether to stop early if the
  search is considered to converge.
- `append_log` - boolean, default=False | Whetehr to directly append the log
  records to the input log file if it exists.
- `auto_augment` - boolean, default=True | Whether to automatically
  augment rare classes.
- `min_sample_size` - int, default=MIN_SAMPLE_TRAIN | the minimal sample
  size when sample=True.
- `use_ray` - boolean, default=False | Whether to use ray to run the training
  in separate processes. This can be used to prevent OOM for large
  datasets, but will incur more overhead in time. Only use it if
  you run into OOM failures.
- `**fit_kwargs` - Other key word arguments to pass to fit() function of
  the searched learners, such as sample_weight. Include period as
  a key word argument for 'ts_forecast' task.

