### A basic classification example

```python
from flaml import AutoML
from sklearn.datasets import load_iris
# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 1,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "iris.log",
}
X_train, y_train = load_iris(return_X_y=True)
# Train with labeled input data
automl.fit(X_train=X_train, y_train=y_train,
           **automl_settings)
# Predict
print(automl.predict_proba(X_train))
# Print the best model
print(automl.model.estimator)
```

### Sample of output
```
[flaml.automl: 11-12 18:21:44] {1485} INFO - Data split method: stratified
[flaml.automl: 11-12 18:21:44] {1489} INFO - Evaluation method: cv
[flaml.automl: 11-12 18:21:44] {1540} INFO - Minimizing error metric: 1-accuracy
[flaml.automl: 11-12 18:21:44] {1577} INFO - List of ML learners in AutoML Run: ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree', 'lrl1']
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 0, current learner lgbm
[flaml.automl: 11-12 18:21:44] {1944} INFO - Estimated sufficient time budget=1285s. Estimated necessary time budget=23s.
[flaml.automl: 11-12 18:21:44] {2029} INFO -  at 0.2s,	estimator lgbm's best error=0.0733,	best estimator lgbm's best error=0.0733
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 1, current learner lgbm
[flaml.automl: 11-12 18:21:44] {2029} INFO -  at 0.3s,	estimator lgbm's best error=0.0733,	best estimator lgbm's best error=0.0733
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 2, current learner lgbm
[flaml.automl: 11-12 18:21:44] {2029} INFO -  at 0.4s,	estimator lgbm's best error=0.0533,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 3, current learner lgbm
[flaml.automl: 11-12 18:21:44] {2029} INFO -  at 0.6s,	estimator lgbm's best error=0.0533,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 4, current learner lgbm
[flaml.automl: 11-12 18:21:44] {2029} INFO -  at 0.6s,	estimator lgbm's best error=0.0533,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:44] {1826} INFO - iteration 5, current learner xgboost
[flaml.automl: 11-12 18:21:45] {2029} INFO -  at 0.9s,	estimator xgboost's best error=0.0600,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:45] {1826} INFO - iteration 6, current learner lgbm
[flaml.automl: 11-12 18:21:45] {2029} INFO -  at 1.0s,	estimator lgbm's best error=0.0533,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:45] {1826} INFO - iteration 7, current learner extra_tree
[flaml.automl: 11-12 18:21:45] {2029} INFO -  at 1.1s,	estimator extra_tree's best error=0.0667,	best estimator lgbm's best error=0.0533
[flaml.automl: 11-12 18:21:45] {2242} INFO - retrain lgbm for 0.0s
[flaml.automl: 11-12 18:21:45] {2247} INFO - retrained model: LGBMClassifier(learning_rate=0.2677050123105203, max_bin=127,
               min_child_samples=12, n_estimators=4, num_leaves=4,
               reg_alpha=0.001348364934537134, reg_lambda=1.4442580148221913,
               verbose=-1)
[flaml.automl: 11-12 18:21:45] {1608} INFO - fit succeeded
[flaml.automl: 11-12 18:21:45] {1610} INFO - Time taken to find the best model: 0.3756711483001709
```

### Log of trials

Content of "iris.log":
```
{"record_id": 0, "iter_per_learner": 1, "logged_metric": null, "trial_time": 0.12717914581298828, "wall_clock_time": 0.1728971004486084, "validation_loss": 0.07333333333333332, "config": {"n_estimators": 4, "num_leaves": 4, "min_child_samples": 20, "learning_rate": 0.09999999999999995, "log_max_bin": 8, "colsample_bytree": 1.0, "reg_alpha": 0.0009765625, "reg_lambda": 1.0}, "learner": "lgbm", "sample_size": 150}
{"record_id": 1, "iter_per_learner": 3, "logged_metric": null, "trial_time": 0.07027268409729004, "wall_clock_time": 0.3756711483001709, "validation_loss": 0.05333333333333332, "config": {"n_estimators": 4, "num_leaves": 4, "min_child_samples": 12, "learning_rate": 0.2677050123105203, "log_max_bin": 7, "colsample_bytree": 1.0, "reg_alpha": 0.001348364934537134, "reg_lambda": 1.4442580148221913}, "learner": "lgbm", "sample_size": 150}
{"curr_best_record_id": 1}
```

1. `iter_per_learner` means how many models have been tried for each learner. The reason you see records like `iter_per_learner=3` for `record_id=1` is that flaml only logs better configs than the previous iters by default, i.e., `log_type='better'`. If you use `log_type='all'` instead, all the trials will be logged.
1. `trial_time` means the time taken to train and evaluate one config in that trial. `total_search_time` is the total time spent from the beginning of `fit()`.
1. flaml will adjust the `n_estimators` for lightgbm etc. according to the remaining budget and check the time budget constraint and stop in several places. Most of the time that makes `fit()` stops before the given budget. Occasionally it may run over the time budget slightly. But the log file always contains the best config info and you can recover the best model until any time point using `retrain_from_log()`.