
### AutoML with minimal customization
* Tasks:
    - Classification: `classification`
    - Regression: `regression`
    - Time series forecasting: `ts_forecast`
    - Rank: `rank`

### Save and load model

### Feature importance

### Training log

1. `iter_per_learner` means how many models have been tried for each learner. The reason you see records like `iter_per_learner=13` for `record_id=2` is that flaml only logs better configs than the previous iters by default, i.e., `log_type='better'`. 
2. `trial_time` means the time taken to train and evaluate one config in that trial. `total_search_time` is the total time spent from the beginning of `fit()`.
3. Because `log_type='better'` by default, `config` is equal to `best_config`. If you use `log_type='all'` instead, all the trials will be logged. And then `config` corresponds to the config in that iteration, and `best_config` is the best config so far.
4. flaml will adjust the `n_estimators` for lightgbm etc. according to the remaining budget and check the time budget constraint and stop in several places. Most of the time that makes `fit()` stops before the given budget. Occasionally it may run over the time budget slightly. But the log file always contains the best config info and you can recover the best model until any time point using `retrain_from_log()`.

### How to set time budget?

* If you have an exact constraint for the total search time, set it as the time budget.
* If you have flexible time constraints, for example, your desirable time budget is t1=60s, and the longest time budget you can tolerate is t2=3600s, you can try the following two ways:
1. set t1 as the time budget, and check the message in the console log in the end. If the budget is too small, you will see a warning like 
> WARNING - Time taken to find the best model is 91% of the provided time budget and not all estimators' hyperparameter search converged. Consider increasing the time budget.
2. set t2 as the time budget, and also set `early_stop=True`. If the early stopping is triggered, you will see a warning like
> WARNING - All estimator hyperparameters local search has converged at least once, and the total search time exceeds 10 times the time taken to find the best model.

> WARNING - Stopping search as early_stop is set to True.

### How long is required to find the best model?
If you want to get a sense of how long is required to find the best model, you can use `max_iter=1` and `retrain_full=False` to perform one trial first. The message will be like:
> INFO - iteration 0, current learner lgbm

> INFO - Estimated sufficient time budget=145194s. Estimated necessary time budget=2118s.

> INFO -  at 2.6s,  estimator lgbm's best error=0.4459,     best estimator lgbm's best error=0.4459

You will see that the time to finish the first and cheapest trial is 2.6 seconds. The estimated necessary time budget is 2118 seconds, and the estimated sufficient time budget is 145194 seconds. Note that this is only an estimated range to help you decide your budget.
