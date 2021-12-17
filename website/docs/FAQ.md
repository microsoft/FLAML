# Frequently Asked Questions

### About `low_cost_partial_config` in `tune`.

- Definition and purpose: The `low_cost_partial_config` is a dictionary of subset of the hyperparameter coordinates whose value corresponds to a configuration with known low-cost (i.e., low computation cost for training the corresponding model).  The concept of low/high-cost is meaningful in the case where a subset of the hyperparameters to tune directly affects the computation cost for training the model. For example, `n_estimators` and `max_leaves` are known to affect the training cost of tree-based learners. We call this subset of hyperparameters, *cost-related hyperparameters*. In such scenarios, if you are aware of low-cost configurations for the cost-related hyperparameters, you are recommended to set them as the `low_cost_partial_config`. Using the tree-based method example again, since we know that small `n_estimators` and  `max_leaves` generally correspond to simpler models and thus lower cost, we set `{'n_estimators': 4, 'max_leaves': 4}` as the `low_cost_partial_config` by default (note that `4` is the lower bound of search space for these two hyperparameters), e.g., in [LGBM](https://github.com/microsoft/FLAML/blob/main/flaml/model.py#L215).  Configuring `low_cost_partial_config` helps the search algorithms make more cost-efficient choices.  
In AutoML, the `low_cost_init_value` in `search_space()` function for each estimator serves the same role.

- Usage in practice: It is recommended to configure it if there are cost-related hyperparameters in your tuning task and you happen to know the low-cost values for them, but it is not required( It is fine to leave it the default value, i.e., `None`).

- How does it work: `low_cost_partial_config` if configured, will be used as an initial point of the search. It also affects the search trajectory. For more details about how does it play a role in the search algorithms, please refer to the papers about the search algorithms used: Section 2 of [Frugal Optimization for Cost-related Hyperparameters (CFO)](https://arxiv.org/pdf/2005.01571.pdf) and Section 3 of [Economical Hyperparameter Optimization with Blended Search Strategy (BlendSearch)](https://openreview.net/pdf?id=VbLH04pRA3).


### How does FLAML handle imbalanced data (unequal distribution of target classes in classification task)?

Currently FLAML does several things for imbalanced data.

1. When a class contains fewer than 20 examples, we repeatedly add these examples to the training data until the count is at least 20.
2. We use stratified sampling when doing holdout and kf.
3. We make sure no class is empty in both training and holdout data.
4. We allow users to pass `sample_weight` to `AutoML.fit()`.


### How to interpret model performance? Is it possible for me to visualize feature importance, SHAP values, optimization history?

You can use ```automl.model.estimator.feature_importances_``` to get the `feature_importances_` for the best model found by automl. See an [example](Examples/AutoML-for-XGBoost#plot-feature-importance).

Packages such as `azureml-interpret` and `sklearn.inspection.permutation_importance` can be used on `automl.model.estimator` to explain the selected model.
Model explanation is frequently asked and adding a native support may be a good feature. Suggestions/contributions are welcome.

Optimization history can be checked from the [log](Use-Cases/Task-Oriented-AutoML#log-the-trials). You can also [retrieve the log and plot the learning curve](Use-Cases/Task-Oriented-AutoML#plot-learning-curve).
