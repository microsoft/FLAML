# AutoML and Hyperparamter Tuning with FLAML Troubleshooting Guide

### What is FLAML

[FLAML](https://github.com/microsoft/FLAML) is a lightweight Python library for efficient automation of machine learning and AI operations, including selection of models, hyperparameters, and other tunable choices of an application (e.g., inference hyperparameters for foundation models, configurations in MLOps/LMOps workflows, pipelines, mathematical/statistical models, algorithms, computing experiments, software configurations).

On Data Science in Microsoft Fabric, we provide AutoML and Hyperparameter Tuning experiences based on an internal version of [FLAML](https://msdata.visualstudio.com/A365/_git/FLAML-Internal) which is fully compatible with all OSS FLAML functionalities. In addition, we have improved integration with MLFlow and PySpark to enhance the overall user experience.

### Best Practice

- Not all trials' models are logged when `model_history` is True

  Starting from `v2.3.5.post4` (Fabric internal version), FLAML logs models (if `model_history` is True) only when the loss improves by default.

  You can set `log_type` to `all` and `model_history` to `True` to log all models. However, it will significantly slow down the training process.

  ```
  settings = {
      "time_budget": 30,           # Total running time in seconds
      "max_iter": 3,               # Number of trials
      ......
      "model_history": True,       # Keep the model history or not, default True
      "log_type": "better",        # Log all models or only better ones, default "better". Set to "all" to log all models (not recommended).
  }

  automl.fit(dataframe, label="xx", **settings)
  ```

### Known Limitations

- Trials take more time when log model files

  MLFlow with `log_models` to be `True` takes more time than with `log_models` to be `False`.

  MLFlow `log_models` is `True` by default, which adds a few seconds of overhead to each trial. For a training process with a lot of trials, you may want to log only the parameters and metrics to accelerate the training process and save some storage space as well. You can retrain the model with the best parameters once the AutoML process is finished.

  To remove this overhead, you can set `model_history` to `False` in flaml's settings:

  ```
  settings = {
      "time_budget": 30,           # Total running time in seconds
      "max_iter": 3,               # Number of trials
      ......
      "model_history": False,      # Keep the model history or not, default True
  }

  automl.fit(dataframe, label="xx", **settings)
  ```

- `time_budget` and `force_cancel` can't precisely control the training time

  With below code, you want to run the training for no more than 30 seconds, but the training cell could run longer than that.

  ```
  # Create an AutoML instance
  automl = AutoML()

  # Define settings
  settings = {
      "time_budget": 30,           # Total running time in seconds
      "use_spark": True,           # Enable Spark-based parallelism
      "n_concurrent_trials": 3,    # Number of concurrent trials to run
      "force_cancel": True,        # Force stop training once time_budget is used up
      "verbose": 1,
  }

  '''The main flaml automl API'''
  with mlflow.start_run(nested=True, run_name = "parallel_trial"):
      automl.fit(dataframe=pandas_df, label='Exited', **settings)
  ```

  The root cause is that there is some latency in the MLflow logging process which is not counted into the time budget.

  To see more details of the training process, set `verbose` to a value greater than `3`.
  To better control the training process, set `max_iter`.
  To reduce the latency, consider not logging the model files, check section "**Trials take more time when log model files**".
  To turn off MLflow logging, set `mlflow_logging` to `False`.
  For example:

  ```
  settings = {
      "time_budget": 30,           # Total running time in seconds
      "max_iter": 10,              # Number of trials
      "use_spark": True,           # Enable Spark-based parallelism
      "n_concurrent_trials": 3,    # Number of concurrent trials to run
      "force_cancel": True,        # Force stop training once time_budget is used up
      "verbose": 4,                # log to show, higher for more details
      # "model_history": False,    # Keep the model history or not, default True
      # "mlflow_logging": False,   # Use mlflow logging or not, default True
  }
  ```

<!--
- Nested runs will appear as individual runs.

    Hierarchical view is not supported in UI side yet, so nested runs will appear as individual runs.

- Trials take more time with autologging enabled.

    MLFlow autologging is enabled by default, which adds a few seconds of overhead to each trial. To remove this overhead, you just need to disable autologging by inserting the following code before proceeding to train:

    ```
    mlflow.autolog(disable=True)
    ```
-->

- Running multiple AutoML/Tuning trainings with the same experiment name at the same time can result in mixed parent and child runs.

  For example, if two notebooks with different training functions are running simultaneously and both are using the same experiment name for MLflow logging, it may become difficult to determine which MLflow run belongs to which notebook's training trial.

  *Since the mlflow experiment is a workspace-level artifact, it means that two experiments with the same name but from different workspaces are considered as distinct experiments. Therefore, there is no need to worry about this issue when running notebooks in different workspaces.*

  *The issue has been resolved with [mlflow PR #9114](https://github.com/mlflow/mlflow/pull/9114), update mlflow to version > 2.5.0 would fix it.*

<!-- - Each AutoML/Tuning training must not exceed 5000 trials. Beyond this number, there may be issues with some of the runs. Modify SEARCH_MAX_RESULTS in _mlflow.py to update the limits, but mlflow search runs will be very slow if the value is too large. -->

<!-- Resolved -->

<!-- - Long-Running AutoML/Tuning trial throws Error.

    Long-running trials (>30 mins) may throw below error:
    ```
    Py4JJavaError: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServeWithJobGroup.
    : org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 74.0 failed 4 times, most recent failure: Lost task 0.3 in stage 74.0 (TID 193) (vm-e1936055 executor 2): org.apache.spark.api.python.PythonException:
    Traceback (most recent call last):
    File "/home/trusted-service-user/cluster-env/trident_env/lib/python3.8/site-packages/mlflow/store/tracking/rest_store.py", line 285, in get_experiment_by_name response_proto = self._call_endpoint(GetExperimentByName, req_body)
    File "/home/trusted-service-user/cluster-env/trident_env/lib/python3.8/site-packages/mlflow/store/tracking/rest_store.py", line 56, in _call_endpoint return call_endpoint(self.get_host_creds(), endpoint, method, json_body, response_proto)
    File "/home/trusted-service-user/cluster-env/trident_env/lib/python3.8/site-packages/mlflow/utils/rest_utils.py", line 256, in call_endpoint response = verify_rest_response(response, endpoint)
    File "/home/trusted-service-user/cluster-env/trident_env/lib/python3.8/site-packages/mlflow/utils/rest_utils.py", line 185, in verify_rest_response raise RestException(json.loads(response.text))
    mlflow.exceptions.RestException: CUSTOMER_UNAUTHORIZED: Response: {'Message': 'User Aad Token is expired.', 'Source': 'GENERAL', 'error_code': 'CUSTOMER_UNAUTHORIZED'}
    ```

    This error is caused by the expiration of the MLflow token in Spark executors. To avoid this error, you can either increase the `n_concurrent_trials` to finish the trials in less than 30 minutes or set `use_spark=false` and `n_concurrent_trials=1` to use sequential training. -->

- Missing metrics in autolog

  When mlflow autolog is enabled, metrics, parameters and models should be logged automatically in mlflow runs. However, metrics and parameters for specific models may not be logged. For instances, no metrics will be logged for [XGBoost](https://mlflow.org/docs/latest/tracking.html#xgboost), [LightGBM](https://mlflow.org/docs/latest/tracking.html#lightgbm), [Spark](https://mlflow.org/docs/latest/tracking.html#spark) and SynapseML models by default.

- Spark dataframe is supported in AutoML. However, `pyspark.sql.DataFrame` should be explicitly converted to `pyspark.pandas.DataFrame` before feeding to AutoML.

  ```
  from flaml.automl.spark.utils import to_pandas_on_spark
  psdf = to_pandas_on_spark(sdf)
  automl.fit(dataframe=psdf, label='Bankrupt?', labelCol="Bankrupt?", isUnbalance=True, **settings)
  ```

  *There is NO such limitations in Tuning scenarios.*

- Add a customized learner/metric for parallel tuning with Spark needs writing the code into a file.

  It's a little bit different from adding customized learners for sequential training. In sequential training, we can define the customized learner in a notebook cell. However, in spark training, we have to import it from a file so that Spark can use it in executors. We can easily do it by leveraging `broadcast_code` function in `flaml.tune.spark.utils`.

  ```
  custom_code = """
  import numpy as np
  from flaml.model import LGBMEstimator
  from flaml import tune


  ''' define your customized objective function '''
  def my_loss_obj(y_true, y_pred):
      c = 0.5
      residual = y_pred - y_true
      grad = c * residual /(np.abs(residual) + c)
      hess = c ** 2 / (np.abs(residual) + c) ** 2
      # rmse grad and hess
      grad_rmse = residual
      hess_rmse = 1.0

      # mae grad and hess
      grad_mae = np.array(residual)
      grad_mae[grad_mae > 0] = 1.
      grad_mae[grad_mae <= 0] = -1.
      hess_mae = 1.0

      coef = [0.4, 0.3, 0.3]
      return coef[0] * grad + coef[1] * grad_rmse + coef[2] * grad_mae, \
          coef[0] * hess + coef[1] * hess_rmse + coef[2] * hess_mae


  ''' create a customized LightGBM learner class with your objective function '''
  class MyLGBM(LGBMEstimator):
      '''LGBMEstimator with my_loss_obj as the objective function
      '''

      def __init__(self, **config):
          super().__init__(objective=my_loss_obj, **config)
  """

  from flaml.tune.spark.utils import broadcast_code
  custom_learner_path = broadcast_code(custom_code=custom_code)
  print(custom_learner_path)
  from flaml.tune.spark.mylearner import MyLGBM
  ```

- Number of parallel trails is not as expected.

  You may find that the number of parallel running trials is not the same as `n_concurrent_trials`. For instance, with below code:

  ```
  import scipy.sparse

  automl_experiment = AutoML()
  automl_settings = {
      "time_budget": 30,
      "metric": "ap",
      "task": "classification",
      "estimator_list": ["xgboost"],
      "n_concurrent_trials": 4,
      "use_spark": True,
  }
  X_train = scipy.sparse.eye(1000)
  y_train = np.random.randint(2, size=1000)

  automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
  ```

  It's possible that you see below results:

  ```
  [flaml.tune.tune: 06-29 09:02:52] {728} INFO - Number of trials: 1/1000000, 1 RUNNING, 0 TERMINATED
  [flaml.tune.tune: 06-29 09:02:59] {751} INFO - Brief result: {'pred_time': 1.5501882515701592e-05, 'wall_clock_time': 7.824495077133179, 'metric_for_logging': {'pred_time': 1.5501882515701592e-05}, 'val_loss': 0.5196078431372548, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d450f9d0>}
  [flaml.tune.tune: 06-29 09:02:59] {728} INFO - Number of trials: 2/1000000, 1 RUNNING, 1 TERMINATED
  [flaml.tune.tune: 06-29 09:03:02] {751} INFO - Brief result: {'pred_time': 1.5576680501302082e-05, 'wall_clock_time': 11.52685832977295, 'metric_for_logging': {'pred_time': 1.5576680501302082e-05}, 'val_loss': 0.5196078431372548, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d2c356f0>}
  [flaml.tune.tune: 06-29 09:03:02] {728} INFO - Number of trials: 3/1000000, 1 RUNNING, 2 TERMINATED
  ```

  `1 RUNNING` means only 1 trial is running, but we expect it to be 4 since we set `n_concurrent_trials` to be 4. This could be caused by either the cluster settings or the algorithm settings. In order to override those settings and forcefully trigger parallel trials, we just need to add below code before calling automl_experiment.fit:

  ```
  import os

  os.environ["FLAML_MAX_CONCURRENT"] = "16"
  ```

  The actual parallelism will be the minimum of `FLAML_MAX_CONCURRENT` and `n_concurrent_trials`. An example result is as below:

  ```
  [flaml.tune.tune: 06-29 09:16:52] {728} INFO - Number of trials: 4/1000000, 4 RUNNING, 0 TERMINATED
  [flaml.tune.tune: 06-29 09:17:00] {751} INFO - Brief result: {'pred_time': 1.5908596562404257e-05, 'wall_clock_time': 10.396050453186035, 'metric_for_logging': {'pred_time': 1.5908596562404257e-05}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39daaeba90>}
  [flaml.tune.tune: 06-29 09:17:00] {751} INFO - Brief result: {'pred_time': 1.9936000599580654e-05, 'wall_clock_time': 10.620550155639648, 'metric_for_logging': {'pred_time': 1.9936000599580654e-05}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f3a4aabc340>}
  [flaml.tune.tune: 06-29 09:17:00] {751} INFO - Brief result: {'pred_time': 1.963213378307866e-05, 'wall_clock_time': 10.687861204147339, 'metric_for_logging': {'pred_time': 1.963213378307866e-05}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d450f280>}
  [flaml.tune.tune: 06-29 09:17:00] {751} INFO - Brief result: {'pred_time': 2.064658146278531e-05, 'wall_clock_time': 10.433406352996826, 'metric_for_logging': {'pred_time': 2.064658146278531e-05}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d2aff010>}
  [flaml.tune.tune: 06-29 09:17:00] {728} INFO - Number of trials: 8/1000000, 4 RUNNING, 4 TERMINATED
  [flaml.tune.tune: 06-29 09:17:05] {751} INFO - Brief result: {'pred_time': 0.0001536747988532571, 'wall_clock_time': 15.154952764511108, 'metric_for_logging': {'pred_time': 0.0001536747988532571}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d2c36d40>}
  [flaml.tune.tune: 06-29 09:17:05] {751} INFO - Brief result: {'pred_time': 0.0001352861815807866, 'wall_clock_time': 15.131330251693726, 'metric_for_logging': {'pred_time': 0.0001352861815807866}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39daae90c0>}
  [flaml.tune.tune: 06-29 09:17:05] {751} INFO - Brief result: {'pred_time': 0.0002753898209216548, 'wall_clock_time': 15.098459720611572, 'metric_for_logging': {'pred_time': 0.0002753898209216548}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d2afce20>}
  [flaml.tune.tune: 06-29 09:17:05] {751} INFO - Brief result: {'pred_time': 1.658177843280867e-05, 'wall_clock_time': 15.06818175315857, 'metric_for_logging': {'pred_time': 1.658177843280867e-05}, 'val_loss': 0.5098039215686274, 'trained_estimator': <flaml.automl.model.XGBoostSklearnEstimator object at 0x7f39d2afac50>}
  [flaml.tune.tune: 06-29 09:17:05] {728} INFO - Number of trials: 12/1000000, 4 RUNNING, 8 TERMINATED
  ```

- Must set `use_docker=False` for autogen `eval_function_completions` function since users have no privilege to run code with docker. `autogen` is a deprecated module in FLAML. Use `https://github.com/microsoft/autogen` instead.

  ```
  from flaml.autogen.code_utils import eval_function_completions

  metrics = eval_function_completions(responses=[response], use_docker=False, **d)
  ```

#### Limitations for python 3.11

- ensemble is not supported

  ```python
  automl_settings = {
      "time_budget": 6,
      "task": "classification",
      "n_jobs": 1,
      "estimator_list": ["catboost", "lrl2"],
      "eval_method": "cv",
      "n_splits": 3,
      "metric": "accuracy",
      "log_training_metric": True,
      "ensemble": True,  # this should be False on python 3.11
  }
  ```

- SynapseML ComputeModelStatistics is not supported yet

  ```python
      from synapse.ml.train import ComputeModelStatistics
      # ComputeModelStatistics doesn't support python 3.11
      # below code will raise errors
      metrics = ComputeModelStatistics(
          evaluationMetric="classification",
          labelCol="Bankrupt?",
          scoredLabelsCol="prediction",
      ).transform(predictions)
      metrics.show()
  ```

### Trouble Shooting

- OSError: No such file or directory: 'mlruns/x/x/artifacts/automl_pipeline/MLmodel'

  Starting from `v2.3.5.post4` (Fabric internal version), FLAML only log models/pipelines into artifact path `model` to better integrate with [Prediction](https://learn.microsoft.com/en-us/fabric/data-science/model-scoring-predict).
  Please replace "automl_pipeline" with "model" in your code.

- AutoML models depend on libraries not available publicly

  You may see some packages that are not available publicly in the logged model's `requirements.txt` file. Such as below:

  ```
  flaml==2.3.4.post3
  synapseml-cognitive==1.0.10.dev1
  synapseml-core==1.0.10.dev1
  synapseml-deep-learning==1.0.10.dev1
  synapseml-internal==1.0.10.1.dev1
  synapseml-lightgbm==1.0.10.dev1
  synapseml-mlflow==1.0.30.post1
  synapseml-opencv==1.0.10.dev1
  synapseml-vw==1.0.10.dev1
  pyspark==3.4.1.5.3.20230713
  ```

  Starting from `v2.3.5.post4` (Fabric internal version), those `synapseml-*` packages are removed from the requirements list as they're not actually needed.

  If you'd like to use the models out of Fabric, you can just try install the OSS FLAML and Pyspark.
  It should work in most cases, but we don't have a solution for you if it doesn't work.

- How to resolve out-of-memory error in `AutoML.fit()`

  - Set `free_mem_ratio` a float between 0 and 1. For example, 0.2 means try to keep free memory above 20% of total memory. Training may be early stopped for memory consumption reason when this is set.
  - Set `model_history` False.
  - If your data are already preprocessed, set `skip_transform` False. If you can preprocess the data before the fit starts, this setting can save memory needed for preprocessing in `fit`.
  - If the OOM error only happens for some particular trials:
    - set `use_spark` True. This will increase the overhead per trial but can keep the AutoML process running when a single trial fails due to OOM error.
    - provide a more accurate [`size`](https://microsoft.github.io/FLAML/docs/reference/automl/model#size) function for the memory bytes consumption of each config for the estimator causing this error.
    - modify the [search space](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#a-shortcut-to-override-the-search-space) for the estimators causing this error.
    - or remove this estimator from the `estimator_list`.
  - If the OOM error happens when ensembling, consider disabling ensemble, or use a cheaper ensemble option. ([Example](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#ensemble)).
