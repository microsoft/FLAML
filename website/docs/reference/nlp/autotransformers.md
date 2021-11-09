---
sidebar_label: autotransformers
title: nlp.autotransformers
---

## AutoTransformers Objects

```python
class AutoTransformers()
```

The AutoTransformers class

**Example**:

  
  .. code-block:: python
  
  autohf = AutoTransformers()
  autohf_settings = {
- `&quot;resources_per_trial&quot;` - {&quot;cpu&quot;: 1, &quot;gpu&quot;: 1},
- `&quot;num_samples&quot;` - -1,
- `&quot;time_budget&quot;` - 60,
  }
  
  validation_metric, analysis = autohf.fit(**autohf_settings)

#### prepare\_data

```python
def prepare_data(data_root_path, jobid_config=None, is_wandb_on=False, server_name=None, max_seq_length=128, fold_name=None, resplit_portion=None, **custom_data_args)
```

Prepare data

**Example**:

  
  .. code-block:: python
  
  preparedata_setting = {&quot;server_name&quot;: &quot;tmdev&quot;, &quot;data_root_path&quot;: &quot;data/&quot;, &quot;max_seq_length&quot;: 128,
- `&quot;jobid_config&quot;` - jobid_config, &quot;wandb_utils&quot;: wandb_utils,
- `&quot;resplit_portion&quot;` - {&quot;source&quot;: [&quot;train&quot;, &quot;validation&quot;],
- `&quot;train&quot;` - [0, 0.8], &quot;validation&quot;: [0.8, 0.9], &quot;test&quot;: [0.9, 1.0]}}
  
  autohf.prepare_data(**preparedata_setting)
  

**Arguments**:

  server_name:
  A string variable, which can be tmdev or azureml
  data_root_path:
  The root path for storing the checkpoints and output results, e.g., &quot;data/&quot;
  jobid_config:
  A JobID object describing the profile of job
  wandb_utils:
  A WandbUtils object for wandb operations
  max_seq_length (optional):
  Max_seq_lckpt_per_epochength for the huggingface, this hyperparameter must be specified
  at the data processing step
  resplit_portion:
  The proportion for resplitting the train and dev data when split_mode=&quot;resplit&quot;.
  If args.resplit_mode = &quot;rspt&quot;, resplit_portion is required
  is_wandb_on:
  A boolean variable indicating whether wandb is used

#### fit

```python
def fit(num_samples, time_budget, custom_metric_name=None, custom_metric_mode_name=None, ckpt_per_epoch=1, fp16=True, ray_verbose=1, transformers_verbose=10, resources_per_trial=None, ray_local_mode=False, **custom_hpo_args)
```

Fine tuning the huggingface using the hpo setting

**Example**:

  
  .. code-block:: python
  
  autohf_settings = {&quot;resources_per_trial&quot;: {&quot;cpu&quot;: 1},
- `&quot;num_samples&quot;` - 1,
- `&quot;time_budget&quot;` - 100000,
- `&quot;ckpt_per_epoch&quot;` - 1,
- `&quot;fp16&quot;` - False,
  }
  
  validation_metric, analysis = autohf.fit(**autohf_settings)
  

**Arguments**:

  resources_per_trial:
  A dict showing the resources used by each trial,
  e.g., {&quot;gpu&quot;: 4, &quot;cpu&quot;: 4}
  num_samples:
  An int variable of the maximum number of trials
  time_budget:
  An int variable of the maximum time budget
  custom_metric_name:
  A string of the dataset name or a function,
  e.g., &#x27;accuracy&#x27;, &#x27;f1&#x27;, &#x27;loss&#x27;
  custom_metric_mode_name:
  A string of the mode name,
  e.g., &quot;max&quot;, &quot;min&quot;, &quot;last&quot;, &quot;all&quot;
  ckpt_per_epoch:
  An integer value of number of checkpoints per epoch, default = 1
  ray_verbose:
  An integer, default=1 | verbosit of ray,
  transformers_verbose:
  An integer, default=transformers.logging.INFO | verbosity of transformers, must be chosen from one of
  transformers.logging.ERROR, transformers.logging.INFO, transformers.logging.WARNING,
  or transformers.logging.DEBUG
  fp16:
  A boolean, default = True | whether to use fp16
  ray_local_mode:
  A boolean, default = False | whether to use the local mode (debugging mode) for ray tune.run
  custom_hpo_args:
  The additional keyword arguments, e.g., custom_hpo_args = {&quot;points_to_evaluate&quot;: [{
- `&quot;num_train_epochs&quot;` - 1, &quot;per_device_train_batch_size&quot;: 128, }]}
  

**Returns**:

  
- `validation_metric` - A dict storing the validation score
  
- `analysis` - A ray.tune.analysis.Analysis object storing the analysis results from tune.run

#### predict

```python
def predict(ckpt_json_dir=None, **kwargs)
```

Predict label for test data.

An example:
predictions, test_metric = autohf.predict()

**Arguments**:

  ckpt_json_dir:
  the checkpoint for the fine-tuned huggingface if you wish to override
  the saved checkpoint in the training stage under self.path_utils._result_dir_per_run
  

**Returns**:

  A numpy array of shape n * 1 - - each element is a predicted class
  label for an instance.

#### output\_prediction

```python
def output_prediction(predictions=None, output_prediction_path=None, output_zip_file_name=None)
```

When using the original GLUE split, output the prediction on test data,
and prepare the .zip file for submission

**Example**:

  local_archive_path = self.autohf.output_prediction(predictions,
  output_prediction_path= self.console_args.data_root_dir + &quot;result/&quot;,
  output_zip_file_name=azure_save_file_name)
  

**Arguments**:

  predictions:
  A list of predictions, which is the output of AutoTransformers.predict()
  output_prediction_path:
  Output path for the prediction
  output_zip_file_name:
  An string, which is the name of the output zip file
  

**Returns**:

  The path of the output .zip file

