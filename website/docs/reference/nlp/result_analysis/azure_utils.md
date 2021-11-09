---
sidebar_label: azure_utils
title: nlp.result_analysis.azure_utils
---

## JobID Objects

```python
@dataclass
class JobID()
```

The class for specifying the config of a job, includes the following fields:

dat:
    A list which is the dataset name
subdat:
    A string which is the sub dataset name
mod:
    A string which is the module, e.g., &quot;grid&quot;, &quot;hpo&quot;
spa:
    A string which is the space mode, e.g., &quot;uni&quot;, &quot;gnr&quot;
arg:
    A string which is the mode for setting the input argument of a search algorithm, e.g., &quot;cus&quot;, &quot;dft&quot;
alg:
    A string which is the search algorithm name
pru:
    A string which is the scheduler name
pre_full:
    A string which is the full name of the pretrained language model
pre:
    A string which is the abbreviation of the pretrained language model
presz:
    A string which is the size of the pretrained language model
spt:
    A string which is the resplit mode, e.g., &quot;ori&quot;, &quot;rspt&quot;
rep:
    An integer which is the repetition id
sddt:
    An integer which is the seed for data shuffling in the resplit mode
sdhf:
    An integer which is the seed for transformers

#### set\_unittest\_config

```python
def set_unittest_config()
```

set the JobID config for unit test

#### is\_match

```python
def is_match(partial_jobid)
```

Return a boolean variable whether the current object matches the partial jobid defined in partial_jobid.

**Example**:

  
  .. code-block:: python
  
  self = JobID(dat = [&#x27;glue&#x27;], subdat = &#x27;cola&#x27;, mod = &#x27;bestnn&#x27;, spa = &#x27;buni&#x27;, arg = &#x27;cus&#x27;, alg = &#x27;bs&#x27;,
  pru = &#x27;None&#x27;, pre = &#x27;funnel&#x27;, presz = &#x27;xlarge&#x27;, spt = &#x27;rspt&#x27;, rep = 0, sddt = 43, sdhf = 42)
  
  partial_jobid1 = JobID(dat = [&#x27;glue&#x27;],
  subdat = &#x27;cola&#x27;,
  mod = &#x27;hpo&#x27;)
  
  partial_jobid2 = JobID(dat = [&#x27;glue&#x27;],
  subdat = &#x27;cola&#x27;,
  mod = &#x27;bestnn&#x27;)
  
  return False for partial_jobid1 and True for partial_jobid2

#### to\_wandb\_string

```python
def to_wandb_string()
```

Preparing for the job ID for wandb

#### to\_jobid\_string

```python
def to_jobid_string()
```

Convert the current JobID into a blob name string which contains all the fields

#### to\_partial\_jobid\_string

```python
def to_partial_jobid_string()
```

Convert the current JobID into a blob name string which only contains the fields whose values are not &quot;None&quot;

#### blobname\_to\_jobid\_dict

```python
@staticmethod
def blobname_to_jobid_dict(keytoval_str)
```

Converting an azure blobname to a JobID config,
e.g., blobname = &quot;dat=glue_subdat=cola_mod=bestnn_spa=buni_arg=cus_
alg=bs_pru=None_pre=funnel_presz=xlarge_spt=rspt_rep=0.json&quot;

the converted jobid dict = {dat = [&#x27;glue&#x27;], subdat = &#x27;cola&#x27;, mod = &#x27;bestnn&#x27;,
                       spa = &#x27;buni&#x27;, arg = &#x27;cus&#x27;, alg = &#x27;bs&#x27;, pru = &#x27;None&#x27;,
                       pre = &#x27;funnel&#x27;, presz = &#x27;xlarge&#x27;, spt = &#x27;rspt&#x27;,
                       rep = 0, sddt = 43, sdhf = 42)

#### set\_jobid\_from\_arg\_list

```python
def set_jobid_from_arg_list(**jobid_list)
```

Set the jobid from a dict object

#### convert\_blobname\_to\_jobid

```python
@staticmethod
def convert_blobname_to_jobid(blobname)
```

Converting a blobname string to a JobID object

#### get\_full\_data\_name

```python
@staticmethod
def get_full_data_name(dataset_name: Union[list, str], subdataset_name=None)
```

Convert a dataset name and sub dataset name to a full dataset name

#### get\_jobid\_full\_data\_name

```python
def get_jobid_full_data_name()
```

Get the full dataset name of the current JobID object

## AzureUtils Objects

```python
class AzureUtils()
```

#### \_\_init\_\_

```python
def __init__(root_log_path=None, azure_key_path=None, data_root_dir=None, autohf=None, jobid_config=None)
```

This class is for saving the output files (logs, predictions) for HPO, uploading it to an azure storage
blob, and performing analysis on the saved blobs. To use the cloud storage, you need to specify a key
and upload the output files to azure. For example, when running jobs in a cluster, this class can
help you store all the output files in the same place. If a key is not specified, this class will help you
save the files locally but not uploading to the cloud. After the outputs are uploaded, you can use this
class to perform analysis on the uploaded blob files.

**Examples**:

  
  Example 1 (saving and uploading):
  
  validation_metric, analysis = autohf.fit(**autohf_settings) # running HPO
  predictions, test_metric = autohf.predict()
  
  azure_utils = AzureUtils(root_log_path=&quot;logs_test/&quot;,
  autohf=autohf,
  azure_key_path=&quot;../../&quot;)
  # passing the azure blob key from key.json under azure_key_path
  
  azure_utils.write_autohf_output(valid_metric=validation_metric,
  predictions=predictions,
  duration=autohf.last_run_duration)
  # uploading the output to azure cloud, which can be used for analysis afterwards
  
  Example 2 (analysis):
  
  jobid_config = JobID()
  jobid_config.mod = &quot;grid&quot;
  jobid_config.pre = &quot;funnel&quot;
  jobid_config.presz = &quot;xlarge&quot;
  
  azure_utils = AzureUtils(root_log_path= &quot;logs_test/&quot;,
  azure_key_path = &quot;../../&quot;,
  jobid_config=jobid_config)
  
  # continue analyzing all files in azure blob that matches jobid_config
  

**Arguments**:

  root_log_path:
  The local root log folder name, e.g., root_log_path=&quot;logs_test/&quot; will create a directory
  &quot;logs_test/&quot; locally
  
  azure_key_path:
  The path for storing the azure keys. The azure key, and container name are stored in a local file
  azure_key_path/key.json. The key_path.json file should look like this:
  
  {
- `&quot;container_name&quot;` - &quot;container_name&quot;,
- `&quot;azure_key&quot;` - &quot;azure_key&quot;,
  }
  
  To find out the container name and azure key of your blob, please refer to:
  https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal
  
  If the container name and azure key are not specified, the output will only be saved locally,
  not synced to azure blob.
  
  data_root_dir:
  The directory for outputing the predictions, e.g., packing the predictions into a .zip file for
  uploading to the glue website
  
  autohf:
  The AutoTransformers object, which contains the output of an HPO run. AzureUtils will save the
  output (analysis results, predictions) from AzureTransformers.
  
  jobid_config:
  The jobid config for analysis. jobid_config specifies the jobid config of azure blob files
  to be analyzed, if autohf is specified, jobid_config will be overwritten by autohf.jobid_config

#### extract\_configscore\_list\_from\_analysis

```python
def extract_configscore_list_from_analysis(analysis)
```

Extracting a json object for storing the key information returned from tune.run

#### write\_autohf\_output

```python
def write_autohf_output(configscore_list=None, valid_metric=None, predictions=None, duration=None)
```

Write the key info from a job and upload to azure blob storage

#### generate\_local\_json\_path

```python
def generate_local_json_path()
```

Return a path string for storing the json file locally

#### create\_local\_prediction\_and\_upload

```python
def create_local_prediction_and_upload(local_json_file, predictions)
```

Store predictions (a .zip file) locally and upload

#### get\_configblob\_from\_partial\_jobid

```python
def get_configblob_from_partial_jobid(root_log_path, partial_jobid, earliest_time: Tuple[int, int, int] = None)
```

Get all blobs whose jobid configs match the partial_jobid

#### get\_config\_and\_score\_from\_partial\_jobid

```python
def get_config_and_score_from_partial_jobid(root_log_path: str, partial_jobid: JobID, earliest_time: Tuple[int, int, int] = None)
```

Extract the config and score list from a partial config id

**Arguments**:

  root_log_path:
  The root log path in azure blob storage, e.g., &quot;logs_seed/&quot;
  
  partial_jobid:
  The partial jobid for matching the blob list
  
  earliest_time (optional):
  The earliest starting time for any matched blob, for filtering out out-dated jobs,
- `format` - (YYYY, MM, DD)
  

**Returns**:

  a ConfigScore list object which stores the config and scores list for each matched blob lists

#### get\_config\_and\_score\_from\_matched\_blob\_list

```python
def get_config_and_score_from_matched_blob_list(matched_blob_list, earliest_time: Tuple[int, int, int] = None)
```

Extract the config and score list of one or multiple blobs

**Arguments**:

  matched_blob_list:
  matched blob list
  

**Returns**:

  a ConfigScore list object which stores the config and scores list for each matched blob lists

