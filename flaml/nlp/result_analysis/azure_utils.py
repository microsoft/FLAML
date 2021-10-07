import re
import pathlib
import os
from datetime import datetime
from dataclasses import dataclass, field
import json
from typing import Tuple, List, Union, Optional
import argparse


class ConfigScore:
    trial_id: str = field(default=None)
    start_time: float = field(default=None)
    last_update_time: float = field(default=None)
    config: dict = field(default=None)
    metric_score: dict = field(default=None)
    time_stamp: float = field(default=None)

    def __init__(
        self,
        trial_id: str = None,
        start_time: float = None,
        last_update_time: float = None,
        config: dict = None,
        metric_score: dict = None,
        time_stamp: float = None,
        all_ckpts: list = None,
    ):
        self.trial_id = trial_id
        self.start_time = start_time
        self.last_update_time = last_update_time
        self.config = config
        self.metric_score = metric_score
        self.time_stamp = time_stamp
        self.all_ckpts = all_ckpts


class ConfigScoreList:
    def __init__(
        self,
        config_score_list: List[ConfigScore],
        jobid_config=None,
        blob_file=None,
        test_metric=None,
    ):
        self._config_score_list = config_score_list
        self._blob_file = blob_file
        self._jobid_config = jobid_config
        self._test_metric = test_metric

    def sorted(self, sort_method="unsorted", metric_mode="max"):
        if sort_method == "unsorted":
            self._config_score_list = self._config_score_list
        elif sort_method == "sort_time":
            self._config_score_list = sorted(
                self._config_score_list, key=lambda x: x.start_time, reverse=False
            )
        else:
            self._config_score_list = sorted(
                self._config_score_list,
                key=lambda x: getattr(x, "metric_score")[metric_mode],
                reverse=True,
            )

    def get_best_config(self, metric_mode="max"):
        filtered_config_score_list = [
            x
            for x in self._config_score_list
            if isinstance(x.metric_score, dict)
        ]
        return max(
            filtered_config_score_list,
            key=lambda x: x.metric_score[metric_mode],
        )


@dataclass
class JobID:
    """
    The class for specifying the config of a job, includes the following fields:

    Notice the naming of class JobID is a shorter version of the naming of console arguments (flaml/nlp/utils.py).
    This existence of the shortened namings in JobID class is for shortening the file names in azure blob storage.

    dat:
        A list which is the dataset name
    subdat:
        A string which is the sub dataset name
    mod:
        A string which is the module, e.g., "grid", "hpo"
    spa:
        A string which is the space mode, e.g., "uni", "gnr"
    arg:
        A string which is the mode for setting the input argument of a search algorithm, e.g., "cus", "dft"
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
        A string which is the resplit mode, e.g., "ori", "rspt"
    rep:
        An integer which is the repetition id
    sddt:
        An integer which is the seed for data shuffling in the resplit mode
    sdhf:
        An integer which is the seed for transformers
    """

    dat: list = field(default=None)
    subdat: str = field(default=None)
    mod: str = field(default=None)
    spa: str = field(default=None)
    arg: str = field(default=None)
    alg: str = field(default=None)
    pru: str = field(default=None)
    pre_full: str = field(default=None)
    pre: str = field(default=None)
    presz: str = field(default=None)
    spt: str = field(default=None)
    rep: int = field(default=0)
    sddt: int = field(default=None)
    sdhf: int = field(default=None)
    sdbs: int = field(default=None)
    var1: Optional[set] = field(default=None)
    var2: Optional[set] = field(default=None)

    def __init__(self, console_args=None):
        if console_args:
            self.set_jobid_from_console_args(console_args)

    def reset_pre_full(self, pre_full):
        self.pre_full = pre_full
        self.pre = self._extract_model_type_with_keywords_match(pre_full)

    def check_model_type_consistency(self):
        assert (
            self._extract_model_type_with_keywords_match(self.pre_full) == self.pre
        ), (
            "The full name and the model type of the pre-trained model"
            " must be consistent"
        )

    def set_dataset_and_model(
        self,
        dat: List[str],
        subdat: str,
        pre_full: str,
        pre: str,
        presz: str,
        alg=None,
        arg=None,
    ):
        """
        set the dataset and model only for the JobID config

        Args:
            dat:
                A list of string. For example, ["glue"], ["race", "high"]
            subdat:
                The name of the sub dataset. For example, "mrpc"
            pre_full:
                The full name of the pre-trained language model in HuggingFace,
                e.g., "google/electra-small-discriminator"
            pre:
                The model type of the pre-trained language model, e.g., "electra"
            presz:
                The size of the pre-trained language model, e.g., "small"
        """
        self.dat = dat
        self.subdat = subdat
        self.mod = "hpo"
        self.spa = "gnr"
        self.arg = "dft" if arg is None else arg
        self.alg = "bs" if alg is None else alg
        self.pru = "asha"
        self.pre_full = pre_full
        self.pre = pre
        self.presz = presz
        self.spt = "rspt"
        self.rep = 0
        self.sddt = 43
        self.sdhf = 42

    def set_unittest_config(self):
        """
        set the JobID config for unit test
        """
        self.dat = ["glue"]
        self.subdat = "mrpc"
        self.mod = "hpo"
        self.spa = "gnr_test"
        self.arg = "cus"
        self.alg = "bs"
        self.pru = "None"
        self.pre_full = "albert-base-v1"
        self.pre = "albert"
        self.presz = "small"
        self.spt = "rspt"
        self.rep = 0
        self.sddt = 43
        self.sdhf = 42

    def is_match(self, partial_jobid):
        """Return a boolean variable whether the current object matches the partial jobid defined in partial_jobid.

        Example:

            .. code-block:: python

                self = JobID(dat = ['glue'], subdat = 'cola', mod = 'bestnn', spa = 'uni', arg = 'cus', alg = 'bs',
                             pru = 'None', pre = 'funnel', presz = 'xlarge', spt = 'rspt', rep = 0, sddt = 43, sdhf = 42)

                partial_jobid1 = JobID(dat = ['glue'],
                                       subdat = 'cola',
                                       mod = 'hpo')

                partial_jobid2 = JobID(dat = ['glue'],
                                       subdat = 'cola',
                                       mod = 'bestnn')

         return False for partial_jobid1 and True for partial_jobid2
        """
        match = True
        for key, val in partial_jobid.__dict__.items():
            if key == "pre":
                # skip matching the abbreviated model name, only match the full model name
                continue
            if val is None:
                continue
            if isinstance(val, set):
                val = set([str(x) for x in val])
                this_value = getattr(self, key)
                if isinstance(this_value, int) or isinstance(this_value, str):
                    this_value = set([str(this_value)])
                is_subset = this_value.issubset(val)
                if is_subset is False:
                    match = False
            else:
                each_val = getattr(self, key)
                if key == "dat":  # replace underline with dash for dataset name
                    # because underline is unsed for separating each attribute
                    each_val = [x.replace("_", "-") for x in each_val]
                    val = [x.replace("_", "-") for x in val]
                if key == "pre_full":  # replace "/" with dash in the full model name
                    # because "/" is the directory separator of the azure file system
                    val = val.replace("/", "-")
                if each_val != val:
                    match = False
        return match

    def to_wandb_string(self):
        """
        Preparing for the job ID for wandb
        """
        field_dict = self.__dict__
        keytoval_str = "_".join(
            [
                JobID.dataset_list_to_str(value)
                if type(value) == list
                else str(value)
                for key, value in field_dict.items()
                if key
                != "pre"  # skip the abbreviated model name in naming of the file, use the full model name instead
            ]
        )
        return keytoval_str

    def to_jobid_string(self):
        """
        Convert the current JobID into a blob name string which contains all the fields
        """
        list_keys = list(JobID.__dataclass_fields__.keys())
        field_dict = self.__dict__
        keytoval_list = []
        for key in list_keys:
            try:
                if (
                    key != "pre"
                ):  # skip the abbreviated model name in naming of the file, use the full model name instead
                    if key == "pre_full":
                        keytoval_list.append(
                            key + "=" + str(field_dict[key].replace("/", "-"))
                        )
                    elif type(field_dict[key]) == list:
                        keytoval_list.append(
                            key
                            + "="
                            + JobID.dataset_list_to_str_for_output(field_dict[key])
                        )
                    elif type(field_dict[key]) == set:
                        keytoval_list.append(
                            key + "=" + JobID.set_to_str(field_dict[key])
                        )
                    else:
                        keytoval_list.append(key + "=" + str(field_dict[key]))
            except KeyError:
                continue
        return "_".join(keytoval_list)

    def to_partial_jobid_string(self):
        """
        Convert the current JobID into a blob name string which only contains the fields whose values are not "None"
        """
        list_keys = list(JobID.__dataclass_fields__.keys())
        field_dict = (
            self.__dict__
        )  # field_dict contains fields whose values are not None
        keytoval_str = "_".join(
            [
                key + "=" + JobID.dataset_list_to_str(field_dict[key])
                if type(field_dict[key]) == list
                else key + "=" + JobID.set_to_str(field_dict[key])
                if type(field_dict[key]) == set
                else key + "=" + str(field_dict[key])
                for key in list_keys
                if key in field_dict.keys()
            ]
        )
        return keytoval_str

    @staticmethod
    def blobname_to_jobid_dict(keytoval_str):
        """
        Converting an azure blobname to a JobID config,
        e.g., blobname = "dat=glue_subdat=cola_mod=bestnn_spa=uni_arg=cus_
        alg=bs_pru=None_pre=funnel_presz=xlarge_spt=rspt_rep=0.json"

        the converted jobid dict = {dat = ['glue'], subdat = 'cola', mod = 'bestnn',
                               spa = 'uni', arg = 'cus', alg = 'bs', pru = 'None',
                               pre = 'funnel', presz = 'xlarge', spt = 'rspt',
                               rep = 0, sddt = 43, sdhf = 42)
        """
        # skip the abbreviated model name in naming of the file, use the full model name instead
        field_keys = [
            key for key in JobID.__dataclass_fields__ if key != "pre"
        ]
        regex_expression = ".*"
        is_first = True
        for key in field_keys:
            if is_first:
                prefix = ""
                is_first = False
            else:
                prefix = "_"
            if key.startswith("sd") or key.startswith("var"):
                regex_expression += "(" + prefix + key + "=(?P<" + key + ">[^_]*))?"
            else:
                regex_expression += prefix + key + "=(?P<" + key + ">[^_]*)"
        regex_expression += ".(json|zip)"
        result = re.search(regex_expression, keytoval_str)
        if result:
            result_dict = {}
            for key in field_keys:
                if key == "dat":
                    result_dict[key] = [result.group(key)]
                elif key in ("var1", "var2"):
                    if result.group(key) != "":
                        result_dict[key] = sorted(list(set([result.group(key)])))
                    else:
                        result_dict[key] = []
                elif key in ("rep", "sddt", "sdhf", "sdbs"):
                    try:
                        try:
                            try:
                                result_dict[key] = int(result.group(key))
                            except TypeError:
                                # print("int() argument is a NoneType, continuing")
                                result_dict[key] = -1
                        except IndexError:
                            print("No group {} in the regex result".format(key))
                            result_dict[key] = -1
                    except ValueError:
                        print("Cannot parse integer {}".format(result.group(key)))
                        result_dict[key] = -1
                else:
                    result_dict[key] = result.group(key)
            return result_dict
        else:
            return None

    @staticmethod
    def dataset_list_to_str(dataset_name, key="dat"):
        if isinstance(dataset_name, list):
            return "-".join(dataset_name)
        else:
            return dataset_name

    @staticmethod
    def dataset_list_to_str_for_output(dataset_name, key="dat"):
        if isinstance(dataset_name, list):
            return "-".join([x.replace("_", "-") for x in dataset_name])
        else:
            return dataset_name.replace("_", "-")

    @staticmethod
    def set_to_str(value_set):
        return min(value_set)

    def set_jobid_from_arg_list(self, **jobid_list):
        """
        Set the jobid from a dict object
        """
        for key in jobid_list:
            assert key in JobID.__dataclass_fields__.keys()
            setattr(self, key, jobid_list[key])
        if self.mod == "grid":
            self.alg = "grid"

    @staticmethod
    def convert_blobname_to_jobid(blobname):
        """
        Converting a blobname string to a JobID object
        """
        jobconfig_dict = JobID.blobname_to_jobid_dict(blobname)
        if jobconfig_dict:
            jobconfig = JobID()
            jobconfig.set_jobid_from_arg_list(**jobconfig_dict)
            return jobconfig
        else:
            return None

    @staticmethod
    def get_full_data_name(dataset_name: Union[list, str], subdataset_name=None):
        """
        Convert a dataset name and sub dataset name to a full dataset name
        """
        if isinstance(dataset_name, list):
            full_dataset_name = JobID.dataset_list_to_str(dataset_name)
        else:
            full_dataset_name = dataset_name
        if subdataset_name:
            full_dataset_name = full_dataset_name + "_" + subdataset_name
        return full_dataset_name

    def get_jobid_full_data_name(self):
        """
        Get the full dataset name of the current JobID object
        """
        return JobID.get_full_data_name(
            JobID.dataset_list_to_str(self.dat), self.subdat
        )

    @staticmethod
    def _extract_model_type_with_keywords_match(pre_full):
        from ..hpo.grid_searchspace_auto import HF_MODEL_LIST

        matched_model_type = []
        for each_model_type in HF_MODEL_LIST:
            if each_model_type in pre_full:
                matched_model_type.append(each_model_type)
        assert len(matched_model_type) > 0
        return max(enumerate(matched_model_type), key=lambda x: len(x[1]))[1]

    @staticmethod
    def extract_model_type(full_model_name):
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(full_model_name)
        config_json_file = model_config.get_config_dict(full_model_name)[0]
        try:
            model_type = config_json_file["model_type"]
        except KeyError:
            print(
                "config_json_file does not contain model_type, re-extracting with keywords matching"
            )
            model_type = JobID._extract_model_type_with_keywords_match(full_model_name)
        return model_type

    @staticmethod
    def get_attrval_from_arg_or_dict(
        console_args: Union[argparse.ArgumentParser, dict], each_key
    ):
        if type(console_args) == argparse.Namespace:
            return getattr(console_args, each_key)
        else:
            return console_args[each_key]

    def set_jobid_from_console_args(
        self, console_args: Union[argparse.ArgumentParser, dict]
    ):
        from ..utils import dataset_subdataset_name_format_check

        console_to_jobid_key_mapping = {
            "pretrained_model_size": "pre",
            "dataset_subdataset_name": "dat",
            "algo_mode": "mod",
            "space_mode": "spa",
            "search_alg_args_mode": "arg",
            "algo_name": "alg",
            "pruner": "pru",
            "resplit_mode": "spt",
            "rep_id": "rep",
            "seed_data": "sddt",
            "seed_transformers": "sdhf",
            "seed_bs": "sdbs",
        }
        for each_key in console_to_jobid_key_mapping.keys():
            try:
                try:
                    if each_key == "dataset_subdataset_name":
                        dataset_subdataset_name_format_check(
                            JobID.get_attrval_from_arg_or_dict(console_args, each_key)
                        )
                        try:
                            self.dat = (
                                JobID.get_attrval_from_arg_or_dict(
                                    console_args, each_key
                                )
                                .split(":")[0]
                                .split(",")
                            )
                        except AttributeError:
                            self.dat = []
                        try:
                            self.subdat = JobID.get_attrval_from_arg_or_dict(
                                console_args, each_key
                            ).split(":")[1]
                        except AttributeError:
                            self.subdat = ""
                    elif each_key == "pretrained_model_size":
                        try:
                            self.pre_full = JobID.get_attrval_from_arg_or_dict(
                                console_args, each_key
                            )[0]
                            self.pre = JobID.extract_model_type(self.pre_full)
                        except IndexError:
                            self.pre_full = ""
                            self.pre = ""
                        try:
                            self.presz = JobID.get_attrval_from_arg_or_dict(
                                console_args, each_key
                            )[1]
                        except IndexError:
                            self.presz = ""
                    else:
                        jobid_key = console_to_jobid_key_mapping.get(each_key, "")
                        attrval = JobID.get_attrval_from_arg_or_dict(
                            console_args, each_key
                        )
                        setattr(self, jobid_key, attrval)
                except AttributeError:
                    print("console_args has no attribute {}, continue".format(each_key))
                    continue
            except KeyError:
                print("console_args has no attribute {}, continue".format(each_key))
                continue
        if self.mod == "grid":
            self.alg = "grid"


class AzureUtils:
    def __init__(
        self,
        root_log_path=None,
        azure_key_path=None,
        data_root_dir=None,
        autohf=None,
        jobid_config=None,
        jobid_config_rename=None,
    ):
        """This class is for saving the output files (logs, predictions) for HPO, uploading it to an azure storage
        blob, and performing analysis on the saved blobs. To use the cloud storage, you need to specify a key
        and upload the output files to azure. For example, when running jobs in a cluster, this class can
        help you store all the output files in the same place. If a key is not specified, this class will help you
        save the files locally but not uploading to the cloud. After the outputs are uploaded, you can use this
        class to perform analysis on the uploaded blob files.

        Examples:

            Example 1 (saving and uploading):

                validation_metric, analysis = autohf.fit(**autohf_settings) # running HPO
                predictions, test_metric = autohf.predict()

                azure_utils = AzureUtils(root_log_path="logs_test/",
                                         autohf=autohf,
                                         azure_key_path="../../")
                                         # passing the azure blob key from key.json under azure_key_path

                azure_utils.write_autohf_output(valid_metric=validation_metric,
                                                predictions=predictions,
                                                duration=autohf.last_run_duration)
                                # uploading the output to azure cloud, which can be used for analysis afterwards

            Example 2 (analysis):

                    jobid_config = JobID()
                    jobid_config.mod = "grid"
                    jobid_config.pre = "funnel"
                    jobid_config.presz = "xlarge"

                    azure_utils = AzureUtils(root_log_path= "logs_test/",
                                             azure_key_path = "../../",
                                             jobid_config=jobid_config)

                    # continue analyzing all files in azure blob that matches jobid_config

        Args:
            root_log_path:
                The local root log folder name, e.g., root_log_path="logs_test/" will create a directory
                "logs_test/" locally

            azure_key_path:
                The path for storing the azure keys. The azure key, and container name are stored in a local file
                azure_key_path/key.json. The key_path.json file should look like this:

                {
                    "container_name": "container_name",
                    "azure_key": "azure_key",
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
        """
        if root_log_path:
            self.root_log_path = root_log_path
        else:
            self.root_log_path = "logs_azure/"
        if jobid_config_rename is not None:
            self.jobid = jobid_config_rename
        elif autohf is not None:
            self.jobid = autohf.jobid_config
        else:
            assert jobid_config is not None, (
                "jobid_config must be passed either through autohf.jobid_config"
                " or jobid_config"
            )
            self.jobid = jobid_config
        self.data_root_dir = data_root_dir
        self.autohf = autohf
        if azure_key_path:
            azure_key, container_name = AzureUtils.get_azure_key(azure_key_path)
            self._container_name = container_name
            self._azure_key = azure_key
        else:
            self._container_name = self._azure_key = ""

    @staticmethod
    def get_azure_key(key_path):
        try:
            try:
                with open(os.path.join(key_path, "key.json"), "r") as fin:
                    key_json = json.load(fin)
                    azure_key = key_json["azure_key"]
                    azure_container_name = key_json["container_name"]
                    return azure_key, azure_container_name
            except FileNotFoundError:
                print(
                    "Your output will not be synced to azure because key.json is not found under key_path"
                )
                return "", ""
        except KeyError:
            print(
                "Your output will not be synced to azure because azure key and container name are not specified"
            )
            return "", ""

    def _get_complete_connection_string(self):
        try:
            return (
                "DefaultEndpointsProtocol=https;AccountName=docws5141197765;AccountKey="
                + self._azure_key
                + ";EndpointSuffix=core.windows.net"
            )
        except AttributeError:
            return (
                "DefaultEndpointsProtocol=https;AccountName=docws5141197765;AccountKey="
                ";EndpointSuffix=core.windows.net"
            )

    def _init_azure_clients(self):
        try:
            from azure.storage.blob import ContainerClient

            connection_string = self._get_complete_connection_string()
            try:
                container_client = ContainerClient.from_connection_string(
                    conn_str=connection_string, container_name=self._container_name
                )
                return container_client
            except ValueError:
                print(
                    "Your output will not be synced to azure because azure key and container name are not specified"
                )
                return None
        except ImportError:
            print(
                "Your output will not be synced to azure because azure-blob-storage is not installed"
            )

    def _init_blob_client(self, local_file_path):
        try:
            from azure.storage.blob import BlobServiceClient

            connection_string = self._get_complete_connection_string()
            blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
            try:
                blob_client = blob_service_client.get_blob_client(
                    container=self._container_name, blob=local_file_path
                )
                return blob_client
            except ValueError:
                print(
                    "Your output will not be synced to azure because azure key and container name are not specified"
                )
                return None
        except ImportError:
            print(
                "Your output will not be synced to azure because azure-storage-blob is not installed"
            )

    def upload_local_file_to_azure(self, local_file_path):
        try:
            from azure.core.exceptions import HttpResponseError

            try:
                blob_client = self._init_blob_client(local_file_path)
                if blob_client:
                    with open(local_file_path, "rb") as fin:
                        blob_client.upload_blob(fin, overwrite=True)
            except HttpResponseError as err:
                print(
                    "Cannot upload blob due to {}: {}".format(
                        "azure.core.exceptions.HttpResponseError", err
                    )
                )
        except ImportError:
            print(
                "Your output will not be synced to azure because azure-blob-storage is not installed"
            )

    def download_azure_blob(self, blobname):
        blob_client = self._init_blob_client(blobname)
        pathlib.Path(blobname).parent.mkdir(parents=True, exist_ok=True)
        with open(blobname, "wb") as fout:
            # pathlib.Path(
            #     re.search("(?P<parent_path>^.*)/[^/]+$", blobname).group("parent_path")
            # ).mkdir(parents=True, exist_ok=True)
            fout.write(
                blob_client.download_blob().readall()
            ) if blob_client else fout.write(b"{}")

    def _get_all_checkpoint_results(self, analysis, each_trial, this_trial_config):
        import math

        all_ckpts = analysis.get_trial_checkpoints_paths(each_trial)

        try:
            ckpt_path = re.search(
                r"^(?P<ckpt_path>.*)/checkpoint_\d+/", all_ckpts[0][0]
            ).group("ckpt_path")

            with open(os.path.join(ckpt_path, "result.json"), "r") as fin:
                all_ckpt_results = []
                epochs = set([])
                max_epoch = -1
                min_epoch = 10000
                for line in fin:
                    result_json = json.loads(line)
                    if result_json["epoch"] in epochs:
                        continue
                    epochs.add(result_json["epoch"])
                    all_ckpt_results.append(
                        {
                            "epoch": result_json["epoch"],
                            "score": result_json["eval_" + analysis.default_metric],
                        }
                    )
                    max_epoch = max(max_epoch, result_json["epoch"])
                    min_epoch = min(min_epoch, result_json["epoch"])
                is_early_stop = AzureUtils.is_early_stop(
                    max_epoch, min_epoch, this_trial_config
                )
        except IndexError:
            is_early_stop = False
            max_epoch = 0
            all_ckpt_results = []
        return {
            "max_epoch": max_epoch,
            "is_early_stop": is_early_stop,
            "all_ckpt_results": all_ckpt_results,
        }

    @staticmethod
    def is_early_stop(max_epoch, min_epoch, this_trial_config):
        import math

        log_max_epoch = math.log(max_epoch / min_epoch, 2)
        return bool(
            max_epoch < this_trial_config["num_train_epochs"]
            and int(log_max_epoch) == log_max_epoch
        )

    def extract_configscore_list_from_analysis(self, analysis):
        """
        Extracting a json object for storing the key information returned from tune.run
        """
        configscore_list = []
        for each_trial in analysis.trials:
            trial_id = each_trial.trial_id
            start_time = each_trial.start_time
            last_update_time = each_trial.last_update_time
            config = each_trial.config
            metric_score = each_trial.metric_analysis.get(
                "eval_" + analysis.default_metric, 0
            )
            time_stamp = each_trial.metric_analysis["timestamp"]
            all_ckpts_results = self._get_all_checkpoint_results(
                analysis, each_trial, config
            )
            configscore_list.append(
                ConfigScore(
                    trial_id=trial_id,
                    start_time=start_time,
                    last_update_time=last_update_time,
                    config=config,
                    metric_score=metric_score,
                    time_stamp=time_stamp,
                    all_ckpts=all_ckpts_results,
                )
            )
        return configscore_list

    def write_autohf_output(
        self,
        configscore_list=None,
        valid_metric=None,
        predictions=None,
        duration=None,
        local_file_path=None,
        other_results=None,
        gitsha=None,
        console_args=None,
    ):
        """
        Write the key info from a job and upload to azure blob storage
        """
        if local_file_path is None:
            local_file_path = self.generate_local_json_path()
        output_json = {}
        if configscore_list:
            if isinstance(configscore_list[0], ConfigScore):
                output_json["val_log"] = [
                    configscore.__dict__ for configscore in configscore_list
                ]
            else:
                output_json["val_log"] = [
                    [configscore.__dict__ for configscore in each_configscore_list]
                    for each_configscore_list in configscore_list
                ]
        if valid_metric:
            output_json["valid_metric"] = valid_metric
        if duration:
            output_json["duration"] = duration
        if other_results:
            output_json["other_results"] = other_results
        if gitsha:
            output_json["gitsha"] = gitsha
        if console_args:
            output_json["console_args"] = console_args
        if len(output_json) > 0:
            self.create_local_json_and_upload(output_json, local_file_path)
        if predictions is not None:
            self.create_local_prediction_and_upload(local_file_path, predictions)

    def generate_local_json_path(self):
        """
        Return a path string for storing the json file locally
        """
        full_dataset_name = self.jobid.get_jobid_full_data_name()
        jobid_str = self.jobid.to_jobid_string()
        local_file_path = os.path.join(
            self.root_log_path, full_dataset_name, jobid_str + ".json"
        )
        pathlib.Path(os.path.join(self.root_log_path, full_dataset_name)).mkdir(
            parents=True, exist_ok=True
        )
        return local_file_path

    def create_local_json_and_upload(self, result_json, local_file_path):
        with open(local_file_path, "w") as fout:
            fout.write(json.dumps(result_json))
            fout.flush()
            self.upload_local_file_to_azure(local_file_path)

    def create_local_prediction_and_upload(self, local_json_file, predictions):
        """
        Store predictions (a .zip file) locally and upload
        """
        azure_save_file_name = local_json_file.split("/")[-1][:-5]
        if self.data_root_dir is None:
            from ..utils import load_dft_args

            console_args = load_dft_args()
            output_dir = getattr(console_args, "data_root_dir")
            print(
                "The path for saving the prediction .zip file is not specified, "
                "setting to {} by default".format(output_dir)
            )
        else:
            output_dir = self.data_root_dir
        local_archive_path = self.autohf.output_prediction(
            predictions,
            output_prediction_path=output_dir + "result/",
            output_zip_file_name=azure_save_file_name,
        )
        if local_archive_path is not None:
            self.upload_local_file_to_azure(local_archive_path)

    @staticmethod
    def is_after_earliest_time(this_blob, earliest_time: Tuple[int, int, int]):
        import pytz

        utc = pytz.UTC
        return this_blob.last_modified >= utc.localize(
            datetime(earliest_time[0], earliest_time[1], earliest_time[2])
        )

    @staticmethod
    def append_blob_list(
        each_blob, root_log_path, partial_jobid, earliest_time, blob_list
    ):
        if each_blob.name.startswith(root_log_path):
            each_jobconfig = JobID.convert_blobname_to_jobid(each_blob.name)
            is_append = False
            if each_jobconfig:
                if each_jobconfig.is_match(partial_jobid):
                    is_append = True
                if earliest_time and not AzureUtils.is_after_earliest_time(
                    each_blob, earliest_time
                ):
                    is_append = False
                if is_append:
                    blob_list.append((each_jobconfig, each_blob))

    def get_configblob_from_partial_jobid(
        self, root_log_path, partial_jobid, earliest_time: Tuple[int, int, int] = None
    ):
        """
        Get all blobs whose jobid configs match the partial_jobid
        """
        blob_list = []
        container_client = self._init_azure_clients()
        if container_client:
            for each_blob in container_client.list_blobs():
                AzureUtils.append_blob_list(
                    each_blob, root_log_path, partial_jobid, earliest_time, blob_list
                )
        return blob_list

    def rename_one_file(self, root_log_path: str, old_jobid: JobID, new_jobid: JobID):
        """
        Rename one file to another name

        Args:
            root_log_path:
                The root log path in azure blob storage, e.g., "logs_seed/"

            old_jobid:
                The old jobid for matching the blob list

            new_jobid (optional):
                The new jobid for matching the blob list

        """
        from azure.storage.blob import BlobLeaseClient

        matched_blob_list = self.get_configblob_from_partial_jobid(
            root_log_path, old_jobid
        )
        assert len(matched_blob_list) == 1

        # step 1: copy from source to destination

        local_file_path = self.generate_local_json_path()
        source_blob = self._init_blob_client(local_file_path)

        lease = BlobLeaseClient(source_blob)
        lease.acquire()

        # Get the source blob's properties and display the lease state.
        source_props = source_blob.get_blob_properties()
        print("Lease state: " + source_props.lease.state)

        self.jobid = new_jobid
        local_file_path = self.generate_local_json_path()

        dst_blob = self._init_blob_client(local_file_path)
        dst_blob.start_copy_from_url(source_blob.url)

        properties = dst_blob.get_blob_properties()
        copy_props = properties.copy

        print("Copy status: " + copy_props["status"])
        print("Copy progress: " + copy_props["progress"])
        print("Completion time: " + str(copy_props["completion_time"]))
        print("Total bytes: " + str(properties.size))

        if source_props.lease.state == "leased":
            # Break the lease on the source blob.
            lease.break_lease()

            # Update the destination blob's properties to check the lease state.
            source_props = source_blob.get_blob_properties()
            print("Lease state: " + source_props.lease.state)

        # step 2: delete

    def get_config_and_score_from_partial_jobid(
        self,
        root_log_path: str,
        partial_jobid: JobID,
        earliest_time: Tuple[int, int, int] = None,
    ):
        """
        Extract the config and score list from a partial config id

        Args:
            root_log_path:
                The root log path in azure blob storage, e.g., "logs_seed/"

            partial_jobid:
                The partial jobid for matching the blob list

            earliest_time (optional):
                The earliest starting time for any matched blob, for filtering out out-dated jobs,
                format: (YYYY, MM, DD)

        Return:
            a ConfigScore list object which stores the config and scores list for each matched blob lists

        """
        assert isinstance(root_log_path, str), "root_log_path must be of type str"
        assert isinstance(partial_jobid, JobID), "partial_jobid must be of type JobID"
        if earliest_time:
            assert isinstance(
                earliest_time, tuple
            ), "earliest_time must be a tuple of (YYYY, MM, DD)"

        matched_blob_list = self.get_configblob_from_partial_jobid(
            root_log_path, partial_jobid, earliest_time=earliest_time
        )
        return self.get_config_and_score_from_matched_blob_list(
            matched_blob_list, earliest_time
        )

    def get_config_and_score_from_matched_blob_list(
        self, matched_blob_list, earliest_time: Tuple[int, int, int] = None
    ):
        """
        Extract the config and score list of one or multiple blobs

        Args:
            matched_blob_list:
                matched blob list

        Return:
            a ConfigScore list object which stores the config and scores list for each matched blob lists

        """
        matched_config_score_lists = []
        for (each_jobconfig, each_blob) in matched_blob_list:
            self.download_azure_blob(each_blob.name)
            with open(each_blob.name, "r") as fin:
                data_json = json.load(fin)
                test_metric = data_json.get("valid_metric", {}).get("test")
                each_config_and_score_list = ConfigScoreList(
                    jobid_config=each_jobconfig,
                    blob_file=each_blob,
                    config_score_list=[
                        ConfigScore(**each_dict)
                        for each_dict in data_json.get("val_log", [])
                    ],
                    test_metric=test_metric,
                )
                matched_config_score_lists.append(each_config_and_score_list)
        return matched_config_score_lists

    def get_plot_data(self, sorted_ts, this_ts2score):
        xs = []
        ys = []
        max_score_sofar = 0
        for each_ts in sorted_ts:
            max_score_ts = max(this_ts2score[each_ts])
            max_score_sofar = max(max_score_sofar, max_score_ts)
            xs.append(each_ts - sorted_ts[0])
            ys.append(max_score_sofar)
        return xs, ys

    def get_all_xs_ys(self, each_matchedblob_lists, metric_mode="max"):
        all_xs_ys = []
        all_xs = set([])
        for each_matchedblob_list in each_matchedblob_lists:
            this_ts2score = {}
            for each_trial_entry in each_matchedblob_list._config_score_list:
                try:
                    this_ts = each_trial_entry.time_stamp[metric_mode]
                    this_ts2score.setdefault(this_ts, [])
                    this_ts2score[this_ts].append(
                        each_trial_entry.metric_score[metric_mode]
                    )
                except TypeError:
                    pass
            sorted_ts_list = sorted(this_ts2score.keys())
            xs, ys = self.get_plot_data(sorted_ts_list, this_ts2score)
            all_xs_ys.append((xs, ys))
            all_xs.update(xs)
        return all_xs, all_xs_ys

    def plot_one_curve(self, all_xs, all_xs_ys, this_color, this_label):
        import matplotlib.pyplot as plt

        plt.step(
            [4 * x for x in all_xs_ys], color=this_color, label=this_label, where="post"
        )

    def plot_avg_curve(self, all_xs, all_xs_ys, this_color, this_label):
        import matplotlib.pyplot as plt
        import bisect
        import numpy as np

        all_xs = sorted(all_xs)
        means = []
        stds = []
        for each_tick in all_xs:
            all_ys = []
            for i in range(len(all_xs_ys)):
                xs = all_xs_ys[i][0]
                ys = all_xs_ys[i][1]
                if len(ys) == 0:
                    continue
                y_pos = max(0, min(bisect.bisect_left(xs, each_tick), len(ys) - 1))
                this_y = ys[y_pos]
                all_ys.append(this_y)
            avg_y = np.mean(all_ys)
            std_y = np.std(all_ys)
            means.append(avg_y)
            stds.append(std_y)
        (line1,) = plt.step(
            all_xs, means, color=this_color, label=this_label, where="post"
        )
        plt.fill_between(
            all_xs,
            np.subtract(means, stds),
            np.add(means, stds),
            color=this_color,
            alpha=0.2,
        )
        plt.legend()

    def plot_each_matchedblob_lists(self, each_matchedblob_lists):
        return self.get_all_xs_ys(each_matchedblob_lists)

    def plot_one_or_multiple_curves(self, all_xs, all_xs_ys, this_color, this_label):
        if len(all_xs) == 1:
            self.plot_one_curve(all_xs, all_xs_ys, this_color, this_label)
        else:
            self.plot_avg_curve(all_xs, all_xs_ys, this_color, this_label)

    def plot_hpo_curves(self, config2matched_blob_list, config2color, plot_title):
        import matplotlib.pyplot as plt

        assert isinstance(config2matched_blob_list, dict) and isinstance(
            list(config2matched_blob_list.values())[0], list
        ), (
            "The input of plot_hpo_curve must be"
            "a dictionary from configuration (e.g., hpo name) to a list of matched blob list"
        )
        for (each_label, each_matchedblob_lists) in config2matched_blob_list.items():
            all_xs, all_xs_ys = self.plot_each_matchedblob_lists(each_matchedblob_lists)
            self.plot_one_or_multiple_curves(
                all_xs, all_xs_ys, config2color[each_label], each_label
            )
        plt.xlabel("wall clock time (s)")
        plt.ylabel("validation score")
        plt.title(plot_title)
        # plt.legend(loc=2)
        plt.ylim(0.8, 0.96)
        plt.show()
