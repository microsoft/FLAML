import re, pathlib, os
from azure.storage.blob import BlobServiceClient, ContainerClient
from transformers import AutoConfig

from ..utils import get_wandb_azure_key
from datetime import datetime
from dataclasses import dataclass, field
from ..hpo.grid_searchspace_auto import HF_MODEL_LIST
import json

@dataclass
class JobID:
    dat: list = field(default = None)
    subdat: str = field(default=None)
    mod: str = field(default = None)
    spa: str = field(default = None)
    arg: str = field(default = None)
    alg: str = field(default = None)
    pru: str = field(default = None)
    pre_full: str = field(default = None)
    pre: str = field(default = None)
    presz: str = field(default = None)
    spt: str = field(default = None)
    rep: int = field(default = 0)

    def __init__(self,
                 console_args):
        self.load_console_args(console_args)

    def to_wandb_string(self):
        field_dict = self.__dict__
        keytoval_str = "_".join([str(field_dict[key][0])
                                 if type(field_dict[key]) == list
                                 else str(field_dict[key])
                                 for key in field_dict.keys() if not key.endswith("_full")])
        return keytoval_str

    def to_jobid_string(self):
        field_dict = self.__dict__
        keytoval_str = "_".join([key + "=" + str(field_dict[key][0])
                                 if type(field_dict[key]) == list
                                 else key + "=" + str(field_dict[key])
                                 for key in field_dict.keys() if not key.endswith("_full")])
        return keytoval_str

    @staticmethod
    def blobname_to_jobid(keytoval_str):
        result = re.search(".*_dat=(?P<dat>.*)_subdat=(?P<subdat>.*)_mod=(?P<mod>.*)_spa=(?P<spa>.*)_arg=(?P<arg>.*)"
                           "_alg=(?P<alg>.*)_pru=(?P<pru>.*)_pre=(?P<pre>.*)_presz=(?P<presz>.*)_spt=(?P<spt>.*)_rep=(?P<rep>\d+).*", keytoval_str)
        dat = [result.group("dat")]
        subdat = result.group("subdat")
        mod = result.group("mod")
        spa = result.group("spa")
        arg = result.group("arg")
        alg = result.group("alg")
        pru = result.group("pru")
        pre = result.group("pre")
        presz = result.group("presz")
        spt = result.group("spt")
        rep = int(result.group("rep"))
        return dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep

    def set_jobid(self,
                  dat=None,
                  subdat=None,
                  mod=None,
                  spa=None,
                  arg=None,
                  alg=None,
                  pru=None,
                  pre=None,
                  presz=None,
                  spt=None,
                  rep=None
                  ):
        self.dat = dat
        self.subdat = subdat
        self.mod = mod
        self.spa = spa
        self.arg = arg
        self.alg = alg
        self.pru = pru
        self.pre = pre
        self.presz = presz
        self.spt = spt
        self.rep = rep

    def from_blobname(self, blobname):
        dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep = JobID.blobname_to_jobid(blobname)
        self.set_jobid(dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep)

    @staticmethod
    def get_full_data_name(dataset_name, subdataset_name=None):
        full_dataset_name = dataset_name
        if subdataset_name:
            full_dataset_name = full_dataset_name + "_" + subdataset_name
        return full_dataset_name

    def get_jobid_full_data_name(self):
        return JobID.get_full_data_name(self.dat[0], self.subdat)

    @staticmethod
    def _extract_model_type_with_keywords_match(pre_full):
        matched_model_type = []
        for each_model_type in HF_MODEL_LIST:
            if each_model_type in pre_full:
                matched_model_type.append(each_model_type)
        assert len(matched_model_type) > 0
        return max(enumerate(matched_model_type), key=lambda x: len(x[1]))[1]

    @staticmethod
    def extract_model_type(full_model_name):
        model_config = AutoConfig.from_pretrained(full_model_name)
        config_json_file = model_config.get_config_dict(full_model_name)[0]
        try:
            model_type = config_json_file["model_type"]
        except:
            model_type = JobID._extract_model_type_with_keywords_match()
        return model_type

    def load_console_args(self, console_args):
        self.dat = console_args.dataset_subdataset_name.split(":")[0].split(",")
        self.subdat = console_args.dataset_subdataset_name.split(":")[1]
        self.mod = console_args.algo_mode
        self.spa = console_args.space_mode
        self.arg = console_args.search_alg_args_mode
        self.alg = console_args.algo_name
        self.pru = console_args.pruner
        self.pre_full = console_args.pretrained_model_size.split(":")[0]
        self.pre = JobID.extract_model_type(self.pre_full)
        self.presz = console_args.pretrained_model_size.split(":")[1]
        self.spt = console_args.resplit_mode
        self.rep = console_args.rep_id

    @staticmethod
    def legacy_old_blobname_to_new_blobname(self,
                                            old_blobname):
        spa_id2val = {
            0: "gnr",
            1: "uni"
        }
        alg_id2val = {
            0: "bs",
            1: "optuna",
            2: "cfo"
        }
        pre_id2val = {
            0: "xlnet-base-cased",
            1: "albert-large-v1",
            2: "distilbert-base-uncased",
            3: "microsoft/deberta-base",
            4: "funnel-transformer/small-base",
            5: "microsoft/deberta-large",
            6: "funnel-transformer/large-base",
            7: "funnel-transformer/intermediate-base",
            8: "funnel-transformer/xlarge-base"
        }
        presz_id2val = {
            0: "base",
            1: "small",
            2: "base",
            3: "base",
            4: "base",
            5: "large",
            6: "large",
            7: "intermediate",
            8: "xlarge"
        }
        spt_id2val = {
            0: "rspt",
            1: "ori"
        }
        result_grid = re.search(".*_mod(el)?(?P<model_id>\d+)_None_None(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log",
                                old_blobname)
        result = re.search(
            ".*_mod(el)?(?P<model_id>\d+)_(alg)?(?P<algo_id>\d+)_(spa)?(?P<space_id>\d+)(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log",
            old_blobname)
        if result_grid:
            dat = [old_blobname.split("/")[1].split("_")[0]]
            subdat = old_blobname.split("/")[1].split("_")[1]
            mod = "hpo"
            spa = None
            arg = None
            alg = None
            pru = None
            pre = pre_id2val[int(result_grid.group("model_id"))]
            presz = presz_id2val[int(result_grid.group("model_id"))]
            try:
                spt = spt_id2val[int(result_grid.group("split_id"))]
            except:
                spt = spt_id2val[0]
            rep = None
            self.set_jobid(dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep)
            return self.to_jobid_string()
        if result:
            dat = [old_blobname.split("/")[1].split("_")[0]]
            subdat = old_blobname.split("/")[1].split("_")[1]
            mod = "hpo"
            spa = spa_id2val[int(result.group("space_id"))]
            arg = "dft"
            alg = alg_id2val[int(result.group("algo_id"))]
            pru = "None"
            pre = pre_id2val[int(result_grid.group("model_id"))]
            presz = presz_id2val[int(result_grid.group("model_id"))]
            try:
                spt = spt_id2val[int(result_grid.group("split_id"))]
            except:
                spt = spt_id2val[0]
            rep = int(result.group("rep_id"))
            self.set_jobid(dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep)
            return self.to_jobid_string()
        return None

class AzureUtils:
    def __init__(self,
                 console_args,
                 jobid,
                 autohf):
        self.jobid = jobid
        self.console_args = console_args
        self.autohf = autohf
        wandb_key, azure_key, container_name = get_wandb_azure_key(console_args.key_path)
        self._container_name = container_name
        self._azure_key = azure_key

    def _get_complete_connection_string(self):
        return "DefaultEndpointsProtocol=https;AccountName=docws5141197765;AccountKey=" \
            + self._azure_key + ";EndpointSuffix=core.windows.net"

    def _init_azure_clients(self):
        connection_string = self._get_complete_connection_string()
        container_client = ContainerClient.from_connection_string(conn_str=connection_string,
                            container_name= self._container_name)
        return container_client

    def _init_blob_client(self,
                          local_file_path):
        connection_string = self._get_complete_connection_string()
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container= self._container_name, blob= local_file_path)
        return blob_client

    def store_azure_uploaded_files_in_db(self):
        import pandas
        container_client = self._init_azure_clients()
        self.all_azure_files = pandas.DataFrame(columns = ["dat", "subdat", "mod", "spa", "arg", "alg", "pru", "pre", "presz", "spt", "rep", "blob_name"])
        for blob in container_client.list_blobs():
            dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep = self.jobid.blobname_to_jobid(blob.name)
            existing_row_count = len(self.all_azure_files)
            self.all_azure_files.loc[existing_row_count] = [dat, subdat, mod, spa, arg, alg, pru, pre, presz, spt, rep, blob.name]

    def upload_local_file_to_azure(self, local_file_path):
        blob_client = self._init_blob_client(local_file_path)
        with open(local_file_path, "rb") as fin:
            blob_client.upload_blob(fin, overwrite=True)

    def download_azure_blob(self, blobname):
        blob_client = self._init_blob_client(blobname)
        pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", blobname).group("parent_path")).mkdir(
            parents=True, exist_ok=True)
        with open(blobname, "wb") as fout:
            fout.write(blob_client.download_blob().readall())

    def write_exception(self):
        result_json = {
                       "timestamp": datetime.now(),
                       }
        local_file_path = self.generate_local_json_path()
        self.create_local_json_and_upload(result_json, local_file_path)

    def extract_log_from_analysis(self,
                                  analysis):
        json_log = []
        for each_trial in analysis.trials:
            trial_id = each_trial.trial_id
            last_update_time = each_trial.last_update_time
            config = each_trial.config
            print("analysis.default_metric.keys():")
            print(each_trial.metric_analysis.keys())
            metric_score = each_trial.metric_analysis["eval_" + analysis.default_metric]
            time_stamp = each_trial.metric_analysis['timestamp']
            json_log.append({"trial_id": trial_id, "last_update_time": last_update_time, "config": config, "metric_score": metric_score, "time_stamp": time_stamp})
        return json_log

    def write_autohf_output(self,
                      json_log = None,
                      test_metric = None,
                      predictions = None):
        local_file_path = self.generate_local_json_path()
        output_json = {}
        if json_log:
            output_json["val_log"] = json_log
        if test_metric:
            output_json["test_metric"] = test_metric
        if len(output_json) > 0:
            self.create_local_json_and_upload(output_json, local_file_path)
        if predictions:
            self.create_local_prediction_and_upload(local_file_path, predictions)

    def generate_local_json_path(self):
        full_dataset_name = self.jobid.get_jobid_full_data_name()
        jobid_str = self.jobid.to_jobid_string()
        local_file_path = os.path.join("logs_azure/", full_dataset_name, jobid_str + ".json")
        pathlib.Path(os.path.join("logs_azure/", full_dataset_name)).mkdir(parents=True, exist_ok=True)
        return local_file_path

    def create_local_json_and_upload(self, result_json, local_file_path):
        with open(local_file_path, "w") as fout:
            fout.write(json.dumps(result_json))
            fout.flush()
            self.upload_local_file_to_azure(local_file_path)

    def legacy_to_json(self):
        container_client = self._init_azure_clients()
        for old_blob in container_client.list_blobs():
            new_jobid_str = self.jobid.legacy_old_blobname_to_new_blobname(old_blob.name)
            if new_jobid_str:
                self.download_azure_blob(old_blob.name)
                with open(old_blob.name, "r") as fin:
                    alllines = fin.readlines()
                    wandb_group_name = alllines[0].rstrip("\n:")
                    timestamp = re.search("timestamp:(?P<timestamp>.*):", alllines[1].strip("\n")).group("timestamp")
                    duration = re.search("duration:(?P<duration>.*)$", alllines[3].strip("\n")).group("duration")
                    sample_num = int(re.search("sample_num: (?P<sample_num>\d+)$", alllines[4].strip("\n")).group("sample_num"))
                    validation = {"accuracy": float(re.search("validation accuracy: (?P<validation>.*)$", alllines[2].strip("\n")).group("validation"))}
                    test = None
                    if len(alllines) > 6:
                        result_test = re.search("test accuracy:(?P<test>.*)$", alllines[6].strip("\n"))
                        if result_test:
                            test = json.loads(result_test.group("test"))
                    yml_file = None
                    if len(alllines) > 8:
                        if alllines[8].startswith("aml"):
                            yml_file = alllines[8].strip("\n")
                    new_json = {"wandb_group_name": wandb_group_name,
                               "validation": validation,
                               "test": test,
                               "timestamp": timestamp,
                               "duration": duration,
                               "sample_num": sample_num,
                               "yml_file": yml_file}
                    full_dataset_name = self.jobid.get_jobid_full_data_name()
                    new_blobname = os.path.join("logs_azure/", full_dataset_name, new_jobid_str + ".json")
                    self.create_local_json_and_upload(new_json, new_blobname)

    def create_local_prediction_and_upload(self,
                         local_json_file,
                         predictions):
        azure_save_file_name = local_json_file.split("/")[-1][:-4]
        local_archive_path = self.autohf.output_prediction(predictions,
                                      output_prediction_path= self.console_args.data_root_dir + "result/",
                                      output_dir_name=azure_save_file_name)
        self.upload_local_file_to_azure(local_archive_path)

    def get_ranked_configs_from_azure_file(self, metric_mode):
        azure_file_path = self.generate_local_json_path()
        self.download_azure_blob(azure_file_path)

        json_log = json.load(open(azure_file_path, "r"))
        assert "val_log" in json_log

        trialid_to_score = {}
        trialid_to_config = {}

        for each_entry in json_log["val_log"]:
            trial_id = each_entry["trial_id"]
            config = each_entry["config"]
            this_score = each_entry["metric_score"][metric_mode]
            trialid_to_config[trial_id] = config
            trialid_to_score[trial_id] = this_score

        sorted_trialid_to_score = sorted(trialid_to_score.items(), key = lambda x:x[1], reverse=True)
        return [trialid_to_config[entry[0]] for entry in sorted_trialid_to_score]
