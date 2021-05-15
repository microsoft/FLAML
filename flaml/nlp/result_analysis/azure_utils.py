import re, pathlib, os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from ..utils import get_wandb_azure_key
from datetime import datetime
import json

class AzureUtils:
    def __init__(self,
                 args,
                 autohf):
        self.args = args
        self.autohf = autohf
        wandb_key, azure_key, container_name = get_wandb_azure_key(args.key_path)
        self._container_name = container_name
        self._azure_key = azure_key
        self.fout = None

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

    def init_azure_log(self,
                       dataset_names,
                       subdataset_names):
        self._azure_log_path = self.generate_azurepath(dataset_names, subdataset_names)
        self.fout = open(self._azure_log_path, "a")
        if self.autohf.split_mode == "origin":
            self._output_prediction_path = self.generate_output_prediction_path()

    def get_all_azure_uploaded_files(self):
        container_client = self._init_azure_clients()
        task2blobs = {}
        bloblist = []
        for blob in container_client.list_blobs():
            parse_result = self.parse_azurepath(blob.name)
            if parse_result[0] != "null":
                hpo_mode, task_name, model_id, algo_id, space_id, split_id, rep_id = parse_result
                task2blobs.setdefault(split_id, {})
                task2blobs[split_id].setdefault(task_name, {})
                task2blobs[split_id][task_name].setdefault(model_id, {})
                task2blobs[split_id][task_name][model_id].setdefault(algo_id, {})
                if algo_id == -1:
                    task2blobs[split_id][task_name][model_id][algo_id] = blob.name
                else:
                    task2blobs[split_id][task_name][model_id][algo_id].setdefault(space_id, {})
                    task2blobs[split_id][task_name][model_id][algo_id][space_id][rep_id] = blob.name
            bloblist.append(blob.name)
        return task2blobs, bloblist

    def generate_output_prediction_path(self):
        azure_save_file_name = self._azure_log_path.split("/")[-1][:-4]
        self._output_prediction_path = os.path.join(self.args.data_root_dir + "result/", azure_save_file_name + ".zip")

    def generate_azurepath(self,
                          dataset_names,
                          subdataset_names):
        full_dataset_name = dataset_names[self.args.dataset_idx][0] + "_" + subdataset_names[self.args.dataset_idx]
        path_for_subdataset = os.path.join("logs/", full_dataset_name)
        if not os.path.exists(path_for_subdataset):
            pathlib.Path(path_for_subdataset).mkdir(parents=True, exist_ok=True)

        if self.args.algo_idx != None:
            return os.path.join(path_for_subdataset,
                     "log_" + full_dataset_name
                     + "_mod" + str(self.args.pretrained_idx)
                     + "_alg" + str(self.args.algo_idx)
                     + "_spa" + str(self.args.space_idx)
                     + "_spt" + str(self.args.resplit_idx)
                     + "_rep" + str(self.args.rep_id) + ".log")
        else:
            return os.path.join(path_for_subdataset,
                                "log_" + full_dataset_name
                                + "_mod" + str(self.args.pretrained_idx)
                                + "_" + str(self.args.algo_idx)
                                + "_" + str(self.args.space_idx)
                                + "_spt" + str(self.args.resplit_idx)
                                + "_rep" + str(self.args.rep_id) + ".log")

    def parse_azurepath(self,
                        blobname):
        result_grid = re.search(".*_mod(el)?(?P<model_id>\d+)_None_None(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log",
                                blobname)
        result = re.search(
            ".*_mod(el)?(?P<model_id>\d+)_(alg)?(?P<algo_id>\d+)_(spa)?(?P<space_id>\d+)(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log",
            blobname)
        if result_grid:
            task_name = blobname.split("/")[1]
            model_id = int(result_grid.group("model_id"))
            algo_id = space_id = rep_id = -1
            try:
                split_id = int(result_grid.group("split_id"))
            except:
                split_id = 0
            return "grid", task_name, model_id, algo_id, space_id, split_id, rep_id
        if result:
            task_name = blobname.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            try:
                split_id = int(result.group("split_id"))
            except:
                split_id = 0
            return "hpo", task_name, model_id, algo_id, space_id, split_id, rep_id
        return ("null")

    def flush_and_upload_score_to_azure(self):
        self.fout.flush()
        blob_client = self._init_blob_client(self._azure_log_path)
        with open(self._azure_log_path, "rb") as fin:
            blob_client.upload_blob(fin, overwrite=True)

    def flush_and_upload_prediction(self):
        self.fout.flush()
        blob_client = self._init_blob_client(self._output_prediction_path)
        with open(self._azure_log_path, "rb") as fin:
            blob_client.upload_blob(fin, overwrite=True)

    def write_exception(self):
        wandb_group_name = self.autohf.wandb_utils.wandb_group_name()
        self.fout.write(wandb_group_name + ":\n")
        self.fout.write("timestamp:" + str(str(datetime.now())) + ":\n")
        self.fout.write("failed, no checkpoint found\n")
        self.flush_and_upload_score_to_azure()

    def write_regular(self,
                      validation_metric,
                      final_sample_num=None):
        wandb_group_name = self.autohf.wandb_utils.wandb_group_name()
        self.fout.write(wandb_group_name + ":\n")
        self.fout.write("timestamp:" + str(str(datetime.now())) + ":\n")
        self.fout.write("validation " + (self.autohf.metric_name) + ":" + json.dumps(validation_metric) + "\n")
        self.fout.write("duration:" + str(self.autohf.last_run_duration) + "\n")
        if not final_sample_num:
            final_sample_num = 0
        self.fout.write("sample_num: " + str(final_sample_num) + "\n")
        self.fout.write(wandb_group_name.split("_")[-1] + "," + str(final_sample_num) + "," + str(self.autohf.last_run_duration) + "," + str(
                validation_metric) + ",")
        self.flush_and_upload_score_to_azure()

    def output_predict(self,
                       test_dataset):
        if test_dataset:
            predictions, output_metric = self.autohf.predict(test_dataset)
            if output_metric:
                self.fout.write(str(output_metric[self.autohf.metric_name]) + "\n")
                self.fout.write("test " + (self.autohf.metric_name) + ":" + json.dumps(output_metric) + "\n")
                self.fout.write(self.args.yml_file + "\n\n")
                self.flush_and_upload_score_to_azure()
            else:
                self.fout.write("\n\n" + self.args.yml_file + "\n\n")
                self.fout.flush()
                self.flush_and_upload_score_to_azure()
            if self.autohf.split_mode == "origin":
                azure_save_file_name = self._azure_log_path.split("/")[-1][:-4]
                self.autohf.output_prediction(predictions,
                                              output_prediction_path= self.args.data_root_dir + "result/",
                                              output_dir_name=azure_save_file_name)
                self.flush_and_upload_prediction()

    def retrieve_wandb_group_name(self):
