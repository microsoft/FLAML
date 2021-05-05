import wandb, re, pathlib, os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

connection_string = "DefaultEndpointsProtocol=https;AccountName=docws5141197765;AccountKey=S/HM+Pz0MqIkJsBSkjj6IQcmOoElXvYYoXQAisBHBjBtqJvtKnBWTPuiwIfCHh4e700QmGJ/geL/MOFKLGFRIA==;EndpointSuffix=core.windows.net"

def init_azure_clients():
    container_name = "hpoexperiments"
    container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)
    return container_client

def init_blob_client():

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# def get_all_runs(args):
#     api = wandb.Api()
#     runs = api.runs("liususan/upload_file_" + args.server_name)
#     task2files = {}
#     filelist = []
#     for file in runs[0].files():
#         result = re.search(".*_model(?P<model_id>\d+)_(?P<algo_id>\d+)_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log", file.name)
#         if result:
#             task_name = file.name.split("/")[1]
#             model_id = int(result.group("model_id"))
#             space_id = int(result.group("space_id"))
#             rep_id = int(result.group("rep_id"))
#             algo_id = int(result.group("algo_id"))
#             task2files.setdefault(task_name, {})
#             task2files[task_name].setdefault(model_id, {})
#             task2files[task_name][model_id].setdefault(algo_id, {})
#             task2files[task_name][model_id][algo_id].setdefault(space_id, {})
#             task2files[task_name][model_id][algo_id][space_id][rep_id] = file
#             filelist.append(file)
#     return task2files, filelist

def extract_ts(timestamp_str):
    result = re.search("(?P<year>\d+)-(?P<month>\d+)-(?P<date>\d+)", timestamp_str)
    return {"month": int(result.group("month")),
            "date": int(result.group("date"))}

def compare_dates(time1, time2):
    if time1["month"] < time2["month"]: return -1
    if time1["month"] > time2["month"]: return 1
    if time1["date"] < time2["date"]: return -1
    if time1["date"] > time2["date"]: return 1
    return 0

def get_azurepath(args, dataset_names, subdataset_names):
    full_dataset_name = dataset_names[args.dataset_idx][0] + "_" + subdataset_names[args.dataset_idx]
    path_for_subdataset = os.path.join("./logs/",
            full_dataset_name)
    if not os.path.exists(path_for_subdataset):
        pathlib.Path(path_for_subdataset).mkdir(parents=True, exist_ok=True)

    return os.path.join(path_for_subdataset,
                 "log_" + full_dataset_name + "_model"
                 + str(args.pretrained_idx)
                 + "_" + str(args.algo_idx)
                 + "_" + str(args.space_idx)
                 + "_rep" + str(args.rep_id) + ".log")

def search_blob_to_delete(args, dataset_names, subdataset_names, mode="delete_one"):
    blob_service_client, container_client = init_azure_clients()
    blobs_to_delete = []
    for blob in container_client.list_blobs():
        result = re.search(".*_model(?P<model_id>\d+)_(?P<algo_id>\d+)_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log", blob.name)
        if result:
            task_name = blob.name.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            is_delete = False
            if mode == "delete_one":
                is_delete = task_name == dataset_names[args.dataset_idx][0] + "_" + subdataset_names[args.dataset_idx] \
                and model_id == args.pretrained_idx \
                and space_id == args.space_idx \
                and rep_id == args.rep_id \
                and algo_id == args.algo_idx
            elif mode == "delete_all":
                is_delete = task_name == dataset_names[args.dataset_idx][0] + "_" + subdataset_names[args.dataset_idx] \
                            and model_id == args.pretrained_idx \
                            and space_id == args.space_idx \
                            and algo_id == args.algo_idx
            if is_delete:
                blobs_to_delete.append(blob)
    return blobs_to_delete


def flush_and_upload(fout, args, azure_log_path):
    fout.flush()
    api = wandb.Api()
    runs = api.runs("liususan/upload_file_" + args.server_name)
    runs[0].upload_file(wandb_log_path)

    blob_c = BlobClient.from_connection_string(conn_str="<connection_string>", container_name="my_container",
                                               blob_name="my_blob")
    blob_service_client, container_client = init_azure_clients()
