import re, pathlib, os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

container_name = "05eedc85-eda3-47b4-a5c4-a4c104810d8f"

def get_complete_connection_string(account_key):
    return "DefaultEndpointsProtocol=https;AccountName=docws5141197765;AccountKey=" + account_key + ";EndpointSuffix=core.windows.net"

def init_azure_clients(azure_key):
    connection_string = get_complete_connection_string(azure_key)
    container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)
    return container_client

def init_blob_client(azure_key, local_file_name):
    connection_string = get_complete_connection_string(azure_key)
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
    return blob_client

def get_all_runs(args):
    container_client = init_azure_clients(args.azure_key)
    task2blobs = {}
    bloblist = []
    for blob in container_client.list_blobs():
        result_grid = re.search(".*_mod(el)?(?P<model_id>\d+)_None_None(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log",
                                blob.name)
        result = re.search(".*_mod(el)?(?P<model_id>\d+)_(alg)?(?P<algo_id>\d+)_(spa)?(?P<space_id>\d+)(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log", blob.name)
        if result_grid:
            bloblist.append(blob.name)
            continue
        if result:
            task_name = blob.name.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            try:
                split_id = int(result.group("split_id"))
            except:
                split_id = 0
            task2blobs.setdefault(split_id, {})
            task2blobs[split_id].setdefault(task_name, {})
            task2blobs[split_id][task_name].setdefault(model_id, {})
            task2blobs[split_id][task_name][model_id].setdefault(algo_id, {})
            task2blobs[split_id][task_name][model_id][algo_id].setdefault(space_id, {})
            task2blobs[split_id][task_name][model_id][algo_id][space_id][rep_id] = blob.name
            bloblist.append(blob.name)
    return task2blobs, bloblist

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
    path_for_subdataset = os.path.join("logs/",
            full_dataset_name)
    if not os.path.exists(path_for_subdataset):
        pathlib.Path(path_for_subdataset).mkdir(parents=True, exist_ok=True)

    return os.path.join(path_for_subdataset,
                 "log_" + full_dataset_name
                 + "_mod" + str(args.pretrained_idx)
                 + "_alg" + str(args.algo_idx)
                 + "_spa" + str(args.space_idx)
                 + "_spt" + str(args.resplit_idx)
                 + "_rep" + str(args.rep_id) + ".log")

def search_blob_to_delete(args, dataset_names, subdataset_names, mode="delete_one"):
    container_client = init_azure_clients(args.azure_key)
    blobs_to_delete = []
    for blob in container_client.list_blobs():
        result = re.search(".*_mod(el)?(?P<model_id>\d+)_(alg)?(?P<algo_id>\d+)_(spa)?(?P<space_id>\d+)(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log", blob.name)
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
                blobs_to_delete.append(blob.name)
    return blobs_to_delete

def clean_outdated_results(args, dataset_names, subdataset_names):
    if args.is_rerun:
        blobs_to_delete = search_blob_to_delete(args,
                                                dataset_names,
                                                subdataset_names,
                                                mode = "delete_one")
    else:
        if args.rep_id == 0:
            blobs_to_delete = search_blob_to_delete(args,
                                                    dataset_names,
                                                    subdataset_names,
                                                    mode="delete_all")
        else:
            blobs_to_delete = search_blob_to_delete(args,
                                                    dataset_names,
                                                    subdataset_names,
                                                    mode="delete_one")
    for each_blob_name in blobs_to_delete:
        blob_client = init_blob_client(args.azure_key, each_blob_name)
        blob_client.delete_blob()

def flush_and_upload(fout, args, azure_log_path):
    fout.flush()
    blob_client = init_blob_client(args.azure_key, local_file_name=azure_log_path)
    with open(azure_log_path, "rb") as fin:
        blob_client.upload_blob(fin, overwrite=True)

def flush_and_upload_prediction(args, result_path):
    blob_client = init_blob_client(args.azure_key, local_file_name=result_path)
    with open(result_path, "rb") as fin:
        blob_client.upload_blob(fin, overwrite=True)