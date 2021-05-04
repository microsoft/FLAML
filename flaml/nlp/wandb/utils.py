import wandb, re, pathlib, os

def get_all_runs():
    api = wandb.Api()
    runs = api.runs("liususan/upload_file_azureml")
    task2files = {}
    filelist = []
    for file in runs[0].files():
        result = re.search(".*_model(?P<model_id>\d+)_(?P<algo_id>\d+)_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log", file.name)
        if result:
            task_name = file.name.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            task2files.setdefault(task_name, {})
            task2files[task_name].setdefault(model_id, {})
            task2files[task_name][model_id].setdefault(algo_id, {})
            task2files[task_name][model_id][algo_id].setdefault(space_id, {})
            task2files[task_name][model_id][algo_id][space_id][rep_id] = file
            filelist.append(file)
    return task2files, filelist

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

def get_wandpath(args, dataset_names, subdataset_names):
    full_dataset_name = dataset_names[args.dataset_idx] + "_" + subdataset_names[args.dataset_idx]
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

def search_file_to_delete(args, this_dataset_name, this_subdataset_name, mode="delete_one"):
    api = wandb.Api()
    runs = api.runs("liususan/upload_file_azureml")
    files_to_delete = []
    for file in runs[0].files():
        result = re.search(".*_model(?P<model_id>\d+)_(?P<algo_id>\d+)_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log",
                           file.name)
        if result:
            task_name = file.name.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            is_delete = False
            if mode == "delete_one":
                is_delete = task_name == this_dataset_name + "_" + this_subdataset_name \
                and model_id == args.pretrained_idx \
                and space_id == args.space_idx \
                and rep_id == args.rep_id \
                and algo_id == args.algo_idx
            elif mode == "delete_all":
                is_delete = task_name == this_dataset_name + "_" + this_subdataset_name \
                            and model_id == args.pretrained_idx \
                            and space_id == args.space_idx \
                            and algo_id == args.algo_idx
            if is_delete:
                files_to_delete.append(file)
    return files_to_delete