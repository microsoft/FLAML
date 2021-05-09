import pandas
import pathlib,re
from flaml.nlp.wandbazure.utils import get_all_runs
import pandas as pd
import argparse
from flaml.nlp.wandbazure.utils import extract_ts, compare_dates
from flaml.nlp.wandbazure.utils import get_all_runs, init_blob_client

algo_space_to_summarize = [(0, 1), (2, 1), (4, 1)]

repid_max = 4
modelid_max = 5

COLUMN_OFFSET=ROW_OFFSET=1

def remove_by_date(tasklist, earliest_ts):
    earliest_time = extract_ts(earliest_ts)
    for each_file in tasklist:
        try:
            each_file.download(replace=True)
        except:
            continue
        with open(each_file.name, "r") as fin:
            alllines = fin.readlines()
            this_time = extract_ts(alllines[1])
        if compare_dates(this_time, earliest_time) == -1:
            each_file.delete()

def generate_result_csv(args, bloblist, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes):

    tab = pd.DataFrame(
        columns=["full_dataset", "rep_id", "space", "init_config", "wandb_hash", "model", "model_size", "algorithm", "pruner", "sample_num", "time", "val_acc", "test_acc"]
        , index=[x for x in range(len(bloblist))])
    for blob_id in range(len(bloblist)):
        blobname = bloblist[blob_id]
        result = re.search(".*_model(?P<model_id>\d+)_(?P<algo_id>\d+)_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log",
                           blobname)
        model_id = int(result.group("model_id"))
        space_id = int(result.group("space_id"))
        rep_id = int(result.group("rep_id"))
        algo_id = int(result.group("algo_id"))
        full_dataset = blobname.split("/")[1]

        try:
            blob_client = init_blob_client(args.azure_key, blobname)
            pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", blobname).group("parent_path")).mkdir(
                parents=True, exist_ok=True)
            with open(blobname, "wb") as fout:
                fout.write(blob_client.download_blob().readall())
        except:
            break
        with open(blobname, "r") as fin:
            alllines = fin.readlines()

            space = hpo_searchspace_modes[space_id]
            init_config = search_algo_args_modes[space_id]
            if len(alllines) < 5: continue
            wandb_hash = alllines[0].rstrip("\n:").split("_")[-1]
            model = pretrained_models[model_id][0]
            model_size = pretrained_models[model_id][1]
            algo = search_algos[algo_id]
            pruner = scheduler_names[algo_id]
            sample_num = int(alllines[5].rstrip("\n").split(",")[1])
            time = float(alllines[5].rstrip("\n").split(",")[2])
            val_acc = float(alllines[5].rstrip("\n").split(",")[3])
            test_acc = float(alllines[5].rstrip("\n").split(",")[4])

            tab.loc[blob_id] = [full_dataset, rep_id, space, init_config, wandb_hash, model, model_size, algo, pruner, sample_num, time, val_acc, test_acc]

    tab.to_csv("result.csv", index = False)