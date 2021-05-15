import pandas
import pathlib,re
from .utils import get_all_runs
import pandas as pd
import argparse
from .utils import extract_ts, compare_dates
from .utils import get_all_runs, init_blob_client

algo_space_to_summarize = [(0, 1), (2, 1), (4, 1)]

repid_max = 4
modelid_max = 5

COLUMN_OFFSET=ROW_OFFSET=1

def generate_result_csv(args, bloblist, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, split_modes):

    tab = pd.DataFrame(
        columns=["full_dataset", "rep_id", "space", "init_config", "wandb_hash", "model", "model_size", "algorithm", "pruner", "sample_num", "time", "split_mode", "yml_file", "val_acc", "test_acc"]
        , index=[x for x in range(len(bloblist))])
    blobname2ymlfile = {}
    for blob_id in range(len(bloblist)):
        blobname = bloblist[blob_id]
        result_grid = re.search(".*_mod(el)?(?P<model_id>\d+)_None_None(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log", blobname)
        if result_grid:
            model_id = int(result_grid.group("model_id"))
            rep_id = int(result_grid.group("rep_id"))
            try:
                split_id = int(result_grid.group("split_id"))
            except:
                split_id = 0
            algo = "grid"
            space = "None"
            init_config = "None"
            pruner = "None"
            split_mode = split_modes[split_id]
        else:
            result = re.search(".*_mod(el)?(?P<model_id>\d+)_(alg)?(?P<algo_id>\d+)_(spa)?(?P<space_id>\d+)(_spt(?P<split_id>\d+))?_rep(?P<rep_id>\d+).log", blobname)
            model_id = int(result.group("model_id"))

            space_id = int(result.group("space_id"))
            space = hpo_searchspace_modes[space_id]
            init_config = search_algo_args_modes[space_id]
            rep_id = int(result.group("rep_id"))
            algo_id = int(result.group("algo_id"))
            algo = search_algos[algo_id]
            pruner = scheduler_names[algo_id]
            try:
                split_id = int(result.group("split_id"))
            except:
                split_id = 0
            split_mode = split_modes[split_id]
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
            if len(alllines) < 5: continue
            wandb_hash = alllines[0].rstrip("\n:").split("_")[-1]
            model = pretrained_models[model_id][0]
            model_size = pretrained_models[model_id][1]
            if len(alllines) > 7 and alllines[7].rstrip("\n") != "" and alllines[7].rstrip("\n").endswith("yml"):
                yml_file = alllines[7].rstrip("\n")
            else:
                yml_file = None
            sample_num = int(alllines[5].rstrip("\n").split(",")[1])
            time = float(alllines[5].rstrip("\n").split(",")[2])
            val_acc = float(alllines[5].rstrip("\n").split(",")[3])
            if split_id == 1 or alllines[5].rstrip("\n").split(",")[4] == "":
                test_acc = -1
            else:
                test_acc = float(alllines[5].rstrip("\n").split(",")[4])

            if rep_id == 0 or algo == "grid":
                blobname2ymlfile[blobname] = yml_file
                tab.loc[blob_id] = [full_dataset, rep_id, space, init_config, wandb_hash, model, model_size, algo, pruner, sample_num, time, split_mode, str(yml_file), val_acc, test_acc]
            else:
                key_str = re.sub("(.*)rep(?P<rep_id>\d).log", r"\1rep0.log", blobname)
                try:
                    rep0ymlfile = blobname2ymlfile[key_str]
                    if yml_file == rep0ymlfile:
                        tab.loc[blob_id] = [full_dataset, rep_id, space, init_config, wandb_hash, model, model_size, algo,
                                            pruner, sample_num, time, split_mode, str(yml_file), val_acc, test_acc]
                except KeyError:
                    pass

    nan_value = float("NaN")
    tab.replace("", nan_value, inplace=True)
    tab.dropna(inplace=True)
    tab.to_csv("result.csv", index = False)