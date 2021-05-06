import pandas
import pathlib,re
from flaml.nlp.wandbazure.utils import get_all_runs
import pandas as pd
import argparse
from flaml.nlp.wandbazure.utils import extract_ts, compare_dates
from flaml.nlp.wandbazure.utils import get_all_runs, init_blob_client

all_tasks = ["glue_mrpc", "glue_rte", "glue_cola"]
search_algos = ["BlendSearch", "BlendSearch", "Optuna", "RandomSearch", "CFO"]
scheduler_names = ["None", "ASHA", "None", "ASHA", "None"]

hpo_searchspace_modes = ["generic", "gridunion"]

algo_space_to_summarize = [(4, 1)]

repid_max = 4
modelid_max = 5

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

def generate_result_csv(args, task2blobs):
    for task_id in range(len(all_tasks)):
        all_blobs = task2blobs[all_tasks[task_id]]
        tasktab = None
        is_first = True
        new_columnnames = []

        for each_algo_space_id in range(len(algo_space_to_summarize)):
            algo_id, space_id = algo_space_to_summarize[each_algo_space_id]
            subtab_name = search_algos[algo_id].lower() + "_" + scheduler_names[algo_id] + "_" + hpo_searchspace_modes[space_id]
            new_columnnames = new_columnnames + [subtab_name, "", "", "", "", "", ""]
            offset = each_algo_space_id * 7
            each_df = pd.DataFrame(
                index = [x for x in range(modelid_max * (repid_max + 1))],
                columns=[offset + x for x in range(5)])
            for model_id in range(modelid_max):
                for rep_id in range(repid_max):
                    insert_id = model_id * (repid_max + 1) + rep_id
                    try:
                        this_blob_file = all_blobs[model_id][algo_id][space_id][rep_id]
                        blob_client = init_blob_client(args.azure_key, this_blob_file)
                        pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", this_blob_file).group("parent_path")).mkdir(parents=True, exist_ok=True)
                        with open(this_blob_file, "wb") as fout:
                            fout.write(blob_client.download_blob().readall())
                    except:
                        break
                    with open(this_blob_file, "r") as fin:
                        alllines = fin.readlines()
                        result = alllines[5].rstrip("\n").split(",")
                        for x in range(5):
                            each_df[offset + x][insert_id] = result[x]

            each_df = each_df.reindex(columns=each_df.columns.tolist()
                                    + [offset + 5,
                                       offset + 6])
            if is_first:
                tasktab = each_df
                is_first = False
            else:
                tasktab = pandas.concat([tasktab, each_df], axis=1)

        tasktab.columns = new_columnnames
        tasktab.to_csv(all_tasks[task_id] + ".csv", index=False)