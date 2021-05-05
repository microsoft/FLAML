import pandas
import wandb
from flaml.nlp.wandbazure.utils import get_all_runs
import pandas as pd
import argparse
from flaml.nlp.wandbazure.utils import extract_ts, compare_dates

all_tasks = ["mrpc", "rte", "cola"]
search_algos = ["BlendSearch", "BlendSearch", "Optuna", "RandomSearch", "CFO"]
scheduler_names = ["None", "ASHA", "None", "ASHA", "None"]

hpo_searchspace_modes = ["generic", "gridunion"]

algo_space_to_summarize = [(0, 0), (0, 1), (2, 0), (2, 1), (4, 1)]

repid_max = 4
modelid_max = 5

api = wandb.Api()

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

def generate_result_csv(task2blobs):
    for task_id in range(len(all_tasks)):
        all_files = task2blobs[all_tasks[task_id]]
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
                        this_file = all_files[model_id][algo_id][space_id][rep_id]
                        this_file.download(replace=True)
                    except:
                        break
                    with open(this_file.name, "r") as fin:
                        alllines = fin.readlines()
                        this_group_name = alllines[0].rstrip(":\n")
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

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"])
    args = arg_parser.parse_args()
    task2blobs, tasklist = get_all_runs(args)

    generate_result_csv(task2blobs)
