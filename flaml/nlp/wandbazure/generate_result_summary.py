import pandas
import pathlib,re
from flaml.nlp.wandbazure.utils import get_all_runs
import pandas as pd
import argparse
from flaml.nlp.wandbazure.utils import extract_ts, compare_dates
from flaml.nlp.wandbazure.utils import get_all_runs, init_blob_client

# all_tasks = ["glue_mrpc", "glue_rte", "glue_cola"]
# search_algos = ["BlendSearch", "BlendSearch", "Optuna", "RandomSearch", "CFO"]
# scheduler_names = ["None", "ASHA", "None", "ASHA", "None"]
#
# hpo_searchspace_modes = ["generic", "gridunion"]

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

def generate_result_csv(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes):
    id2model = {}
    for task_id in range(3): #len(dataset_names)):
        full_name = dataset_names[task_id][0] + "_" + subdataset_names[task_id]
        all_blobs = task2blobs[full_name]
        tasktab = None
        arrayformulatab = None

        is_first = True
        new_columnnames = []

        for each_algo_space_id in range(len(algo_space_to_summarize)):
            algo_id, space_id = algo_space_to_summarize[each_algo_space_id]
            subtab_name = search_algos[algo_id].lower() + "_" + scheduler_names[algo_id] + "_" + hpo_searchspace_modes[space_id]
            new_columnnames = new_columnnames + [subtab_name, "", "", "", "", "", ""]
            offset = each_algo_space_id * 7
            each_df = pd.DataFrame(
                index = [x for x in range((modelid_max) * (repid_max + 1))],
                columns=[offset + x for x in range(5)])
            each_arrayforula = pd.DataFrame(index=[0], columns=[offset + x for x in range(5)])
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

            arrayformula_indices = [offset + 3, offset + 4]
            for each_index in arrayformula_indices:
                first_part = int((each_index + COLUMN_OFFSET + 1) / 26)
                second_part = (each_index + COLUMN_OFFSET + 1) % 26
                first_char = "" if first_part == 0 else chr(first_part + 64)
                second_char = chr(65 + second_part)
                arrayformula_col = first_char + second_char
                arrayformula_row1 = 2 + ROW_OFFSET
                arrayformula_row2 = ROW_OFFSET + (modelid_max) * (repid_max + 1)
                start_cell = arrayformula_col + str(arrayformula_row1)
                end_cell = arrayformula_col + str(arrayformula_row2)
                each_arrayforula[each_index] = "=average(" + start_cell + ":" + end_cell + ")"

            each_df = each_df.reindex(columns=each_df.columns.tolist()
                                    + [offset + 5,
                                       offset + 6])
            each_arrayforula = each_arrayforula.reindex(columns = each_arrayforula.columns.tolist() + [offset + 5, offset + 6])

            if is_first:
                tasktab = each_df
                arrayformulatab = each_arrayforula
                is_first = False
            else:
                tasktab = pandas.concat([tasktab, each_df], axis=1)
                arrayformulatab = pandas.concat([arrayformulatab, each_arrayforula], axis = 1)

        tasktab.index = [pretrained_models[int (x / (repid_max + 1))][0] if x %(repid_max + 1) == 0 else "" for x in range(modelid_max * (repid_max + 1))]
        tasktab.columns = new_columnnames
        tasktab.to_csv(full_name + ".csv", index=True)
        arrayformulatab.to_csv(full_name + "_arrayformula.csv", index = False)