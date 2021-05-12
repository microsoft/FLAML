import bisect
import argparse
import pathlib
import re

import wandb
import matplotlib.pyplot as plt
import numpy as np
from flaml.nlp.result_analysis.utils import get_all_runs, init_blob_client
from utils import get_wandb_azure_key

api = wandb.Api()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"])
    arg_parser.add_argument('--azure_key', type=str, help='azure key', required=False)
    args = arg_parser.parse_args()

    wandb_key, args.azure_key = get_wandb_azure_key()

    task2ylim = {"glue_mrpc":
                     {
                        "xlnet": [0.82, 0.86],
                         "albert": [0.84, 0.87],
                         "distilbert": [0.83, 0.86],
                         "deberta": [0.85, 0.9],
                         "funnel": [0.85, 0.9]
                     },
                "glue_rte": {
                       "xlnet": [0.65, 0.75],
                        "albert": [0.74, 0.81],
                        "distilbert": [0.6, 0.68],
                        "deberta": [0.6, 0.81],
                        "funnel": [0.75, 0.81]
                },
                "glue_cola": {
                   "xlnet": [0.2, 0.6],
                    "albert": [0.45, 0.65],
                    "distilbert": [0.45, 0.6],
                    "deberta": [0.5, 0.75],
                    "funnel": [0.6, 0.75]
                }
                }

    all_run_names = [("glue_mrpc", "eval/accuracy"), ("glue_rte", "eval/accuracy"), ("glue_cola", "eval/matthews_correlation")]
    run_idx = 1

    tovar2model2ticks = {}
    tovar2model2reps = {}

    task2blobs, _ = get_all_runs(args)
    tovar2color_list = ["blue", "red", "green", "black"]
    tovar2color = {}
    tovar_idx = 0

    model2id = {}
    id2model = {}

    resplit_id = 1
    for run_idx in range(1, 2):
        all_runs = []
        task_name = all_run_names[run_idx][0]
        eval_name = all_run_names[run_idx][1]

        fig, axs = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
        fig.tight_layout()
        fig.suptitle(task_name, fontsize=14)

        all_blobs = task2blobs[resplit_id][task_name]
        print("downloading files for task " + task_name)
        fixed_var = ""
        #try:
        for model_id in range(6, 7):
            for algo_id in [-1] + [x for x in range(2, 3, 2)]:
                for space_id in range(1, 2):
                    for rep_id in range(1):
                        if algo_id == -1:
                            this_blob_file = all_blobs[model_id][algo_id]
                        else:
                            this_blob_file = all_blobs[model_id][algo_id][space_id][rep_id]
                        blob_client = init_blob_client(args.azure_key, this_blob_file)
                        pathlib.Path(re.search("(?P<parent_path>^.*)/[^/]+$", this_blob_file).group("parent_path")).mkdir(
                            parents=True, exist_ok=True)
                        with open(this_blob_file, "wb") as fout:
                            fout.write(blob_client.download_blob().readall())
                        with open(this_blob_file, "r") as fin:
                            this_group_name = fin.readline().rstrip(":\n")
                            all_runs.append((task_name, this_group_name, model_id))

        run_count = len(all_runs)
        print("there are " + str(run_count) + " runs ")
        for idx in range(0, run_count):
            proj_name = all_runs[idx][0]
            group_name = all_runs[idx][1]
            model_id = all_runs[idx][2]

            print("collecting data for the " + str(idx) + "th project")
            dims_to_var = [group_name.split("_")[4], group_name.split("_")[8]]
            model = group_name.split("_")[2]
            #fixed_var = group_name.split("_")[8]
            ts2acc = {}
            model2id[model] = model_id
            id2model[model_id] = model

            runs = api.runs('liususan/' + proj_name, filters={"group": group_name})
            is_this_run_recorded = False
            for idx in range(0, len(runs)):
                run=runs[idx]
                for i, row in run.history().iterrows():
                    try:
                        if not np.isnan(row[eval_name]):
                            ts = row["_timestamp"]
                            acc = row[eval_name]
                            is_this_run_recorded = True
                            ts2acc.setdefault(ts,  [])
                            ts2acc[ts].append(acc)
                    except KeyError:
                        pass
            sorted_ts = sorted(ts2acc.keys())
            max_acc_sofar = 0
            xs = []
            ys = []
            for each_ts in sorted_ts:
                max_acc_ts = max(ts2acc[each_ts])
                max_acc_sofar = max(max_acc_sofar, max_acc_ts)
                xs.append(each_ts - sorted_ts[0])
                ys.append(max_acc_sofar)

            to_var_str = "_".join(dims_to_var)
            tovar2model2reps.setdefault(to_var_str, {})
            tovar2model2reps[to_var_str].setdefault(model, [])
            tovar2model2reps[to_var_str][model].append((xs, ys))
            tovar2model2ticks.setdefault(to_var_str, {})
            tovar2model2ticks[to_var_str].setdefault(model, set([]))
            tovar2model2ticks[to_var_str][model].update(xs)

        model2bounds = {}
        for each_model in model2id.keys():
            ys_all_methods = []
            tovar_idx = 0
            tovar2color = {}

            for tovar in tovar2model2reps.keys():
                try:
                    sorted_ticks = sorted(tovar2model2ticks[tovar][each_model])
                    means = []
                    stds = []
                    for each_tick in sorted_ticks:
                        all_ys = []
                        for i in range(len(tovar2model2reps[tovar][each_model])):
                            xs = tovar2model2reps[tovar][each_model][i][0]
                            ys = tovar2model2reps[tovar][each_model][i][1]
                            if len(ys) == 0: continue
                            y_pos = max(0, min(bisect.bisect_left(xs, each_tick), len(ys) - 1))
                            this_y = ys[y_pos]
                            all_ys.append(this_y)
                            ys_all_methods.append(this_y)
                        avg_y = np.mean(all_ys)
                        std_y = np.std(all_ys)
                        means.append(avg_y)
                        stds.append(std_y)
                    model_id = model2id[each_model] - 5
                    first_ax_id = int(model_id / 2)
                    second_ax_id = model_id % 2
                    try:
                        this_color = tovar2color[tovar]
                    except KeyError:
                        tovar2color[tovar] = tovar2color_list[tovar_idx]
                        tovar_idx += 1
                        this_color = tovar2color[tovar]
                    line1, = axs[first_ax_id, second_ax_id].plot(sorted_ticks, means, color=this_color,
                                                                 label=tovar)
                    axs[first_ax_id, second_ax_id].fill_between(sorted_ticks, np.subtract(means, stds), np.add(means, stds), color= this_color, alpha=0.2)
                    axs[first_ax_id, second_ax_id].legend(loc=4)
                except KeyError:
                    pass
            sorted_ys_all_methods = sorted(ys_all_methods)
            upper = sorted_ys_all_methods[-int(0.01 * len(sorted_ys_all_methods))] + 0.05
            lower = sorted_ys_all_methods[int(0.03 * len(sorted_ys_all_methods))]
            model2bounds[each_model] = [lower, upper]
        for model_id in id2model.keys():
            first_ax_id = int((model_id - 5) / 2)
            second_ax_id = (model_id - 5) % 2
            model = id2model[model_id]
            axs[first_ax_id, second_ax_id].set(ylabel='validation acc')
            axs[first_ax_id, second_ax_id].set_title(id2model[model_id])
            axs[first_ax_id, second_ax_id].axis(ymin=model2bounds[model][0],ymax=model2bounds[model][1])
        plt.savefig(task_name + ".png")
        plt.show()