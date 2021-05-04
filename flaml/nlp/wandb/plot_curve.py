import bisect

import wandb
import matplotlib.pyplot as plt
import numpy as np
from flaml.nlp.wandb.utils import get_all_runs

api = wandb.Api()

if __name__ == "__main__":

    task2ylim = {"mrpc":
                     {
                        "xlnet": [0.8, 0.88],
                         "albert": [0.82, 0.88],
                         "distilbert": [0.8, 0.88],
                         "deberta": [0.7, 0.9],
                         "funnel": [0.7, 0.9]
                     },
                "rte": {
                       "xlnet": [0.5, 0.75],
                        "albert": [0.65, 0.81],
                        "distilbert": [0.5, 0.65],
                        "deberta": [0.55, 0.81],
                        "funnel": [0.6, 0.81]
                },
                "cola": {
                   "xlnet": [0.0, 0.6],
                    "albert": [0.3, 0.65],
                    "distilbert": [0.1, 0.6],
                    "deberta": [0.3, 0.75],
                    "funnel": [0.1, 0.75]
                }
                }

    all_run_names = [("mrpc", "eval/accuracy"), ("rte", "eval/accuracy"), ("cola", "eval/matthews_correlation")]
    run_idx = 1

    tovar2model2ticks = {}
    tovar2model2reps = {}

    task2files = get_all_runs()
    #tovar2color = {"optuna": "blue", "blendsearch": "green"}
    tovar2color = {"gridunion": "blue", "generic": "green"}

    model2id = {}
    id2model = {}

    for run_idx in range(2, 3):
        all_runs = []
        task_name = all_run_names[run_idx][0]
        eval_name = all_run_names[run_idx][1]

        fig, axs = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
        fig.tight_layout()
        fig.suptitle(task_name, fontsize=14)

        all_files = task2files[task_name]
        print("downloading files for task " + task_name)
        for model_id in range(5):
            for space_id in range(2): #3, 2):
                for rep_id in range(3):
                    this_file = all_files[model_id][2][space_id][rep_id]
                    this_file.download(replace = True)
                    with open(this_file.name, "r") as fin:
                        this_group_name = fin.readline().rstrip(":\n")
                        all_runs.append(("glue_" + task_name, this_group_name, model_id))

        run_count = len(all_runs)
        print("there are " + str(run_count) + " runs ")
        for idx in range(0, run_count):
            proj_name = all_runs[idx][0]
            group_name = all_runs[idx][1]
            model_id = all_runs[idx][2]

            print("collecting data for the " + str(idx) + "th project")
            dim_to_var = group_name.split("_")[8]
            model = group_name.split("_")[2]
            fixed_algo = group_name.split("_")[4]
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

            tovar2model2reps.setdefault(dim_to_var, {})
            tovar2model2reps[dim_to_var].setdefault(model, [])
            tovar2model2reps[dim_to_var][model].append((xs, ys))
            tovar2model2ticks.setdefault(dim_to_var, {})
            tovar2model2ticks[dim_to_var].setdefault(model, set([]))
            tovar2model2ticks[dim_to_var][model].update(xs)

        for each_model in model2id.keys():
            for tovar in tovar2model2reps.keys():
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
                    avg_y = np.mean(all_ys)
                    std_y = np.std(all_ys)
                    means.append(avg_y)
                    stds.append(std_y)
                model_id = model2id[each_model]
                first_ax_id = int(model_id / 2)
                second_ax_id = model_id % 2
                line1, = axs[first_ax_id, second_ax_id].plot(sorted_ticks, means, color=tovar2color[tovar], label= "bo_" + tovar)
                axs[first_ax_id, second_ax_id].fill_between(sorted_ticks, np.subtract(means, stds), np.add(means, stds), color=tovar2color[tovar], alpha=0.2)
                axs[first_ax_id, second_ax_id].legend(loc=4)
        for model_id in range(max(id2model.keys()) + 1):
            first_ax_id = int(model_id / 2)
            second_ax_id = model_id % 2
            model = id2model[model_id]
            axs[first_ax_id, second_ax_id].set(ylabel='validation acc')
            axs[first_ax_id, second_ax_id].set_title(id2model[model_id])
            axs[first_ax_id, second_ax_id].axis(ymin=task2ylim[task_name][model][0],ymax=task2ylim[task_name][model][1])
        plt.show()