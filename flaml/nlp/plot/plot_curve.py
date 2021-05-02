import bisect

import wandb
import matplotlib.pyplot as plt
import numpy as np
import argparse, re
api = wandb.Api()

def get_all_runs():
    api = wandb.Api()
    runs = api.runs("liususan/upload_file_azureml")
    task2files = {}
    for file in runs[0].files():
        result = re.search(".*_model(?P<model_id>\d+)_0_(?P<space_id>\d+)_rep(?P<rep_id>\d+).log", file.name)
        if result:
            task_name = file.name.split("/")[1]
            model_id = int(result.group("model_id"))
            space_id = int(result.group("space_id"))
            rep_id = int(result.group("rep_id"))
            task2files.setdefault(task_name, {})
            task2files[task_name].setdefault(model_id, {})
            task2files[task_name][model_id].setdefault(space_id, {})
            task2files[task_name][model_id][space_id][rep_id] = file

    return task2files

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_name', type=str, help='scheduler name', required=False)

    task2ylim = {"mrpc": [0.8, 0.9], "rte": [0.6, 0.81], "cola": [0.4, 0.7]}

    all_run_names = [("mrpc", "eval/accuracy"), ("rte", "eval/accuracy"), ("cola", "eval/matthews_correlation")]
    run_idx = 1

    # space2ticks = {}
    # space2reps = {}

    space2model2ticks = {}
    space2model2reps = {}

    task2files = get_all_runs()
    space2color = {"gridunion": "blue", "generic": "green"}

    fig, axs = plt.subplots(1, 2)
    model2id = {}

    for run_idx in range(1, 2):
        all_runs = []
        task_name = all_run_names[run_idx][0]
        eval_name = all_run_names[run_idx][1]
        all_files = task2files[task_name]
        print("downloading files for task " + task_name)
        for model_id in range(2):
            for space_id in range(2):
                for rep_id in range(2):
                    this_file = all_files[model_id][space_id][rep_id]
                    this_file.download(replace = True)
                    with open(this_file.name, "r") as fin:
                        proj_name = fin.readline().rstrip(":\n")
                        all_runs.append(("glue_" + task_name, proj_name, model_id))

        run_count = len(all_runs)
        print("there are " + str(run_count) + " runs ")
        for idx in range(0, run_count):
            proj_name = all_runs[idx][0]
            group_name = all_runs[idx][1]
            model_id = all_runs[idx][2]

            print("collecting data for the " + str(idx) + "th project")
            space = group_name.split("_")[8]
            model = group_name.split("_")[2]
            ts2acc = {}
            model2id[model] = model_id

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

            space2model2reps.setdefault(space, {})
            space2model2reps[space].setdefault(model, [])
            space2model2reps[space][model].append((xs, ys))
            space2model2ticks.setdefault(space, {})
            space2model2ticks[space].setdefault(model, set([]))
            space2model2ticks[space][model].update(xs)

        for each_model in model2id.keys():
            for space in space2model2reps.keys():
                sorted_ticks = sorted(space2model2ticks[space][each_model])
                means = []
                stds = []
                for each_tick in sorted_ticks:
                    all_ys = []
                    for i in range(len(space2model2reps[space][each_model])):
                        xs = space2model2reps[space][each_model][i][0]
                        ys = space2model2reps[space][each_model][i][1]
                        if len(ys) == 0: continue
                        y_pos = max(0, min(bisect.bisect_left(xs, each_tick), len(ys) - 1))
                        this_y = ys[y_pos]
                        all_ys.append(this_y)
                    avg_y = np.mean(all_ys)
                    std_y = np.std(all_ys)
                    means.append(avg_y)
                    stds.append(std_y)
                model_id = model2id[each_model]
                line1, = axs[0, model_id].plot(sorted_ticks, means, color= space2color[space], label=space)
                axs[0, model_id].fill_between(sorted_ticks, np.subtract(means, stds), np.add(means, stds), color=space2color[space], alpha=0.2)
                axs[0, model_id].legend()
        for ax in axs.flat:
            ax.set(xlabel='wall clock time (s)', ylabel='validation acc')
            ax.title()
            ax.ylim(task2ylim[task_name][0], task2ylim[task_name][1])