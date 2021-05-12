import matplotlib.pyplot as plt
import pathlib, re
from .utils import get_all_runs, init_blob_client
import wandb
import numpy as np
import bisect,subprocess

def plot_walltime_curve(args):
    subprocess.run(["wandb", "login", "--relogin", args.wandb_key])
    api = wandb.Api()
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