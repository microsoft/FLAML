import bisect

import wandb
import matplotlib.pyplot as plt
import numpy as np
import argparse
api = wandb.Api()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_name', type=str, help='scheduler name', required=False)

    mrpc = [
        ("glue_mrpc", "glue_mrpc_xlnet_base_blendsearch_none_hpo_space_generic_17sgijnj"),
        ("glue_mrpc", "glue_mrpc_albert_small_blendsearch_none_hpo_space_generic_252gex3m"),
        ("glue_mrpc", "glue_mrpc_xlnet_base_blendsearch_none_hpo_space_gridunion_other_22xw4ni2"),
        ("glue_mrpc", "glue_mrpc_albert_small_blendsearch_none_hpo_space_gridunion_other_3r5hfdef"),
    ]

    all_runs = [("mrpc", mrpc, "eval/acc")]

    run_idx = 1

    space2ticks = {}
    space2reps = {}

    for run_idx in range(0, 1):
        task_name = all_runs[run_idx][0]
        eval_name = all_runs[run_idx][2]
        run_count = len(all_runs[run_idx][1])

        for idx in range(0, run_count):
            proj_name = all_runs[run_idx][1][idx][0]
            group_name = all_runs[run_idx][1][idx][1]

            print("collecting data for " + proj_name)
            space = group_name.split("_")[8]
            ts2acc = {}

            runs = api.runs('liususan/' + proj_name, filters={"group": group_name})
            for idx in range(0, len(runs)):
                run=runs[idx]
                for i, row in run.history().iterrows():
                    if not np.isnan(row[eval_name]):
                        ts = row["_timestamp"]
                        acc = row[eval_name]
                        ts2acc.setdefault(ts,  [])
                        ts2acc[ts].append(acc)
            sorted_ts = sorted(ts2acc.keys())
            max_acc_sofar = 0
            xs = []
            ys = []
            for each_ts in sorted_ts:
                max_acc_ts = max(ts2acc[each_ts])
                max_acc_sofar = max(max_acc_sofar, max_acc_ts)
                xs.append(each_ts - sorted_ts[0])
                ys.append(max_acc_sofar)

            space2reps.setdefault(space, [])
            space2reps[space].append((xs, ys))
            space2ticks.setdefault(space, set([]))
            space2ticks[space].update(xs)

        plt.figure(figsize=(10, 6))

        space2color={"rs": "blue", "optuna": "green", "grid": "red"}

        for algo in space2reps.keys():
            if len(space2reps[algo]) == 1:
                plt.plot([4*x for x in space2reps[algo][0][0]], space2reps[algo][0][1], color=space2color[algo], label=algo)
            else:
                sorted_ticks = sorted(space2ticks[algo])
                means = []
                stds = []
                for each_tick in sorted_ticks:
                    all_ys = []
                    for i in range(len(space2reps[algo])):
                        xs = space2reps[algo][i][0]
                        ys = space2reps[algo][i][1]
                        if len(ys) == 0: continue
                        y_pos = max(0, min(bisect.bisect_left(xs, each_tick), len(ys) - 1))
                        this_y = ys[y_pos]
                        all_ys.append(this_y)
                    avg_y = np.mean(all_ys)
                    std_y = np.std(all_ys)
                    means.append(avg_y)
                    stds.append(std_y)
                line1, = plt.plot(sorted_ticks, means, color= space2color[algo], label=algo)
                plt.fill_between(sorted_ticks, np.subtract(means, stds), np.add(means, stds), color=space2color[algo], alpha=0.2)
                plt.legend()
        plt.xlabel("wall clock time (s)")
        plt.ylabel("validation acc")
        plt.title(task_name + "-deberta")
        #plt.legend(loc=2)
        plt.ylim(0.4, 0.7)
        plt.show()
        stop = 0