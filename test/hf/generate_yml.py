from shutil import copyfile
import os, sys
from datetime import datetime

subdataset_names = ["cola", "mrpc", "rte"]
is_first = True

now = datetime.now()
suffix = sys.argv[1] + "_" + now.strftime("%m%d%H%M")

copyfile("./yml_files/amlk8s_header.yml", "./yml_files/amlk8s_" + str(suffix) + ".yml")
with open("./yml_files/amlk8s_" + str(suffix) + ".yml", "a") as fout:
        for data_idx in range(0, 3):
            names = [subdataset_names[data_idx] + "_hpo1" + str(x) + "_" + sys.argv[1] for x in range(5)]
            algo_modes = ["hpo"] * 5
            space_idxs = [1] * 5
            algo_idxs = [2] * 5
            pretrained_idxs = [0, 1, 2, 3, 4]
            if data_idx == 2 or data_idx == 1:
                time_budget = 3600
            else:
                time_budget = 7200
            for name_idx in range(0, len(names)):
                space_idx = space_idxs[name_idx]
                this_name = names[name_idx]
                this_algo_mode = algo_modes[name_idx]
                this_algo_idx = algo_idxs[name_idx]
                this_pretrained_idx = pretrained_idxs[name_idx]

                fout.write("- name: " + this_name + "\n")
                fout.write("  sku: 32G4\n")
                fout.write("  command:\n")
                fout.write("  - python run_autohf.py --server_name azureml --algo " + this_algo_mode + " "
                           "--dataset_idx " + str(data_idx) + " --suffix " + this_name + " "
                           "--data_root_dir './data/' --sample_num 100000 --time_budget " + str(time_budget))
                if space_idx is not None:
                    fout.write(" --space_idx " + str(space_idx))
                if this_algo_idx is not None:
                    fout.write(" --algo_idx " + str(this_algo_idx))
                if this_pretrained_idx is not None:
                    fout.write(" --pretrained_idx " + str(this_pretrained_idx))
                fout.write("\n")
                if is_first:
                    is_first = False
                    fout.write("  submit_args: &retry_args\n    max_attempts: 0\n\n")
                else:
                    fout.write("  submit_args:\n    <<: *retry_args\n\n")