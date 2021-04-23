from shutil import copyfile

subdataset_names = ["cola", "mrpc", "rte"]
is_first = True

copyfile("amlk8s_header.yml", "amlk8s.yml")
with open("amlk8s.yml", "a") as fout:
    for data_idx in range(0, 3):
        names = [subdataset_names[data_idx] + "_grid",
                 subdataset_names[data_idx] + "_hpo0",
                 subdataset_names[data_idx] + "_hpo10",
                 subdataset_names[data_idx] + "_hpo11"]
        algo_modes = ["grid_search_bert",
                 "hpo",
                 "hpo",
                 "hpo"]
        space_idxs = [None, 0, 1, 1]
        algo_idxs = [None, 0, 0, 1]
        if data_idx == 2 or data_idx == 1:
            time_budget = 3600
        else:
            time_budget = 7200
        for name_idx in range(3, len(names)):
            space_idx = space_idxs[name_idx]
            this_name = names[name_idx]
            this_algo_mode = algo_modes[name_idx]
            this_algo_idx = algo_idxs[name_idx]

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
            fout.write("\n")
            if is_first:
                is_first = False
                fout.write("  submit_args: &retry_args\n    max_attempts: 0\n\n")
            else:
                fout.write("  submit_args:\n    <<: *retry_args\n\n")