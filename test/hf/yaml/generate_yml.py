from shutil import copyfile

subdataset_names = ["rte", "mrpc", "cola", "sst2"]
is_first = True

copyfile("amlk8s_header.yml", "amlk8s.yml")
with open("amlk8s.yml", "a") as fout:
    for data_idx in range(0, 3):
        names = [subdataset_names[data_idx] + "_grid", subdataset_names[data_idx] + "_hpo0", subdataset_names[data_idx] + "_hpo1"]
        algos = ["grid_search_bert", "hpo", "hpo"]
        for name_idx in range(1): #len(names)):
            if data_idx == 0 or data_idx == 1:
                time_budget = 3600
            else:
                time_budget = 7200
            space_idx = name_idx - 1

            fout.write("- name: " + names[name_idx] + "\n")
            fout.write("  sku: 32G4\n")
            fout.write("  command:\n")
            fout.write("  - python run_autohf.py --server_name azureml --algo " + algos[name_idx] + " "
                       "--dataset_idx " + str(data_idx) + " --suffix " + names[name_idx] + " "
                       "--data_root_dir './data/' --sample_num 64 --time_budget " + str(time_budget))
            if algos[name_idx] == "hpo":
                fout.write(" --space_idx " + str(space_idx))
            fout.write("\n")
            if is_first:
                is_first = False
                fout.write("  submit_args: &retry_args\n    max_attempts: 0\n\n")
            else:
                fout.write("  submit_args:\n    <<: *retry_args\n\n")