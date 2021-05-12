'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import argparse,os
import subprocess

from flaml.nlp import generate_result_csv
from flaml.nlp import plot_walltime_curve
from flaml.nlp import get_all_runs
from utils import get_wandb_azure_key
from run_autohf import dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"])
    arg_parser.add_argument('--azure_key', type=str, help='azure key', required=False)
    arg_parser.add_argument('--mode', type=str, help='analysis mode', required=True, choices=["summary", "analysis", "plot"])
    args = arg_parser.parse_args()

    wandb_key, args.azure_key = get_wandb_azure_key(os.path.abspath("../../"))
    task2blobs, bloblist = get_all_runs(args)

    if args.mode == "analysis":
        from flaml.nlp.result_analysis.analysis_modelsize import analysis_model_size
        subprocess.run(["wandb", "login", "--relogin", wandb_key])
        analysis_model_size(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes)
    elif args.mode == "summary":
        generate_result_csv(args, bloblist, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes)
    elif args.mode == "plot":
        plot_walltime_curve(args)